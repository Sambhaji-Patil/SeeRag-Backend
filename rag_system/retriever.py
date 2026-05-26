"""
Retrieval Strategies:
1. VECTOR - pure cosine similarity on FAISS
2. BM25 - sparse keyword match (great for exact terms)
3. HYBRID - linear combination of BM25 + vector scores (RRF)
4. MMR - Maximal Marginal Relevance for diversity
5. RERANKER - cross-encoder reranking of initial retrieval pool
6. PARENT-CHILD - expand narrow child chunk -> surrounding parent context
"""
import logging
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .config import get_settings
from .embeddings import embed_query, cosine_similarity
from .vector_store import similarity_search_with_scores, get_store

logger = logging.getLogger(__name__)
settings = get_settings()

#BM25 corpus cache per collection (rebuilt on first retrieval)
_bm25_cache: dict[str, tuple[BM25Okapi, list[Document]]] = {}

#BM25

def _get_bm25(collection: str) -> tuple[BM25Okapi, list[Document]]:
    """Build or retrieve cached BM25 index from FAISS doc store"""
    if collection not in _bm25_cache:
        store = get_store(collection)
        if store is None:
            raise ValueError(f"Collection '{collection}' not loaded")
        all_docs = list(store.docstore._dict.values())
        tokenised = [d.page_content.lower().split() for d in all_docs]
        _bm25_cache[collection] = (BM25Okapi(tokenised),all_docs)
        logger.info(f"Built BM25 index for '{collection}' ({len(all_docs)} docs)")
    return _bm25_cache[collection]

def bm25_retrieve(query: str, collection: str,k: int) -> list[tuple[Document,float]]:
    bm25, docs = _get_bm25(collection)
    scores = bm25.get_scores(query.lower().split())
    top_idx = np.argsort(scores)[::-1][:k] #select top k scores
    results = [(docs[i],float(scores[i])) for i in top_idx if scores[i] > 0]
    return results

#Reciprocal Rank Fusion
def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank + 1) #here 1 is added to handle rank 1 which here comes as 0


def _resolve_hybrid_weights(
    bm25_weight: Optional[float],
    vector_weight: Optional[float],
) -> tuple[float, float]:
    bw = settings.bm25_weight if bm25_weight is None else bm25_weight
    vw = settings.vector_weight if vector_weight is None else vector_weight
    total = bw + vw
    if total <= 0:
        bw = settings.bm25_weight
        vw = settings.vector_weight
        total = bw + vw
    return bw / total, vw / total

#Hybrid Retrieval
def hybrid_retrieve(
    query: str,
    collection: str,
    k: int,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
) -> list[tuple[Document,float]]:
    """Reciprocal Rank fusion of BM25 and Vector results"""
    bw, vw = _resolve_hybrid_weights(bm25_weight, vector_weight)
    pool_size = k*3 #casting a wide net before fusing
    vec_results = similarity_search_with_scores(query,collection,k=pool_size)
    bm25_results = bm25_retrieve(query,collection,k=pool_size)

    rrf_scores: dict[str,float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(vec_results):
        did = doc.metadata.get("doc_id",id(doc))
        rrf_scores[did] = rrf_scores.get(did,0) + vw * _rrf_score(rank) #check the existing score first and then add the fresh score
        doc_map[did] = doc
    
    for rank, (doc, _) in enumerate(bm25_results):
        did = doc.metadata.get("doc_id",id(doc))
        rrf_scores[did] = rrf_scores.get(did,0) + bw * _rrf_score(rank) 
        doc_map[did] = doc
    
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
    return [(doc_map[did],rrf_scores[did]) for did in sorted_ids]

#MMR
async def mmr_retrieve(
    query: str,
    collection: str,
    k: int,
    lambda_mult: Optional[float] = None,
    fetch_k: Optional[int] = None,
) -> list[tuple[Document, float]]:
    """
    Maximal Marginal Relevance: balance relevance vs diversity.
    """
    # 1. Setup parameters
    lam = lambda_mult or settings.mmr_lambda
    fetch_k = fetch_k or settings.top_k_retrieval
    store = get_store(collection)

    # 2. Get original scores to map them back later
    # LangChain's MMR method doesn't return scores by default
    pool = await store.asimilarity_search_with_relevance_scores(query, k=fetch_k)
    score_map = {doc.metadata.get("doc_id"): score for doc, score in pool}

    # 3. Perform the actual MMR search
    mmr_docs = await store.amax_marginal_relevance_search(
        query, 
        k=k, 
        fetch_k=fetch_k, 
        lambda_mult=lam
    )

    # 4. Re-attach scores and return
    return [
        (doc, score_map.get(doc.metadata.get("doc_id"), 0.0))
        for doc in mmr_docs
    ]

#cross encoder reranker
class CrossEncoderReranker:
    """
    Lightweight reranker using a sentence-transformer cross-encoder.
    Scores (query, passage) pairs directly — much more accurate than
    bi-encoder similarity for final-stage ranking.

    Falls back gracefully if sentence-transformers is not installed.
    """
    def __init__(self,model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(model_name)
            self.available = True
            logger.info(f"Cross-encoder loaded: {model_name}")
        except ImportError:
            self.available = False
            logger.warning("sentence-transformers not installed - reranking disabled")

    def rerank(
        self,
        query: str,
        docs: list[tuple[Document,float]],
        top_k: int,
    ) -> list[tuple[Document,float]]:
        if not self.available or not docs:
            return docs[:top_k]
        pairs = [(query, d.page_content) for d, _ in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs,scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for (doc, _), score in ranked[:top_k]]
    
_reranker = CrossEncoderReranker()

#Parent child context expansion
def expand_to_parent_context(
    docs: list[tuple[Document,float]],
    collection: str
) -> list[tuple[Document, float]]:
    """
    For each retrieved child chunk, stitch in prev/next chunks if available.
    This gives the LLM wider context around the matched passage without
    embedding huge parent documents.
    """
    store = get_store(collection)
    if store is None:
        return docs
    
    all_stored = store.docstore._dict #{faiss_id: Document}
    by_doc_id = {d.metadata.get("doc_id"): d for d in all_stored.values()}

    expanded = []
    seen_ids: set[str] = set()

    for doc, score in docs:
        did = doc.metadata.get("doc_id")
        if did in seen_ids:
            continue
        seen_ids.add(did)

        prev_id = doc.metadata.get("prev_chunk_id")
        next_id = doc.metadata.get("next_chunk_id")

        #stitch surrounding context
        parts = []
        if prev_id and prev_id in by_doc_id:
            parts.append(by_doc_id[prev_id].page_content)
        parts.append(doc.page_content)
        if next_id and next_id in by_doc_id:
            parts.append(by_doc_id[next_id].page_content)

        expanded_doc = Document(
            page_content="\n".join(parts),
            metadata=doc.metadata,
        )
        expanded.append((expanded_doc, score))
    
    return expanded

#Main retrieval entrypoint
async def retrieve(
    query: str,
    collection: str = "default",
    mode: str = "hybrid",
    top_k: Optional[int] = None,
    top_k_retrieval: Optional[int] = None,
    mmr_lambda: Optional[float] = None,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
    use_reranker: bool = True,
    expand_context: bool = True,
) -> list[tuple[Document,float]]:
    k_retrieve = top_k_retrieval or settings.top_k_retrieval
    k_final = top_k or settings.top_k_rerank

    if mode == "vector":
        results = similarity_search_with_scores(query,collection,k=k_retrieve)
    elif mode == "bm25":
        results = bm25_retrieve(query,collection,k=k_retrieve)
    elif mode == "mmr":
        results = await mmr_retrieve(query,collection,k=k_final,lambda_mult=mmr_lambda,fetch_k=k_retrieve)
        return results # MMR alrady handles diversity , skip reranker
    else: #go with hybrid
        results = hybrid_retrieve(
            query,
            collection,
            k=k_retrieve,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )

    #Rerank
    if use_reranker:
        results = _reranker.rerank(query,results,top_k=k_final)
    else:
        results = results[:k_final]
    
    # Expand to surrounding context(Parent-Child)
    if expand_context:
        results = expand_to_parent_context(results,collection)
    
    return results

import re as _re

_COMPARISON_RE = _re.compile(
    r"\b(compare|comparison|contrast|difference|differ|both|versus|\bvs\b|between|across|"
    r"each\s+(document|doc|file)|all\s+(documents?|docs?|files?)|"
    r"what\s+do\s+(both|all)|how\s+do\s+.{0,20}(differ|compare))\b",
    _re.IGNORECASE,
)


def detect_query_scope(query: str, collections: list[str]) -> list[str]:
    """
    Returns which sub-collections to search for this query.
    Tier 1 — comparison keywords → all collections.
    Tier 2 — explicit doc name in query → matched collection(s).
    Default → all collections (safest fallback).
    """
    if len(collections) <= 1:
        return collections

    if _COMPARISON_RE.search(query):
        return collections

    query_lower = query.lower()
    matched = [
        c for c in collections
        if c.split("__")[-1].replace("_", " ").replace("-", " ").lower() in query_lower
    ]
    return matched if matched else collections


async def multi_collection_retrieve(
    query: str,
    collections: list[str],
    mode: str = "hybrid",
    k_per_collection: int = 5,
    top_k_retrieval: Optional[int] = None,
    mmr_lambda: Optional[float] = None,
    bm25_weight: Optional[float] = None,
    vector_weight: Optional[float] = None,
    use_reranker: bool = True,
    expand_context: bool = True,
) -> list[tuple[Document, float]]:
    """
    Retrieves from each collection independently, guaranteeing each document gets
    at least k_per_collection candidates before the pool is reranked.
    """
    pool: list[tuple[Document, float]] = []

    for coll in collections:
        try:
            results = await retrieve(
                query=query,
                collection=coll,
                mode=mode,
                top_k=k_per_collection,
                top_k_retrieval=top_k_retrieval,
                mmr_lambda=mmr_lambda,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
                use_reranker=False,   # defer reranking until after merge
                expand_context=expand_context,
            )
            pool.extend(results)
        except Exception as e:
            logger.warning(f"Retrieval skipped for '{coll}': {e}")

    if not pool:
        return []

    # Deduplicate by doc_id
    seen: set[str] = set()
    deduped: list[tuple[Document, float]] = []
    for doc, score in pool:
        did = str(doc.metadata.get("doc_id", id(doc)))
        if did not in seen:
            seen.add(did)
            deduped.append((doc, score))

    k_final = k_per_collection * len(collections)
    if use_reranker and _reranker.available:
        return _reranker.rerank(query, deduped, top_k=k_final)
    return deduped[:k_final]

print("[retriever] Module ready")