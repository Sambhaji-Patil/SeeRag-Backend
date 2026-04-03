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
from langchain.schema import Document
from rank_bm25 import BM25Okapi

from config import get_settings
from embeddings import embed_query, cosine_similarity
from vector_store import similarity_search_with_scores, get_store

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

#Hybrid Retrieval
def hybrid_retrieve(
    query: str,
    collection: str,
    k: int
) -> list[tuple[Document,float]]:
    """Reciprocal Rank fusion of BM25 and Vector results"""
    pool_size = k*3 #casting a wide net before fusing
    vec_results = similarity_search_with_scores(query,collection,k=pool_size)
    bm25_results = bm25_retrieve(query,collection,k=pool_size)

    rrf_scores: dict[str,float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(vec_results):
        did = doc.metadata.get("doc_id",id(doc))
        rrf_scores[did] = rrf_scores.get(did,0) + settings.vector_weight * _rrf_score(rank) #check the existing score first and then add the fresh score
        doc_map[did] = doc
    
    for rank, (doc, _) in enumerate(bm25_results):
        did = doc.metadata.get("doc_id",id(doc))
        rrf_scores[did] = rrf_scores.get(did,0) + settings.bm25_weight * _rrf_score(rank) 
        doc_map[did] = doc
    
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:k]
    return [(doc_map[did],rrf_scores[did]) for did in sorted_ids]
