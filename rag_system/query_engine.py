"""
Core RAG query pipeline:
1. Resolve standalone question (multi-turn)
2. Rewrite query for better retrieval
3. Retrieve + rerank
4. Build prompt with context
5. Generate answer (sync or streaming)
6. Return answer + sources
"""
import hashlib
import logging
import re
import time
from typing import AsyncIterator, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from .config import get_settings
from .prompt import SYSTEM_PROMPT, QUERY_REWRITE_PROMPT, MULTI_DOC_SYSTEM_PROMPT
from .models import QueryRequest, QueryResponse, SourceDocument
from .retriever import retrieve, detect_query_scope, multi_collection_retrieve
from .vector_store import resolve_embedding_mode_for_collections
from .memory import resolve_standalone_question,trim_history_to_budget, build_lc_messages
from .guardrails import check_query, check_context, redact_pii
from .cache import (
    CACHE_EMBEDDING_MODE,
    get_exact,
    set_exact,
    get_semantic,
    set_semantic,
)
from .embeddings import embed_query

logger = logging.getLogger(__name__)
settings = get_settings()

# LLM Singleton
def _build_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        openai_api_key = settings.openai_api_key,
        streaming=streaming,
        callbacks=[StreamingStdOutCallbackHandler()] if streaming else None,
    )

_llm = _build_llm()

_SECTION_REF_RE = re.compile(r"\b\d+\.\d+\b")
_SECTION_HINT_RE = re.compile(r"\b(section|clause|exclusion|code|excl)\b", re.IGNORECASE)


def _should_preserve_exact_reference(query: str) -> bool:
    """
    Preserve exact retrieval query when user asks about numbered clauses/sections,
    e.g. "7.14 exclusion". Rewriting often dilutes these anchors.
    """
    return bool(_SECTION_REF_RE.search(query) and _SECTION_HINT_RE.search(query))


def _is_try_docs_scope(collections: list[str]) -> bool:
    prefix = settings.try_docs_prefix
    return bool(collections) and all(c.startswith(prefix) for c in collections)


def _cache_collection_key(collections: list[str]) -> str:
    raw = "|".join(sorted(collections))
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _fmt_param(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _cache_params_key_v2(
    mode: str,
    top_k: Optional[int],
    top_k_retrieval: Optional[int],
    mmr_lambda: Optional[float],
    bm25_weight: Optional[float],
    vector_weight: Optional[float],
) -> str:
    k_final = top_k if top_k is not None else settings.top_k_rerank
    k_retrieve = top_k_retrieval if top_k_retrieval is not None else settings.top_k_retrieval
    return (
        f"{mode}:{k_final}:{k_retrieve}:"
        f"{_fmt_param(mmr_lambda)}:{_fmt_param(bm25_weight)}:{_fmt_param(vector_weight)}"
    )

# Query rewriting
async def rewrite_query(query: str) -> str:
    """
    HyDE-lite: rewrite the query to be more retrieval-friendly.
    For full HyDE, generate a hypothetical answer and embed that instead
    """
    prompt = QUERY_REWRITE_PROMPT.format(query=query)
    response = await _llm.ainvoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip()
    logger.debug(f"Rewritten query: '{rewritten}'")
    return rewritten

# HyDE (Hypothetical Document Embeddings)
async def hyde_query_expansion(query: str) -> str:
    """
    Generate a hypothetical answer to the question, then embed that
    answer for retrieval. Often finds more relevant chunks than embedding 
    the question alone
    """
    prompt = (
        f"Write a short factual paragraph that would answer the following question.\n"
        f"Question: {query}"
        f"Answer:"
    )
    response = await _llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()

# Context builder
def build_context_block(docs_with_scores: list) -> tuple[str, list[SourceDocument]]:
    """
    Build the <context> prompt block and source list.
    Wraps in XML tags to help the model distinguish context from instructions.
    """
    context_parts: list[str] = []
    sources: list[SourceDocument] = []

    for doc,score in docs_with_scores:
        doc_id = doc.metadata.get("doc_id","unknown")
        suspicious = check_context(doc.page_content)
        content = doc.page_content
        if suspicious:
            content = redact_pii(content) # sanitize if suspicious

        # Build human-readable attributes for the context tag
        source_id = doc.metadata.get("source_id", "unknown")
        source_label = source_id.replace("\\", "/").split("/")[-1] if source_id != "unknown" else "unknown"
        raw_page = doc.metadata.get("page")
        page_attr = f' page="{int(raw_page) + 1}"' if raw_page is not None else ""

        context_parts.append(
            f'<document source="{source_label}"{page_attr} score="{score:.3f}">\n{content}\n</document>'
        )
        sources.append(SourceDocument(
            doc_id=doc_id,
            content=content[:300]+"..." if len(content) > 300 else content,
            metadata=doc.metadata,
            relevance_score=round(score,4),
        ))
    context_str = "<context>\n" + "\n\n".join(context_parts) + "\n</context>"
    return context_str,sources


def build_grouped_context_block(
    docs_with_scores: list,
) -> tuple[str, list[SourceDocument]]:
    """
    Groups retrieved chunks by source document for multi-doc queries.
    Produces clearly-attributed <document name="..."> blocks so the LLM
    can reason about what each document says independently.
    Falls back to flat build_context_block when all chunks share one source.
    """
    groups: dict[str, list] = {}
    for doc, score in docs_with_scores:
        source_id = doc.metadata.get("source_id", "unknown")
        filename = source_id.replace("\\", "/").split("/")[-1]
        groups.setdefault(filename, []).append((doc, score))

    if len(groups) <= 1:
        return build_context_block(docs_with_scores)

    context_parts: list[str] = []
    sources: list[SourceDocument] = []

    for filename, items in groups.items():
        chunk_xmls: list[str] = []
        for doc, score in items:
            suspicious = check_context(doc.page_content)
            content = redact_pii(doc.page_content) if suspicious else doc.page_content
            raw_page = doc.metadata.get("page")
            page_attr = f' page="{int(raw_page) + 1}"' if raw_page is not None else ""
            chunk_xmls.append(
                f'  <chunk{page_attr} score="{score:.3f}">\n{content}\n  </chunk>'
            )
            doc_id = doc.metadata.get("doc_id", "unknown")
            sources.append(SourceDocument(
                doc_id=doc_id,
                content=content[:300] + "..." if len(content) > 300 else content,
                metadata=doc.metadata,
                relevance_score=round(float(score), 4),
            ))
        context_parts.append(
            f'<document name="{filename}">\n' + "\n".join(chunk_xmls) + "\n</document>"
        )

    context_str = "<documents>\n" + "\n\n".join(context_parts) + "\n</documents>"
    return context_str, sources


#Main Query Pipeline
async def query(
    request: QueryRequest,
    use_hyde: bool = False,
) -> QueryResponse:
    start = time.monotonic()

    collections = request.doc_collections or [request.collection_name]
    embedding_mode = resolve_embedding_mode_for_collections(collections, request.embedding_mode)
    mode_val = request.retrieval_mode.value if hasattr(request.retrieval_mode, "value") else str(request.retrieval_mode)
    cache_allowed = settings.cache_enabled and _is_try_docs_scope(collections)
    cache_collection_key = _cache_collection_key(collections)
    cache_params_key = _cache_params_key_v2(
        mode_val,
        request.top_k,
        request.top_k_retrieval,
        request.mmr_lambda,
        request.bm25_weight,
        request.vector_weight,
    )
    cache_query_vec = None

    # 1. Input guardrail
    guard = check_query(request.query)
    if not guard.allowed:
        return QueryResponse(
            answer=f"Request blocked: {guard.reason}",
            sources = [],
            session_id=request.session_id,
            latency_ms=0
        )
    
    # 2. Exact cache check
    if cache_allowed:
        cached = get_exact(request.query, cache_collection_key, cache_params_key)
        if cached:
            logger.info(f"Exact cache hit for query: '{request.query}'")
            cached["cached"] = True
            cached["latency_ms"] = round((time.monotonic()-start)*1000,2)
            return QueryResponse(**cached)
    
    # 3. Embed query for semantic cache + later retrieval
    if cache_allowed:
        cache_query_vec = await embed_query(request.query, CACHE_EMBEDDING_MODE)
        semantic_hit = get_semantic(cache_query_vec, cache_collection_key, cache_params_key)
        if semantic_hit:
            logger.info(f"Semantic cache hit for query: '{request.query}'")
            semantic_hit["cached"] = True
            semantic_hit["latency_ms"] = round((time.monotonic()-start)*1000,2)
            return QueryResponse(**semantic_hit)
    
    # 4. Resolve standalone question (multi-turn)
    history = [h.model_dump() for h in request.history]
    trimmed_history = trim_history_to_budget(history)
    standalone = await resolve_standalone_question(request.query, trimmed_history, _llm)

    # 5. Query rewrite / HyDE
    if _should_preserve_exact_reference(standalone):
        retrieval_query = standalone
        logger.info("Skipping query rewrite to preserve section/clause reference: '%s'", standalone)
    elif use_hyde:
        retrieval_query = await hyde_query_expansion(standalone)
    else:
        retrieval_query = await rewrite_query(standalone)
    
    # 6. Retrieve — multi-doc aware
    if len(collections) > 1:
        scoped = detect_query_scope(retrieval_query, collections)
        k_per = max(3, (request.top_k or settings.top_k_rerank) // len(scoped))
        docs_with_scores = await multi_collection_retrieve(
            query=retrieval_query,
            collections=scoped,
            mode=request.retrieval_mode.value if hasattr(request.retrieval_mode, "value") else str(request.retrieval_mode),
            k_per_collection=k_per,
            top_k_retrieval=request.top_k_retrieval,
            mmr_lambda=request.mmr_lambda,
            bm25_weight=request.bm25_weight,
            vector_weight=request.vector_weight,
            use_reranker=True,
            expand_context=True,
        )
        is_multi = len(scoped) > 1
    else:
        docs_with_scores = await retrieve(
            query=retrieval_query,
            collection=collections[0],
            mode=request.retrieval_mode,
            top_k=request.top_k,
            top_k_retrieval=request.top_k_retrieval,
            mmr_lambda=request.mmr_lambda,
            bm25_weight=request.bm25_weight,
            vector_weight=request.vector_weight,
            use_reranker=True,
            expand_context=True,
        )
        is_multi = False

    if not docs_with_scores:
        latency_ms = round((time.monotonic() - start) * 1000, 2)
        clarify = (
            "Can you clarify your question with a bit more detail "
            "(topic, document name, section, or timeframe)?"
        )
        return QueryResponse(
            answer=clarify,
            sources=[],
            session_id=request.session_id,
            rewritten_query=retrieval_query if retrieval_query != request.query else None,
            cached=False,
            latency_ms=latency_ms,
        )

    # 7. Build Prompt — grouped for multi-doc, flat for single-doc
    context_str, sources = (
        build_grouped_context_block(docs_with_scores) if is_multi
        else build_context_block(docs_with_scores)
    )
    active_system_prompt = MULTI_DOC_SYSTEM_PROMPT if is_multi else SYSTEM_PROMPT
    user_message = (
        f"{context_str}\n\n"
        f"Question: {request.query}\n\n"
        f"Answer based solely on the context above:"
    )

    try:
        import os
        os.makedirs("context", exist_ok=True)
        with open("context/query_context.txt", "w", encoding="utf-8") as f:
            f.write(f"--- Original Query ---\n{request.query}\n\n")
            f.write(f"--- Rewritten Query ---\n{retrieval_query}\n\n")
            f.write(f"--- Final Context ---\n{context_str}\n")
    except Exception as e:
        logger.warning(f"Failed to write query context to file: {e}")

    messages = build_lc_messages(trimmed_history, active_system_prompt)
    messages.append(HumanMessage(content=user_message))

    # 8. Generate
    response = await _llm.ainvoke(messages)
    answer = response.content.strip()

    latency_ms = round((time.monotonic() - start)*1000,2)

    result = QueryResponse(
        answer = answer,
        sources=sources,
        session_id=request.session_id,
        rewritten_query=retrieval_query if retrieval_query != request.query else None,
        cached = False,
        latency_ms=latency_ms
    )

    # 9. Cache the result
    if cache_allowed:
        result_dict = result.model_dump()
        set_exact(request.query, cache_collection_key, cache_params_key, result_dict)
        if cache_query_vec is None:
            cache_query_vec = await embed_query(request.query, CACHE_EMBEDDING_MODE)
        set_semantic(cache_query_vec, request.query, cache_collection_key, cache_params_key, result_dict)

    return result

# Pipeline-events streaming variant (step-by-step SSE for frontend animation)
async def pipeline_stream_query(request: QueryRequest) -> AsyncIterator[str]:
    """
    Yields structured SSE JSON events for every step of the RAG pipeline,
    then streams LLM tokens one-by-one. Designed to drive frontend animations.

    Event types: pipeline_start, guardrail_check, cache_check, query_rewrite,
                 retrieval_start, chunks_retrieved, context_built,
                 generation_start, token, complete
    """
    import json

    def _default(obj):
        """Fallback serialiser for types json.dumps can't handle natively."""
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return str(obj)

    def emit(event: str, status: str, data: dict = None) -> str:
        payload = {"event": event, "status": status, "data": data or {}}
        return f"data: {json.dumps(payload, default=_default)}\n\n"

    start = time.monotonic()
    mode_val = request.retrieval_mode.value if hasattr(request.retrieval_mode, "value") else str(request.retrieval_mode)
    collections = request.doc_collections or [request.collection_name]
    embedding_mode = resolve_embedding_mode_for_collections(collections, request.embedding_mode)
    cache_allowed = settings.cache_enabled and _is_try_docs_scope(collections)
    cache_collection_key = _cache_collection_key(collections)
    cache_params_key = _cache_params_key_v2(
        mode_val,
        request.top_k,
        request.top_k_retrieval,
        request.mmr_lambda,
        request.bm25_weight,
        request.vector_weight,
    )

    yield emit("pipeline_start", "in_progress", {
        "query": request.query,
        "collection": request.collection_name,
        "mode": mode_val,
        "embedding_mode": embedding_mode,
    })

    try:
        # --- Guardrail check ---
        guard = check_query(request.query)
        if not guard.allowed:
            yield emit("guardrail_check", "blocked", {"reason": guard.reason})
            yield emit("complete", "blocked", {
                "answer": f"Request blocked: {guard.reason}",
                "sources": [],
                "latency_ms": round((time.monotonic() - start) * 1000, 2),
            })
            yield "data: [DONE]\n\n"
            return
        yield emit("guardrail_check", "passed", {})

        # --- Cache check ---
        cache_query_vec = None
        if cache_allowed:
            cached = get_exact(request.query, cache_collection_key, cache_params_key)
            if cached:
                cached["cached"] = True
                cached["latency_ms"] = round((time.monotonic() - start) * 1000, 2)
                yield emit("cache_check", "hit", {"type": "exact"})
                yield emit("complete", "done", cached)
                yield "data: [DONE]\n\n"
                return

            cache_query_vec = await embed_query(request.query, CACHE_EMBEDDING_MODE)
            semantic_hit = get_semantic(cache_query_vec, cache_collection_key, cache_params_key)
            if semantic_hit:
                semantic_hit["cached"] = True
                semantic_hit["latency_ms"] = round((time.monotonic() - start) * 1000, 2)
                yield emit("cache_check", "hit", {"type": "semantic"})
                yield emit("complete", "done", semantic_hit)
                yield "data: [DONE]\n\n"
                return
            yield emit("cache_check", "miss", {})
        else:
            yield emit("cache_check", "skipped", {})

        # --- Standalone question resolution (multi-turn) ---
        history = [h.model_dump() for h in request.history]
        trimmed_history = trim_history_to_budget(history)
        standalone = await resolve_standalone_question(request.query, trimmed_history, _llm)

        # --- Query rewrite ---
        if _should_preserve_exact_reference(standalone):
            retrieval_query = standalone
            yield emit("query_rewrite", "skipped", {
                "reason": "section/clause reference preserved",
                "query": standalone,
            })
        else:
            retrieval_query = await rewrite_query(standalone)
            yield emit("query_rewrite", "done", {
                "original": request.query,
                "rewritten": retrieval_query,
            })

        # --- Document routing (multi-doc) ---
        if len(collections) > 1:
            scoped = detect_query_scope(retrieval_query, collections)
            is_multi = len(scoped) > 1
            yield emit("doc_routing", "done", {
                "total_docs": len(collections),
                "selected": [c.split("__")[-1] for c in scoped],
                "mode": "comparison" if is_multi else "targeted",
            })
        else:
            scoped = collections
            is_multi = False

        # --- Retrieval ---
        yield emit("retrieval_start", "in_progress", {
            "mode": mode_val,
            "top_k": request.top_k or settings.top_k_rerank,
            "top_k_retrieval": request.top_k_retrieval or settings.top_k_retrieval,
            "collections": len(scoped),
        })

        if is_multi:
            k_per = max(3, (request.top_k or settings.top_k_rerank) // len(scoped))
            docs_with_scores = await multi_collection_retrieve(
                query=retrieval_query,
                collections=scoped,
                mode=mode_val,
                k_per_collection=k_per,
                top_k_retrieval=request.top_k_retrieval,
                mmr_lambda=request.mmr_lambda,
                bm25_weight=request.bm25_weight,
                vector_weight=request.vector_weight,
                use_reranker=True,
                expand_context=True,
            )
        else:
            docs_with_scores = await retrieve(
                query=retrieval_query,
                collection=scoped[0],
                mode=request.retrieval_mode,
                top_k=request.top_k,
                top_k_retrieval=request.top_k_retrieval,
                mmr_lambda=request.mmr_lambda,
                bm25_weight=request.bm25_weight,
                vector_weight=request.vector_weight,
                use_reranker=True,
                expand_context=True,
            )

        if not docs_with_scores:
            yield emit("chunks_retrieved", "empty", {"count": 0})
            yield emit("complete", "done", {
                "answer": "Can you clarify your question with a bit more detail (topic, document name, section, or timeframe)?",
                "sources": [],
                "rewritten_query": retrieval_query,
                "latency_ms": round((time.monotonic() - start) * 1000, 2),
                "session_id": request.session_id,
                "cached": False,
            })
            yield "data: [DONE]\n\n"
            return

        chunk_previews = [
            {
                "doc_id": doc.metadata.get("doc_id", "unknown")[:12],
                "score": round(float(score), 4),
                "preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                "source": doc.metadata.get("source_id", doc.metadata.get("source", "unknown")),
                "chunk_index": int(doc.metadata.get("chunk_index", 0)),
            }
            for doc, score in docs_with_scores
        ]
        yield emit("chunks_retrieved", "done", {
            "count": len(docs_with_scores),
            "chunks": chunk_previews,
        })

        # --- Context building ---
        context_str, sources = (
            build_grouped_context_block(docs_with_scores) if is_multi
            else build_context_block(docs_with_scores)
        )
        active_system_prompt = MULTI_DOC_SYSTEM_PROMPT if is_multi else SYSTEM_PROMPT
        estimated_tokens = len(context_str) // 4

        yield emit("context_built", "done", {
            "chunks_used": len(sources),
            "estimated_tokens": estimated_tokens,
            "sources": [{"doc_id": s.doc_id, "score": s.relevance_score} for s in sources],
        })

        # --- LLM generation ---
        user_message = (
            f"{context_str}\n\n"
            f"Question: {request.query}\n\n"
            f"Answer based solely on the context above:"
        )
        messages = build_lc_messages(trimmed_history, active_system_prompt)
        messages.append(HumanMessage(content=user_message))

        yield emit("generation_start", "in_progress", {"model": settings.chat_model})

        llm_stream = _build_llm(streaming=True)
        full_answer = ""
        async for chunk in llm_stream.astream(messages):
            token = chunk.content
            if token:
                full_answer += token
                yield f"data: {json.dumps({'event': 'token', 'status': 'in_progress', 'data': {'text': token}})}\n\n"

        latency_ms = round((time.monotonic() - start) * 1000, 2)
        sources_data = [s.model_dump() for s in sources]

        # Cache result — failure must not crash the stream
        if cache_allowed:
            try:
                result_dict = {
                    "answer": full_answer,
                    "sources": sources_data,
                    "session_id": request.session_id,
                    "rewritten_query": retrieval_query if retrieval_query != request.query else None,
                    "cached": False,
                    "latency_ms": latency_ms,
                    "eval_scores": None,
                }
                if cache_query_vec is None:
                    cache_query_vec = await embed_query(request.query, CACHE_EMBEDDING_MODE)
                set_exact(request.query, cache_collection_key, cache_params_key, result_dict)
                set_semantic(cache_query_vec, request.query, cache_collection_key, cache_params_key, result_dict)
            except Exception:
                logger.warning("Cache write failed (non-fatal)", exc_info=True)

        yield emit("complete", "done", {
            "answer": full_answer,
            "sources": sources_data,
            "rewritten_query": retrieval_query if retrieval_query != request.query else None,
            "latency_ms": latency_ms,
            "session_id": request.session_id,
            "cached": False,
        })
        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.exception("pipeline_stream_query crashed mid-stream")
        try:
            yield emit("complete", "failed", {
                "answer": f"Pipeline error: {exc}",
                "sources": [],
                "latency_ms": round((time.monotonic() - start) * 1000, 2),
            })
            yield "data: [DONE]\n\n"
        except Exception:
            pass


# Streaming variant
async def stream_query(request: QueryRequest) -> AsyncIterator[str]:
    """
    SSE-compatible streaming answer generator.
    Yields answer tokens as they arrive from OpenAI.
    Sources are emitted as a final JSON event.
    """
    guard = check_query(request.query)
    if not guard.allowed:
        yield f"data: {guard.reason}\n\n"
        return

    standalone = await resolve_standalone_question(
        request.query,
        [h.model_dump() for h in request.history],
        _llm,
    )
    if _should_preserve_exact_reference(standalone):
        retrieval_query = standalone
        logger.info("Skipping query rewrite to preserve section/clause reference: '%s'", standalone)
    else:
        retrieval_query = await rewrite_query(standalone)
    docs_with_scores = await retrieve(
        retrieval_query, request.collection_name, request.retrieval_mode.value
    )
    context_str, sources = build_context_block(docs_with_scores)
    user_message = f"{context_str}\n\nQuestion: {request.query}\nAnswer:"

    llm_stream = _build_llm(streaming=True)
    async for chunk in llm_stream.astream([HumanMessage(content=user_message)]):
        token = chunk.content
        if token:
            yield f"data: {token}\n\n"
    
    import json
    sources_payload = [{"doc_id":s.doc_id, "score":s.relevance_score} for s in sources]
    yield f"data: [SOURCES]{json.dumps(sources_payload)}\n\n"
    yield "data: [DONE]\n\n"

print("[query_engine] Module ready")