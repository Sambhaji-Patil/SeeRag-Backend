"""
Core RAG query pipeline:
1. Resolve standalone question (multi-turn)
2. Rewrite query for better retrieval
3. Retrieve + rerank
4. Build prompt with context
5. Generate answer (sync or streaming)
6. Return answer + sources
"""
import logging
import time
from typing import AsyncIterator, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from .config import get_settings
from .prompt import SYSTEM_PROMPT, QUERY_REWRITE_PROMPT
from .models import QueryRequest, QueryResponse, SourceDocument
from .retriever import retrieve
from .memory import resolve_standalone_question,trim_history_to_budget, build_lc_messages
from .guardrails import check_query, check_context, redact_pii
from .cache import get_exact,set_exact,get_semantic,set_semantic
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
        
        context_parts.append(
            f"<document id=\"{doc_id}\" score=\"{score:.3f}\">\n{content}\n</document>"
        )
        sources.append(SourceDocument(
            doc_id=doc_id,
            content=content[:300]+"..." if len(content) > 300 else content,
            metadata=doc.metadata,
            relevance_score=round(score,4),
        ))
    context_str = "<context>\n" + "\n\n".join(context_parts) + "\n</context>"
    return context_str,sources

#Main Query Pipeline
async def query(
    request: QueryRequest,
    use_hyde: bool = False,
) -> QueryResponse:
    start = time.monotonic()

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
    cached = get_exact(request.query, request.collection_name, request.retrieval_mode)
    if cached:
        cached["cached"] = True
        cached["latency_ms"] = round((time.monotonic()-start)*1000,2)
        return QueryResponse(**cached)
    
    # 3. Embed query for semantic cache + later retrieval
    query_vec = await embed_query(request.query)
    semantic_hit = get_semantic(query_vec)
    if semantic_hit:
        semantic_hit["cached"] = True
        semantic_hit["latency_ms"] = round((time.monotonic()-start)*1000,2)
        return QueryResponse(**semantic_hit)
    
    # 4. Resolve standalone question (multi-turn)
    history = [h.model_dump() for h in request.history]
    trimmed_history = trim_history_to_budget(history)
    standalone = await resolve_standalone_question(request.query, trimmed_history, _llm)

    # 5. Query rewrite / HyDE
    if use_hyde:
        retrieval_query = await hyde_query_expansion(standalone)
    else:
        retrieval_query = await rewrite_query(standalone)
    
    # 6. Retrieve
    docs_with_scores = await retrieve(
        query = retrieval_query,
        collection = request.collection_name,
        mode = request.retrieval_mode,
        top_k = request.top_k,
        use_reranker=True,
        expand_context=True,
    )

    # 7. Build Prompt
    context_str, sources = build_context_block(docs_with_scores)
    user_message = (
        f"{context_str}\n\n"
        f"Question: {request.query}\n\n"
        f"Answer based solely on the context above:"
    )

    messages = build_lc_messages(trimmed_history,SYSTEM_PROMPT)
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
    result_dict = result.model_dump()
    set_exact(request.query, request.collection_name, request.retrieval_mode, result_dict)
    set_semantic(query_vec,request.query, result_dict)

    return result

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
    retrieval_query = await rewrite_query(standalone)
    docs_with_scores = await retrieve(
        retrieval_query, request.collection_name, request.retrieval_mode.value
    )
    context_str, sources = build_context_block(docs_with_scores)
    user_message = f"{context_str}\n\nQuestion: {request.query}\nAnswer:"

    llm_stream = _build_llm(streamin=True)
    async for chunk in llm_stream.astream([HumanMessage(content=user_message)]):
        token = chunk.content
        if token:
            yield f"data: {token}\n\n"
    
    import json
    sources_payload = [{"doc_id":s.doc_id, "score":s.relevance_score} for s in sources]
    yield f"data: [SOURCES]{json.dumps(sources_payload)}\n\n"
    yield "data: [DONE]\n\n"

print("[query_engine] Module ready")