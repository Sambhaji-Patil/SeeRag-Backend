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
from langchain.schema import HumanMessage,SystemMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from config import get_settings
from prompt import SYSTEM_PROMPT, QUERY_REWRITE_PROMPT
from models import QueryRequest, QueryResponse, SourceDocument
from retriever import retrieve
from memory import resolve_standalone_question,trim_history_to_budget, build_lc_messages
from guardrails import check_query, check_context, redact_pii
from cache import get_exact,set_exact,get_semantic,set_semantic
from embeddings import embed_query

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