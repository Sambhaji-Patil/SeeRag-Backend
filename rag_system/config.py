import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    #openai llm service
    openai_api_key: str
    chat_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Embeddings (GPU vs CPU auto selection)
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimensions: int = 1024
    embedding_model_cpu: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions_cpu: int = 384
    embedding_model_openai: str = "text-embedding-3-small"
    embedding_dimensions_openai: int = 1536
    embedding_device: str = "auto"
    embedding_batch_size: int = 32
    embedding_normalize: bool = True

    # Try Docs (pre-indexed demo docs)
    try_docs_path: str = os.path.join(os.path.dirname(__file__), "Try Docs")
    try_docs_prefix: str = "try__"

    #FAISS
    faiss_index_path: str = "./faiss_indexes"
    faiss_index_name: str = "prod_rag"

    #Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_size: int = 100

    #retrieval
    top_k_retrieval: int = 20
    top_k_rerank: int = 6
    mmr_lambda: float = 0.6
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    #memory
    max_history_turns: int = 10
    context_window_tokens: int = 8000

    #cache
    cache_enabled: bool = False
    redis_url: str = "redis://localhost:6379"
    cache_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.95

    #api
    api_title: str = "Production RAG API"
    api_version: str = "1.0.0"
    cors_origins: list[str] = ["*"]
    rate_limit_per_minute: int = 60

    #guardrails
    guardrails_use_llama_guard: bool = True
    guardrails_model_id: str = "meta-llama/Llama-Guard-3-1B"
    guardrails_max_new_tokens: int = 32
    guardrails_local_model_path: str | None = None
    guardrails_local_files_only: bool = True
    guardrails_download_if_missing: bool = True
    guardrails_require_harm_intent_for_llama_unsafe: bool = True
    guardrails_risk_block_threshold: float = 0.50
    guardrails_unsafe_base_score: float = 0.20
    hf_token: str | None = None

    #evaluation
    faithfullness_threshold: float = 0.7
    answer_relevance_threshold: float = 0.7

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False
    )

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
print(
    "[Config] Loaded. Model: "
    f"{settings.chat_model}, Embedding GPU: {settings.embedding_model} ({settings.embedding_dimensions}), "
    f"Embedding CPU: {settings.embedding_model_cpu} ({settings.embedding_dimensions_cpu}), "
    f"Embedding OpenAI: {settings.embedding_model_openai} ({settings.embedding_dimensions_openai}), "
    f"Device: {settings.embedding_device}"
)