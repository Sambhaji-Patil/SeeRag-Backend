from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any, Literal
from enum import Enum
import uuid

class RetrievalMode(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    MMR = "mmr"

EmbeddingMode = Literal["bge-large", "bge-small", "openai-small", "auto"]

# Ingestion 

class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Raw text chunks to ingest")
    metadatas: Optional[list[dict[str,Any]]] =  None
    collection_name: str = Field(default="default",pattern=r"^[a-z0-9_-]+$")
    force_reindex: bool = False
    embedding_mode: Optional[EmbeddingMode] = None

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls,v):
        if any(not t.strip() for t in v):
            raise ValueError("All text entries must be non-empty")
        return v

class IngestResponse(BaseModel):
    success: bool
    docs_indexed: int
    collection_name: str
    message: str
    job_id: Optional[str] = None

# Query

class ChatMessage(BaseModel):
    role: str = Field(...,pattern=r"^(user|assistant)$")
    content: str

class QueryRequest(BaseModel):
    query: str = Field(...,min_length=1,max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    collection_name: str = Field(default="default")
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    embedding_mode: Optional[EmbeddingMode] = None
    top_k: Optional[int] = None
    top_k_retrieval: Optional[int] = None
    mmr_lambda: Optional[float] = None
    bm25_weight: Optional[float] = None
    vector_weight: Optional[float] = None
    doc_collections: Optional[list[str]] = None  # per-doc sub-collections; None = legacy single-collection mode
    history: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False

    @field_validator("query")
    @classmethod
    def sanitize_query(cls,v):
        return v.strip()

    @field_validator("top_k", "top_k_retrieval")
    @classmethod
    def validate_top_k(cls, v):
        if v is None:
            return v
        if v < 1:
            raise ValueError("top_k values must be >= 1")
        return v

    @field_validator("mmr_lambda", "bm25_weight", "vector_weight")
    @classmethod
    def validate_weights(cls, v):
        if v is None:
            return v
        if v < 0 or v > 1:
            raise ValueError("weights must be between 0 and 1")
        return v

class SourceDocument(BaseModel):
    doc_id: str
    content: str
    metadata: dict[str,Any]
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    session_id: str
    rewritten_query: Optional[str] = None
    cached: bool = False
    latency_ms:float
    eval_scores: Optional[dict[str,float]] = None

# Evaluation

class EvalRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    ground_truth: Optional[str] = None

class EvalResponse(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    passed: bool

# Health
class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool
    cache_connected: bool
    model: str

print("[Models] Pydantic schemas loaded")