from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any
from enum import Enum
import uuid

class RetrievalMode(str, Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    MMR = "mmr"

# Ingestion 

class IngestRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Raw text chunks to ingest")
    metadatas: Optional[list[dict[str,Any]]] =  None
    collection_name: str = Field(default="default",pattern=r"^[a-z0-9_-]+$")
    force_reindex: bool = False

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
    top_k: Optional[int] = None
    doc_collections: Optional[list[str]] = None  # per-doc sub-collections; None = legacy single-collection mode
    history: list[ChatMessage] = Field(default_factory=list)
    stream: bool = False

    @field_validator("query")
    @classmethod
    def sanitize_query(cls,v):
        return v.strip()

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