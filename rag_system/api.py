import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware

from config import get_settings
from models import (
    IngestRequest,IngestResponse,
    QueryRequest,QueryResponse,
    EvalRequest,EvalResponse,
    HealthResponse
)
from document_processor import process_texts,process_file
from vector_store import add_documents, load_or_create_store, is_loaded
from query_engine import query as run_query, stream_query
from eval import evaluate
from cache import cache_connected

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()

# Lifespan (startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG API...")
    # Here we will pre load default FAISS Index if it exists on disk
    load_or_create_store("default")
    logger.info("RAG API ready!")
    yield
    logger.info("Shutting down RAG API")

# App
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Production RAG system: ingest documents, query with advanced retrieval",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request,call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = round((time.monotonic()-start)*1000,2)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response

# Helath
@app.get("/health",response_model=HealthResponse,tags=["ops"])
async def health():
    return HealthResponse(
        status="ok",
        vector_store_loaded=is_loaded("default"),
        cache_connected=cache_connected(),
        model=settings.chat_model
    )