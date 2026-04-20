import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware

from .config import get_settings
from .models import (
    IngestRequest,IngestResponse,
    QueryRequest,QueryResponse,
    EvalRequest,EvalResponse,
    HealthResponse
)
from .document_processor import process_texts,process_file
from .vector_store import add_documents, load_or_create_store, is_loaded
from .query_engine import query as run_query, stream_query
from .eval import evaluate
from .cache import cache_connected, get_cache_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("system_logs.txt", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
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

@app.get("/cache_stats", tags=["ops"])
async def cache_stats():
    """Information about where the cache is stored and how many queries have been cached"""
    return get_cache_stats()

# Ingest raw texts 
@app.post("/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest_texts(req: IngestRequest):
    """
    Ingest raw text strings into a named collection.
    - Chunks, embeds, and upserts into FAISS
    - Persists index to disk
    """
    try:
        docs = process_texts(
            texts = req.texts,
            metadatas=req.metadatas,
            source_id=req.collection_name
        )
        add_documents(docs, collection=req.collection_name, force_reindex=req.force_reindex)
        return IngestResponse(
            success=True,
            docs_indexed=len(docs),
            collection_name=req.collection_name,
            message=f"Indexed {len(docs)} chunks into '{req.collection_name}'.",
        )
    except Exception as e:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500,detail=str(e))

# Ingest file upload
@app.post("/ingest/file",response_model=IngestResponse,tags=["ingest"])
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: str = "default",
    background_tasks: BackgroundTasks = None
):
    """
    Upload a PDF, TXT or markdown file, Processing runs in the background.
    """
    import tempfile, os

    suffix = "." + file.filename.rsplit(".",1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    def _process():
        try:
            docs = process_file(tmp_path)
            add_documents(docs,collection=collection_name)
            logger.info(f"Background ingest complete: {file.filename} -> {len(docs)} chunks")
        finally:
            os.unlink(tmp_path)
    
    if background_tasks:
        background_tasks.add_task(_process)
        return IngestResponse(
            success=True,
            docs_indexed=-1, # unknown until background completes
            collection_name=collection_name,
            message=f"File '{file.filename}' queued for background processing",
        )

    _process()
    return IngestResponse(
            success=True,
            docs_indexed=0, # unknown until background completes
            collection_name=collection_name,
            message=f"File '{file.filename}' processed",
        )

# Query
@app.post("/query", response_model=QueryResponse,tags=["query"])
async def query_endpoint(req: QueryRequest):
    """
    Main RAG query endpoint.
    Supports multi-turn history, hybrid retrieval, and semantic caching.
    Set stream=true in body to get an SSE stream instead.
    """
    # first check if the collection loaded into the memory, else load it
    if not is_loaded(req.collection_name):
        load_or_create_store(req.collection_name)
    # if it fails to load than , first we have to ingest it
    if not is_loaded(req.collection_name):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{req.collection_name}' not found. Ingest docs first",
        )
    
    if req.stream:
        return StreamingResponse(
            stream_query(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering":"no"}
        )
    
    try:
        result = await run_query(req)
        return result
    except Exception as e:
        logger.exception("Query Failed")
        raise HTTPException(status_code=500, detail=str(e))

# Evaluate
@app.post("/evaluate",response_model=EvalResponse,tags=["eval"])
async def evaluate_endpoint(req: EvalRequest):
    """
    Run RAGAS-style evaluation on a (question,answer,contexts) triple.
    Use this in CI pipelines to gate quality regressions.
    """
    try:
        return await evaluate(req)
    except Exception as e:
        logger.exception("Eval failed")
        raise HTTPException(status_code=500, detail=str(e))

# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "rag_system.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )

print("[api] FastAPI app configured.")