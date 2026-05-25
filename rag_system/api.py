import logging
import time
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware

from .config import get_settings
from .models import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse,
    EvalRequest, EvalResponse,
    HealthResponse,
)
from .document_processor import process_texts, process_file
from .vector_store import (
    add_documents, load_or_create_store, is_loaded,
    list_collections, get_collection_stats, delete_collection,
    cleanup_stale_collections,
)
from .query_engine import query as run_query, stream_query, pipeline_stream_query
from .eval import evaluate
from .cache import cache_connected, get_cache_stats
from .embeddings import get_embeddings
from .guardrails import _load_llama_guard
from .retriever import _reranker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("system_logs.txt", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)
settings = get_settings()

# In-memory job registry for background ingestion tasks
_ingest_jobs: dict[str, dict] = {}


def _safe_coll_name(filename: str) -> str:
    """Convert a filename to a safe FAISS collection name component."""
    from pathlib import Path as _Path
    import re as _re
    stem = _Path(filename).stem if filename else "doc"
    safe = _re.sub(r'[^a-z0-9-]', '_', stem.lower())
    safe = _re.sub(r'_+', '_', safe).strip('_')[:40]
    return safe or 'doc'


# Lifespan (startup / shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG API...")

    logger.info("Preloading models...")
    get_embeddings()
    _load_llama_guard()
    if getattr(_reranker, "available", False):
        logger.info("Reranker model preloaded")
    else:
        logger.info("Reranker unavailable; skipping preload")

    from pathlib import Path
    base_path = Path(settings.faiss_index_path)
    if base_path.exists():
        for d in base_path.iterdir():
            if d.is_dir():
                load_or_create_store(d.name)

    logger.info("RAG API ready!")

    # Background session-cleanup loop: remove collections idle > 30 min
    import asyncio

    async def _session_cleanup_loop():
        while True:
            await asyncio.sleep(300)  # check every 5 minutes
            removed = cleanup_stale_collections(ttl_seconds=1800)
            if removed:
                logger.info(f"Session cleanup removed {len(removed)} stale collection(s): {removed}")

    cleanup_task = asyncio.create_task(_session_cleanup_loop())

    yield

    cleanup_task.cancel()
    logger.info("Shutting down RAG API")


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
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.monotonic() - start) * 1000, 2))
    return response


# ── Ops ──────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="ok",
        vector_store_loaded=is_loaded(None),
        cache_connected=cache_connected(),
        model=settings.chat_model,
    )


@app.get("/cache_stats", tags=["ops"])
async def cache_stats():
    return get_cache_stats()


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest_texts(req: IngestRequest):
    """Ingest raw text strings into a named collection."""
    try:
        docs = process_texts(
            texts=req.texts,
            metadatas=req.metadatas,
            source_id=req.collection_name,
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestResponse, tags=["ingest"])
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: str = "default",
    background_tasks: BackgroundTasks = None,
):
    """
    Upload a PDF, TXT, or Markdown file.
    Returns a job_id immediately; processing runs in the background.
    Poll GET /ingest/jobs/{job_id} or subscribe to GET /ingest/jobs/{job_id}/events.
    """
    import tempfile, os

    job_id = str(_uuid.uuid4())
    suffix = "." + file.filename.rsplit(".", 1)[-1].lower()
    doc_collection = f"{collection_name}__{_safe_coll_name(file.filename)}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    _ingest_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "collection_name": doc_collection,
        "filename": file.filename,
        "chunks_created": 0,
        "message": "File received, extracting text...",
        "progress": 5,
    }

    def _process():
        try:
            _ingest_jobs[job_id]["progress"] = 20
            _ingest_jobs[job_id]["message"] = "Extracting and chunking text..."
            docs = process_file(tmp_path, display_name=file.filename)

            _ingest_jobs[job_id]["progress"] = 60
            _ingest_jobs[job_id]["message"] = f"Embedding and indexing {len(docs)} chunks..."
            add_documents(docs, collection=doc_collection)

            _ingest_jobs[job_id]["progress"] = 100
            _ingest_jobs[job_id]["status"] = "done"
            _ingest_jobs[job_id]["chunks_created"] = len(docs)
            _ingest_jobs[job_id]["message"] = f"Indexed {len(docs)} chunks into '{doc_collection}'"
            logger.info(f"Ingest job {job_id} complete: {file.filename} -> {len(docs)} chunks")
        except Exception as e:
            _ingest_jobs[job_id]["status"] = "failed"
            _ingest_jobs[job_id]["message"] = str(e)
            logger.exception(f"Ingest job {job_id} failed")
        finally:
            os.unlink(tmp_path)

    if background_tasks:
        background_tasks.add_task(_process)
        return IngestResponse(
            success=True,
            docs_indexed=-1,
            collection_name=doc_collection,
            message=f"Job '{job_id}' started for '{file.filename}'",
            job_id=job_id,
        )

    _process()
    return IngestResponse(
        success=True,
        docs_indexed=_ingest_jobs[job_id].get("chunks_created", 0),
        collection_name=doc_collection,
        message=_ingest_jobs[job_id].get("message", "Done"),
        job_id=job_id,
    )


@app.get("/ingest/jobs", tags=["ingest"])
async def list_ingest_jobs():
    """List all ingestion jobs (most recent first)."""
    return {"jobs": list(reversed(list(_ingest_jobs.values())))}


@app.get("/ingest/jobs/{job_id}", tags=["ingest"])
async def get_ingest_job(job_id: str):
    """Get the current status of an ingestion job."""
    job = _ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/ingest/jobs/{job_id}/events", tags=["ingest"])
async def ingest_job_events(job_id: str):
    """
    SSE stream of ingestion progress events.
    Emits the job dict every 300 ms until status is 'done' or 'failed'.
    """
    import asyncio, json

    async def generate():
        while True:
            job = _ingest_jobs.get(job_id)
            if not job:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                return
            yield f"data: {json.dumps(job)}\n\n"
            if job["status"] in ("done", "failed"):
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Query ─────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query_endpoint(req: QueryRequest):
    """
    Main RAG query endpoint.
    Supports multi-turn history, hybrid retrieval, semantic caching, and multi-doc routing.
    Set stream=true in body to get a plain SSE token stream.
    """
    collections = req.doc_collections or [req.collection_name]
    for coll in collections:
        if not is_loaded(coll):
            load_or_create_store(coll)
    if not any(is_loaded(c) for c in collections):
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found. Ingest documents first.",
        )

    if req.stream:
        return StreamingResponse(
            stream_query(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await run_query(req)
        return result
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/pipeline", tags=["query"])
async def pipeline_query_endpoint(req: QueryRequest):
    """
    Pipeline-events SSE endpoint — supports multi-doc routing.
    Streams a structured JSON event for every RAG step (guardrail → cache →
    rewrite → doc_routing → retrieval → context → generation), then streams LLM tokens.
    """
    collections = req.doc_collections or [req.collection_name]
    for coll in collections:
        if not is_loaded(coll):
            load_or_create_store(coll)
    if not any(is_loaded(c) for c in collections):
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found. Ingest documents first.",
        )
    return StreamingResponse(
        pipeline_stream_query(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Collections ───────────────────────────────────────────────────────────────

@app.get("/collections", tags=["collections"])
async def list_collections_endpoint():
    """List all collections with chunk count and disk size."""
    names = list_collections()
    return {"collections": [get_collection_stats(n) for n in names]}


@app.get("/collections/{collection_name}", tags=["collections"])
async def get_collection_endpoint(collection_name: str):
    """Get detailed stats for a specific collection."""
    if collection_name not in list_collections():
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    return get_collection_stats(collection_name)


@app.delete("/collections/{collection_name}", tags=["collections"])
async def delete_collection_endpoint(collection_name: str):
    """Permanently delete a collection from memory and disk."""
    deleted = delete_collection(collection_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    return {"success": True, "message": f"Collection '{collection_name}' deleted"}


# ── Evaluate ──────────────────────────────────────────────────────────────────

@app.post("/evaluate", response_model=EvalResponse, tags=["eval"])
async def evaluate_endpoint(req: EvalRequest):
    """Run RAGAS-style evaluation on a (question, answer, contexts) triple."""
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
        workers=1,
    )

print("[api] FastAPI app configured.")
