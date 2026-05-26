import logging
import time
import uuid as _uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Body
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
    cleanup_stale_collections, get_collection_embedding_mode,
    pin_collection,
)
from .query_engine import query as run_query, stream_query, pipeline_stream_query
from .eval import evaluate
from .cache import cache_connected, get_cache_stats
from .embeddings import get_embeddings, get_embeddings_runtime_info
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

# Raw file bytes for document preview: collection_name -> (bytes, content_type)
_doc_files: dict[str, tuple[bytes, str]] = {}

# Try Docs file paths: collection_name -> path
_try_doc_paths: dict[str, "Path"] = {}

_FILE_CONTENT_TYPES: dict[str, str] = {
    '.pdf': 'application/pdf',
    '.txt': 'text/plain; charset=utf-8',
    '.md':  'text/markdown; charset=utf-8',
}

# Viz cache: per collection, stores fitted PCA + 2D projected points
_viz_cache: dict[str, dict] = {}


def _compute_viz(collection: str) -> dict:
    """PCA-project all chunk embeddings to 2D. Cached per collection."""
    if collection in _viz_cache:
        return _viz_cache[collection]

    from .vector_store import get_store
    store = get_store(collection)
    if store is None or store.index.ntotal == 0:
        return {"points": [], "pca": None, "vectors": None}

    import numpy as np
    from sklearn.decomposition import PCA

    n = store.index.ntotal
    d = store.index.d
    try:
        vectors = store.index.reconstruct_n(0, n).astype(np.float32)
    except Exception:
        return {"points": [], "pca": None, "vectors": None}

    n_components = min(2, n, d)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(vectors)

    points = []
    for i in range(n):
        doc_id = store.index_to_docstore_id.get(i)
        if not doc_id:
            continue
        doc = store.docstore._dict.get(doc_id)
        if not doc:
            continue
        cx = float(coords[i, 0]) if n_components >= 1 else 0.0
        cy = float(coords[i, 1]) if n_components >= 2 else 0.0
        points.append({
            "doc_id": doc_id,
            "x": cx,
            "y": cy,
            "preview": doc.page_content[:100],
            "page": doc.metadata.get("page"),
            "source": str(doc.metadata.get("source_id", "")),
            "chunk_index": int(doc.metadata.get("chunk_index", i)),
        })

    result = {"points": points, "pca": pca, "vectors": vectors}
    _viz_cache[collection] = result
    return result


def _safe_coll_name(filename: str) -> str:
    """Convert a filename to a safe FAISS collection name component."""
    from pathlib import Path as _Path
    import re as _re
    stem = _Path(filename).stem if filename else "doc"
    safe = _re.sub(r'[^a-z0-9-]', '_', stem.lower())
    safe = _re.sub(r'_+', '_', safe).strip('_')[:40]
    return safe or 'doc'


def _try_doc_collection_name(filename: str) -> str:
    return f"{settings.try_docs_prefix}{_safe_coll_name(filename)}"


def _load_try_docs() -> None:
    """Load pre-indexed Try Docs into memory and cache raw bytes for preview."""
    from pathlib import Path

    try_dir = Path(settings.try_docs_path)
    if not try_dir.exists():
        logger.info("Try Docs folder not found at %s", try_dir)
        return

    for path in sorted(try_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in _FILE_CONTENT_TYPES:
            continue

        collection = _try_doc_collection_name(path.name)
        _try_doc_paths[collection] = path

        store = load_or_create_store(collection)
        if store is not None:
            pin_collection(collection)
        else:
            logger.warning("Try Doc index missing for '%s' (%s)", path.name, collection)

        try:
            _doc_files[collection] = (path.read_bytes(), _FILE_CONTENT_TYPES[suffix])
        except Exception:
            logger.warning("Failed to cache Try Doc file bytes for '%s'", path.name)


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

    _load_try_docs()

    logger.info("RAG API ready!")

    # Background session-cleanup loop: remove collections idle > 30 min
    import asyncio

    async def _session_cleanup_loop():
        while True:
            await asyncio.sleep(300)  # check every 5 minutes
            removed = cleanup_stale_collections(ttl_seconds=1800)
            if removed:
                logger.info(f"Session cleanup removed {len(removed)} stale collection(s): {removed}")
                for coll in removed:
                    _doc_files.pop(coll, None)

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

@app.get("/", tags=["ops"])
async def root():
    return {"status": "ok", "service": settings.api_title}

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


@app.get("/embeddings/info", tags=["ops"])
async def embeddings_info():
    return get_embeddings_runtime_info()


# ── Try Docs ─────────────────────────────────────────────────────────────────

@app.get("/try_docs", tags=["try_docs"])
async def list_try_docs():
    """List pre-indexed Try Docs available to add to a session."""
    from pathlib import Path

    try_dir = Path(settings.try_docs_path)
    if not try_dir.exists():
        return {"docs": []}

    docs = []
    for path in sorted(try_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in _FILE_CONTENT_TYPES:
            continue

        collection = _try_doc_collection_name(path.name)
        _try_doc_paths.setdefault(collection, path)

        index_path = Path(settings.faiss_index_path) / collection
        stats = get_collection_stats(collection) if index_path.exists() else None

        docs.append({
            "filename": path.name,
            "collection": collection,
            "chunks": stats["chunk_count"] if stats else 0,
            "embedding_mode": stats["embedding_mode"] if stats else None,
            "size_mb": stats["size_mb"] if stats else 0.0,
            "ready": stats is not None,
        })

    return {"docs": docs}


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
        add_documents(
            docs,
            collection=req.collection_name,
            force_reindex=req.force_reindex,
            embedding_mode=req.embedding_mode,
        )
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
    embedding_mode: str | None = None,
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

    _doc_files[doc_collection] = (content, _FILE_CONTENT_TYPES.get(suffix, 'application/octet-stream'))

    _ingest_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "collection_name": doc_collection,
        "filename": file.filename,
        "embedding_mode": embedding_mode,
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
            add_documents(docs, collection=doc_collection, embedding_mode=embedding_mode)
            _viz_cache.pop(doc_collection, None)  # invalidate stale viz

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
    _doc_files.pop(collection_name, None)
    _viz_cache.pop(collection_name, None)
    return {"success": True, "message": f"Collection '{collection_name}' deleted"}


# ── Viz ───────────────────────────────────────────────────────────────────────

@app.get("/collections/{collection_name}/viz", tags=["viz"])
async def get_collection_viz(collection_name: str):
    """Return PCA 2D projection of all chunk embeddings for scatter-plot visualization."""
    if not is_loaded(collection_name):
        load_or_create_store(collection_name)
    if not is_loaded(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
    result = _compute_viz(collection_name)
    return {"collection": collection_name, "points": result["points"]}


@app.post("/collections/{collection_name}/query_similarity", tags=["viz"])
async def get_query_similarity(collection_name: str, body: dict = Body(...)):
    """
    Project a query into the chunk embedding PCA space.
    Returns query 2D position + all chunks with cosine similarity scores.
    Enables the live similarity animation as the user types.
    """
    query = (body.get("query") or "").strip()
    if not query:
        return {"query": None, "chunks": []}

    if not is_loaded(collection_name):
        load_or_create_store(collection_name)
    if not is_loaded(collection_name):
        raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

    result = _compute_viz(collection_name)
    if not result["points"] or result["pca"] is None:
        return {"query": None, "chunks": []}

    import numpy as np

    embedding_mode = get_collection_embedding_mode(collection_name)
    q_vec = np.array(get_embeddings(embedding_mode).embed_query(query), dtype=np.float32).reshape(1, -1)
    q_2d = result["pca"].transform(q_vec)[0]

    vectors = result["vectors"]
    norms = np.linalg.norm(vectors, axis=1)
    q_norm = float(np.linalg.norm(q_vec))
    with np.errstate(divide='ignore', invalid='ignore'):
        sims = (vectors @ q_vec.T).flatten() / (norms * q_norm + 1e-10)

    chunks = []
    for i, pt in enumerate(result["points"]):
        chunks.append({**pt, "score": float(sims[i]) if i < len(sims) else 0.0})
    chunks.sort(key=lambda c: c["score"], reverse=True)

    return {
        "query": {"x": float(q_2d[0]), "y": float(q_2d[1])},
        "chunks": chunks,
    }


# ── Documents ─────────────────────────────────────────────────────────────────

@app.get("/documents/{collection_name}/raw", tags=["documents"])
async def get_document_raw(collection_name: str):
    """Serve raw document bytes for in-browser preview."""
    from fastapi.responses import Response
    entry = _doc_files.get(collection_name)
    if not entry and collection_name in _try_doc_paths:
        path = _try_doc_paths[collection_name]
        try:
            suffix = path.suffix.lower()
            _doc_files[collection_name] = (path.read_bytes(), _FILE_CONTENT_TYPES.get(suffix, 'application/octet-stream'))
            entry = _doc_files.get(collection_name)
        except Exception:
            entry = None
    if not entry:
        raise HTTPException(status_code=404, detail=f"Document '{collection_name}' not available for preview")
    data, media_type = entry
    # Derive a human-readable filename from the collection key
    display_name = collection_name.split("__")[-1] if "__" in collection_name else collection_name
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{display_name}"'},
    )


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
