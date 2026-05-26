#faiss index management
import json
import logging
import time
import os
from pathlib import Path
from typing import Optional

import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from .config import get_settings
from .embeddings import (
    get_embeddings,
    get_embedding_info,
    get_default_embedding_mode,
    infer_embedding_mode_from_dim,
    normalize_embedding_mode,
)

logger = logging.getLogger(__name__)
settings = get_settings()

_stores: dict[str, FAISS] = {}
# Tracks last-used timestamp per collection (epoch seconds) for TTL-based cleanup
_last_used: dict[str, float] = {}
_collection_embeddings: dict[str, str] = {}

_EMBEDDING_META_FILE = "embedding.json"


def _embedding_meta_path(collection: str) -> Path:
    return Path(_index_path(collection)) / _EMBEDDING_META_FILE


def _read_embedding_meta(collection: str) -> dict | None:
    meta_path = _embedding_meta_path(collection)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read embedding metadata for '%s'", collection)
        return None


def _write_embedding_meta(collection: str, info: dict) -> None:
    meta_path = _embedding_meta_path(collection)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def get_collection_embedding_mode(collection: str) -> Optional[str]:
    if collection in _collection_embeddings:
        return _collection_embeddings[collection]

    meta = _read_embedding_meta(collection)
    if meta and isinstance(meta, dict):
        mode = meta.get("mode") or meta.get("embedding_mode")
        if isinstance(mode, str):
            _collection_embeddings[collection] = mode
            return mode
    return None


def resolve_embedding_mode_for_collections(
    collections: list[str],
    requested_mode: Optional[str] = None,
) -> str:
    requested = None
    if requested_mode and requested_mode != "auto":
        requested = normalize_embedding_mode(requested_mode)

    modes = []
    for coll in collections:
        mode = get_collection_embedding_mode(coll)
        if mode:
            modes.append(mode)

    if requested and modes and any(m != requested for m in modes):
        logger.warning(
            "Embedding mode mismatch (requested=%s, existing=%s). Using existing.",
            requested,
            sorted(set(modes)),
        )
        return modes[0]

    if requested:
        return requested

    if modes:
        if any(m != modes[0] for m in modes):
            logger.warning("Multiple embedding modes across collections: %s", sorted(set(modes)))
        return modes[0]

    return get_default_embedding_mode()

def _index_path(collection: str) -> str:
    return str(Path(settings.faiss_index_path)/ collection)

#load or create
def load_or_create_store(collection: str = "default") -> FAISS:
    """
    Load index from disk if it exists, otherwise return an empty placeholder.
    Stores are registered globally so the API reuses them without re-loading.
    """
    if collection in _stores:
        _last_used[collection] = time.time()
        return _stores[collection]

    path = _index_path(collection)
    embedding_mode = resolve_embedding_mode_for_collections([collection])
    embeddings = get_embeddings(embedding_mode)

    if Path(path).exists():
        logger.info(f"Loading FAISS index from {path}")
        store = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        expected_dim = int(get_embedding_info(embedding_mode)["dimensions"])
        if store.index.d != expected_dim:
            inferred_mode = infer_embedding_mode_from_dim(store.index.d)
            if inferred_mode and inferred_mode != embedding_mode:
                embeddings = get_embeddings(inferred_mode)
                store = FAISS.load_local(
                    path,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                embedding_mode = inferred_mode
                expected_dim = int(get_embedding_info(embedding_mode)["dimensions"])

        if store.index.d != expected_dim:
            logger.error(
                "Embedding dim mismatch for collection '%s': index dim=%s, expected=%s. "
                "Re-ingest with force_reindex or delete the collection.",
                collection,
                store.index.d,
                expected_dim,
            )
            _stores[collection] = None
        else:
            _stores[collection] = store
            _collection_embeddings[collection] = embedding_mode
            _write_embedding_meta(collection, get_embedding_info(embedding_mode))
    else:
        logger.warning(f"No index at {path}. Will create on first Ingest.")
        _stores[collection] = None

    _last_used[collection] = time.time()
    return _stores[collection]

#Ingest
def add_documents(
    docs: list[Document],
    collection: str = "default",
    force_reindex: bool = False,
    embedding_mode: Optional[str] = None,
) -> FAISS:
    """
    Adding docs to a FAISS collection.
    - force_reindex: wipe exiting index and rebuild from scratch
    - Persists to disk after every write
    """

    existing_mode = get_collection_embedding_mode(collection)
    selected_mode = resolve_embedding_mode_for_collections([collection], embedding_mode)
    if existing_mode and existing_mode != selected_mode and not force_reindex:
        raise ValueError(
            f"Embedding mode mismatch for '{collection}': existing={existing_mode}, requested={selected_mode}. "
            "Use force_reindex to rebuild."
        )

    embeddings = get_embeddings(selected_mode)
    path = _index_path(collection)

    existing = None if force_reindex else _stores.get(collection)

    if existing is not None:
        logger.info(f"Merging {len(docs)} docs into existing collection '{collection}'")
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]
        existing.add_texts(texts, metadatas=metas)
        store = existing
    else:
        logger.info(f"Creating a new FAISS index for collection '{collection}' with {len(docs)} docs")
        store = FAISS.from_documents(docs, embeddings)
    
    #persist
    Path(path).mkdir(parents=True, exist_ok=True)
    store.save_local(path)
    _stores[collection] = store
    _last_used[collection] = time.time()
    _collection_embeddings[collection] = selected_mode
    _write_embedding_meta(collection, get_embedding_info(selected_mode))
    
    # Prebuild BM25 index on ingest
    from .retriever import _bm25_cache, _get_bm25
    if collection in _bm25_cache:
        del _bm25_cache[collection]
    _get_bm25(collection)
    
    logger.info(f"Index Saved at {path}")
    return store

#rettrieval helpers
def similarity_search_with_scores(
    query=str,
    collection: str = "default",
    k: int = 20,
) -> list[tuple[Document, float]]:
    store = _stores.get(collection)
    if store is None:
        raise ValueError(f"Collection '{collection}' not loaded. Ingest documents first.")
    _last_used[collection] = time.time()
    return store.similarity_search_with_relevance_scores(query, k=k)

def get_store(collection: str = "default") -> Optional[FAISS]:
    return _stores.get(collection)

def is_loaded(collection: str = None) -> bool:
    if collection is None:
        return any(s is not None for s in _stores.values())
    return _stores.get(collection) is not None


def list_collections() -> list[str]:
    """Return all collection names that have a persisted index on disk or are loaded in memory."""
    base = Path(settings.faiss_index_path)
    on_disk = [d.name for d in base.iterdir() if d.is_dir()] if base.exists() else []
    in_memory = [name for name, store in _stores.items() if store is not None]
    return sorted(set(on_disk + in_memory))


def get_collection_stats(collection: str) -> dict:
    """Return chunk count, size-on-disk, and load status for a collection."""
    store = load_or_create_store(collection)
    path = _index_path(collection)

    embedding_mode = get_collection_embedding_mode(collection)
    embedding_info = get_embedding_info(embedding_mode) if embedding_mode else None

    chunk_count = 0
    if store is not None and hasattr(store, "index"):
        chunk_count = store.index.ntotal

    size_mb = 0.0
    p = Path(path)
    if p.exists():
        size_mb = round(
            sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024),
            3,
        )

    return {
        "name": collection,
        "chunk_count": chunk_count,
        "size_mb": size_mb,
        "loaded": store is not None,
        "index_path": path,
        "embedding_mode": embedding_mode,
        "embedding_dimensions": embedding_info["dimensions"] if embedding_info else None,
        "embedding_provider": embedding_info["provider"] if embedding_info else None,
    }


def cleanup_stale_collections(ttl_seconds: int = 1800) -> list[str]:
    """
    Delete all collections that have not been accessed within ttl_seconds.
    Called periodically by the API to reclaim memory and disk from idle sessions.
    Returns the list of collection names that were removed.
    """
    cutoff = time.time() - ttl_seconds
    stale = [name for name, ts in list(_last_used.items()) if ts < cutoff]
    for name in stale:
        logger.info(f"Cleaning up stale collection '{name}' (idle > {ttl_seconds}s)")
        delete_collection(name)
    return stale


def delete_collection(collection: str) -> bool:
    """Remove a collection from memory and delete its index directory from disk."""
    import shutil

    path = _index_path(collection)
    if collection in _stores:
        del _stores[collection]
    if collection in _collection_embeddings:
        del _collection_embeddings[collection]

    # Local import to avoid circular dependency with retriever
    from .retriever import _bm25_cache
    if collection in _bm25_cache:
        del _bm25_cache[collection]

    p = Path(path)
    if p.exists():
        shutil.rmtree(path)
        return True
    return False


def get_session_collections(session_id: str) -> list[str]:
    """Return all per-doc sub-collections for this session (format: {session_id}__{docname})."""
    prefix = f"{session_id}__"
    found: set[str] = set()
    for name, store in _stores.items():
        if name.startswith(prefix) and store is not None:
            found.add(name)
    base = Path(settings.faiss_index_path)
    if base.exists():
        for d in base.iterdir():
            if d.is_dir() and d.name.startswith(prefix):
                found.add(d.name)
    return sorted(found)

print("[vector_store] Module ready.")