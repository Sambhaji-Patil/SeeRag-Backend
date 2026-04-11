#faiss index management
from gc import collect
import logging 
import os
from pathlib import Path
from typing import Optional

import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from openai import embeddings

from .config import get_settings
from .embeddings import get_embeddings

logger = logging.getLogger(__name__)
settings = get_settings()

_stores: dict[str,FAISS] = {}

def _index_path(collection: str) -> str:
    return str(Path(settings.faiss_index_path)/ collection)

#load or create
def load_or_create_store(collection: str = "default") -> FAISS:
    """
    Load index from disk if it exists , otherwise return an empty placeholder.
    Stores are registered globally so the API reuses them without re-loading
    """
    if collection in _stores:
        return _stores[collection]
    
    path = _index_path(collection)
    embeddings = get_embeddings()

    if Path(path).exists():
        logger.info(f"Loading FAISS index from {path}")
        store = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        _stores[collection] = store
    else:
        logger.warning(f"No index at {path}. Will create on first Ingest.")
        _stores[collection] = None

    return _stores[collection]

#Ingest
def add_documents(
    docs: list[Document],
    collection: str = "default",
    force_reindex: bool = False
) -> FAISS:
    """
    Adding docs to a FAISS collection.
    - force_reindex: wipe exiting index and rebuild from scratch
    - Persists to disk after every write
    """

    embeddings = get_embeddings()
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
    Path(path).mkdir(parents=True,exist_ok=True)
    store.save_local(Path)
    _stores[collection] = store
    logger.info(f"Index Saved at {path}")
    return store

#rettrieval helpers
def similarity_search_with_scores(
    query = str,
    collection: str = "default",
    k: int = 20
) -> list[tuple[Document,float]]:
    store = _stores.get(collection)
    if store is None:
        raise ValueError(f"Collection '{collection}' not loaded. Ingest documents first.")
    return store.similarity_search_with_relevance_scores(query,k=k)

def get_store(collection: str = "default") -> Optional[FAISS]:
    return _stores.get(collection)

def is_loaded(collection: str = "default") -> bool:
    return _stores.get(collection) is not None

print("[vector_store] Module ready.")