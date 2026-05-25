import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader
}

#Loaders
def load_file(file_path: str) -> list[Document]:
    """This function auto detects the file type and loads to the langchain documents"""
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")
    loader = loader_cls(file_path)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages from {file_path}")
    return docs

def load_url(url: str) -> list[Document]:
    """Scrape a webpage and return Documents"""
    loader = WebBaseLoader(url)
    logger.info(f"Loaded data from {url}")
    return loader.load()

#Cleaning
def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars
    text = re.sub(r"[ \t]+", " ", text)     # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse excess blank lines
    return text.strip()

#Splitter
def build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function = len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )

#Metadata Enrichment
def _stable_hash(text:str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]

def enrich_metadata(
        chunks: list[Document],
        source_id: Optional[str] = None,
        extra_meta: Optional[dict] = None
) -> list[Document]:
    """
    Production enrichment:
    - stable doc_id from content hash (dedup-safe)
    - chunk_index for indexing
    - char_count for downstream token budget checks
    - prev/next chunk IDs for context stitching
    """
    chunk_ids = [_stable_hash(c.page_content) for c in chunks]
    enriched = []
    for i, (doc,cid) in enumerate(zip(chunks,chunk_ids)):
        meta = {
            **doc.metadata,
            "doc_id": cid,
            "chunk_index": i,
            "char_count": len(doc.page_content),
            "prev_chunk_id": chunk_ids[i-1] if i > 0 else None,
            "next_chunk_id": chunk_ids[i+1] if i < len(chunks) - 1 else None,
            "source_id": source_id or "unknown"
        }
        if extra_meta:
            meta.update(extra_meta)
        enriched.append(Document(page_content=doc.page_content,metadata=meta))
    return enriched

#Main pipeline
def process_texts(
    texts: list[str],
    metadatas: Optional[list[dict]] = None,
    source_id: Optional[str] = None
) -> list[Document]:
    """
    Full ingestion Pipeline:
    1. Wrap raw strings in Documents
    2. Clean_text
    3. Split into Chunks
    4. Filter junk chunks
    5. Enrich Metadata
    """
    splitter = build_splitter()

    raw_docs = [
        Document(page_content=clean_text(t), metadata = m or {})
        for t,m in zip(texts,metadatas or [{}]*len(texts))
    ]

    chunks = splitter.split_documents(raw_docs)

    #drop tiny or near to empty chunks
    chunks = [
        c for c in chunks
        if len(c.page_content.strip()) >= settings.min_chunk_size
    ]

    chunks = enrich_metadata(chunks,source_id=source_id)
    logger.info(f"Processed {len(texts)} texts -> {len(chunks)} chunks")
    return chunks

def process_file(file_path: str, display_name: str | None = None) -> list[Document]:
    """End to end ingestion of file path. display_name overrides the temp path as source_id."""
    docs = load_file(file_path)
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    source = display_name if display_name else file_path
    return process_texts(texts, metas, source_id=source)

print("[document_processor] Module ready")