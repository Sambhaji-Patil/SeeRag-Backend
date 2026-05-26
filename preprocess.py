"""
Preprocess Try Docs into FAISS indices.

Usage:
  python preprocess.py
  python preprocess.py --embedding-mode bge-large
  python preprocess.py --docs-dir "rag_system/Try Docs"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rag_system.config import get_settings
from rag_system.document_processor import process_file
from rag_system.vector_store import add_documents


SUPPORTED_EXTS = {".pdf", ".txt", ".md"}


def safe_coll_name(filename: str) -> str:
    import re
    stem = Path(filename).stem if filename else "doc"
    safe = re.sub(r"[^a-z0-9-]", "_", stem.lower())
    safe = re.sub(r"_+", "_", safe).strip("_")[:40]
    return safe or "doc"


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Preprocess Try Docs into FAISS indices")
    parser.add_argument(
        "--docs-dir",
        default=settings.try_docs_path,
        help="Folder containing Try Docs (PDF/TXT/MD)",
    )
    parser.add_argument(
        "--embedding-mode",
        default="auto",
        help="Embedding mode (auto, bge-large, bge-small, openai-small)",
    )
    parser.add_argument(
        "--prefix",
        default=settings.try_docs_prefix,
        help="Collection prefix for Try Docs",
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        print(f"Try Docs folder not found: {docs_dir}")
        return 1

    embedding_mode = None if args.embedding_mode == "auto" else args.embedding_mode

    files = [p for p in sorted(docs_dir.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        print(f"No supported files found in: {docs_dir}")
        return 0

    for path in files:
        collection = f"{args.prefix}{safe_coll_name(path.name)}"
        print(f"Indexing: {path.name} -> {collection}")

        docs = process_file(str(path), display_name=path.name)
        add_documents(
            docs,
            collection=collection,
            force_reindex=True,
            embedding_mode=embedding_mode,
        )
        print(f"  chunks: {len(docs)}")

    print("Done. Try Docs indices are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
