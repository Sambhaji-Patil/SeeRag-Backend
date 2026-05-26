"""
Local embedding model with GPU/CPU auto selection.
- GPU: BAAI/bge-large-en-v1.5 (1024-dim output)
- CPU fallback: BAAI/bge-small-en-v1.5 (384-dim output)
- Runs fully local via sentence-transformers — zero API calls, zero cost
- BGE requires a special query prefix: 'Represent this sentence for searching'
    (documents are embedded as-is; only queries get the prefix)
- LangChain's HuggingFaceEmbeddings handles the prefix automatically
"""

import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic.warnings import UnsupportedFieldAttributeWarning

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Suppress known third-party warning noise triggered inside sentence-transformers stack.
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

#thread pool for running blocking sentence-transformers calls
#inside async contexts without blocking the event loop

_executor = ThreadPoolExecutor(max_workers=2)

def _resolve_device() -> str:
    device = (settings.embedding_device or "auto").lower()
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            logger.warning("Torch unavailable or CUDA check failed; falling back to CPU")
        return "cpu"

    if device != "cpu":
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available; falling back to CPU")
                return "cpu"
        except Exception:
            logger.warning("Torch unavailable or CUDA check failed; falling back to CPU")
            return "cpu"

    return device


def _select_embedding_config() -> tuple[str, int, str]:
    device = _resolve_device()
    if device == "cpu":
        model_name = settings.embedding_model_cpu or settings.embedding_model
        dimensions = settings.embedding_dimensions_cpu or settings.embedding_dimensions
    else:
        model_name = settings.embedding_model
        dimensions = settings.embedding_dimensions

    return model_name, dimensions, device


def get_embedding_info() -> dict[str, str | int]:
    model_name, dimensions, device = _select_embedding_config()
    return {
        "model_name": model_name,
        "dimensions": dimensions,
        "device": device,
    }


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Singleton embedding model

    encode_kwargs:
        normalize_embeddings=True -> required for cosine similarity to work correctly
    
    query_encode_kwargs:
        BGE was finetuned with an instruction-like query prefix.
        We pass that prefix for query encoding only; documents remain unchanged.
    """
    model_name, dimensions, device = _select_embedding_config()

    logger.info(f"Loading embedding model: {model_name} on {device}")
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "normalize_embeddings": settings.embedding_normalize,
            "batch_size": settings.embedding_batch_size,
        },
        query_encode_kwargs={
            "prompt": "Represent this sentence for searching relevant passages: ",
        },
    )
    logger.info(f"Embedding model loaded. Output dim={dimensions}")
    return model

#Async wrappers
# sentence-transformers is synchronous/blocking. We run it in a
# thread pool so FastAPI's async event loop stays unblocked.

async def embed_texts(
    texts: list[str],
    batch_size: int = None
) -> list[list[float]]:
    model = get_embeddings()
    bs = batch_size or settings.embedding_batch_size
    loop = asyncio.get_event_loop()

    all_embeddings: list[list[float]] = []
    for i in range(0,len(texts),bs):
        batch = texts[i:i+bs] #so this will process 32 chunks in one go
        #now run blocking call in thread pool
        vecs = await loop.run_in_executor(
            _executor,
            model.embed_documents,
            batch,
        )
        all_embeddings.extend(vecs)
        logger.debug(f"Embedded batch {i}–{i + len(batch)} ({len(batch)} docs)")
    return all_embeddings

async def embed_query(text: str) -> list[float]:
    model = get_embeddings()
    loop = asyncio.get_event_loop()
    vec = await loop.run_in_executor(
        _executor,
        model.embed_query,
        text
    )
    return vec

#utility function
def cosine_similarity(a:list[float],b:list[float]) -> float:
    a_np, b_np = np.array(a), np.array(b)
    denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if denom == 0:
        return 0.0
    return float(np.dot(a_np,b_np)/denom)

print("[embeddings] Module ready. Model will load on first embed call")
#the model can be preloaded using a warmup call at start