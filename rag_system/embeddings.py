"""
Local + API embedding models with GPU/CPU aware defaults.
- BGE-large (1024-dim) for GPU
- BGE-small (384-dim) for fast local CPU
- OpenAI text-embedding-3-small (1536-dim) for fast CPU via API
- BGE requires a special query prefix for queries only
"""

import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from pydantic.warnings import UnsupportedFieldAttributeWarning

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Suppress known third-party warning noise triggered inside sentence-transformers stack.
warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

#thread pool for running blocking sentence-transformers calls
#inside async contexts without blocking the event loop

_executor = ThreadPoolExecutor(max_workers=2)

EMBEDDING_MODES: tuple[str, ...] = ("bge-large", "bge-small", "openai-small")
_EMBEDDING_CACHE: dict[str, Embeddings] = {}


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


def get_default_embedding_mode() -> str:
    return "bge-large" if _resolve_device() == "cuda" else "openai-small"


def normalize_embedding_mode(mode: str | None) -> str:
    if mode is None or mode == "auto":
        return get_default_embedding_mode()
    if mode not in EMBEDDING_MODES:
        raise ValueError(f"Unknown embedding mode: {mode}")
    return mode


def infer_embedding_mode_from_dim(dim: int) -> str | None:
    dim_map = {
        int(settings.embedding_dimensions): "bge-large",
        int(settings.embedding_dimensions_cpu): "bge-small",
        int(settings.embedding_dimensions_openai): "openai-small",
    }
    return dim_map.get(int(dim))


def _embedding_spec(mode: str) -> dict[str, Any]:
    if mode == "bge-large":
        return {
            "provider": "local",
            "model_name": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "device": _resolve_device(),
        }
    if mode == "bge-small":
        return {
            "provider": "local",
            "model_name": settings.embedding_model_cpu,
            "dimensions": settings.embedding_dimensions_cpu,
            "device": _resolve_device(),
        }
    return {
        "provider": "openai",
        "model_name": settings.embedding_model_openai,
        "dimensions": settings.embedding_dimensions_openai,
        "device": "api",
    }


def get_embedding_info(mode: str | None = None) -> dict[str, str | int]:
    resolved = normalize_embedding_mode(mode)
    spec = _embedding_spec(resolved)
    return {
        "mode": resolved,
        "provider": spec["provider"],
        "model_name": spec["model_name"],
        "dimensions": int(spec["dimensions"]),
        "device": spec["device"],
    }


def get_embeddings_runtime_info() -> dict[str, Any]:
    default_mode = get_default_embedding_mode()
    options = []
    for mode in EMBEDDING_MODES:
        info = get_embedding_info(mode)
        options.append({
            "id": mode,
            "model_name": info["model_name"],
            "dimensions": info["dimensions"],
            "provider": info["provider"],
            "recommended": mode == default_mode,
        })
    return {
        "default_mode": default_mode,
        "device": _resolve_device(),
        "options": options,
    }


def get_embeddings(mode: str | None = None) -> Embeddings:
    """
    Singleton embedding model per mode.

    encode_kwargs:
        normalize_embeddings=True -> required for cosine similarity to work correctly
    
    query_encode_kwargs:
        BGE was finetuned with an instruction-like query prefix.
        We pass that prefix for query encoding only; documents remain unchanged.
    """
    resolved = normalize_embedding_mode(mode)
    cached = _EMBEDDING_CACHE.get(resolved)
    if cached is not None:
        return cached

    spec = _embedding_spec(resolved)
    logger.info("Loading embedding model: %s on %s", spec["model_name"], spec["device"])
    if spec["provider"] == "openai":
        model = OpenAIEmbeddings(
            model=spec["model_name"],
            dimensions=int(spec["dimensions"]),
            openai_api_key=settings.openai_api_key,
        )
    else:
        model = HuggingFaceEmbeddings(
            model_name=spec["model_name"],
            model_kwargs={
                "device": spec["device"],
            },
            encode_kwargs={
                "normalize_embeddings": settings.embedding_normalize,
                "batch_size": settings.embedding_batch_size,
            },
            query_encode_kwargs={
                "prompt": "Represent this sentence for searching relevant passages: ",
            },
        )

    logger.info("Embedding model loaded. Output dim=%s", spec["dimensions"])
    _EMBEDDING_CACHE[resolved] = model
    return model

#Async wrappers
# sentence-transformers is synchronous/blocking. We run it in a
# thread pool so FastAPI's async event loop stays unblocked.

async def embed_texts(
    texts: list[str],
    batch_size: int = None,
    embedding_mode: str | None = None,
) -> list[list[float]]:
    model = get_embeddings(embedding_mode)
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

async def embed_query(text: str, embedding_mode: str | None = None) -> list[float]:
    model = get_embeddings(embedding_mode)
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