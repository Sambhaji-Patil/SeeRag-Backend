"""
Try Docs cache (Redis-only):
1. Exact-match cache in Redis
2. Semantic cache using Redis Vector Search (RediSearch)
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Optional

import numpy as np

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

CACHE_EMBEDDING_MODE = "openai-small"
EXACT_PREFIX = "rag:try:exact:"
SEMANTIC_INDEX = "rag:try:semantic"
SEMANTIC_PREFIX = "rag:try:sem:"
VECTOR_FIELD = "vector"
PAYLOAD_FIELD = "payload"
COLLECTION_FIELD = "collection_key"
PARAMS_FIELD = "params_key"


def _build_redis_client():
    try:
        import redis

        client = redis.from_url(settings.redis_url, decode_responses=False)
        client.ping()
        logger.info("Redis cache connected")
        return client
    except Exception as exc:
        logger.warning("Redis unavailable (%s); caching disabled", exc)
        return None


_client = _build_redis_client()
_semantic_ready = False
_semantic_failed = False


def _ensure_semantic_index() -> bool:
    global _semantic_ready, _semantic_failed
    if _semantic_ready:
        return True
    if _semantic_failed or _client is None:
        return False

    try:
        _client.execute_command("FT.INFO", SEMANTIC_INDEX)
        _semantic_ready = True
        return True
    except Exception:
        pass

    try:
        dim = str(int(settings.embedding_dimensions_openai))
        _client.execute_command(
            "FT.CREATE",
            SEMANTIC_INDEX,
            "ON",
            "HASH",
            "PREFIX",
            1,
            SEMANTIC_PREFIX,
            "SCHEMA",
            "query",
            "TEXT",
            COLLECTION_FIELD,
            "TAG",
            PARAMS_FIELD,
            "TAG",
            VECTOR_FIELD,
            "VECTOR",
            "HNSW",
            6,
            "TYPE",
            "FLOAT32",
            "DIM",
            dim,
            "DISTANCE_METRIC",
            "COSINE",
        )
        _semantic_ready = True
        return True
    except Exception as exc:
        logger.warning("Failed to create Redis vector index: %s", exc)
        _semantic_failed = True
        return False


def _exact_key(query: str, collection_key: str, params_key: str) -> str:
    payload = f"{query}::{collection_key}::{params_key}"
    return EXACT_PREFIX + hashlib.sha256(payload.encode()).hexdigest()[:32]


def _tag_hash(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:16]


def _vector_bytes(vec: list[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()


def _decode(value: bytes | str | None) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


# Exact cache

def get_exact(query: str, collection_key: str, params_key: str) -> Optional[dict]:
    if _client is None:
        return None
    key = _exact_key(query, collection_key, params_key)
    raw = _client.get(key)
    if not raw:
        return None
    payload = _decode(raw)
    if payload is None:
        return None
    return json.loads(payload)


def set_exact(query: str, collection_key: str, params_key: str, value: dict) -> None:
    if _client is None:
        return
    key = _exact_key(query, collection_key, params_key)
    serialized = json.dumps(value)
    _client.set(key, serialized.encode("utf-8"), ex=settings.cache_ttl_seconds)


# Semantic cache

def get_semantic(query_vec: list[float], collection_key: str, params_key: str) -> Optional[dict]:
    if _client is None:
        return None
    if not _ensure_semantic_index():
        return None

    vec = _vector_bytes(query_vec)
    k = 4
    params_tag = _tag_hash(params_key)
    query = (
        f"(@{COLLECTION_FIELD}:{{{collection_key}}} "
        f"@{PARAMS_FIELD}:{{{params_tag}}})"
        f"=>[KNN {k} @{VECTOR_FIELD} $vec AS score]"
    )

    try:
        res = _client.execute_command(
            "FT.SEARCH",
            SEMANTIC_INDEX,
            query,
            "PARAMS",
            2,
            "vec",
            vec,
            "RETURN",
            2,
            PAYLOAD_FIELD,
            "score",
            "DIALECT",
            2,
        )
    except Exception as exc:
        logger.warning("Redis semantic search failed: %s", exc)
        return None

    if not res or res[0] == 0:
        return None

    best_similarity = 0.0
    best_payload = None

    for i in range(1, len(res), 2):
        fields = res[i + 1]
        payload = None
        distance = None
        for j in range(0, len(fields), 2):
            name = _decode(fields[j]) or ""
            value = fields[j + 1]
            if name == PAYLOAD_FIELD:
                payload = _decode(value)
            elif name == "score":
                distance = float(_decode(value) or 0)
        if payload is None or distance is None:
            continue
        similarity = 1.0 - distance
        if similarity > best_similarity:
            best_similarity = similarity
            best_payload = payload

    if best_payload and best_similarity >= settings.semantic_cache_threshold:
        logger.info("Semantic cache hit (score=%.3f)", best_similarity)
        return json.loads(best_payload)

    return None


def set_semantic(
    query_vec: list[float],
    query: str,
    collection_key: str,
    params_key: str,
    response: dict,
) -> None:
    if _client is None:
        return
    if not _ensure_semantic_index():
        return

    key = f"{SEMANTIC_PREFIX}{uuid.uuid4().hex}"
    payload = json.dumps(response)

    _client.hset(
        key,
        mapping={
            "query": query,
            COLLECTION_FIELD: collection_key,
            PARAMS_FIELD: _tag_hash(params_key),
            VECTOR_FIELD: _vector_bytes(query_vec),
            PAYLOAD_FIELD: payload,
        },
    )
    _client.expire(key, settings.cache_ttl_seconds)


def cache_connected() -> bool:
    if _client is None:
        return False
    try:
        return bool(_client.ping())
    except Exception:
        return False


def _info_to_dict(raw: list) -> dict:
    info = {}
    if not raw:
        return info
    for i in range(0, len(raw), 2):
        key = _decode(raw[i]) or ""
        info[key] = raw[i + 1]
    return info


def get_cache_stats() -> dict:
    if _client is None:
        return {"system": "disabled", "exact_matches_cached": 0, "semantic_matches_cached": 0}

    stats = {"system": "redis"}
    try:
        stats["exact_matches_cached"] = _client.dbsize()
    except Exception:
        stats["exact_matches_cached"] = "unknown"

    try:
        info = _client.execute_command("FT.INFO", SEMANTIC_INDEX)
        info_map = _info_to_dict(info)
        num_docs = info_map.get("num_docs", 0)
        stats["semantic_matches_cached"] = int(num_docs) if num_docs is not None else 0
    except Exception:
        stats["semantic_matches_cached"] = "unknown"

    return stats


print("[cache] Module ready")
