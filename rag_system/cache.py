"""
Two-layer cache:
1. Exact-match hash cache (Redis/in-memory fallback)
2. Semantic near-duplicate cache using cosine similarity on query embeddings

Semantic caching prevents re-querying the LLM for paraphrased versions of the 
same question - a major cost & latency win in production
"""

import hashlib
import json 
import logging
import time
from typing import Optional

from google_crc32c import value
import numpy as np

from config import get_settings
from embeddings import cosine_similarity

logger = logging.getLogger(__name__)
settings = get_settings()

# In memory fallback (used when Redis is unavailable)

class InMemoryCache:
    def __init__(self,ttl: int = 3600, max_size: int = 1000):
        self._store: dict[str,tuple[str, float]] = {} # Key -> (value, expiry)
        self.ttl = ttl
        self.max_size = max_size
    
    def get(self,key: str) -> Optional[str]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expiry = entry
        if time.time() > expiry:
            del self._store[key]
            return None
        return value
    
    def set(self,key: str, value: str) -> None:
        if len(self._store) >= self.max_size:
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[key] = (value, time.time() + self.ttl)
    
    def ping(self) -> bool:
        return True
    
def _build_redis_client():
    try:
        import redis
        client = redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        logger.info("Redis cache connected")
        return client
    except Exception as e:
        logger.warning(f"Redis unavalaible ({e}) - using in-memory cache.")
        return InMemoryCache(ttl=settings.cache_ttl_seconds)

_cache_client = _build_redis_client()

# Exact match cache
def _cache_key(query: str, collection: str, mode: str) -> str:
    payload = f"{query}::{collection}::{mode}"
    return "rag:exact:" + hashlib.sha256(payload.encode()).hexdigest()[:32]

def get_exact(query: str, collection: str, mode: str) -> Optional[dict]:
    key = _cache_key(query, collection, mode)
    raw = _cache_client.get(key)
    if raw:
        logger.debug(f"Exact cache hit: {key[:16]}...")
        return json.loads(raw)
    return None

def set_exact(query: str, collection: str, mode: str, value: str) -> None:
    key = _cache_key(query,collection,mode)
    serialized = json.dumps(value)
    if hasattr(_cache_client,"setex"):
        _cache_client.setex(key,settings.cache_ttl_seconds,serialized)
    else:
        _cache_client.set(key,serialized)

# Semantic Cache
# stores (embedding, serialized_response) pairs keyed by short hash
_semantic_index: list[tuple[list[float],str,dict]] = [] # (vec,key,response)

def get_semantic(query_vec: list[float]) -> Optional[dict]:
    """Return the cache response if cosine similarity > threshold"""
    best_score = 0.0
    best_response = None
    for vec, _,response in _semantic_index:
        score = cosine_similarity(query_vec,vec)
        if score > best_score:
            best_score = score
            best_response = response
    if best_score >= settings.semantic_cache_threshold:
        logger.info(f"Semantic Cache hit (score={best_score:.3f})")
        return best_response
    return None

def set_semantic(query_vec: list[float], query: str, response: dict) -> None:
    h = hashlib.md5(query.encode()).hexdigest()[:8]
    _semantic_index.append((query_vec, h, response))
    if len(_semantic_index) > 5000: # cap memory
        _semantic_index.pop(0)

def cache_connected() -> bool:
    try:
        return bool(_cache_client.ping())
    except Exception:
        return False

print("[cache] Module ready")