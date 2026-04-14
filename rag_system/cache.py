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