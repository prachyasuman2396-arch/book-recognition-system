"""Cache abstraction used by tools for Google Books / Gemini / recommendations.

`CacheBackend` is the interface tools depend on. `RedisCache` is used when
`CACHE_BACKEND=redis` and a Redis server is reachable; otherwise (or on any
connection failure) we transparently fall back to `InMemoryCache` so local
dev and tests never require a running Redis instance.
"""
from __future__ import annotations

import abc
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from app.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def hash_key(*parts: str) -> str:
    """Stable hash for cache keys built from multiple parts (e.g. image bytes)."""
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


class CacheBackend(abc.ABC):
    @abc.abstractmethod
    async def get(self, key: str) -> Any | None: ...

    @abc.abstractmethod
    async def set(self, key: str, value: Any, ttl: int) -> None: ...

    @abc.abstractmethod
    async def delete(self, key: str) -> None: ...

    async def aclose(self) -> None:
        """Release any underlying connections. No-op by default."""
        return None


class InMemoryCache(CacheBackend):
    """LRU + TTL in-memory cache. Safe default with no external dependency."""

    def __init__(self, max_size: int = 5000) -> None:
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max_size = max_size

    async def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    async def set(self, key: str, value: Any, ttl: int) -> None:
        if key in self._store:
            del self._store[key]
        elif len(self._store) >= self._max_size:
            self._store.popitem(last=False)
        self._store[key] = (time.time() + ttl, value)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)


class RedisCache(CacheBackend):
    """Thin async Redis wrapper; falls back to in-memory on connection errors."""

    def __init__(self, redis_url: str, fallback: InMemoryCache) -> None:
        self._redis_url = redis_url
        self._fallback = fallback
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import redis.asyncio as redis  # type: ignore

            self._client = redis.from_url(self._redis_url, decode_responses=True)
            return self._client
        except ImportError:
            logger.warning("redis package not installed; using in-memory cache fallback")
            return None

    async def get(self, key: str) -> Any | None:
        client = self._get_client()
        if client is None:
            return await self._fallback.get(key)
        try:
            raw = await client.get(key)
            return json.loads(raw) if raw is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis GET failed (%s); falling back to memory cache", exc)
            return await self._fallback.get(key)

    async def set(self, key: str, value: Any, ttl: int) -> None:
        client = self._get_client()
        if client is None:
            await self._fallback.set(key, value, ttl)
            return
        try:
            await client.set(key, json.dumps(value, default=str), ex=ttl)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis SET failed (%s); falling back to memory cache", exc)
            await self._fallback.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        client = self._get_client()
        if client is None:
            await self._fallback.delete(key)
            return
        try:
            await client.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis DELETE failed (%s)", exc)
            await self._fallback.delete(key)

    async def aclose(self) -> None:
        if self._client is None:
            return
        try:
            # redis-py >=5 renamed `close()` to `aclose()`; support both so
            # this works across the version range in requirements.txt.
            closer = getattr(self._client, "aclose", None) or self._client.close
            await closer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error closing Redis client (%s)", exc)
        finally:
            self._client = None


_cache_singleton: CacheBackend | None = None


def get_cache(settings: Settings | None = None) -> CacheBackend:
    global _cache_singleton
    if _cache_singleton is not None:
        return _cache_singleton

    settings = settings or get_settings()
    fallback = InMemoryCache(max_size=settings.CACHE_SIZE)
    if settings.CACHE_BACKEND == "redis":
        _cache_singleton = RedisCache(settings.REDIS_URL, fallback)
    else:
        _cache_singleton = fallback
    return _cache_singleton
