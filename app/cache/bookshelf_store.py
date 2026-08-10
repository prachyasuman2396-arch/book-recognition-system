"""BookshelfStore -- persists a request's validated bookshelf between calls.

`/books/recognize` and `/books/recommend` are two *separate* HTTP requests
(the "Wait for User Query" step in the product spec is literally the human
looking at the recognized shelf and then deciding which book they liked).
LangGraph's own checkpointer isn't a good fit for bridging that gap in
production: `MemorySaver` is per-process (breaks across replicas/restarts)
and the sqlite checkpointer is per-node-local-disk (breaks across replicas
in a typical Kubernetes deployment -- see `deploy/k8s-deployment.yaml`).

Instead we reuse the existing `CacheBackend` abstraction (already the
project's answer to "state that must be readable by any worker": Redis in
production, in-memory for local/dev/tests) purely as a keyed store for the
bookshelf, with the same fallback semantics tools already rely on.
"""
from __future__ import annotations

from app.cache.cache_backend import CacheBackend
from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.domain import ValidatedBook

logger = get_logger(__name__)


class BookshelfStore:
    """Thin, typed wrapper around `CacheBackend` for `list[ValidatedBook]`."""

    def __init__(self, cache: CacheBackend, settings: Settings | None = None) -> None:
        self.cache = cache
        self.settings = settings or get_settings()

    def _key(self, request_id: str) -> str:
        return f"{self.settings.BOOKSHELF_STORE_KEY_PREFIX}:{request_id}"

    async def save(self, request_id: str, books: list[ValidatedBook]) -> None:
        payload = [book.model_dump(mode="json") for book in books]
        await self.cache.set(
            self._key(request_id), payload, ttl=self.settings.CACHE_TTL_BOOKSHELF_SECONDS
        )
        logger.info("Stored bookshelf for request_id=%s (%d books)", request_id, len(books))

    async def load(self, request_id: str) -> list[ValidatedBook] | None:
        raw = await self.cache.get(self._key(request_id))
        if raw is None:
            return None
        return [ValidatedBook(**item) for item in raw]
