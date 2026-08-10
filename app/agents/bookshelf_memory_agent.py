"""BookshelfMemoryAgent -- persists the recognized shelf for later recall.

Runs after `EmbeddingGenerationAgent`. This is the "Store every validated
book in memory during the request" step from the product spec, plus the
durability needed to serve `POST /api/v1/books/recommend` as an
independent follow-up request (see `app.cache.bookshelf_store`).
"""
from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.cache.bookshelf_store import BookshelfStore
from app.models.state import PipelineState


class BookshelfMemoryAgent(BaseAgent):
    name = "bookshelf_memory_agent"

    def __init__(self, bookshelf_store: BookshelfStore) -> None:
        super().__init__()
        self.bookshelf_store = bookshelf_store

    async def run(self, state: PipelineState) -> dict[str, Any]:
        validated_books = state.get("validated_books", [])
        if not validated_books:
            return {}

        await self.bookshelf_store.save(state["request_id"], validated_books)
        return {}
