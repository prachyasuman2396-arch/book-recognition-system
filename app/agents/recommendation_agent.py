"""RecommendationAgent -- resolves `liked_book_title` against the
bookshelf and returns ranked, bookshelf-only recommendations.

MODIFIED from the original: no longer unconditionally recommends for every
validated book against the internet. It now requires a `liked_book_title`
(set on `PipelineState` only for the `/books/recommend` follow-up call --
see `app.graph.runner.run_recommendation`) and matches it, by fuzzy title
comparison, against `state["validated_books"]` (the bookshelf). During the
initial `/books/recognize` request `liked_book_title` is unset, so this
agent is a no-op there (returns no recommendations) -- recommendations
only happen once the user has told us which book they liked.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from app.agents.base_agent import BaseAgent
from app.config import Settings, get_settings
from app.models.domain import ValidatedBook
from app.models.state import PipelineState
from app.tools.recommendation_tool import RecommendationInput, RecommendationTool

#: Below this fuzzy-match ratio, we treat the title as "not on this shelf"
#: rather than guessing which book the user meant.
_TITLE_MATCH_THRESHOLD = 0.6


class RecommendationAgent(BaseAgent):
    name = "recommendation_agent"

    def __init__(self, recommendation_tool: RecommendationTool, settings: Settings | None = None) -> None:
        super().__init__()
        self.recommendation_tool = recommendation_tool
        self.settings = settings or get_settings()

    async def run(self, state: PipelineState) -> dict[str, Any]:
        bookshelf = state.get("validated_books", [])
        liked_title = state.get("liked_book_title")

        if not liked_title or not bookshelf:
            # No user query yet (e.g. mid-`/books/recognize`) -- nothing to do.
            return {"recommendations": []}

        liked_book = self.find_book_by_title(bookshelf, liked_title)
        if liked_book is None:
            return {
                "recommendations": [],
                "warnings": ["Book not found in uploaded bookshelf."],
            }

        top_k = state.get("recommendation_top_k") or self.settings.RECOMMENDATION_DEFAULT_TOP_K
        recommendations = await self.recommendation_tool.run(
            RecommendationInput(liked_book=liked_book, bookshelf=bookshelf, top_k=top_k)
        )
        return {"recommendations": recommendations}

    @staticmethod
    def find_book_by_title(bookshelf: list[ValidatedBook], title: str) -> ValidatedBook | None:
        """Fuzzy, case-insensitive title match against the bookshelf.

        Exact (normalized) match wins outright; otherwise the closest match
        above `_TITLE_MATCH_THRESHOLD` is returned, so "atomic habit" or a
        minor typo still resolves to "Atomic Habits" without over-matching
        an unrelated title.
        """
        normalized = title.strip().lower()
        for book in bookshelf:
            if book.title.strip().lower() == normalized:
                return book

        best_book: ValidatedBook | None = None
        best_score = 0.0
        for book in bookshelf:
            score = SequenceMatcher(None, book.title.strip().lower(), normalized).ratio()
            if score > best_score:
                best_score = score
                best_book = book

        return best_book if best_score >= _TITLE_MATCH_THRESHOLD else None
