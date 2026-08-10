"""GoogleBooksTool -- the source of truth for validating recognized books.

Any title/author pair Gemini proposes must be corroborated here via fuzzy
string matching against real Google Books volumes before it's allowed to
flow further downstream. This is how hallucinated titles get rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.core.exceptions import GoogleBooksAPIError
from app.models.domain import ValidatedBook, VisionExtraction
from app.tools.base_tool import BaseTool


@dataclass
class GoogleBooksLookupInput:
    candidate: VisionExtraction


class GoogleBooksTool(BaseTool[GoogleBooksLookupInput, ValidatedBook | None]):
    """Looks up + validates a single candidate. Returns `None` if rejected."""

    name = "google_books_tool"

    def __init__(self, settings: Settings | None = None, cache=None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.timeout_seconds = self.settings.GOOGLE_BOOKS_TIMEOUT_SECONDS
        self.max_retries = self.settings.GOOGLE_BOOKS_MAX_RETRIES
        self.cache = cache

    async def run(self, payload: GoogleBooksLookupInput) -> ValidatedBook | None:
        return await self._execute(self._validate, payload, retry_on=(GoogleBooksAPIError,))

    async def _validate(self, payload: GoogleBooksLookupInput) -> ValidatedBook | None:
        candidate = payload.candidate
        query = f'intitle:"{candidate.title}" inauthor:"{candidate.author}"'

        cache_key = f"gbooks:{query}"
        if self.cache is not None:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                data = cached
            else:
                data = await self._fetch(query)
                await self.cache.set(cache_key, data, ttl=self.settings.CACHE_TTL_BOOKS_SECONDS)
        else:
            data = await self._fetch(query)

        best_match = self._best_match(data, candidate)
        if best_match is None:
            self.logger.info(
                "Rejecting unvalidated candidate '%s' by '%s' (no Google Books match)",
                candidate.title,
                candidate.author,
            )
            return None
        return best_match

    async def _fetch(self, query: str) -> dict[str, Any]:
        params = {"q": query, "maxResults": 5}
        if self.settings.GOOGLE_BOOKS_API_KEY:
            params["key"] = self.settings.GOOGLE_BOOKS_API_KEY

        async with httpx.AsyncClient(timeout=self.settings.GOOGLE_BOOKS_TIMEOUT_SECONDS) as client:
            try:
                resp = await client.get(self.settings.GOOGLE_BOOKS_URL, params=params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as exc:
                raise GoogleBooksAPIError(f"Google Books lookup failed: {exc}") from exc

    def _best_match(
        self, data: dict[str, Any], candidate: VisionExtraction
    ) -> ValidatedBook | None:
        items = data.get("items", [])
        best_score = 0.0
        best_item: dict[str, Any] | None = None

        for item in items:
            info = item.get("volumeInfo", {})
            title = info.get("title", "")
            authors = info.get("authors", [])
            author_str = authors[0] if authors else ""

            title_sim = self._similarity(title, candidate.title)
            author_sim = self._similarity(author_str, candidate.author)

            if (
                title_sim >= self.settings.VALIDATION_MIN_TITLE_SIMILARITY
                and author_sim >= self.settings.VALIDATION_MIN_AUTHOR_SIMILARITY
            ):
                score = (title_sim * 0.65) + (author_sim * 0.35)
                if score > best_score:
                    best_score = score
                    best_item = item

        if best_item is None:
            return None

        return self._to_validated_book(best_item, candidate, best_score)

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

    def _to_validated_book(
        self, item: dict[str, Any], candidate: VisionExtraction, match_confidence: float
    ) -> ValidatedBook:
        info = item.get("volumeInfo", {})
        industry_ids = {i["type"]: i["identifier"] for i in info.get("industryIdentifiers", [])}
        published_date = info.get("publishedDate", "")
        published_year = None
        if published_date and published_date[:4].isdigit():
            published_year = int(published_date[:4])

        return ValidatedBook(
            detection_id=candidate.detection_id,
            title=info.get("title", candidate.title),
            author=(info.get("authors") or [candidate.author])[0],
            publisher=info.get("publisher"),
            isbn_10=industry_ids.get("ISBN_10"),
            isbn_13=industry_ids.get("ISBN_13"),
            cover_url=(info.get("imageLinks") or {}).get("thumbnail"),
            categories=info.get("categories", []),
            description=info.get("description"),
            published_year=published_year,
            average_rating=info.get("averageRating"),
            ratings_count=info.get("ratingsCount"),
            google_volume_id=item.get("id", ""),
            match_confidence=round(match_confidence, 4),
            is_validated=True,
        )
