"""RecommendationTool -- recommends books ONLY from the user's own bookshelf.

This is a full rewrite of what used to be a Google-Books-backed "find
similar books on the internet" tool. The product goal changed: recommend
exclusively from the set of books the user actually photographed. Same
class name/module path/`BaseTool` contract, so `RecommendationAgent` and
`Container` need only a constructor-signature change, not a rewire.

Strategy
--------
1. Primary: cosine similarity between the liked book's Gemini embedding
   and every other bookshelf book's embedding (title + subtitle + author +
   categories + description -- see `GeminiEmbeddingTool`).
2. Fallback: for any book missing an embedding (embedding generation
   failed/was skipped upstream), or if the liked book itself has no
   embedding, ask Gemini directly to reason about topical similarity and
   return a score -- same idea, no vector math required.
3. Both paths produce the same `BookshelfRecommendation` shape so the
   caller never needs to know which path ran; `method` on each result
   records which one did.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from app.config import Settings, get_settings
from app.core.exceptions import GeminiAPIError
from app.models.domain import BookshelfRecommendation, ValidatedBook
from app.tools.base_tool import BaseTool

_LLM_SYSTEM_PROMPT = (
    "You compare a 'liked' book against a short list of candidate books "
    "from the same reader's bookshelf and estimate topical/thematic "
    "similarity for each candidate. Respond only with JSON matching the "
    "provided schema -- no prose, no markdown fences."
)

_LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "detection_id": {"type": "string"},
                    "similarity_score": {"type": "number"},
                    "reason": {"type": "string"},
                    "common_topics": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["detection_id", "similarity_score", "reason", "common_topics"],
            },
        }
    },
    "required": ["scores"],
}


@dataclass
class RecommendationInput:
    liked_book: ValidatedBook
    bookshelf: list[ValidatedBook]
    top_k: int = 5


class RecommendationTool(BaseTool[RecommendationInput, list[BookshelfRecommendation]]):
    name = "recommendation_tool"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.timeout_seconds = self.settings.GEMINI_TIMEOUT_SECONDS
        self.max_retries = 1  # LLM fallback failure degrades gracefully; no need to hammer retries
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        from google import genai

        self._client = genai.Client(api_key=self.settings.GEMINI_API_KEY)
        return self._client

    async def run(self, payload: RecommendationInput) -> list[BookshelfRecommendation]:
        return await self._execute(self._recommend, payload, retry_on=(GeminiAPIError,))

    async def _recommend(self, payload: RecommendationInput) -> list[BookshelfRecommendation]:
        candidates = [
            b for b in payload.bookshelf if b.detection_id != payload.liked_book.detection_id
        ]
        if not candidates:
            return []

        results: list[BookshelfRecommendation] = []
        embedded_candidates = [b for b in candidates if b.embedding]
        needs_fallback = [b for b in candidates if not b.embedding]

        if payload.liked_book.embedding and embedded_candidates:
            results.extend(self._cosine_rank(payload.liked_book, embedded_candidates))
        else:
            # Liked book itself has no embedding: nothing to compare against
            # via cosine similarity, so *every* candidate needs the LLM path.
            needs_fallback = candidates

        if needs_fallback:
            try:
                results.extend(await self._llm_fallback_rank(payload.liked_book, needs_fallback))
            except GeminiAPIError as exc:
                self.logger.warning("LLM fallback ranking failed, continuing with embedding-only results: %s", exc)

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        above_threshold = [
            r for r in results if r.similarity_score >= self.settings.RECOMMENDATION_MIN_SIMILARITY
        ]
        final = above_threshold or results  # small shelves: better to show something than nothing
        top_k = min(payload.top_k, self.settings.RECOMMENDATION_MAX_TOP_K)
        return final[:top_k]

    # ------------------------------------------------------------ cosine
    def _cosine_rank(
        self, liked: ValidatedBook, candidates: list[ValidatedBook]
    ) -> list[BookshelfRecommendation]:
        liked_vec = np.asarray(liked.embedding, dtype=np.float64)
        liked_norm = np.linalg.norm(liked_vec)
        if liked_norm == 0:
            return []

        out: list[BookshelfRecommendation] = []
        for book in candidates:
            vec = np.asarray(book.embedding, dtype=np.float64)
            norm = np.linalg.norm(vec)
            similarity = float(np.dot(liked_vec, vec) / (liked_norm * norm)) if norm else 0.0
            similarity = max(0.0, min(1.0, similarity))
            common_topics = sorted(set(liked.categories) & set(book.categories))

            out.append(
                BookshelfRecommendation(
                    title=book.title,
                    author=book.author,
                    cover_url=book.cover_url,
                    similarity_score=round(similarity, 4),
                    categories=book.categories,
                    reason_for_recommendation=self._embedding_reason(similarity, common_topics),
                    common_topics=common_topics,
                    rating=book.average_rating,
                    description=book.description,
                    published_year=book.published_year,
                    isbn_13=book.isbn_13,
                    google_volume_id=book.google_volume_id,
                    method="embedding",
                )
            )
        return out

    @staticmethod
    def _embedding_reason(similarity: float, common_topics: list[str]) -> str:
        if common_topics:
            topics = ", ".join(common_topics[:3])
            return f"Shares themes with your liked book ({topics}); {similarity:.0%} semantic match"
        return f"{similarity:.0%} semantic match on title/description/author"

    # -------------------------------------------------------------- llm
    async def _llm_fallback_rank(
        self, liked: ValidatedBook, candidates: list[ValidatedBook]
    ) -> list[BookshelfRecommendation]:
        import asyncio

        client = self._get_client()
        prompt = self._build_llm_prompt(liked, candidates)
        loop = asyncio.get_event_loop()

        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.settings.GEMINI_MODEL,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                    config={
                        "system_instruction": _LLM_SYSTEM_PROMPT,
                        "temperature": self.settings.GEMINI_TEMPERATURE,
                        "max_output_tokens": self.settings.GEMINI_MAX_OUTPUT_TOKENS,
                        "response_mime_type": "application/json",
                        "response_schema": _LLM_RESPONSE_SCHEMA,
                    },
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise GeminiAPIError(f"Recommendation LLM fallback failed: {exc}") from exc

        return self._parse_llm_response(response, candidates)

    def _build_llm_prompt(self, liked: ValidatedBook, candidates: list[ValidatedBook]) -> str:
        lines = [
            "Liked book:",
            f"- title: {liked.title}",
            f"- author: {liked.author}",
            f"- categories: {', '.join(liked.categories) or 'unknown'}",
            f"- description: {(liked.description or '')[:400]}",
            "",
            "Candidate books (score each 0.0-1.0 for topical/thematic similarity "
            "to the liked book above; return one entry per detection_id):",
        ]
        for c in candidates:
            lines.append(
                f"- detection_id={c.detection_id} | title: {c.title} | author: {c.author} | "
                f"categories: {', '.join(c.categories) or 'unknown'} | "
                f"description: {(c.description or '')[:300]}"
            )
        return "\n".join(lines)

    def _parse_llm_response(
        self, response, candidates: list[ValidatedBook]
    ) -> list[BookshelfRecommendation]:
        try:
            data = json.loads(response.text)
        except (AttributeError, json.JSONDecodeError, TypeError) as exc:
            raise GeminiAPIError(f"Could not parse recommendation LLM response: {exc}") from exc

        by_id = {c.detection_id: c for c in candidates}
        out: list[BookshelfRecommendation] = []
        for entry in data.get("scores", []):
            book = by_id.get(entry.get("detection_id", ""))
            if book is None:
                continue
            score = max(0.0, min(1.0, float(entry.get("similarity_score", 0.0))))
            out.append(
                BookshelfRecommendation(
                    title=book.title,
                    author=book.author,
                    cover_url=book.cover_url,
                    similarity_score=round(score, 4),
                    categories=book.categories,
                    reason_for_recommendation=entry.get("reason", "Similar themes (LLM reasoning)"),
                    common_topics=entry.get("common_topics", []),
                    rating=book.average_rating,
                    description=book.description,
                    published_year=book.published_year,
                    isbn_13=book.isbn_13,
                    google_volume_id=book.google_volume_id,
                    method="llm_fallback",
                )
            )
        return out
