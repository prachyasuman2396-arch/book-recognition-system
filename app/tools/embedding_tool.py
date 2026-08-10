"""GeminiEmbeddingTool -- generates semantic embeddings for bookshelf books.

Mirrors `GeminiVisionTool`'s structure exactly (same `google-genai` client,
same executor-offload pattern for the blocking SDK call, same retry/timeout
via `BaseTool._execute`) so there is nothing new to learn here if you've
already read that file.

Embedding text is built from title + subtitle + author + categories +
description so cosine similarity in `RecommendationTool` reflects topical
closeness, not just title string similarity. Per-book results are cached
(keyed by a hash of the embedding input text) using the same `CacheBackend`
already used for Google Books/recommendation lookups, since embeddings for
a given book's metadata are stable and reused across every future request
that happens to detect the same book.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.cache.cache_backend import CacheBackend, hash_key
from app.config import Settings, get_settings
from app.core.exceptions import EmbeddingAPIError
from app.models.domain import TokenUsage, ValidatedBook
from app.tools.base_tool import BaseTool


@dataclass
class EmbeddingInput:
    books: list[ValidatedBook]


@dataclass
class EmbeddingOutput:
    # detection_id -> embedding vector
    embeddings: dict[str, list[float]]
    token_usage: TokenUsage


class GeminiEmbeddingTool(BaseTool[EmbeddingInput, EmbeddingOutput]):
    name = "gemini_embedding_tool"

    def __init__(self, settings: Settings | None = None, cache: CacheBackend | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.timeout_seconds = self.settings.GEMINI_EMBEDDING_TIMEOUT_SECONDS
        self.max_retries = self.settings.GEMINI_EMBEDDING_MAX_RETRIES
        self.cache = cache
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        from google import genai  # official Google GenAI SDK (already a dependency)

        self._client = genai.Client(api_key=self.settings.GEMINI_API_KEY)
        return self._client

    async def run(self, payload: EmbeddingInput) -> EmbeddingOutput:
        return await self._execute(self._embed_all, payload, retry_on=(EmbeddingAPIError,))

    async def _embed_all(self, payload: EmbeddingInput) -> EmbeddingOutput:
        results = await asyncio.gather(
            *(self._embed_one(book) for book in payload.books)
        )

        embeddings: dict[str, list[float]] = {}
        total_input_tokens = 0
        for detection_id, vector, input_tokens in results:
            if vector:
                embeddings[detection_id] = vector
            total_input_tokens += input_tokens

        usage = TokenUsage(
            model=self.settings.GEMINI_EMBEDDING_MODEL,
            input_tokens=total_input_tokens,
            output_tokens=0,
        )
        return EmbeddingOutput(embeddings=embeddings, token_usage=usage)

    async def _embed_one(self, book: ValidatedBook) -> tuple[str, list[float], int]:
        text = self._build_embedding_text(book)
        cache_key = f"gembed:{hash_key(self.settings.GEMINI_EMBEDDING_MODEL, text)}"

        if self.cache is not None:
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return book.detection_id, cached.get("vector", []), 0

        vector, input_tokens = await self._embed_text(text)

        if self.cache is not None and vector:
            await self.cache.set(
                cache_key,
                {"vector": vector},
                ttl=self.settings.CACHE_TTL_GEMINI_SECONDS,
            )
        return book.detection_id, vector, input_tokens

    async def _embed_text(self, text: str) -> tuple[list[float], int]:
        client = self._get_client()
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.models.embed_content(
                    model=self.settings.GEMINI_EMBEDDING_MODEL,
                    contents=text,
                    config={"task_type": self.settings.GEMINI_EMBEDDING_TASK_TYPE},
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingAPIError(f"Gemini embedding request failed: {exc}") from exc

        vector = self._extract_vector(response)
        # The embeddings API doesn't return token usage as consistently as
        # generate_content; approximate input tokens from text length
        # (~4 chars/token) purely for cost-estimation display purposes.
        approx_input_tokens = max(1, len(text) // 4)
        return vector, approx_input_tokens

    @staticmethod
    def _extract_vector(response) -> list[float]:
        embeddings = getattr(response, "embeddings", None)
        if embeddings:
            first = embeddings[0]
            values = getattr(first, "values", None)
            if values:
                return list(values)
        # Some SDK versions expose a single `.embedding.values` instead.
        embedding = getattr(response, "embedding", None)
        values = getattr(embedding, "values", None) if embedding is not None else None
        return list(values) if values else []

    def _build_embedding_text(self, book: ValidatedBook) -> str:
        parts = [book.title, book.subtitle or "", book.author]
        if book.categories:
            parts.append(", ".join(book.categories))
        if book.description:
            parts.append(book.description)
        text = " | ".join(p for p in parts if p)
        return text[: self.settings.GEMINI_EMBEDDING_MAX_CHARS]
