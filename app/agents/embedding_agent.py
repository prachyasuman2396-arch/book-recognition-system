"""EmbeddingGenerationAgent -- attaches a semantic vector to every book.

Runs immediately after `ValidationAgent`. Failure here is non-fatal: if
embeddings can't be generated (quota, network, malformed response), the
books simply keep an empty `embedding` list, and `RecommendationTool`
transparently falls back to LLM-reasoning-based recommendations (see that
module's docstring) instead of cosine similarity.
"""
from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.config import Settings, get_settings
from app.models.state import PipelineState
from app.tools.embedding_tool import EmbeddingInput, GeminiEmbeddingTool


class EmbeddingGenerationAgent(BaseAgent):
    name = "embedding_generation_agent"

    def __init__(self, embedding_tool: GeminiEmbeddingTool, settings: Settings | None = None) -> None:
        super().__init__()
        self.embedding_tool = embedding_tool
        self.settings = settings or get_settings()

    async def run(self, state: PipelineState) -> dict[str, Any]:
        validated_books = state.get("validated_books", [])
        if not validated_books:
            return {}

        output = await self.embedding_tool.run(EmbeddingInput(books=validated_books))

        updated_books = [
            book.model_copy(update={"embedding": output.embeddings.get(book.detection_id, [])})
            for book in validated_books
        ]

        missing = [b.title for b in updated_books if not b.embedding]
        delta: dict[str, Any] = {
            "validated_books": updated_books,
            "token_usage": [output.token_usage],
            "estimated_cost_usd": self._estimate_cost(output.token_usage),
        }
        if missing:
            delta["warnings"] = [
                f"No embedding generated for {len(missing)} book(s); "
                "recommendations for them will use LLM-reasoning fallback"
            ]
        return delta

    def _estimate_cost(self, usage) -> float:
        return round(
            (usage.input_tokens / 1000) * self.settings.GEMINI_EMBEDDING_COST_PER_1K_TOKENS_USD, 6
        )
