"""ValidationAgent -- rejects hallucinated books; the pipeline's source of truth.

Only `VisionExtraction` candidates that are corroborated by a real Google
Books volume (via `GoogleBooksTool`'s fuzzy matching) become
`ValidatedBook`s. Everything else is dropped and recorded as a warning.
"""
from __future__ import annotations

import asyncio
from typing import Any

from app.agents.base_agent import BaseAgent
from app.models.state import PipelineState
from app.tools.google_books_tool import GoogleBooksLookupInput, GoogleBooksTool


class ValidationAgent(BaseAgent):
    name = "validation_agent"

    def __init__(self, google_books_tool: GoogleBooksTool) -> None:
        super().__init__()
        self.google_books_tool = google_books_tool

    async def run(self, state: PipelineState) -> dict[str, Any]:
        candidates = state.get("vision_results", [])
        if not candidates:
            return {"validated_books": []}

        results = await asyncio.gather(
            *(
                self.google_books_tool.run(GoogleBooksLookupInput(candidate=c))
                for c in candidates
            )
        )

        validated = [r for r in results if r is not None]
        rejected_count = len(results) - len(validated)

        delta: dict[str, Any] = {"validated_books": validated}
        if rejected_count:
            delta["warnings"] = [
                f"Rejected {rejected_count} unvalidated/hallucinated book candidate(s)"
            ]
        return delta
