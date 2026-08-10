"""FinalResponseAgent -- aggregates pipeline state into the final metrics.

Does not itself build `BookRecognitionResponse` (that's the API layer's
job, since it needs the fully-merged terminal state) -- instead it
computes the final `PipelineMetrics` snapshot so the API layer only has to
read it off, without re-deriving counts.
"""
from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.models.state import PipelineMetrics, PipelineState


class FinalResponseAgent(BaseAgent):
    name = "final_response_agent"

    async def run(self, state: PipelineState) -> dict[str, Any]:
        detections = state.get("detections", [])
        validated_books = state.get("validated_books", [])
        vision_results = state.get("vision_results", [])
        recommendations = state.get("recommendations", [])

        books_rejected = max(0, len(vision_results) - len(validated_books))

        trace = state.get("execution_trace", [])
        total_duration_ms = sum(step.duration_ms or 0.0 for step in trace)

        metrics = PipelineMetrics(
            total_duration_ms=round(total_duration_ms, 2),
            detections_found=len(detections),
            books_validated=len(validated_books),
            books_rejected=books_rejected,
            recommendations_count=len(recommendations),
        )
        return {"metrics": metrics}
