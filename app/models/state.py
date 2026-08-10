"""Typed shared state for the LangGraph pipeline.

`PipelineState` is the single source of truth passed node-to-node. It is
implemented as a `TypedDict` (LangGraph's native state contract) whose
values are themselves Pydantic models, giving us both LangGraph
compatibility and Pydantic validation/serialization.
"""
from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field

from app.models.domain import (
    BookshelfRecommendation,
    Detection,
    EnhancedImage,
    ExecutionStep,
    ImageQualityReport,
    TokenUsage,
    ToolDecision,
    ValidatedBook,
    VisionExtraction,
)


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged.update(right)
    return merged


class PipelineMetrics(BaseModel):
    total_duration_ms: float = 0.0
    detections_found: int = 0
    books_validated: int = 0
    books_rejected: int = 0
    recommendations_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class PipelineState(TypedDict, total=False):
    """Shared state threaded through every LangGraph node.

    Fields using `Annotated[..., operator.add]` are append-only / reducer
    based so concurrent branches merge safely; everything else is
    last-write-wins (the default LangGraph channel behavior).
    """

    request_id: str
    original_image_path: str

    detections: list[Detection]
    cropped_images: list[str]

    quality_reports: list[ImageQualityReport]
    tool_decisions: list[ToolDecision]
    enhanced_images: list[EnhancedImage]

    vision_results: list[VisionExtraction]
    validated_books: list[ValidatedBook]
    recommendations: list[BookshelfRecommendation]

    # Set only when this state represents a `/books/recommend` follow-up
    # call (see `run_recommendation` in `app.graph.runner`), never during
    # the initial `/books/recognize` run. `RecommendationAgent` reads these
    # instead of `user_preferences` to stay strictly typed.
    liked_book_title: str | None
    recommendation_top_k: int

    execution_trace: Annotated[list[ExecutionStep], operator.add]
    metrics: PipelineMetrics
    token_usage: Annotated[list[TokenUsage], operator.add]
    estimated_cost_usd: float

    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]

    timestamps: Annotated[dict[str, str], _merge_dicts]

    user_preferences: dict[str, Any]


def new_pipeline_state(request_id: str, original_image_path: str) -> PipelineState:
    """Factory producing a fully-initialized, empty `PipelineState`."""
    now = datetime.now(timezone.utc).isoformat()
    return PipelineState(
        request_id=request_id,
        original_image_path=original_image_path,
        detections=[],
        cropped_images=[],
        quality_reports=[],
        tool_decisions=[],
        enhanced_images=[],
        vision_results=[],
        validated_books=[],
        recommendations=[],
        liked_book_title=None,
        recommendation_top_k=0,
        execution_trace=[],
        metrics=PipelineMetrics(),
        token_usage=[],
        estimated_cost_usd=0.0,
        errors=[],
        warnings=[],
        timestamps={"pipeline_started_at": now},
        user_preferences={},
    )


class BookRecognitionResponse(BaseModel):
    """The final, production-grade API response shape."""

    request_id: str
    status: str
    validated_books: list[ValidatedBook]
    recommendations: list[BookshelfRecommendation]
    execution_trace: list[ExecutionStep]
    metrics: PipelineMetrics
    token_usage: list[TokenUsage]
    estimated_cost_usd: float
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: str


class BookRecommendRequest(BaseModel):
    """Request body for `POST /api/v1/books/recommend`."""

    request_id: str
    liked_book: str = Field(..., min_length=1, description="Title of a book from the uploaded bookshelf")
    top_k: int = Field(default=5, ge=1, le=20)


class BookRecommendResponse(BaseModel):
    """Response body for `POST /api/v1/books/recommend`."""

    request_id: str
    status: str
    liked_book: ValidatedBook | None = None
    recommendations: list[BookshelfRecommendation] = Field(default_factory=list)
    reasoning: list[str] = Field(default_factory=list)
    message: str | None = None
    generated_at: str
