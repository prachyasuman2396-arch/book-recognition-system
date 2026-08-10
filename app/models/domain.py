"""Domain models shared across agents, tools, and the API layer.

All models are Pydantic v2 and strictly typed. These are the contracts
tools/agents pass between each other -- never bare dicts.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


class Detection(BaseModel):
    """A single detected book region from YOLODetectionTool."""

    detection_id: str
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    crop_path: str
    padded: bool = True


class QualityDecision(str, Enum):
    ACCEPT = "accept"
    ENHANCE = "enhance"
    REJECT = "reject"


class ImageQualityReport(BaseModel):
    detection_id: str
    blur_score: float
    brightness_score: float
    contrast_score: float
    noise_score: float
    resolution_score: float
    perspective_score: float
    quality_score: float = Field(ge=0.0, le=1.0)
    recommendation: str
    decision: QualityDecision


class RouteTarget(str, Enum):
    GEMINI_DIRECT = "gemini_direct"
    SUPER_RESOLUTION = "super_resolution"
    REJECTED = "rejected"


class ToolDecision(BaseModel):
    detection_id: str
    route: RouteTarget
    reason: str
    thresholds_used: dict[str, float]


class EnhancedImage(BaseModel):
    detection_id: str
    original_crop_path: str
    enhanced_path: str
    scale_factor: int
    model_used: str


class VisionExtraction(BaseModel):
    """Structured, schema-enforced output from GeminiVisionAgent."""

    detection_id: str
    title: str
    author: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    language: str
    visible_text: str


class ValidatedBook(BaseModel):
    """Ground-truth book record confirmed against Google Books."""

    detection_id: str
    title: str
    author: str
    publisher: str | None = None
    isbn_10: str | None = None
    isbn_13: str | None = None
    cover_url: str | None = None
    categories: list[str] = Field(default_factory=list)
    description: str | None = None
    published_year: int | None = None
    average_rating: float | None = None
    ratings_count: int | None = None
    google_volume_id: str
    match_confidence: float = Field(ge=0.0, le=1.0)
    is_validated: bool = True
    subtitle: str | None = None
    embedding: list[float] = Field(
        default_factory=list,
        description=(
            "Gemini embedding vector over title/subtitle/description/"
            "categories/author, populated by EmbeddingGenerationAgent. "
            "Empty until that agent runs; recommendation logic falls back "
            "to LLM reasoning when this is empty."
        ),
    )


class Recommendation(BaseModel):
    title: str
    author: str
    cover_url: str | None = None
    rating: float | None = None
    description: str | None = None
    buy_link: str | None = None
    preview_link: str | None = None
    source_book_title: str = ""
    match_reason: str = ""
    score: float = Field(ge=0.0, le=1.0, default=0.0)


class BookshelfRecommendation(BaseModel):
    """A recommendation drawn ONLY from the user's own uploaded bookshelf.

    Replaces the old Google-Books-backed `Recommendation` model as the
    pipeline's recommendation output. `Recommendation` is kept below,
    unmodified, purely for backward compatibility with anything still
    importing it (e.g. existing test mocks); it is no longer produced by
    `RecommendationTool`.
    """

    title: str
    author: str
    cover_url: str | None = None
    similarity_score: float = Field(ge=0.0, le=1.0, default=0.0)
    categories: list[str] = Field(default_factory=list)
    reason_for_recommendation: str = ""
    common_topics: list[str] = Field(default_factory=list)
    rating: float | None = None
    description: str | None = None
    published_year: int | None = None
    isbn_13: str | None = None
    google_volume_id: str = ""
    method: str = "embedding"  # "embedding" | "llm_fallback"


class ExecutionStep(BaseModel):
    agent_name: str
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "running"
    duration_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
