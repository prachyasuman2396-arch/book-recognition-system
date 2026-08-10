"""Centralized application configuration.

All tunables live here and are sourced from environment variables / .env.
Nothing in the agent or tool layer should hardcode a threshold, model name,
timeout, or credential -- everything is injected from `Settings`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application settings.

    Values are resolved in this order of precedence:
      1. Explicit environment variables
      2. `.env` file at the project root
      3. Defaults declared below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------- app
    APP_NAME: str = "book-recognition-system"
    APP_ENV: Literal["local", "dev", "staging", "production"] = "local"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_JSON: bool = True
    REQUEST_ID_HEADER: str = "X-Request-ID"

    # --------------------------------------------------------------- data
    DATA_DIR: Path = Path("data")
    UPLOAD_DIR: Path = Path("data/uploads")
    CROP_DIR: Path = Path("data/crops")
    ENHANCED_DIR: Path = Path("data/enhanced")
    CHECKPOINT_DIR: Path = Path("data/checkpoints")
    MAX_UPLOAD_SIZE_MB: int = 15
    ALLOWED_IMAGE_TYPES: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")

    # ---------------------------------------------------------------- yolo
    YOLO_MODEL_PATH: str = "models/yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.35
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    YOLO_CROP_PADDING_RATIO: float = 0.06
    YOLO_MAX_DETECTIONS: int = 25

    # ------------------------------------------------------- image quality
    BLUR_THRESHOLD: float = 100.0
    BRIGHTNESS_MIN: float = 40.0
    BRIGHTNESS_MAX: float = 220.0
    CONTRAST_THRESHOLD: float = 30.0
    MIN_WIDTH: int = 200
    MIN_HEIGHT: int = 200
    NOISE_THRESHOLD: float = 25.0
    PERSPECTIVE_SCORE_THRESHOLD: float = 0.55
    QUALITY_SCORE_ACCEPT: float = 0.72
    QUALITY_SCORE_ENHANCE: float = 0.40

    # --------------------------------------------------------- super-res
    SUPER_RES_MODEL_NAME: str = "RealESRGAN_x4plus"
    SUPER_RES_SCALE: int = 4
    SUPER_RES_TILE: int = 256
    SUPER_RES_DEVICE: Literal["cpu", "cuda"] = "cpu"

    # ------------------------------------------------------------- gemini
    GEMINI_API_KEY: str = Field(default="", repr=False)
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_TEMPERATURE: float = 0.1
    GEMINI_MAX_OUTPUT_TOKENS: int = 4096
    GEMINI_BATCH_SIZE: int = 6
    GEMINI_TIMEOUT_SECONDS: float = 30.0
    GEMINI_MAX_RETRIES: int = 3

    # ------------------------------------------------------- google books
    GOOGLE_BOOKS_API_KEY: str = Field(default="", repr=False)
    GOOGLE_BOOKS_URL: str = "https://www.googleapis.com/books/v1/volumes"
    GOOGLE_BOOKS_TIMEOUT_SECONDS: float = 10.0
    GOOGLE_BOOKS_MAX_RETRIES: int = 3
    VALIDATION_MIN_TITLE_SIMILARITY: float = 0.62
    VALIDATION_MIN_AUTHOR_SIMILARITY: float = 0.55

    # ---------------------------------------------------- gemini embedding
    # Powers the semantic bookshelf-recommendation engine. Uses the same
    # google-genai SDK/client as GeminiVisionTool -- no new dependency.
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"
    GEMINI_EMBEDDING_TASK_TYPE: str = "SEMANTIC_SIMILARITY"
    GEMINI_EMBEDDING_TIMEOUT_SECONDS: float = 20.0
    GEMINI_EMBEDDING_MAX_RETRIES: int = 3
    GEMINI_EMBEDDING_MAX_CHARS: int = 2000
    GEMINI_EMBEDDING_COST_PER_1K_TOKENS_USD: float = 0.00001

    # ---------------------------------------------------- bookshelf memory
    # Per-request "bookshelf" (the set of validated books detected in the
    # user's uploaded photo) is persisted here so a *later*, separate
    # `/books/recommend` call can look it up by `request_id` without
    # re-running detection/recognition. Backed by the same `CacheBackend`
    # used elsewhere (Redis in production => shared across replicas/workers;
    # in-memory in local/dev).
    BOOKSHELF_STORE_KEY_PREFIX: str = "bookshelf"
    CACHE_TTL_BOOKSHELF_SECONDS: int = 21600  # 6 hours -- long enough for a user to browse + ask

    # --------------------------------------------------- recommendation
    RECOMMENDATION_DEFAULT_TOP_K: int = 5
    RECOMMENDATION_MAX_TOP_K: int = 20
    # Below this cosine similarity, a candidate is not offered even if it
    # would otherwise fill out top_k -- prevents "recommending" a totally
    # unrelated book just because the shelf is small.
    RECOMMENDATION_MIN_SIMILARITY: float = 0.35

    # --------------------------------------------------------- retry/http
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_BASE_SECONDS: float = 0.5
    RETRY_BACKOFF_MAX_SECONDS: float = 8.0
    TIMEOUT: float = 20.0

    # ------------------------------------------------------------- cache
    CACHE_BACKEND: Literal["redis", "memory"] = "memory"
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_SIZE: int = 5000
    CACHE_TTL_SECONDS: int = 3600
    CACHE_TTL_BOOKS_SECONDS: int = 86400
    CACHE_TTL_GEMINI_SECONDS: int = 3600
    CACHE_TTL_RECOMMENDATIONS_SECONDS: int = 3600

    # --------------------------------------------------------- security
    API_KEY_ENABLED: bool = True
    API_KEYS: tuple[str, ...] = ("dev-local-key",)
    RATE_LIMIT_PER_MINUTE: int = 60
    CORS_ALLOW_ORIGINS: tuple[str, ...] = ("*",)
    # Empty tuple disables TrustedHostMiddleware (local/dev default). Set to
    # concrete hostnames (e.g. the ALB DNS name / API domain) in production.
    ALLOWED_HOSTS: tuple[str, ...] = ()

    # ---------------------------------------------------------- LangGraph
    GRAPH_CHECKPOINT_BACKEND: Literal["memory", "sqlite"] = "sqlite"
    GRAPH_CHECKPOINT_DB_PATH: str = "data/checkpoints/graph_state.db"
    GRAPH_RECURSION_LIMIT: int = 50

    # ------------------------------------------------------- observability
    OTEL_ENABLED: bool = False
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"
    PROMETHEUS_ENABLED: bool = True
    METRICS_NAMESPACE: str = "book_recognition"

    # ---------------------------------------------------------- cost/est.
    GEMINI_COST_PER_1K_INPUT_TOKENS_USD: float = 0.000075
    GEMINI_COST_PER_1K_OUTPUT_TOKENS_USD: float = 0.0003

    @field_validator("UPLOAD_DIR", "CROP_DIR", "ENHANCED_DIR", "CHECKPOINT_DIR", mode="after")
    @classmethod
    def _ensure_dir_exists(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings singleton."""
    return Settings()
