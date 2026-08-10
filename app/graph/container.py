"""Dependency-injection container.

Single place where concrete tools are constructed and wired into agents.
Swapping an implementation (e.g. a mock GoogleBooksTool in tests) means
constructing a different `Container`, never editing agent code.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.agents.book_detection_agent import BookDetectionAgent
from app.agents.bookshelf_memory_agent import BookshelfMemoryAgent
from app.agents.decision_agent import DecisionAgent
from app.agents.embedding_agent import EmbeddingGenerationAgent
from app.agents.final_response_agent import FinalResponseAgent
from app.agents.gemini_vision_agent import GeminiVisionAgent
from app.agents.image_quality_agent import ImageQualityAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.super_resolution_agent import SuperResolutionAgent
from app.agents.validation_agent import ValidationAgent
from app.cache.bookshelf_store import BookshelfStore
from app.cache.cache_backend import CacheBackend, get_cache
from app.config import Settings, get_settings
from app.tools.embedding_tool import GeminiEmbeddingTool
from app.tools.gemini_vision_tool import GeminiVisionTool
from app.tools.google_books_tool import GoogleBooksTool
from app.tools.image_quality_tool import ImageQualityTool
from app.tools.recommendation_tool import RecommendationTool
from app.tools.super_resolution_tool import SuperResolutionTool
from app.tools.yolo_detection_tool import YOLODetectionTool


@dataclass
class Container:
    """Constructs and holds every tool/agent instance for one process."""

    settings: Settings = field(default_factory=get_settings)
    cache: CacheBackend | None = None

    def __post_init__(self) -> None:
        if self.cache is None:
            self.cache = get_cache(self.settings)

        # ---- tools ----
        self.yolo_tool = YOLODetectionTool(settings=self.settings)
        self.quality_tool = ImageQualityTool(settings=self.settings)
        self.super_res_tool = SuperResolutionTool(settings=self.settings)
        self.gemini_tool = GeminiVisionTool(settings=self.settings)
        self.google_books_tool = GoogleBooksTool(settings=self.settings, cache=self.cache)
        self.embedding_tool = GeminiEmbeddingTool(settings=self.settings, cache=self.cache)  # NEW
        # No longer takes `cache` -- it never calls an external API, just
        # ranks the in-memory bookshelf (embeddings themselves are cached
        # one layer down, inside GeminiEmbeddingTool).
        self.recommendation_tool = RecommendationTool(settings=self.settings)

        # ---- bookshelf persistence (NEW) ----
        self.bookshelf_store = BookshelfStore(cache=self.cache, settings=self.settings)

        # ---- agents ----
        self.book_detection_agent = BookDetectionAgent(yolo_tool=self.yolo_tool)
        self.image_quality_agent = ImageQualityAgent(quality_tool=self.quality_tool)
        self.decision_agent = DecisionAgent(settings=self.settings)
        self.super_resolution_agent = SuperResolutionAgent(super_res_tool=self.super_res_tool)
        self.gemini_vision_agent = GeminiVisionAgent(
            gemini_tool=self.gemini_tool, settings=self.settings
        )
        self.validation_agent = ValidationAgent(google_books_tool=self.google_books_tool)
        self.embedding_generation_agent = EmbeddingGenerationAgent(  # NEW
            embedding_tool=self.embedding_tool, settings=self.settings
        )
        self.bookshelf_memory_agent = BookshelfMemoryAgent(  # NEW
            bookshelf_store=self.bookshelf_store
        )
        self.recommendation_agent = RecommendationAgent(
            recommendation_tool=self.recommendation_tool, settings=self.settings
        )
        self.final_response_agent = FinalResponseAgent()

    def warm_up(self) -> None:
        """Eagerly load model weights (YOLO, Real-ESRGAN) synchronously.

        Intended to be called once at process startup via
        `asyncio.to_thread(container.warm_up)` (see `app.main.lifespan`) so
        that blocking, CPU-bound weight loading happens *before* the app
        accepts traffic and off the event loop -- never inside a request.
        Safe to call multiple times; each tool caches its own loaded model
        and is a no-op on subsequent calls.
        """
        self.yolo_tool.warm_up()
        self.super_res_tool.warm_up()

    async def aclose(self) -> None:
        """Release process-wide resources (e.g. the Redis connection pool)."""
        if self.cache is not None:
            await self.cache.aclose()


_container_singleton: Container | None = None


def get_container() -> Container:
    global _container_singleton
    if _container_singleton is None:
        _container_singleton = Container()
    return _container_singleton
