"""LangGraph workflow builder for the book recognition pipeline.

Graph shape (matches the CORE WORKFLOW spec):

    book_detection -> image_quality -> decision --+--> super_resolution --> gemini_vision
                                                   +--> gemini_vision (direct)
                                                   +--> final_response (all rejected)
    gemini_vision -> validation --+--> embedding_generation -> bookshelf_memory
                                   |        -> recommendation -> final_response
                                   +--> final_response (nothing validated)

`embedding_generation` and `bookshelf_memory` are the two NEW nodes added
for the Personal Bookshelf Assistant (see module docstrings on
`EmbeddingGenerationAgent` / `BookshelfMemoryAgent`). `recommendation`
keeps its original *position* in the graph -- still directly before
`final_response` -- but is a no-op during `/books/recognize` because no
`liked_book_title` is set yet on `PipelineState` at that point (see
`RecommendationAgent`). `POST /books/recommend` does its real work by
calling `container.recommendation_agent.execute(...)` directly against a
small ad-hoc state built from the persisted bookshelf (see
`app.graph.runner.run_recommendation`) rather than resuming this compiled
graph -- LangGraph's checkpointer is per-node-local (sqlite) or
per-process (memory), neither of which reliably bridges two separate HTTP
requests across a multi-replica production deployment, whereas the
`BookshelfStore` (Redis-backed in production) does. This keeps the graph
shape stable per the "do not restructure" requirement while accommodating
the new two-step (recognize, then recommend) product flow.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langgraph.graph import END, StateGraph

from app.config import Settings, get_settings
from app.graph.container import Container
from app.graph.router import (
    route_after_decision,
    route_after_detection,
    route_after_validation,
)
from app.models.state import PipelineState

NODE_BOOK_DETECTION = "book_detection"
NODE_IMAGE_QUALITY = "image_quality"
NODE_DECISION = "decision"
NODE_SUPER_RESOLUTION = "super_resolution"
NODE_GEMINI_VISION = "gemini_vision"
NODE_VALIDATION = "validation"
NODE_EMBEDDING_GENERATION = "embedding_generation"  # NEW
NODE_BOOKSHELF_MEMORY = "bookshelf_memory"  # NEW
NODE_RECOMMENDATION = "recommendation"
NODE_FINAL_RESPONSE = "final_response"


def build_graph(container: Container):
    """Construct (but do not compile) the `StateGraph` for the pipeline."""
    graph = StateGraph(PipelineState)

    graph.add_node(NODE_BOOK_DETECTION, container.book_detection_agent.execute)
    graph.add_node(NODE_IMAGE_QUALITY, container.image_quality_agent.execute)
    graph.add_node(NODE_DECISION, container.decision_agent.execute)
    graph.add_node(NODE_SUPER_RESOLUTION, container.super_resolution_agent.execute)
    graph.add_node(NODE_GEMINI_VISION, container.gemini_vision_agent.execute)
    graph.add_node(NODE_VALIDATION, container.validation_agent.execute)
    graph.add_node(NODE_EMBEDDING_GENERATION, container.embedding_generation_agent.execute)  # NEW
    graph.add_node(NODE_BOOKSHELF_MEMORY, container.bookshelf_memory_agent.execute)  # NEW
    graph.add_node(NODE_RECOMMENDATION, container.recommendation_agent.execute)
    graph.add_node(NODE_FINAL_RESPONSE, container.final_response_agent.execute)

    graph.set_entry_point(NODE_BOOK_DETECTION)

    graph.add_conditional_edges(
        NODE_BOOK_DETECTION,
        route_after_detection,
        {
            "no_detections": NODE_FINAL_RESPONSE,
            "has_detections": NODE_IMAGE_QUALITY,
        },
    )

    graph.add_edge(NODE_IMAGE_QUALITY, NODE_DECISION)

    graph.add_conditional_edges(
        NODE_DECISION,
        route_after_decision,
        {
            "all_rejected": NODE_FINAL_RESPONSE,
            "needs_super_resolution": NODE_SUPER_RESOLUTION,
            "direct_to_vision": NODE_GEMINI_VISION,
        },
    )

    graph.add_edge(NODE_SUPER_RESOLUTION, NODE_GEMINI_VISION)
    graph.add_edge(NODE_GEMINI_VISION, NODE_VALIDATION)

    graph.add_conditional_edges(
        NODE_VALIDATION,
        route_after_validation,
        {
            "no_validated_books": NODE_FINAL_RESPONSE,
            "has_validated_books": NODE_EMBEDDING_GENERATION,  # was NODE_RECOMMENDATION
        },
    )

    graph.add_edge(NODE_EMBEDDING_GENERATION, NODE_BOOKSHELF_MEMORY)  # NEW
    graph.add_edge(NODE_BOOKSHELF_MEMORY, NODE_RECOMMENDATION)  # NEW
    graph.add_edge(NODE_RECOMMENDATION, NODE_FINAL_RESPONSE)
    graph.add_edge(NODE_FINAL_RESPONSE, END)

    return graph


@asynccontextmanager
async def _build_checkpointer(settings: Settings) -> AsyncIterator[object | None]:
    """Yield a LangGraph checkpointer, or `None` if sqlite backend unavailable."""
    if settings.GRAPH_CHECKPOINT_BACKEND == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()
        return

    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = Path(settings.GRAPH_CHECKPOINT_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
            yield saver
    except ImportError:
        from langgraph.checkpoint.memory import MemorySaver

        yield MemorySaver()


async def compile_pipeline(container: Container, settings: Settings | None = None):
    """Compile the graph with a checkpointer, returning a runnable app.

    Usage:
        async with compile_pipeline(container) as app:
            result = await app.ainvoke(state, config=...)
    """
    settings = settings or get_settings()
    graph = build_graph(container)

    return _CompiledPipelineContext(graph, settings)


class _CompiledPipelineContext:
    """Async context manager wrapping graph compilation + checkpointer lifecycle."""

    def __init__(self, graph: StateGraph, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings
        self._checkpointer_cm = None
        self.compiled = None

    async def __aenter__(self):
        self._checkpointer_cm = _build_checkpointer(self._settings)
        checkpointer = await self._checkpointer_cm.__aenter__()
        self.compiled = self._graph.compile(checkpointer=checkpointer)
        return self.compiled

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._checkpointer_cm is not None:
            await self._checkpointer_cm.__aexit__(exc_type, exc, tb)
