"""High level entrypoint: run one image through the full pipeline."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.agents.recommendation_agent import RecommendationAgent
from app.config import Settings, get_settings
from app.core.exceptions import BookshelfNotFoundError, GraphExecutionError
from app.core.logging import get_logger, log_extra, request_id_ctx
from app.graph.container import Container, get_container
from app.graph.workflow import compile_pipeline
from app.models.state import (
    BookRecognitionResponse,
    BookRecommendResponse,
    PipelineMetrics,
    new_pipeline_state,
)

logger = get_logger(__name__)


async def run_pipeline(
    image_path: str,
    *,
    request_id: str | None = None,
    container: Container | None = None,
    settings: Settings | None = None,
) -> BookRecognitionResponse:
    """Execute the full LangGraph pipeline for a single uploaded image."""
    settings = settings or get_settings()
    container = container or get_container()
    request_id = request_id or str(uuid.uuid4())
    token = request_id_ctx.set(request_id)

    initial_state = new_pipeline_state(request_id=request_id, original_image_path=image_path)

    try:
        async with await compile_pipeline(container, settings) as app:
            config = {
                "configurable": {"thread_id": request_id},
                "recursion_limit": settings.GRAPH_RECURSION_LIMIT,
            }
            log_extra(logger, 20, "Pipeline execution starting", request_id=request_id)
            final_state = await app.ainvoke(initial_state, config=config)
    except Exception as exc:  # noqa: BLE001
        raise GraphExecutionError(f"Pipeline execution failed: {exc}") from exc
    finally:
        request_id_ctx.reset(token)

    return _to_response(final_state, request_id)


def _to_response(state: dict, request_id: str) -> BookRecognitionResponse:
    metrics = state.get("metrics") or PipelineMetrics()
    errors = state.get("errors", [])
    status = "success" if not errors else ("partial_success" if state.get("validated_books") else "failed")

    return BookRecognitionResponse(
        request_id=request_id,
        status=status,
        validated_books=state.get("validated_books", []),
        recommendations=state.get("recommendations", []),
        execution_trace=state.get("execution_trace", []),
        metrics=metrics,
        token_usage=state.get("token_usage", []),
        estimated_cost_usd=state.get("estimated_cost_usd", 0.0),
        errors=errors,
        warnings=state.get("warnings", []),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


async def run_recommendation(
    *,
    request_id: str,
    liked_book: str,
    top_k: int = 5,
    container: Container | None = None,
    settings: Settings | None = None,
) -> BookRecommendResponse:
    """Execute the recommendation step for a *previously recognized*
    bookshelf (`POST /api/v1/books/recommend`).

    Loads the bookshelf that `BookshelfMemoryAgent` persisted for
    `request_id` during the earlier `/books/recognize` call, then invokes
    `RecommendationAgent.execute` directly (not the compiled graph -- see
    `app.graph.workflow` docstring for why) against a minimal ad-hoc
    `PipelineState`. Raises `BookshelfNotFoundError` if `request_id` has no
    stored bookshelf (never ran, or its TTL expired).
    """
    settings = settings or get_settings()
    container = container or get_container()
    token = request_id_ctx.set(request_id)

    try:
        bookshelf = await container.bookshelf_store.load(request_id)
        if bookshelf is None:
            raise BookshelfNotFoundError(
                f"No bookshelf found for request_id={request_id}. "
                "Run /api/v1/books/recognize first, or it may have expired."
            )

        log_extra(
            logger,
            20,
            "Recommendation execution starting",
            request_id=request_id,
            liked_book=liked_book,
            bookshelf_size=len(bookshelf),
        )

        state = new_pipeline_state(request_id=request_id, original_image_path="")
        state["validated_books"] = bookshelf
        state["liked_book_title"] = liked_book
        state["recommendation_top_k"] = top_k

        delta = await container.recommendation_agent.execute(state)
        resolved_liked_book = RecommendationAgent.find_book_by_title(bookshelf, liked_book)
    finally:
        request_id_ctx.reset(token)

    return _to_recommend_response(request_id, delta, resolved_liked_book)


def _to_recommend_response(request_id: str, delta: dict, liked) -> BookRecommendResponse:
    recommendations = delta.get("recommendations", [])
    warnings = delta.get("warnings", [])
    errors = delta.get("errors", [])
    not_found = "Book not found in uploaded bookshelf." in warnings

    if not_found:
        status, message, liked = "not_found", "Book not found in uploaded bookshelf.", None
    elif errors:
        status, message = "failed", "; ".join(errors)
    elif recommendations:
        status, message = "success", None
    else:
        status, message = "no_recommendations", "No similar books found on this shelf."

    return BookRecommendResponse(
        request_id=request_id,
        status=status,
        liked_book=liked,
        recommendations=recommendations,
        reasoning=[r.reason_for_recommendation for r in recommendations],
        message=message,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


async def render_graph_mermaid(container: Container | None = None) -> str:
    """Return a Mermaid diagram of the compiled graph for docs/visualization."""
    container = container or get_container()
    settings = get_settings()
    async with await compile_pipeline(container, settings) as app:
        return app.get_graph().draw_mermaid()
