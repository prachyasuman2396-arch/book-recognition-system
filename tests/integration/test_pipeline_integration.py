"""Integration test: runs the real graph wiring end-to-end, mocking only
the external network-dependent tools (Gemini, Google Books, Embeddings,
Recommendations). YOLO and image-quality run for real (fast, local,
deterministic).
"""
from __future__ import annotations

import pytest

from app.graph.container import Container
from app.graph.runner import run_pipeline, run_recommendation
from tests.mocks.mock_tools import (
    MockEmbeddingTool,
    MockGeminiVisionTool,
    MockGoogleBooksTool,
    MockRecommendationTool,
)


def _rewire_for_mocks(container: Container, tmp_settings) -> None:
    """Container agents were built in `__post_init__` with the real tools
    before the test swapped `container.<x>_tool` above; re-wire the agents
    that hold direct references so the mocks actually get used."""
    from app.agents.embedding_agent import EmbeddingGenerationAgent
    from app.agents.gemini_vision_agent import GeminiVisionAgent
    from app.agents.recommendation_agent import RecommendationAgent
    from app.agents.validation_agent import ValidationAgent

    container.gemini_vision_agent = GeminiVisionAgent(
        gemini_tool=container.gemini_tool, settings=tmp_settings
    )
    container.validation_agent = ValidationAgent(google_books_tool=container.google_books_tool)
    container.embedding_generation_agent = EmbeddingGenerationAgent(
        embedding_tool=container.embedding_tool, settings=tmp_settings
    )
    container.recommendation_agent = RecommendationAgent(
        recommendation_tool=container.recommendation_tool, settings=tmp_settings
    )


@pytest.mark.asyncio
async def test_full_pipeline_happy_path_stores_bookshelf_without_auto_recommending(
    tmp_settings, sample_image_path
):
    """`/books/recognize` should detect + validate + embed + persist the
    bookshelf, but must NOT auto-generate recommendations -- that only
    happens once a `liked_book` query arrives via `/books/recommend`."""
    container = Container(settings=tmp_settings)
    container.gemini_tool = MockGeminiVisionTool()
    container.google_books_tool = MockGoogleBooksTool(should_validate=True)
    container.embedding_tool = MockEmbeddingTool()
    container.recommendation_tool = MockRecommendationTool()
    _rewire_for_mocks(container, tmp_settings)

    result = await run_pipeline(
        image_path=sample_image_path, container=container, settings=tmp_settings
    )

    assert result.status in ("success", "partial_success")
    assert len(result.validated_books) >= 1
    assert result.validated_books[0].title == "Dune"
    assert result.validated_books[0].embedding  # EmbeddingGenerationAgent attached a vector
    assert result.recommendations == []  # no liked_book query yet
    assert result.metrics.detections_found >= 1
    assert result.metrics.books_validated >= 1
    agent_names = [step.agent_name for step in result.execution_trace]
    assert "book_detection_agent" in agent_names
    assert "embedding_generation_agent" in agent_names
    assert "bookshelf_memory_agent" in agent_names
    assert "final_response_agent" in agent_names

    # The bookshelf must now be retrievable for a follow-up /books/recommend call.
    stored = await container.bookshelf_store.load(result.request_id)
    assert stored is not None
    assert stored[0].title == "Dune"


@pytest.mark.asyncio
async def test_recommend_after_recognize_returns_bookshelf_only_recs(
    tmp_settings, sample_image_path
):
    container = Container(settings=tmp_settings)
    container.gemini_tool = MockGeminiVisionTool()
    container.google_books_tool = MockGoogleBooksTool(should_validate=True)
    container.embedding_tool = MockEmbeddingTool()
    container.recommendation_tool = MockRecommendationTool()
    _rewire_for_mocks(container, tmp_settings)

    recognize_result = await run_pipeline(
        image_path=sample_image_path, container=container, settings=tmp_settings
    )

    recommend_result = await run_recommendation(
        request_id=recognize_result.request_id,
        liked_book="Dune",
        top_k=3,
        container=container,
        settings=tmp_settings,
    )

    assert recommend_result.status in ("success", "no_recommendations")
    assert recommend_result.liked_book is not None
    assert recommend_result.liked_book.title == "Dune"


@pytest.mark.asyncio
async def test_recommend_unknown_book_returns_not_found(tmp_settings, sample_image_path):
    container = Container(settings=tmp_settings)
    container.gemini_tool = MockGeminiVisionTool()
    container.google_books_tool = MockGoogleBooksTool(should_validate=True)
    container.embedding_tool = MockEmbeddingTool()
    container.recommendation_tool = MockRecommendationTool()
    _rewire_for_mocks(container, tmp_settings)

    recognize_result = await run_pipeline(
        image_path=sample_image_path, container=container, settings=tmp_settings
    )

    result = await run_recommendation(
        request_id=recognize_result.request_id,
        liked_book="A Completely Unrelated Book Title Nobody Owns",
        top_k=3,
        container=container,
        settings=tmp_settings,
    )

    assert result.status == "not_found"
    assert result.message == "Book not found in uploaded bookshelf."
    assert result.recommendations == []


@pytest.mark.asyncio
async def test_recommend_unknown_request_id_raises_bookshelf_not_found(tmp_settings):
    from app.core.exceptions import BookshelfNotFoundError

    container = Container(settings=tmp_settings)
    with pytest.raises(BookshelfNotFoundError):
        await run_recommendation(
            request_id="does-not-exist",
            liked_book="Dune",
            top_k=3,
            container=container,
            settings=tmp_settings,
        )


@pytest.mark.asyncio
async def test_full_pipeline_rejects_hallucination(tmp_settings, sample_image_path):
    container = Container(settings=tmp_settings)
    container.gemini_tool = MockGeminiVisionTool()
    container.google_books_tool = MockGoogleBooksTool(should_validate=False)

    from app.agents.gemini_vision_agent import GeminiVisionAgent
    from app.agents.validation_agent import ValidationAgent

    container.gemini_vision_agent = GeminiVisionAgent(
        gemini_tool=container.gemini_tool, settings=tmp_settings
    )
    container.validation_agent = ValidationAgent(google_books_tool=container.google_books_tool)

    result = await run_pipeline(
        image_path=sample_image_path, container=container, settings=tmp_settings
    )

    assert result.validated_books == []
    assert result.recommendations == []
    assert any("Rejected" in w for w in result.warnings)
