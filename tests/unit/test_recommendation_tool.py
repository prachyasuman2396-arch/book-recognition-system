from __future__ import annotations

import pytest

from app.agents.recommendation_agent import RecommendationAgent
from app.config import Settings
from app.models.domain import ValidatedBook
from app.tools.recommendation_tool import RecommendationInput, RecommendationTool


def _book(detection_id: str, title: str, embedding: list[float], **kwargs) -> ValidatedBook:
    defaults = dict(
        author="Test Author",
        categories=["Self-Help"],
        description="A book about habits and productivity.",
        google_volume_id=f"vol-{detection_id}",
        match_confidence=0.9,
    )
    defaults.update(kwargs)
    return ValidatedBook(detection_id=detection_id, title=title, embedding=embedding, **defaults)


@pytest.mark.asyncio
async def test_cosine_rank_prefers_closer_vector(tmp_settings: Settings):
    # Disable the min-similarity floor for this test: we want to inspect the
    # *ordering* of both candidates, whereas by default the "far" one would
    # be filtered out entirely (correctly -- see
    # test_min_similarity_threshold_filters_unrelated_books below).
    tool = RecommendationTool(settings=tmp_settings.model_copy(update={"RECOMMENDATION_MIN_SIMILARITY": 0.0}))

    liked = _book("liked", "Atomic Habits", embedding=[1.0, 0.0, 0.0])
    close = _book("close", "Deep Work", embedding=[0.9, 0.1, 0.0])
    far = _book("far", "Unrelated Cookbook", embedding=[0.0, 0.0, 1.0])

    result = await tool.run(
        RecommendationInput(liked_book=liked, bookshelf=[liked, close, far], top_k=5)
    )

    titles_in_order = [r.title for r in result]
    assert titles_in_order[0] == "Deep Work"
    assert all(r.method == "embedding" for r in result)
    assert result[0].similarity_score > result[-1].similarity_score


@pytest.mark.asyncio
async def test_min_similarity_threshold_filters_unrelated_books_when_a_good_match_exists(
    tmp_settings: Settings,
):
    tool = RecommendationTool(settings=tmp_settings)
    liked = _book("liked", "Atomic Habits", embedding=[1.0, 0.0, 0.0])
    close = _book("close", "Deep Work", embedding=[0.9, 0.1, 0.0])
    far = _book("far", "Unrelated Cookbook", embedding=[0.0, 0.0, 1.0])

    result = await tool.run(
        RecommendationInput(liked_book=liked, bookshelf=[liked, close, far], top_k=5)
    )

    titles = [r.title for r in result]
    assert "Deep Work" in titles
    assert "Unrelated Cookbook" not in titles


@pytest.mark.asyncio
async def test_small_shelf_falls_back_to_best_effort_when_nothing_clears_threshold(
    tmp_settings: Settings,
):
    """If literally everything on the shelf scores below the similarity
    floor (e.g. a 2-book shelf with unrelated books), showing nothing would
    be a worse experience than showing the best-available match -- so the
    tool falls back to unfiltered results rather than returning an empty
    list."""
    tool = RecommendationTool(settings=tmp_settings)
    liked = _book("liked", "Atomic Habits", embedding=[1.0, 0.0, 0.0])
    far = _book("far", "Unrelated Cookbook", embedding=[0.0, 0.0, 1.0])

    result = await tool.run(RecommendationInput(liked_book=liked, bookshelf=[liked, far], top_k=5))

    assert len(result) == 1
    assert result[0].title == "Unrelated Cookbook"


@pytest.mark.asyncio
async def test_recommend_excludes_the_liked_book_itself(tmp_settings: Settings):
    tool = RecommendationTool(settings=tmp_settings)
    liked = _book("liked", "Atomic Habits", embedding=[1.0, 0.0])

    result = await tool.run(RecommendationInput(liked_book=liked, bookshelf=[liked], top_k=5))

    assert result == []


def test_find_book_by_title_exact_match():
    books = [_book("a", "Atomic Habits", []), _book("b", "Deep Work", [])]
    found = RecommendationAgent.find_book_by_title(books, "Deep Work")
    assert found is not None
    assert found.detection_id == "b"


def test_find_book_by_title_fuzzy_match_typo():
    books = [_book("a", "Atomic Habits", []), _book("b", "Deep Work", [])]
    found = RecommendationAgent.find_book_by_title(books, "atomic habit")
    assert found is not None
    assert found.detection_id == "a"


def test_find_book_by_title_returns_none_when_not_on_shelf():
    books = [_book("a", "Atomic Habits", [])]
    found = RecommendationAgent.find_book_by_title(books, "A Completely Different Book")
    assert found is None
