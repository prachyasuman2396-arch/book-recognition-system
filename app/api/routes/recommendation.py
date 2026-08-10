"""Bookshelf recommendation endpoint -- the second half of the new flow.

`POST /api/v1/books/recognize` populates a request's bookshelf; this
endpoint takes that `request_id` plus a `liked_book` title and returns
ranked recommendations drawn ONLY from that same bookshelf.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.logging import get_logger, log_extra
from app.graph.container import get_container
from app.graph.runner import run_recommendation
from app.models.state import BookRecommendRequest, BookRecommendResponse
from app.security.auth import verify_api_key

router = APIRouter(tags=["recommendation"])
logger = get_logger(__name__)


@router.post(
    "/api/v1/books/recommend",
    response_model=BookRecommendResponse,
    summary="Recommend books from a previously recognized bookshelf",
    dependencies=[Depends(verify_api_key)],
)
async def recommend_books(payload: BookRecommendRequest) -> BookRecommendResponse:
    """Recommend books from the same uploaded bookshelf as `liked_book`.

    `request_id` must come from a prior `POST /api/v1/books/recognize`
    call. Raises 404 (`bookshelf_not_found`) if that bookshelf has expired
    or never existed; returns HTTP 200 with `status="not_found"` and the
    literal message `"Book not found in uploaded bookshelf."` if
    `liked_book` doesn't match anything detected in the photo -- that's a
    valid, expected outcome, not a server error, so it isn't a 4xx/5xx.
    """
    log_extra(
        logger,
        20,
        "Recommendation request accepted",
        request_id=payload.request_id,
        liked_book=payload.liked_book,
    )

    container = get_container()
    return await run_recommendation(
        request_id=payload.request_id,
        liked_book=payload.liked_book,
        top_k=payload.top_k,
        container=container,
    )
