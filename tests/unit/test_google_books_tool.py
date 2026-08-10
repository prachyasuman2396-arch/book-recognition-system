from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.models.domain import VisionExtraction
from app.tools.google_books_tool import GoogleBooksLookupInput, GoogleBooksTool

_MOCK_RESPONSE = {
    "items": [
        {
            "id": "abc123",
            "volumeInfo": {
                "title": "Dune",
                "authors": ["Frank Herbert"],
                "publisher": "Ace Books",
                "publishedDate": "1965-08-01",
                "industryIdentifiers": [{"type": "ISBN_13", "identifier": "9780441013593"}],
                "imageLinks": {"thumbnail": "https://example.com/cover.jpg"},
                "categories": ["Fiction"],
                "description": "A desert planet saga",
                "averageRating": 4.6,
                "ratingsCount": 50000,
            },
        }
    ]
}


@pytest.mark.asyncio
async def test_validates_matching_candidate(tmp_settings):
    tool = GoogleBooksTool(settings=tmp_settings)
    candidate = VisionExtraction(
        detection_id="d1", title="Dune", author="Frank Herbert", confidence=0.9,
        reasoning="clear cover text", language="en", visible_text="DUNE Frank Herbert",
    )

    with patch.object(tool, "_fetch", new=AsyncMock(return_value=_MOCK_RESPONSE)):
        result = await tool.run(GoogleBooksLookupInput(candidate=candidate))

    assert result is not None
    assert result.title == "Dune"
    assert result.isbn_13 == "9780441013593"
    assert result.is_validated is True


@pytest.mark.asyncio
async def test_rejects_hallucinated_candidate(tmp_settings):
    tool = GoogleBooksTool(settings=tmp_settings)
    candidate = VisionExtraction(
        detection_id="d2", title="Totally Fake Book Title Xyz", author="Nobody Real",
        confidence=0.4, reasoning="guessed", language="en", visible_text="???",
    )

    with patch.object(tool, "_fetch", new=AsyncMock(return_value={"items": []})):
        result = await tool.run(GoogleBooksLookupInput(candidate=candidate))

    assert result is None
