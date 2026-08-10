"""Deterministic mocks for external services (Gemini, Google Books, YOLO)."""
from __future__ import annotations

from app.models.domain import TokenUsage, ValidatedBook, VisionExtraction
from app.tools.gemini_vision_tool import GeminiVisionOutput
from app.tools.google_books_tool import GoogleBooksLookupInput


class MockGeminiVisionTool:
    """Drop-in replacement for `GeminiVisionTool.run`."""

    name = "gemini_vision_tool"

    def __init__(self, fixed_title: str = "Dune", fixed_author: str = "Frank Herbert") -> None:
        self.fixed_title = fixed_title
        self.fixed_author = fixed_author

    async def run(self, payload) -> list[GeminiVisionOutput]:
        extractions = [
            VisionExtraction(
                detection_id=detection_id,
                title=self.fixed_title,
                author=self.fixed_author,
                confidence=0.92,
                reasoning="Mocked extraction for testing",
                language="en",
                visible_text=f"{self.fixed_title} {self.fixed_author}",
            )
            for detection_id in payload.image_paths
        ]
        return [
            GeminiVisionOutput(
                extractions=extractions,
                token_usage=TokenUsage(model="mock-gemini", input_tokens=120, output_tokens=60),
            )
        ]


class MockGoogleBooksTool:
    """Drop-in replacement for `GoogleBooksTool.run`; always validates."""

    name = "google_books_tool"

    def __init__(self, should_validate: bool = True) -> None:
        self.should_validate = should_validate

    async def run(self, payload: GoogleBooksLookupInput) -> ValidatedBook | None:
        if not self.should_validate:
            return None
        candidate = payload.candidate
        return ValidatedBook(
            detection_id=candidate.detection_id,
            title=candidate.title,
            author=candidate.author,
            publisher="Mock Publisher",
            isbn_13="9780000000000",
            cover_url="https://example.com/cover.jpg",
            categories=["Science Fiction"],
            description="Mocked description",
            published_year=1965,
            average_rating=4.6,
            ratings_count=12345,
            google_volume_id="mock-volume-id",
            match_confidence=0.95,
            is_validated=True,
        )


class MockEmbeddingTool:
    """Drop-in replacement for `GeminiEmbeddingTool.run`.

    Returns a deterministic, distinguishable vector per book (based on
    title hash) rather than all-zeros, so cosine-similarity tests exercise
    real math instead of a degenerate zero-vector edge case.
    """

    name = "gemini_embedding_tool"

    async def run(self, payload):
        from app.tools.embedding_tool import EmbeddingOutput

        embeddings = {
            book.detection_id: self._fake_vector(book.title) for book in payload.books
        }
        return EmbeddingOutput(
            embeddings=embeddings,
            token_usage=TokenUsage(model="mock-embedding", input_tokens=42, output_tokens=0),
        )

    @staticmethod
    def _fake_vector(title: str) -> list[float]:
        seed = sum(ord(c) for c in title) or 1
        return [((seed * (i + 1)) % 97) / 97 for i in range(8)]


class MockRecommendationTool:
    """Drop-in replacement for `RecommendationTool.run` (bookshelf-only contract)."""

    name = "recommendation_tool"

    async def run(self, payload):
        from app.models.domain import BookshelfRecommendation

        others = [b for b in payload.bookshelf if b.detection_id != payload.liked_book.detection_id]
        return [
            BookshelfRecommendation(
                title=book.title,
                author=book.author,
                cover_url=book.cover_url,
                similarity_score=0.87,
                categories=book.categories,
                reason_for_recommendation="Mocked: same author/genre",
                common_topics=book.categories[:1],
                rating=book.average_rating,
                description=book.description,
                published_year=book.published_year,
                isbn_13=book.isbn_13,
                google_volume_id=book.google_volume_id,
                method="embedding",
            )
            for book in others[: payload.top_k]
        ]
