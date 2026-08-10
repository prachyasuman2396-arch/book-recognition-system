"""Structured exception hierarchy.

Every raised exception carries a stable `error_code` and a `details` dict so
API error responses and logs are machine-parseable, never bare strings.
"""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all application errors."""

    error_code: str = "app_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# ------------------------------------------------------------------ tools
class ToolError(AppError):
    error_code = "tool_error"
    http_status = 502


class ToolTimeoutError(ToolError):
    error_code = "tool_timeout"
    http_status = 504


class ToolRetryExhaustedError(ToolError):
    error_code = "tool_retry_exhausted"
    http_status = 502


# ----------------------------------------------------------------- agents
class AgentError(AppError):
    error_code = "agent_error"
    http_status = 500


class AgentExecutionError(AgentError):
    error_code = "agent_execution_error"


# ------------------------------------------------------------ validation
class ValidationFailedError(AppError):
    error_code = "validation_failed"
    http_status = 422


class HallucinatedBookError(ValidationFailedError):
    error_code = "hallucinated_book_rejected"


# ---------------------------------------------------------------- input
class InvalidImageError(AppError):
    error_code = "invalid_image"
    http_status = 400


class FileTooLargeError(AppError):
    error_code = "file_too_large"
    http_status = 413


class UnsupportedFileTypeError(AppError):
    error_code = "unsupported_file_type"
    http_status = 415


# --------------------------------------------------------------- external
class ExternalAPIError(ToolError):
    error_code = "external_api_error"
    http_status = 502


class GeminiAPIError(ExternalAPIError):
    error_code = "gemini_api_error"


class GoogleBooksAPIError(ExternalAPIError):
    error_code = "google_books_api_error"


class EmbeddingAPIError(ExternalAPIError):
    error_code = "embedding_api_error"


# ------------------------------------------------------------- bookshelf
class BookshelfNotFoundError(AppError):
    """No stored bookshelf for the given `request_id` (expired/never ran)."""

    error_code = "bookshelf_not_found"
    http_status = 404


class BookNotInBookshelfError(AppError):
    """The queried `liked_book` was not among the books detected in the
    uploaded image for this `request_id`."""

    error_code = "book_not_in_bookshelf"
    http_status = 404


# ------------------------------------------------------------------ auth
class AuthenticationError(AppError):
    error_code = "authentication_error"
    http_status = 401


class RateLimitExceededError(AppError):
    error_code = "rate_limit_exceeded"
    http_status = 429


# ------------------------------------------------------------------ graph
class GraphExecutionError(AppError):
    error_code = "graph_execution_error"
    http_status = 500
