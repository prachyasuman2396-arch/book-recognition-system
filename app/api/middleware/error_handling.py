"""Centralized error-handling middleware.

Converts any `AppError` into a structured JSON response with the right
HTTP status, and converts unexpected exceptions into a generic 500 with a
secure (non-leaky) message -- stack traces stay in the logs, never in the
response body.
"""
from __future__ import annotations

import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger, log_extra, request_id_ctx

logger = get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        settings = get_settings()
        request_id = request.headers.get(settings.REQUEST_ID_HEADER) or str(uuid.uuid4())
        token = request_id_ctx.set(request_id)

        try:
            response = await call_next(request)
            response.headers[settings.REQUEST_ID_HEADER] = request_id
            return response
        except AppError as exc:
            log_extra(logger, 30, "Handled AppError", error=exc.to_dict(), path=str(request.url))
            return JSONResponse(
                status_code=exc.http_status,
                content={"request_id": request_id, **exc.to_dict()},
                headers={settings.REQUEST_ID_HEADER: request_id},
            )
        except Exception as exc:  # noqa: BLE001
            log_extra(
                logger, 50, "Unhandled exception", error=str(exc), path=str(request.url)
            )
            return JSONResponse(
                status_code=500,
                content={
                    "request_id": request_id,
                    "error_code": "internal_server_error",
                    "message": "An unexpected error occurred. Please try again or contact support.",
                    "details": {},
                },
                headers={settings.REQUEST_ID_HEADER: request_id},
            )
        finally:
            request_id_ctx.reset(token)
