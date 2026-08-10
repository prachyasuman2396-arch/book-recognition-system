"""Lightweight per-client rate limiting middleware.

Uses a fixed-window counter keyed by client IP (or API key if present).
Sufficient for a single-process deployment; for multi-instance production
deployments, back this with Redis (the same `CacheBackend` used elsewhere)
by swapping `_hits` for a Redis INCR+EXPIRE.
"""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app) -> None:
        super().__init__(app)
        self._hits: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        settings = get_settings()
        if request.url.path in ("/health", "/ready", "/metrics"):
            return await call_next(request)

        client_key = request.headers.get("x-api-key") or (
            request.client.host if request.client else "unknown"
        )
        now = time.time()
        window_start = now - 60.0

        hits = self._hits[client_key]
        hits[:] = [t for t in hits if t > window_start]

        if len(hits) >= settings.RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=429,
                content={
                    "error_code": "rate_limit_exceeded",
                    "message": f"Rate limit of {settings.RATE_LIMIT_PER_MINUTE} requests/minute exceeded",
                    "details": {},
                },
            )

        hits.append(now)
        return await call_next(request)
