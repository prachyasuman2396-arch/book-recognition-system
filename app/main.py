"""FastAPI application factory.

`create_app()` wires together middleware (error handling, rate limiting,
CORS, gzip compression), routers, and startup/shutdown hooks. Kept as a
factory (rather than a module-level `app`) so tests can construct fresh,
isolated instances.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app import __version__
from app.api.middleware.error_handling import ErrorHandlingMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.middleware.security_headers import SecurityHeadersMiddleware
from app.api.routes import metrics, ops, recognition, recognition_stream, recommendation
from app.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.graph.container import get_container
from app.observability.tracing import configure_tracing

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.LOG_LEVEL, json_output=settings.LOG_JSON)
    configure_tracing()

    # Warm up heavy ML models (YOLO / Real-ESRGAN weight loading) off the
    # event loop and *before* we start accepting traffic. Without this,
    # whichever request happens to arrive first pays the full model-load
    # cost synchronously inside the request path, and -- because that load
    # is blocking, CPU-bound work executed directly inside an `async def`
    # method with no `await` in between -- it stalls the entire event loop
    # (every other in-flight request on this worker) for the duration.
    container = get_container()
    try:
        await asyncio.to_thread(container.warm_up)
        logger.info("Model warm-up complete")
    except Exception:  # noqa: BLE001
        # Warm-up is best-effort: tools fall back to heuristic/CPU paths on
        # their own if weights are missing, so a warm-up failure shouldn't
        # prevent the app from serving traffic -- it'll just retry lazily
        # (and log loudly) on first real request.
        logger.exception("Model warm-up failed; will retry lazily on first request")

    logger.info("Application startup complete: %s v%s (%s)", settings.APP_NAME, __version__, settings.APP_ENV)
    yield

    await container.aclose()
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=__version__,
        description=(
            "Production-grade multi-agent AI Personal Bookshelf Assistant: "
            "YOLO detection, adaptive image-quality routing, Gemini Vision "
            "recognition, Google Books validation, Gemini-embedding-based "
            "semantic recommendations drawn exclusively from the user's own "
            "uploaded bookshelf, orchestrated with LangGraph."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # A wildcard origin ("*") combined with allow_credentials=True is an
    # invalid CORS configuration: the spec forbids reflecting credentialed
    # requests to "any origin", so browsers either reject it outright or
    # (on permissive middleware versions) silently echo the request's
    # Origin back, which defeats origin restriction entirely. Credentials
    # are only enabled when the operator has configured a concrete,
    # non-wildcard origin allowlist.
    cors_origins = list(settings.CORS_ALLOW_ORIGINS)
    allow_wildcard = cors_origins == ["*"]
    if allow_wildcard:
        logger.warning(
            "CORS_ALLOW_ORIGINS is '*' (wildcard) -- disabling allow_credentials. "
            "Set explicit origins in production to enable credentialed cross-origin requests."
        )

    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=not allow_wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if settings.ALLOWED_HOSTS:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.ALLOWED_HOSTS))
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)

    app.include_router(recognition.router)
    app.include_router(recognition_stream.router)
    app.include_router(recommendation.router)
    app.include_router(ops.router)
    app.include_router(metrics.router)

    return app


app = create_app()
