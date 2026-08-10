"""Operational endpoints: liveness, readiness, and version metadata."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.config import get_settings
from app.graph.container import get_container

router = APIRouter(tags=["ops"])


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]


class VersionResponse(BaseModel):
    app_name: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def ready() -> ReadinessResponse:
    settings = get_settings()
    checks: dict[str, str] = {}

    try:
        container = get_container()
        checks["dependency_container"] = "ok" if container is not None else "unavailable"
    except Exception as exc:  # noqa: BLE001
        checks["dependency_container"] = f"error: {exc}"

    checks["gemini_api_key"] = "configured" if settings.GEMINI_API_KEY else "missing"
    checks["google_books_api_key"] = "configured" if settings.GOOGLE_BOOKS_API_KEY else "missing (unauthenticated quota)"

    overall = "ready" if all(v in ("ok", "configured") or v.startswith("missing") for v in checks.values()) else "not_ready"
    return ReadinessResponse(status=overall, checks=checks)


@router.get("/version", response_model=VersionResponse, summary="Build/version metadata")
async def version() -> VersionResponse:
    settings = get_settings()
    return VersionResponse(
        app_name=settings.APP_NAME, version=__version__, environment=settings.APP_ENV
    )
