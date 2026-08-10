"""Prometheus scrape endpoint."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from app.observability.metrics import get_metrics

router = APIRouter(tags=["ops"])


@router.get("/metrics", summary="Prometheus metrics")
async def metrics() -> Response:
    registry = get_metrics()
    return Response(content=registry.render(), media_type="text/plain; version=0.0.4")
