"""API key authentication for FastAPI routes."""
from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import get_settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing `X-API-Key` when `API_KEY_ENABLED=true`."""
    settings = get_settings()
    if not settings.API_KEY_ENABLED:
        return
    if not x_api_key or x_api_key not in settings.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
