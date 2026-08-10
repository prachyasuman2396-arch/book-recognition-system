"""Shared pytest fixtures."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    """An isolated `Settings` instance writing to a scratch directory."""
    return Settings(
        UPLOAD_DIR=tmp_path / "uploads",
        CROP_DIR=tmp_path / "crops",
        ENHANCED_DIR=tmp_path / "enhanced",
        CHECKPOINT_DIR=tmp_path / "checkpoints",
        GRAPH_CHECKPOINT_BACKEND="memory",
        CACHE_BACKEND="memory",
        API_KEY_ENABLED=False,
        GEMINI_API_KEY="test-key",
        GOOGLE_BOOKS_API_KEY="test-key",
    )


@pytest.fixture()
def sample_image_path(tmp_path: Path) -> str:
    path = tmp_path / "sample_book.jpg"
    img = Image.new("RGB", (800, 1200), color=(90, 110, 130))
    img.save(path, format="JPEG", quality=95)
    return str(path)


@pytest.fixture(autouse=True)
def _cleanup_module_singletons():
    """Reset process-wide singletons between tests to avoid cross-test bleed."""
    yield
    import app.cache.cache_backend as cache_module
    import app.graph.container as container_module
    import app.observability.metrics as metrics_module

    cache_module._cache_singleton = None
    container_module._container_singleton = None
    # metrics registry is safe to keep (Prometheus counters aren't test-scoped)
    del metrics_module
