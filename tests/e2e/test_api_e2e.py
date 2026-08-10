"""End-to-end test hitting the FastAPI app via TestClient, mocking only
the external tools so no real network calls are made."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.graph import container as container_module
from app.main import create_app
from tests.mocks.mock_tools import (
    MockGeminiVisionTool,
    MockGoogleBooksTool,
    MockRecommendationTool,
)


@pytest.fixture()
def client(tmp_settings, monkeypatch):
    # `get_settings` is imported with `from app.config import get_settings`
    # in several modules (each binds its own name at import time), so
    # patching only `app.config.settings.get_settings` doesn't reach them.
    # Patch every module-local binding that's actually called on the
    # request path -- most importantly `app.security.auth`, whose
    # `verify_api_key` dependency runs on every request and otherwise
    # falls through to the real (API_KEY_ENABLED=True) settings, causing
    # every test request to fail auth with 401 before reaching the
    # assertions it's meant to check.
    monkeypatch.setattr("app.config.settings.get_settings", lambda: tmp_settings)
    monkeypatch.setattr("app.security.auth.get_settings", lambda: tmp_settings)
    monkeypatch.setattr("app.api.routes.recognition.get_settings", lambda: tmp_settings)
    monkeypatch.setattr("app.api.middleware.rate_limit.get_settings", lambda: tmp_settings)
    monkeypatch.setattr("app.api.middleware.error_handling.get_settings", lambda: tmp_settings)

    container = container_module.Container(settings=tmp_settings)
    container.gemini_tool = MockGeminiVisionTool()
    container.google_books_tool = MockGoogleBooksTool(should_validate=True)
    container.recommendation_tool = MockRecommendationTool()

    from app.agents.gemini_vision_agent import GeminiVisionAgent
    from app.agents.recommendation_agent import RecommendationAgent
    from app.agents.validation_agent import ValidationAgent

    container.gemini_vision_agent = GeminiVisionAgent(
        gemini_tool=container.gemini_tool, settings=tmp_settings
    )
    container.validation_agent = ValidationAgent(google_books_tool=container.google_books_tool)
    container.recommendation_agent = RecommendationAgent(
        recommendation_tool=container.recommendation_tool
    )
    container_module._container_singleton = container

    app = create_app()
    return TestClient(app)


def _sample_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (800, 1200), color=(90, 110, 130)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_health_and_version(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}
    version = client.get("/version").json()
    assert version["app_name"] == "book-recognition-system"


def test_recognize_endpoint_happy_path(client: TestClient):
    files = {"file": ("book.jpg", _sample_jpeg_bytes(), "image/jpeg")}
    response = client.post("/api/v1/books/recognize", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("success", "partial_success")
    assert len(body["validated_books"]) >= 1
    assert body["validated_books"][0]["title"] == "Dune"


def test_recognize_endpoint_rejects_bad_content_type(client: TestClient):
    files = {"file": ("book.txt", b"not an image", "text/plain")}
    response = client.post("/api/v1/books/recognize", files=files)

    assert response.status_code == 415
    assert response.json()["error_code"] == "unsupported_file_type"


def test_recognize_endpoint_rejects_oversized_file(client: TestClient, tmp_settings, monkeypatch):
    tmp_settings.MAX_UPLOAD_SIZE_MB = 0  # force rejection regardless of actual size
    # `recognition.py` resolves settings via `get_settings()` at request time,
    # so patch the binding it actually calls.
    monkeypatch.setattr("app.api.routes.recognition.get_settings", lambda: tmp_settings)

    files = {"file": ("book.jpg", _sample_jpeg_bytes(), "image/jpeg")}
    response = client.post("/api/v1/books/recognize", files=files)
    assert response.status_code == 413
    assert response.json()["error_code"] == "file_too_large"
