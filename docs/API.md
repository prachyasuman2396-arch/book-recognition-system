# API Reference

Full interactive docs are always available at `/docs` (Swagger UI) and
`/redoc` once the service is running. This document is the static reference.

Base URL (local): `http://localhost:8000`

All endpoints except `/health`, `/ready`, and `/metrics` require the
`X-API-Key` header when `API_KEY_ENABLED=true` (default).

---

## POST /api/v1/books/recognize

Upload an image; runs the full pipeline synchronously and returns the
aggregated result.

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | JPEG/PNG/WebP, ≤ `MAX_UPLOAD_SIZE_MB` (default 15MB) |

**Response:** `200 OK`, `application/json` — `BookRecognitionResponse`

```json
{
  "request_id": "a1b2c3d4-...",
  "status": "success",
  "validated_books": [
    {
      "detection_id": "f3a9c1b2",
      "title": "Dune",
      "author": "Frank Herbert",
      "publisher": "Ace Books",
      "isbn_10": null,
      "isbn_13": "9780441013593",
      "cover_url": "https://books.google.com/...",
      "categories": ["Fiction"],
      "description": "...",
      "published_year": 1965,
      "average_rating": 4.6,
      "ratings_count": 50000,
      "google_volume_id": "abc123",
      "match_confidence": 0.94,
      "is_validated": true
    }
  ],
  "recommendations": [
    {
      "title": "Dune Messiah",
      "author": "Frank Herbert",
      "cover_url": "https://books.google.com/...",
      "rating": 4.4,
      "description": "...",
      "buy_link": "https://...",
      "preview_link": "https://...",
      "source_book_title": "Dune",
      "match_reason": "same author",
      "score": 0.87
    }
  ],
  "execution_trace": [
    {"agent_name": "book_detection_agent", "status": "success", "duration_ms": 812.4, "error": null}
  ],
  "metrics": {
    "total_duration_ms": 4210.6,
    "detections_found": 3,
    "books_validated": 2,
    "books_rejected": 1,
    "recommendations_count": 4
  },
  "token_usage": [{"model": "gemini-2.0-flash", "input_tokens": 1450, "output_tokens": 320}],
  "estimated_cost_usd": 0.0002,
  "errors": [],
  "warnings": ["Rejected 1 unvalidated/hallucinated book candidate(s)"],
  "generated_at": "2026-01-15T10:22:31.412Z"
}
```

**Error responses:**

| Status | error_code | Cause |
|---|---|---|
| 400 | `invalid_image` | File isn't a valid/decodable image |
| 401 | `authentication_error` | Missing/invalid `X-API-Key` |
| 413 | `file_too_large` | Exceeds `MAX_UPLOAD_SIZE_MB` |
| 415 | `unsupported_file_type` | Content-Type not in `ALLOWED_IMAGE_TYPES` |
| 429 | `rate_limit_exceeded` | Exceeded `RATE_LIMIT_PER_MINUTE` |
| 500 | `graph_execution_error` | Unrecoverable pipeline failure |

---

## POST /api/v1/books/recognize/stream

Same input as above; returns a `text/event-stream` (SSE) of per-agent
progress events, terminated by a `done` event.

```
event: agent_update
data: {"request_id": "...", "agent": "book_detection", "status": "success"}

event: agent_update
data: {"request_id": "...", "agent": "image_quality", "status": "success"}

event: done
data: {"request_id": "..."}
```

---

## GET /health

Liveness probe. Always `200 {"status": "ok"}` if the process is up.

## GET /ready

Readiness probe. Checks the DI container and reports whether API keys are
configured.

```json
{"status": "ready", "checks": {"dependency_container": "ok", "gemini_api_key": "configured", "google_books_api_key": "configured"}}
```

## GET /version

```json
{"app_name": "book-recognition-system", "version": "1.0.0", "environment": "production"}
```

## GET /metrics

Prometheus exposition format (`text/plain; version=0.0.4`). See
`app/observability/metrics.py` for the full metric list (agent/tool
latency histograms, call counters, token/cost counters, retry counters).
