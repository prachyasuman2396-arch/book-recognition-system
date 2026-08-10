# Troubleshooting Guide

## "ultralytics not installed; using heuristic fallback detector"

**Symptom**: log warning, detection quality is poor on multi-book photos
(treats the whole image as one book).

**Cause**: `ultralytics` isn't installed, or `YOLO_MODEL_PATH` doesn't point
to an existing `.pt` file.

**Fix**:
```bash
pip install ultralytics
# and/or set YOLO_MODEL_PATH to a real weights file (see ENVIRONMENT_SETUP.md)
```

## "Real-ESRGAN unavailable; falling back to Lanczos upscale"

**Cause**: `basicsr`/`realesrgan` not installed, or the weights file at
`weights/{SUPER_RES_MODEL_NAME}.pth` is missing.

**Fix**: install the packages and download weights (see
`docs/ENVIRONMENT_SETUP.md` §3), or accept the Lanczos fallback for
development (functional, just lower quality on genuinely blurry crops).

## `gemini_vision_agent` fails with "cannot import name 'genai' from 'google'"

**Cause**: `google-genai` isn't installed in the environment.

**Fix**: `pip install google-genai`. Note this is a distinct package from
the deprecated `google-generativeai`.

## `GeminiAPIError` / 502 on recognize requests

**Cause**: invalid/missing `GEMINI_API_KEY`, quota exceeded, or a transient
Gemini outage.

**Diagnosis**:
1. Check `GET /ready` — confirms whether `GEMINI_API_KEY` is configured at
   all.
2. Check the response's `errors` array — `gemini_vision_agent: <detail>`
   will contain the underlying SDK error.
3. Check logs for `tool.gemini_vision_tool` retry warnings — the tool
   retries `GEMINI_MAX_RETRIES` times with exponential backoff before
   giving up.

The pipeline degrades gracefully: a Gemini failure means
`vision_results: []`, which cascades to `validated_books: []` and
`status: "failed"`, but the request still returns `200` with full
diagnostic detail rather than crashing.

## All books rejected / `validated_books` always empty

**Likely causes**:
1. `GOOGLE_BOOKS_API_KEY` missing or rate-limited (unauthenticated Google
   Books quota is low) — check `/ready`.
2. `VALIDATION_MIN_TITLE_SIMILARITY` / `VALIDATION_MIN_AUTHOR_SIMILARITY`
   set too high for noisy OCR/vision output — try lowering in `.env`.
3. Gemini is genuinely hallucinating titles that don't exist — check
   `warnings: ["Rejected N unvalidated/hallucinated book candidate(s)"]`
   and inspect `vision_results` (available in logs, not the final response)
   to see what was proposed.

## `rate_limit_exceeded` (429) during load testing

**Cause**: `RateLimitMiddleware` enforces `RATE_LIMIT_PER_MINUTE` per
client IP/API key, in-memory per process.

**Fix**: raise `RATE_LIMIT_PER_MINUTE` in `.env`, or if running multiple
replicas and you need a *global* limit, replace the in-memory counter in
`app/api/middleware/rate_limit.py` with a Redis `INCR`/`EXPIRE` (the
`CacheBackend` abstraction is already available for this).

## SQLite checkpoint errors in Kubernetes

**Cause**: `GRAPH_CHECKPOINT_BACKEND=sqlite` writing to an ephemeral
`emptyDir` that isn't shared across pods, or multiple replicas writing to
the same file concurrently.

**Fix**: either set `GRAPH_CHECKPOINT_BACKEND=memory` (checkpoints are
per-request-lifetime only, fine if you don't need cross-restart recovery),
or mount a `ReadWriteMany` volume / migrate to a distributed checkpointer.

## Upload rejected with `unsupported_file_type` for a file that looks fine

**Cause**: the client didn't set (or mis-set) the `Content-Type` header on
the multipart part. `validate_upload` checks `file.content_type` against
`ALLOWED_IMAGE_TYPES` (`image/jpeg`, `image/png`, `image/webp` by default).

**Fix**: ensure your HTTP client sets the correct MIME type, e.g. with
`requests`: `files={"file": ("book.jpg", data, "image/jpeg")}`.

## High latency on first request after startup

**Cause**: lazy model loading — `YOLODetectionTool`, `SuperResolutionTool`,
and `GeminiVisionTool` all load their underlying model/client on first use,
not at import time, to keep startup fast and avoid loading unused models in
processes that never see traffic.

**Fix**: this is expected. If you need warm starts, add a startup hook in
`app/main.py`'s `lifespan` that invokes each tool once against a tiny dummy
image.
