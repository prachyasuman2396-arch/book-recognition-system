# Environment Setup Guide

## Prerequisites

- Python 3.12+
- (Optional) Docker + Docker Compose for the full stack
- (Optional) an NVIDIA GPU for real-time YOLO / Real-ESRGAN inference — CPU
  works fine for development thanks to the built-in fallbacks

## 1. Clone and bootstrap

```bash
git clone <this-repo>
cd book-recognition-system
./scripts/bootstrap_dev.sh
source .venv/bin/activate
```

This creates a virtualenv, installs `requirements-dev.txt`, installs
pre-commit hooks, and copies `.env.example` to `.env`.

## 2. Configure secrets

Edit `.env`:

```env
GEMINI_API_KEY=your-real-key
GOOGLE_BOOKS_API_KEY=your-real-key   # optional — public endpoint works unauthenticated at low quota
```

Get a Gemini key from Google AI Studio, and a Google Books API key from
Google Cloud Console (enable the "Books API").

## 3. (Optional) Download model weights

```bash
# YOLO — any Ultralytics-compatible weights work; for production, fine-tune
# on a book-spine/cover dataset. For a quick start, a general object model:
mkdir -p models
curl -L -o models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt

# Real-ESRGAN (optional; falls back to Lanczos upscale without it)
pip install basicsr realesrgan
mkdir -p weights
curl -L -o weights/RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

Without these, the service still runs end-to-end (both tools log a warning
and use their fallback implementation) — useful for developing everything
except detection/enhancement accuracy.

## 4. Run the service

```bash
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs.

## 5. Run tests

```bash
./scripts/run_tests.sh
# or directly:
pytest -q
```

## 6. Code quality

```bash
black app tests
isort app tests
ruff check app tests --fix
mypy app
```

Or simply `pre-commit run --all-files` to run everything at once (also runs
automatically on `git commit` after `bootstrap_dev.sh`).

## 7. Redis (optional, for realistic caching behavior locally)

```bash
docker run -p 6379:6379 redis:7-alpine
```

Then set `CACHE_BACKEND=redis` in `.env`. Without Redis running, leave
`CACHE_BACKEND=memory` (the default) — everything still works, just without
cross-process cache sharing.
