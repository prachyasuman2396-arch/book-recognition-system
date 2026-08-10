# AI Personal Bookshelf Assistant

A production-grade, multi-agent AI system that recognizes multiple books from
a single photo of a bookshelf, then recommends other books **from that same
shelf** -- never from the open internet -- once you tell it which one you
liked. Built with **LangGraph** for orchestration, **FastAPI** for the API
layer, **YOLO** for detection, **Google Gemini Vision** for recognition, the
**Google Books API** as the source of truth for validation, and **Gemini
embeddings** for semantic, bookshelf-only recommendations.

## Why this exists

Vision LLMs are powerful but expensive and occasionally hallucinate titles.
This system is engineered around three core ideas:

1. **Don't call a Vision LLM more than necessary.** A fast, local
   `ImageQualityAgent` decides whether a crop is already good enough for
   Gemini, needs super-resolution first, or should be rejected outright.
2. **Never trust a Vision LLM's output on its own.** Every recognized title
   and author is checked against the real Google Books catalog before it's
   allowed to reach the user. Unvalidated ("hallucinated") results are
   dropped, not surfaced.
3. **Recommend from what you actually own, not the internet.** Once your
   shelf is recognized and validated, recommendations are ranked by cosine
   similarity over Gemini embeddings of the *other detected books on your
   shelf* -- title, categories, author, and description -- with an
   LLM-reasoning fallback if an embedding is unavailable. A book is never
   recommended unless it was physically photographed.

## Architecture at a glance

```
Image Upload
   │
   ▼
BookDetectionAgent  (YOLODetectionTool)
   │
   ▼
ImageQualityAgent   (ImageQualityTool)
   │
   ▼
DecisionAgent  ──────────────┐
   │                         │
   │ high quality            │ low quality
   ▼                         ▼
GeminiVisionAgent  ◄── SuperResolutionAgent
   │
   ▼
ValidationAgent          (GoogleBooksTool — source of truth)
   │
   ▼
EmbeddingGenerationAgent (GeminiEmbeddingTool)              ── NEW
   │
   ▼
BookshelfMemoryAgent     (BookshelfStore, keyed by request_id) ── NEW
   │
   ▼
RecommendationAgent      (cosine similarity over the bookshelf,
   │                       LLM-reasoning fallback -- no-op until a
   │                       liked_book query arrives, see below)
   ▼
FinalResponseAgent
   │
   ▼
JSON Response
```

`POST /api/v1/books/recognize` runs the whole graph above and stores the
validated bookshelf. A separate, later call to
`POST /api/v1/books/recommend` (with `request_id` + `liked_book`) resolves
the liked book against that stored shelf and returns ranked recommendations
drawn only from it.

See [`docs/architecture-graph.mmd`](docs/architecture-graph.mmd) for the
Mermaid diagram generated directly from the live LangGraph definition (run
`python scripts/generate_graph_diagram.py` to regenerate it), and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full component and
sequence diagrams.

## Quickstart

```bash
git clone <this-repo>
cd book-recognition-system
./scripts/bootstrap_dev.sh          # venv + deps + pre-commit + .env
source .venv/bin/activate
cp .env.example .env                # then set GEMINI_API_KEY / GOOGLE_BOOKS_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for interactive Swagger docs.

```bash
curl -X POST http://localhost:8000/api/v1/books/recognize \
  -H "X-API-Key: dev-local-key" \
  -F "file=@/path/to/bookshelf.jpg"
```

Then, once you know which detected book you liked:

```bash
curl -X POST http://localhost:8000/api/v1/books/recommend \
  -H "X-API-Key: dev-local-key" \
  -H "Content-Type: application/json" \
  -d '{"request_id": "<request_id from the recognize response>", "liked_book": "Atomic Habits", "top_k": 5}'
```

### Docker

```bash
docker compose up --build
```

This starts the API, Redis (caching), and Prometheus (metrics scraping).

## Project layout

```
app/
  agents/        BaseAgent + 8 pipeline agents (DI, logging, metrics, retry, tracing)
  tools/         BaseTool + 6 stateless tools (YOLO, quality, super-res, Gemini, Books, recs)
  graph/         LangGraph StateGraph, router, DI container, runner
  models/        Pydantic domain models + typed shared PipelineState
  api/           FastAPI routes + middleware (auth, rate limit, error handling)
  cache/         Redis-backed cache with in-memory fallback
  core/          exceptions, logging, retry
  observability/ Prometheus metrics, OpenTelemetry tracing
  security/      API key auth, upload validation/sanitization
  config/        Pydantic Settings (all tunables, zero hardcoded values)
tests/
  unit/          Tool- and agent-level unit tests
  integration/   Full graph runs with mocked external services
  e2e/           FastAPI TestClient tests against the real HTTP surface
  mocks/         Deterministic stand-ins for Gemini / Google Books / YOLO
docs/            Architecture, deployment, API, dev, testing, troubleshooting guides
deploy/          docker-compose services, k8s manifests, Prometheus config
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — components, sequence diagram, state model
- [API Reference](docs/API.md) — endpoints, request/response schemas
- [Deployment Guide](docs/DEPLOYMENT.md) — Docker, Kubernetes, environment config
- [Environment Setup](docs/ENVIRONMENT_SETUP.md) — local dev, model weights
- [Developer Guide](docs/DEVELOPER_GUIDE.md) — conventions, adding agents/tools
- [Testing Guide](docs/TESTING.md) — running/writing tests, mocking strategy
- [Troubleshooting](docs/TROUBLESHOOTING.md) — common failure modes

## Key engineering properties

- **Every agent** gets DI, structured logging, Prometheus metrics, retry
  (via its tools), execution-trace recording, typed I/O, and async support
  for free from `BaseAgent`/`BaseTool` — no per-agent boilerplate.
- **Zero hardcoded thresholds/config** — everything routes through
  `app/config/settings.py`, sourced from environment variables / `.env`.
- **Graceful degradation** — a single agent failure (e.g. Gemini API down)
  is recorded in `execution_trace`/`errors` and the graph still returns a
  partial result instead of crashing.
- **LangGraph checkpointing** — every run is persisted (SQLite by default,
  swappable to memory) keyed by `request_id`, enabling replay/recovery.
- **Caching** — Google Books lookups, Gemini responses, and recommendations
  are cached (Redis with in-memory fallback) to cut latency and cost on
  repeated queries.
- **Security** — API key auth, file type/size validation, image
  sanitization (EXIF stripping + re-encode), rate limiting, secure error
  messages (no stack traces leaked to clients).

## License

Proprietary — internal reference implementation.
