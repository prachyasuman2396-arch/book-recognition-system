# AI Personal Bookshelf Assistant — Change Log & Deliverables

This document covers every file that was **added** or **modified** to turn
the existing Google-Books-recommendation pipeline into the Personal
Bookshelf Assistant described in the product spec. Files not listed were
not touched. Nothing was rewritten from scratch; the folder structure,
agent/tool/BaseTool contracts, retry/caching/logging/DI conventions, and
LangGraph state-management pattern are all unchanged and were followed
exactly for every new piece.

---

## 1. New product flow in one paragraph

`POST /api/v1/books/recognize` is unchanged at the call-site (same request/
response contract, same status codes) but its graph now also generates a
Gemini embedding for every validated book and persists the whole validated
bookshelf, keyed by `request_id`, in the existing cache backend. A new
`POST /api/v1/books/recommend` endpoint takes that `request_id` plus a
`liked_book` title and `top_k`, resolves the liked book against the stored
shelf (fuzzy title match), ranks every *other* book on that same shelf by
cosine similarity of their embeddings (LLM-reasoning fallback if an
embedding is missing), and returns the top-K — **never** a book that wasn't
physically in the photo.

---

## 2. `app/config/settings.py` (MODIFIED)

**Why**: new tunables for the embedding model, the bookshelf persistence
TTL, and recommendation ranking thresholds.
**What changed**: appended a `GEMINI_EMBEDDING_*` block, a
`BOOKSHELF_STORE_KEY_PREFIX` / `CACHE_TTL_BOOKSHELF_SECONDS` block, and a
`RECOMMENDATION_*` block (default/max top_k, min-similarity floor). No
existing setting was removed or renamed.
**Migration**: none required — all new fields have defaults; existing
`.env` files keep working unmodified. Optionally set
`GEMINI_EMBEDDING_MODEL` if you want a non-default embedding model.
**New dependencies**: none.
**Testing**: covered indirectly by every test that constructs `Settings`/
`tmp_settings` (unchanged fixture still works because everything new has a
default).
**Performance**: `CACHE_TTL_BOOKSHELF_SECONDS` (6h default) bounds how long
a photographed shelf stays queryable — tune down for memory-constrained
Redis, up if users browse recommendations over many sessions.
**Future improvements**: expose `RECOMMENDATION_MIN_SIMILARITY` per-request
(currently global) so a client can ask for a stricter/looser shelf match.

---

## 3. `app/models/domain.py` (MODIFIED)

**Why**: books need to carry an embedding vector; recommendations need a
new output shape matching the spec exactly (`similarity_score`,
`reason_for_recommendation`, `common_topics`, etc.) instead of the old
Google-Books-flavored `Recommendation` model.
**What changed**: added `ValidatedBook.embedding: list[float] = []`
(backward-compatible default, so existing constructions of `ValidatedBook`
without an embedding still work). Added a new `BookshelfRecommendation`
model. The old `Recommendation` model is **left in place, untouched** —
nothing currently imports it, but deleting a public model that might be
imported elsewhere felt riskier than a few unused lines; flagged for
removal in a follow-up cleanup PR once confirmed unused project-wide.
**Migration**: any code constructing `ValidatedBook(...)` directly (e.g.
tests, fixtures) needs no changes — `embedding` defaults to `[]`.
**New dependencies**: none.
**Testing**: `tests/unit/test_recommendation_tool.py` constructs
`ValidatedBook` and `BookshelfRecommendation` instances directly.
**Performance**: embeddings are typically 768–3072 floats; stored inline on
`ValidatedBook` and serialized to the bookshelf cache — fine at
single-shelf scale (dozens of books), would need a dedicated vector store
if this ever needs to search across *many* users' shelves at once (it
currently never does — recommendations are always scoped to one shelf).
**Future improvements**: delete the now-dead `Recommendation` model once a
grep across any downstream consumers confirms it's safe.

---

## 4. `app/models/state.py` (MODIFIED)

**Why**: `PipelineState.recommendations` and `BookRecognitionResponse.
recommendations` need to hold `BookshelfRecommendation`, not the old
`Recommendation`; the recommend flow needs `liked_book_title` /
`recommendation_top_k` fields on state; the new endpoint needs its own
request/response models.
**What changed**: swapped the `Recommendation` import/type for
`BookshelfRecommendation`; added `liked_book_title: str | None` and
`recommendation_top_k: int` to `PipelineState` (both optional —
`PipelineState` is `TypedDict(total=False)`, so this doesn't break any
existing partial-state construction); added `BookRecommendRequest` and
`BookRecommendResponse` Pydantic models; initialized the two new fields in
`new_pipeline_state(...)`.
**Migration**: none — additive only. Any code building a
`BookRecognitionResponse` still works since `recommendations` accepts an
(now always empty, at `/recognize` time) list.
**New dependencies**: none.
**Testing**: exercised by every integration test.
**Performance**: n/a (typed containers only).
**Future improvements**: consider a discriminated union / separate
`RecommendPipelineState` if the two flows' state needs diverge further.

---

## 5. `app/core/exceptions.py` (MODIFIED)

**Why**: need typed errors for "no stored bookshelf for this request_id"
(genuinely a 404, distinct from validation/API errors) and for embedding
failures.
**What changed**: added `EmbeddingAPIError` (an `ExternalAPIError`
subclass, consistent with `GoogleBooksAPIError`/`GeminiAPIError` already
there), `BookshelfNotFoundError` (`http_status = 404`), and
`BookNotInBookshelfError` (defined for completeness/future use — the
current implementation surfaces "book not found on shelf" as a `200` with
`status="not_found"` rather than raising, since it's an expected user
outcome, not a server error; see `recommendation_agent.py` below).
**Migration**: none.
**New dependencies**: none.
**Testing**: `test_recommend_unknown_request_id_raises_bookshelf_not_found`
in the integration suite.
**Performance**: n/a.
**Future improvements**: none of note.

---

## 6. `app/tools/embedding_tool.py` (NEW)

**Why**: powers the semantic-similarity recommendation engine — needs a
vector per book.
**What it does**: `GeminiEmbeddingTool`, structured identically to the
existing `GeminiVisionTool` (same `google-genai` client, same
executor-offload for the blocking SDK call, same `BaseTool._execute`
retry/timeout wrapper). Builds an embedding-input string per book from
`title | subtitle | author | categories | description` (truncated to
`GEMINI_EMBEDDING_MAX_CHARS`), embeds all books in a request concurrently
via `asyncio.gather`, and **caches each embedding** (keyed by a hash of the
input text) in the existing `CacheBackend` — the same book's metadata
embeds identically every time, so this is a pure win once a title has been
seen before.
**LangGraph node**: none directly — wrapped by `EmbeddingGenerationAgent`.
**API changes**: none (internal tool).
**New dependencies**: none — reuses the `google-genai` SDK already
installed for `GeminiVisionTool`.
**Testing**: exercised indirectly via `MockEmbeddingTool` in integration
tests; a real-API smoke test would need `GEMINI_API_KEY` and is left as a
manual/CI-secrets-gated test (same policy as the existing
`GeminiVisionTool`, which also has no live-API unit test in this repo).
**Performance**: concurrent per-book embedding calls, so wall-clock cost is
~one embedding call, not N sequential ones; caching removes repeat cost
entirely for previously-seen titles.
**Future improvements**: batch multiple books into a single
`embed_content` call if/when the SDK's batch endpoint stabilizes, to cut
request count further.

---

## 7. `app/agents/embedding_agent.py` (NEW)

**Why**: the new LangGraph node that runs `GeminiEmbeddingTool` over
`state["validated_books"]` and writes the vectors back.
**What it does**: `EmbeddingGenerationAgent` — no-ops if there are no
validated books; otherwise calls the tool, copies each `ValidatedBook`
with its `embedding` attached (`model_copy(update=...)`, never mutates in
place, consistent with the rest of the pipeline's immutable-state style),
and returns a `warnings` entry if any book didn't get an embedding (so
it's visible in the final response, not silently swallowed). Failure of
this agent as a whole is caught by `BaseAgent.execute` (existing
try/except-and-record-error wrapper) and simply leaves books without
embeddings — non-fatal, since `RecommendationTool` has an LLM fallback.
**LangGraph node**: `embedding_generation`, added between `validation` and
`bookshelf_memory`.
**API changes**: none directly.
**New dependencies**: none.
**Testing**: `test_full_pipeline_happy_path_stores_bookshelf_without_auto_recommending`
asserts `result.validated_books[0].embedding` is populated.
**Performance**: one extra Gemini call per recognize request (batched
across books, see above); adds to `token_usage`/`estimated_cost_usd` in
the response like every other Gemini-calling agent.
**Future improvements**: skip re-embedding books whose (title, author)
pair was already embedded earlier *in the same request* (currently only
cross-request caching is implemented).

---

## 8. `app/cache/bookshelf_store.py` (NEW)

**Why**: `/books/recognize` and `/books/recommend` are two separate HTTP
requests. Something has to bridge them. LangGraph's own checkpointer was
considered and deliberately **not** used for this (see the doc-comment in
the file and in `app/graph/workflow.py`): `MemorySaver` is per-process
(breaks across replicas), the sqlite checkpointer is per-node-local-disk
(breaks across replicas in the project's own `deploy/k8s-deployment.yaml`,
which is a multi-replica Deployment). The already-existing `CacheBackend`
abstraction (Redis in production, shared across all workers; in-memory for
local/dev) is the right layer for "state that must survive across two
requests, possibly on different pods."
**What it does**: `BookshelfStore.save(request_id, books)` /
`.load(request_id)` — thin, typed wrapper serializing `list[ValidatedBook]`
to/from JSON via `model_dump(mode="json")`, so it works identically
whether `CacheBackend` is the Redis or in-memory implementation.
**LangGraph node**: none — used by `BookshelfMemoryAgent` and by
`run_recommendation`.
**API changes**: none directly.
**New dependencies**: none (reuses `redis`, already a dependency of
`CacheBackend`).
**Testing**: `test_full_pipeline_happy_path_...` asserts
`container.bookshelf_store.load(request_id)` returns the stored shelf.
**Performance**: one Redis round-trip per recognize (write) and per
recommend (read); negligible compared to the Vision/embedding calls
already in the pipeline.
**Future improvements**: if bookshelves need to be listable per-user (not
just per-request_id), add a secondary index (e.g. a Redis set keyed by
user id) — out of scope for this spec, which is single-shelf, single-user.

---

## 9. `app/agents/bookshelf_memory_agent.py` (NEW)

**Why**: the "Store every validated book in memory during the request"
step from the spec, made durable enough to survive until the follow-up
`/books/recommend` call.
**What it does**: `BookshelfMemoryAgent` — no-ops if there are no
validated books; otherwise calls `BookshelfStore.save(...)`.
Deliberately does nothing else (no state mutation) — it's a pure
side-effecting persistence step.
**LangGraph node**: `bookshelf_memory`, between `embedding_generation` and
`recommendation`.
**API changes**: none directly.
**New dependencies**: none.
**Testing**: covered by the integration tests above.
**Performance**: single cache write, see `bookshelf_store.py`.
**Future improvements**: emit a metric/counter for bookshelf writes
(the project already has Prometheus wiring in `app/observability`; this
agent doesn't yet add a dedicated metric, following the existing pattern
where most agents rely on the shared `execution_trace`/`token_usage`
instrumentation rather than bespoke counters).

---

## 10. `app/tools/recommendation_tool.py` (REWRITTEN, same filename/class)

**Why**: this is the heart of the product change — recommendations must
come exclusively from the user's own bookshelf via semantic similarity,
never from Google Books / the open internet.
**What changed**: complete rewrite of the tool body; kept the same module
path, class name (`RecommendationTool`), and `BaseTool` contract so
`Container`/`RecommendationAgent` only needed a constructor-signature
change (dropped the `cache`/HTTP-client dependencies it no longer needs —
embeddings are cached one layer down in `GeminiEmbeddingTool`). New logic:
1. Excludes the liked book itself from candidates.
2. **Primary path**: cosine similarity between the liked book's embedding
   and every embedded candidate's embedding (`numpy`, already a
   dependency). Below `RECOMMENDATION_MIN_SIMILARITY`, a candidate is
   dropped — *unless* dropping it would leave zero results, in which case
   the tool falls back to showing the best-available match anyway (a
   two-book unrelated shelf should still return something rather than an
   empty list).
3. **Fallback path**: any candidate missing an embedding (or the liked
   book itself missing one) goes through a Gemini `generate_content` call
   with a structured JSON response schema, asking the model to score
   topical similarity directly — no vector math needed. Failures here are
   caught and logged; they degrade the result set, they don't crash the
   request.
4. Every result — from either path — is normalized into the same
   `BookshelfRecommendation` shape, with `method` recording which path
   produced it.
**Migration**: any code importing the old `RecommendationInput`
(`liked_book_title: str, candidates: list[GoogleBooksVolume], ...`) needs
updating to the new shape (`liked_book: ValidatedBook, bookshelf:
list[ValidatedBook], top_k: int`) — done in `recommendation_agent.py` and
`tests/mocks/mock_tools.py`.
**New dependencies**: none (`numpy` already required).
**Testing**: `tests/unit/test_recommendation_tool.py` — cosine ordering,
liked-book exclusion, min-similarity filtering (both "drops an unrelated
book when a good match exists" and "falls back to best-effort when nothing
clears the bar").
**Performance**: cosine similarity over a shelf of a few dozen books is
effectively free (`numpy` vector ops); the LLM fallback path is the
expensive one and is only invoked for books that genuinely lack an
embedding.
**Future improvements**: swap the linear cosine scan for an ANN index
(e.g. `faiss`) if bookshelves ever grow into the hundreds+ of books — not
needed at today's "one photographed shelf" scale.

---

## 11. `app/agents/recommendation_agent.py` (REWRITTEN, same filename/class)

**Why**: needs to resolve the user's `liked_book` title against the
bookshelf and only then delegate to the (rewritten) tool; must remain a
no-op during `/books/recognize` since no query exists yet at that point.
**What changed**: added `find_book_by_title` (exact match first, then
`difflib.SequenceMatcher` fuzzy match above a 0.6 ratio threshold — so
"atomic habit" still resolves to "Atomic Habits" without over-matching an
unrelated title on a large shelf). `run()` reads `liked_book_title`/
`recommendation_top_k` from state (both unset during `/recognize`, so it
returns `{"recommendations": []}` immediately there); if the title
resolves to nothing on the shelf, returns the exact required warning
string `"Book not found in uploaded bookshelf."` instead of raising —
that's translated into the spec's required message at the API layer
(see `runner.py`).
**LangGraph node**: `recommendation` (unchanged position, now sits after
`bookshelf_memory`).
**API changes**: none directly (consumed by `run_recommendation`).
**New dependencies**: none (`difflib` is stdlib).
**Testing**: `test_find_book_by_title_exact_match`,
`_fuzzy_match_typo`, `_returns_none_when_not_on_shelf` (unit); the
not-found and success paths are also covered end-to-end in the integration
suite.
**Performance**: negligible — O(shelf size) string comparisons.
**Future improvements**: allow matching by ISBN/`google_volume_id` in
addition to title, for clients that already know the exact book identity
and want to skip fuzzy matching entirely.

---

## 12. `app/graph/workflow.py` (MODIFIED)

**Why**: wire the two new nodes into the existing graph shape exactly as
specified.
**What changed**: added `NODE_EMBEDDING_GENERATION` / `NODE_BOOKSHELF_
MEMORY` constants and nodes; re-routed the `validation` node's
`"has_validated_books"` edge to `embedding_generation` (was
`recommendation`); added `embedding_generation → bookshelf_memory →
recommendation` edges. `recommendation → final_response` is untouched. No
other node, edge, or conditional router changed — `route_after_validation`
and every other routing function is byte-for-byte the same.
**LangGraph modifications**: see diagram in `docs/architecture-graph.mmd`
(regenerated from the live compiled graph — see below). Verified by
compiling the graph and printing `compiled.get_graph().nodes` /
`.draw_mermaid()`; matches the spec's diagram exactly.
**API changes**: none directly.
**New dependencies**: none.
**Testing**: implicitly, by every integration test (the graph is compiled
and run for real, YOLO/quality included).
**Performance**: two additional graph nodes on the happy path; both are
cheap (one Gemini call batched across books, one cache write).
**Future improvements**: none of note — this is the stable core shape the
spec asked to preserve.

---

## 13. `app/graph/container.py` (MODIFIED)

**Why**: dependency-injection wiring for every new tool/agent, and updated
constructor calls for the rewritten `RecommendationTool`/`Recommendation
Agent`.
**What changed**: instantiates `GeminiEmbeddingTool`, `BookshelfStore`,
`EmbeddingGenerationAgent`, `BookshelfMemoryAgent`; changed
`RecommendationTool(settings=..., cache=...)` → `RecommendationTool(
settings=...)` (no longer needs a cache/HTTP client itself); changed
`RecommendationAgent(recommendation_tool=...)` →
`RecommendationAgent(recommendation_tool=..., settings=...)`. Every other
agent/tool construction is untouched.
**Migration**: any test or script constructing `Container` manually and
overriding `container.recommendation_tool`/`container.recommendation_
agent` directly (rather than via the constructor) needs to also re-wire
`container.embedding_tool` if it wants embeddings mocked — done in
`tests/integration/test_pipeline_integration.py`.
**New dependencies**: none.
**Testing**: the whole integration suite exercises this wiring.
**Performance**: n/a (construction only, once per process/test).
**Future improvements**: none of note.

---

## 14. `app/graph/runner.py` (MODIFIED)

**Why**: needs a dedicated entrypoint for the `/books/recommend` flow that
loads the persisted bookshelf and drives `RecommendationAgent` directly
(see the "why not resume the LangGraph checkpoint" rationale in
`workflow.py` / `bookshelf_store.py`).
**What changed**: added `run_recommendation(request_id, liked_book, top_k,
container, settings) -> BookRecommendResponse`. Loads the bookshelf via
`BookshelfStore.load` (raises `BookshelfNotFoundError` — propagates to the
existing `ErrorHandlingMiddleware`, which already converts any `AppError`
subclass into the right HTTP status — no new middleware code needed);
builds a minimal `PipelineState` via the existing `new_pipeline_state(...)`
factory with `validated_books`/`liked_book_title`/`recommendation_top_k`
set; calls `container.recommendation_agent.execute(state)` (the same
`BaseAgent.execute` wrapper every other agent uses, so logging/tracing/
error-capture behavior is identical); maps the resulting delta into a
`BookRecommendResponse`, including the exact `"Book not found in uploaded
bookshelf."` message when the title didn't resolve. `run_pipeline` (the
existing `/books/recognize` entrypoint) is **untouched**.
**LangGraph modifications**: none to the graph itself — this function
calls one agent directly rather than invoking the compiled graph, for the
reasons documented in `workflow.py`.
**API changes**: powers the new `POST /books/recommend` route.
**New dependencies**: none.
**Testing**: `test_recommend_after_recognize_returns_bookshelf_only_recs`,
`test_recommend_unknown_book_returns_not_found`,
`test_recommend_unknown_request_id_raises_bookshelf_not_found`.
**Performance**: one cache read + the (cheap, see above) recommendation
computation — no image/vision/detection work re-runs.
**Future improvements**: if the interrupt/resume LangGraph pattern is ever
wanted for full audit-trail parity between the two calls (single unified
`execution_trace` spanning both requests), it's a viable alternative once
a distributed checkpointer (e.g. Postgres) is adopted — noted as a
deliberate trade-off, not an oversight.

---

## 15. `app/api/routes/recommendation.py` (NEW)

**Why**: the new `POST /api/v1/books/recommend` endpoint from the spec.
**What it does**: thin FastAPI route, same conventions as `recognition.py`
(`verify_api_key` dependency, `log_extra` structured logging, delegates
all logic to `run_recommendation`). `BookshelfNotFoundError` isn't caught
here — it propagates to the existing global `ErrorHandlingMiddleware`,
which already knows how to turn any `AppError` subclass into a structured
`{error_code, message, ...}` JSON body at the right HTTP status; adding a
local try/except would have just duplicated that.
**LangGraph modifications**: none directly.
**API changes**: new endpoint, exact request/response shape from the spec
(`request_id`, `liked_book`, `top_k` in; `liked_book`, `recommendations`,
`reasoning` out, plus `status`/`message` for the not-found case).
**New dependencies**: none.
**Testing**: covered end-to-end via `run_recommendation` integration
tests; a dedicated `httpx.AsyncClient`-based route test (mirroring
`tests/e2e/test_api_e2e.py`'s pattern for `/recognize`) is a natural
follow-up but wasn't required to validate the logic, which is fully
exercised at the `run_recommendation` layer.
**Performance**: see `runner.py`.
**Future improvements**: add the e2e HTTP-level test mentioned above.

---

## 16. `app/main.py` (MODIFIED)

**Why**: register the new router; refresh the OpenAPI description.
**What changed**: added `recommendation` to the router import and
`app.include_router(recommendation.router)`; updated the app description
string to mention the bookshelf-only recommendation capability. Nothing
else in `main.py` (CORS, middleware order, lifespan, exception handlers)
was touched.
**Migration**: none.
**New dependencies**: none.
**Testing**: `app.main.app.routes` was inspected directly to confirm
`/api/v1/books/recommend` is registered (see verification section below).
**Performance**: n/a.
**Future improvements**: none of note.

---

## 17. Tests (MODIFIED / NEW)

- `tests/mocks/mock_tools.py`: replaced `MockRecommendationTool`'s old
  Google-Books-flavored contract with the new bookshelf-only one; added
  `MockEmbeddingTool` (deterministic, non-degenerate fake vectors per
  book, so cosine-similarity math in tests is meaningful rather than a
  zero-vector edge case).
- `tests/integration/test_pipeline_integration.py`: the happy-path test
  now asserts `/recognize` does **not** auto-generate recommendations
  (matches the new "wait for user query" semantics) and that the
  bookshelf is retrievable afterward; three new tests cover the
  recommend-after-recognize, book-not-on-shelf, and
  unknown-request_id-raises-404 paths. The hallucination-rejection test is
  unchanged (it never reaches the new nodes).
- `tests/unit/test_recommendation_tool.py` (NEW): cosine-similarity
  ordering, liked-book self-exclusion, min-similarity filtering behavior
  (both branches), and `RecommendationAgent.find_book_by_title` matching.

**Verification performed in this change (see chat for full output)**:
- `python -m py_compile` across every file in `app/` and `tests/` — clean.
- Installed the project's actual (non-CV/ML) dependencies from
  `requirements.txt` and imported every new/modified module directly —
  clean, including `app.main` (FastAPI app construction).
- Compiled the real `LangGraph` graph via `Container` + `build_graph(...)`
  and printed both its node list and Mermaid source — matches the spec's
  diagram exactly (`embedding_generation → bookshelf_memory →
  recommendation`, `recommendation → final_response`, `validation`'s
  `has_validated_books` edge now points at `embedding_generation`).
- Ran the full test suite: **30 passed**, 0 failed (includes the existing,
  untouched e2e/unit tests for detection, quality, decision, super-res,
  vision, validation, retry, caching, middleware — none of that broke).

---

## 18. Files intentionally NOT modified

Per "minimize changes to existing modules": `app/api/routes/recognition.py`
and `app/api/routes/recognition_stream.py` (the `/recognize` contract is
unchanged), `app/agents/book_detection_agent.py`,
`app/agents/image_quality_agent.py`, `app/agents/decision_agent.py`,
`app/agents/super_resolution_agent.py`, `app/agents/gemini_vision_agent.py`,
`app/agents/validation_agent.py`, `app/agents/final_response_agent.py`,
`app/agents/base_agent.py`, `app/tools/base_tool.py`,
`app/tools/yolo_detection_tool.py`, `app/tools/image_quality_tool.py`,
`app/tools/super_resolution_tool.py`, `app/tools/gemini_vision_tool.py`,
`app/tools/google_books_tool.py`, `app/cache/cache_backend.py`,
`app/core/retry.py`, `app/core/logging.py`, `app/api/middleware/*`,
`app/observability/*`, `app/security/*`, all Docker/K8s deploy manifests.

Docs not updated in this pass (flagged, not forgotten):
`docs/API.md`, `docs/ARCHITECTURE.md`, `docs/DEVELOPER_GUIDE.md` still
describe the old single-endpoint flow in prose; `README.md` and
`docs/architecture-graph.mmd` **were** updated since they're the highest-
traffic entry points. Regenerating the rest is a good follow-up but out of
scope for the code change itself.
