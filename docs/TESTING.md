# Testing Guide

## Running the suite

```bash
./scripts/run_tests.sh                 # everything, with coverage
pytest tests/unit -q                   # fast, no I/O
pytest tests/integration -q            # full graph, mocked externals
pytest tests/e2e -q                    # HTTP layer, mocked externals
pytest -k test_decision_agent -q       # single test file
pytest --cov=app --cov-report=html     # HTML coverage report in htmlcov/
```

Required env vars for the suite (defaults are test-safe, but the app
requires *some* value): `GEMINI_API_KEY`, `GOOGLE_BOOKS_API_KEY` — set to
any placeholder string, since integration/e2e tests mock the actual network
calls. `scripts/run_tests.sh` sets sensible defaults automatically.

## Test layers

| Layer | What's real | What's mocked | Speed |
|---|---|---|---|
| `tests/unit` | Single tool/agent/router function | External APIs (Gemini, Google Books) | ms |
| `tests/integration` | Full compiled LangGraph, YOLO fallback, image quality | Gemini, Google Books, Recommendations | ~100ms–1s |
| `tests/e2e` | FastAPI `TestClient`, real middleware stack | Same as integration | ~100ms–1s |

## Mocking strategy

`tests/mocks/mock_tools.py` provides `MockGeminiVisionTool`,
`MockGoogleBooksTool`, and `MockRecommendationTool` — drop-in replacements
matching each real tool's `run()` signature. Swap them into a `Container`
instance and re-construct the dependent agents:

```python
container = Container(settings=tmp_settings)
container.gemini_tool = MockGeminiVisionTool()
container.gemini_vision_agent = GeminiVisionAgent(
    gemini_tool=container.gemini_tool, settings=tmp_settings
)
```

This pattern (swap the tool, then rebuild the one agent that owns it) keeps
tests explicit about exactly what's mocked, rather than patching at the
module level.

## Fixtures (`tests/conftest.py`)

- `tmp_settings` — an isolated `Settings` instance writing to `tmp_path`,
  with `CACHE_BACKEND=memory`, `GRAPH_CHECKPOINT_BACKEND=memory`, and
  `API_KEY_ENABLED=False` so tests don't need real auth.
- `sample_image_path` — a synthetic JPEG on disk for tools that need a real
  file.
- `_cleanup_module_singletons` (autouse) — resets the cache/container
  singletons between tests so state doesn't leak across test functions.

## Writing a new test

- **Tool test**: construct the tool with `tmp_settings`, call `.run(...)`
  directly, assert on the typed output. Use `patch.object` on any private
  network method (e.g. `_fetch`) rather than mocking `httpx` globally.
- **Agent test**: build a plain `dict` matching the subset of
  `PipelineState` your agent reads, call `agent.execute(state)`, assert on
  the returned delta dict (including `delta["execution_trace"][0].status`).
- **Router test**: call the router function directly with a hand-built
  state dict — no graph needed.
- **Integration/E2E test**: swap in the mocks as shown above, run the real
  graph/HTTP client, assert on the final `BookRecognitionResponse`.

## Coverage target

CI enforces a minimum via `pytest-cov` (`pyproject.toml` →
`addopts = "--cov=app --cov-report=term-missing"`). Aim for ≥80% on
`app/agents`, `app/graph`, and `app/api` (business-logic-dense, cheap to
test); tool code that wraps optional heavy ML dependencies (Real-ESRGAN,
ultralytics) is allowed lower coverage since its fallback paths are what's
actually exercised in CI.
