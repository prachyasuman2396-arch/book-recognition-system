# Developer Guide

## Code organization principles

- **Clean architecture layering**: `tools` (stateless external-facing
  operations) → `agents` (orchestration/business logic wrapping tools) →
  `graph` (workflow wiring) → `api` (HTTP surface). Dependencies point
  inward; `api` knows about `graph`, `graph` knows about `agents`/`tools`,
  but `tools` never import `agents` or `api`.
- **Dependency injection**: every agent takes its tools as constructor
  arguments (see `app/graph/container.py`). Never instantiate a tool inside
  an agent method — this is what makes mocking in tests trivial.
- **Typed everything**: all tool/agent I/O is a Pydantic model or a
  `@dataclass`; no bare dicts crossing a public boundary except the
  LangGraph `PipelineState` itself (a `TypedDict` by LangGraph's design).

## Adding a new tool

1. Create `app/tools/my_tool.py`, subclass `BaseTool[InputT, OutputT]`.
2. Give it a class-level `name` (used in logs/metrics).
3. Implement `async def run(self, payload: InputT) -> OutputT`, delegating
   the actual work to `self._execute(self._do_thing, payload)` so you get
   retry/timeout/metrics for free.
4. If it calls an external API, raise your own subclass of `ExternalAPIError`
   (see `app/core/exceptions.py`) on failure so retries target it precisely.

```python
class MyTool(BaseTool[MyInput, MyOutput]):
    name = "my_tool"

    async def run(self, payload: MyInput) -> MyOutput:
        return await self._execute(self._do_thing, payload)

    async def _do_thing(self, payload: MyInput) -> MyOutput:
        ...
```

## Adding a new agent

1. Create `app/agents/my_agent.py`, subclass `BaseAgent`.
2. Constructor takes the tool(s) it needs.
3. Implement `async def run(self, state: PipelineState) -> dict[str, Any]`
   returning **only the fields you changed** — never touch
   `execution_trace` (injected automatically) and never return the full
   accumulated value of an `Annotated[list, operator.add]` field, only the
   new items.
4. Register it in `Container.__post_init__` and wire it into
   `app/graph/workflow.py` (new node + edges). If it needs conditional
   routing, add a function to `app/graph/router.py`.
5. Add a unit test exercising `agent.execute(state)` directly with a
   hand-built `state` dict (see `tests/unit/test_decision_agent.py`).

## State field conventions

| Field | Merge strategy | Who writes it |
|---|---|---|
| `detections`, `quality_reports`, `tool_decisions`, `enhanced_images`, `vision_results`, `validated_books`, `recommendations`, `metrics` | last-write-wins | exactly one agent each |
| `execution_trace`, `errors`, `warnings`, `token_usage` | append (`operator.add`) | any agent, via `BaseAgent.execute` for the first three |
| `timestamps` | dict-merge | any agent |

## Logging

Use `app.core.logging.get_logger(__name__)` / `log_extra(logger, level, msg,
**fields)` rather than bare `print` or unstructured `logger.info(f"...")`
calls — this keeps every log line valid JSON with a `request_id` attached
automatically via `contextvars`.

## Error handling

- Tool-level failures → typed `ToolError` subclasses (`app/core/exceptions.py`).
- Agent-level failures are caught by `BaseAgent.execute` and converted to a
  recorded `errors` entry; they don't crash the graph.
- API-level failures → any `AppError` subclass is converted to the right
  HTTP status by `ErrorHandlingMiddleware`.

## Testing philosophy

- **Unit**: one tool/agent/router function at a time, real local
  computation (image quality, YOLO fallback) where cheap and deterministic,
  mocked network calls.
- **Integration**: the real compiled graph, with only the network-bound
  tools (Gemini, Google Books, Recommendations) swapped for deterministic
  mocks from `tests/mocks/mock_tools.py`.
- **E2E**: FastAPI `TestClient` against the real HTTP surface, same mocking
  strategy as integration tests.

See [TESTING.md](TESTING.md) for details.
