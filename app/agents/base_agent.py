"""Abstract base class for all pipeline agents.

Design note on LangGraph state merging
---------------------------------------
`PipelineState` uses `Annotated[list[...], operator.add]` reducers for
append-only fields (`execution_trace`, `errors`, `warnings`, `token_usage`).
LangGraph merges a node's *returned* dict into the accumulated state using
those reducers. That means a node must return only the *new* items for
those fields (e.g. `{"execution_trace": [one_new_step]}`), never the full
accumulated history -- returning the full history would double-count on
every hop.

Contract:
  * `run(state)` receives the full, already-merged `PipelineState` (read
    freely) and returns a **partial update dict** containing only the
    fields this agent changed/added.
  * `BaseAgent.execute(state)` wraps `run` and injects the execution-trace
    entry, error entries, and timing -- subclasses never touch
    `execution_trace` directly.
"""
from __future__ import annotations

import abc
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from app.core.logging import agent_name_ctx, get_logger, log_extra
from app.models.domain import ExecutionStep
from app.models.state import PipelineState
from app.observability.metrics import get_metrics


class BaseAgent(abc.ABC):
    """Common scaffolding shared by every concrete agent."""

    name: str = "base_agent"

    def __init__(self) -> None:
        self.logger = get_logger(f"agent.{self.name}")
        self.metrics = get_metrics()

    async def execute(self, state: PipelineState) -> dict[str, Any]:
        """LangGraph node entrypoint: returns a partial state update dict.

        Failures inside `run` are caught, recorded as a failed
        `ExecutionStep` and an `errors` entry, and swallowed rather than
        propagated -- so a single agent failure degrades the pipeline to a
        partial result instead of crashing the whole graph. Callers can
        inspect `state["errors"]` to detect this.
        """
        token = agent_name_ctx.set(self.name)
        started_at = datetime.now(timezone.utc)
        step = ExecutionStep(agent_name=self.name, started_at=started_at, status="running")
        start_perf = perf_counter()

        log_extra(self.logger, 20, f"{self.name} started", request_id=state.get("request_id"))

        delta: dict[str, Any] = {}
        error_message: str | None = None

        try:
            with self.metrics.time_agent(self.name):
                delta = await self.run(state)
            step.status = "success"
        except Exception as exc:  # noqa: BLE001 - recorded, never re-raised
            step.status = "error"
            step.error = str(exc)
            error_message = f"{self.name}: {exc}"
            log_extra(
                self.logger,
                40,
                f"{self.name} failed",
                request_id=state.get("request_id"),
                error=str(exc),
            )
        finally:
            step.finished_at = datetime.now(timezone.utc)
            step.duration_ms = round((perf_counter() - start_perf) * 1000, 2)
            agent_name_ctx.reset(token)

        delta["execution_trace"] = [step]
        if error_message:
            delta["errors"] = [error_message]
        return delta

    @abc.abstractmethod
    async def run(self, state: PipelineState) -> dict[str, Any]:
        """Subclasses implement business logic and return a partial delta.

        Must NOT include `execution_trace` (injected by `execute`). May
        raise to signal failure; `execute` records it as a non-fatal error
        entry so downstream conditional edges/consumers decide whether to
        halt based on `state["errors"]`.
        """
        raise NotImplementedError
