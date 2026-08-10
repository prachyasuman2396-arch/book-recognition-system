"""Abstract base class for all stateless tools.

Every concrete tool:
  * exposes exactly one public async method (its "entrypoint")
  * is stateless (no per-call mutable instance state)
  * gets retry, timeout, structured logging, and metrics for free by
    calling `self._execute(...)` from its public method.
"""
from __future__ import annotations

import abc
import asyncio
from time import perf_counter
from typing import Awaitable, Callable, Generic, TypeVar

from app.core.exceptions import ToolTimeoutError
from app.core.logging import get_logger, log_extra
from app.core.retry import with_retry
from app.observability.metrics import get_metrics

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class BaseTool(abc.ABC, Generic[InputT, OutputT]):
    """Common scaffolding shared by every tool implementation."""

    name: str = "base_tool"
    timeout_seconds: float = 20.0
    max_retries: int = 3

    def __init__(self) -> None:
        self.logger = get_logger(f"tool.{self.name}")
        self.metrics = get_metrics()

    async def _execute(
        self,
        func: Callable[..., Awaitable[OutputT]],
        *args: object,
        retry_on: tuple[type[Exception], ...] = (Exception,),
        **kwargs: object,
    ) -> OutputT:
        """Run `func` under timeout + retry + metrics + structured logging."""

        @with_retry(
            max_retries=self.max_retries,
            retry_on=retry_on,
            op_name=self.name,
        )
        async def _guarded() -> OutputT:
            return await asyncio.wait_for(
                func(*args, **kwargs), timeout=self.timeout_seconds
            )

        start = perf_counter()
        with self.metrics.time_tool(self.name):
            try:
                result = await _guarded()
            except asyncio.TimeoutError as exc:
                raise ToolTimeoutError(
                    f"{self.name} timed out after {self.timeout_seconds}s"
                ) from exc
        log_extra(
            self.logger,
            20,
            f"{self.name} completed",
            duration_ms=round((perf_counter() - start) * 1000, 2),
        )
        return result

    @abc.abstractmethod
    async def run(self, payload: InputT) -> OutputT:
        """The single public entrypoint every tool must implement."""
        raise NotImplementedError
