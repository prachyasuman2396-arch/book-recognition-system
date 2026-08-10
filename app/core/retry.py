"""Reusable async retry decorator with exponential backoff + jitter.

Every tool wraps its external call with `@with_retry(...)` rather than
hand-rolling retry loops, so backoff behavior is consistent and testable.
"""
from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.core.exceptions import ToolRetryExhaustedError
from app.core.logging import get_logger, log_extra

logger = get_logger(__name__)

T = TypeVar("T")


def with_retry(
    *,
    max_retries: int = 3,
    backoff_base: float = 0.5,
    backoff_max: float = 8.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    op_name: str = "operation",
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorate an async function to retry with exponential backoff + jitter."""

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: object, **kwargs: object) -> T:
            attempt = 0
            last_exc: Exception | None = None
            start = time.monotonic()

            while attempt <= max_retries:
                try:
                    return await func(*args, **kwargs)
                except retry_on as exc:  # noqa: PERF203
                    last_exc = exc
                    attempt += 1
                    if attempt > max_retries:
                        break
                    delay = min(backoff_max, backoff_base * (2 ** (attempt - 1)))
                    delay += random.uniform(0, backoff_base)
                    log_extra(
                        logger,
                        30,
                        f"Retrying {op_name} after failure",
                        attempt=attempt,
                        max_retries=max_retries,
                        delay_seconds=round(delay, 3),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)

            elapsed = round(time.monotonic() - start, 3)
            raise ToolRetryExhaustedError(
                f"{op_name} failed after {max_retries} retries ({elapsed}s elapsed)",
                details={"last_error": str(last_exc), "attempts": attempt},
            ) from last_exc

        return wrapper

    return decorator
