"""Prometheus metrics registry.

Central place for all counters/histograms so agents and tools import one
object instead of re-declaring metrics (which would raise on duplicate
registration). Falls back to a no-op stub if `prometheus_client` isn't
installed, so unit tests never hard-depend on it.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

try:
    from prometheus_client import CollectorRegistry, Counter, Histogram

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only w/o the dependency
    _PROM_AVAILABLE = False


class _NoOpMetric:
    def labels(self, *_a: object, **_kw: object) -> "_NoOpMetric":
        return self

    def inc(self, *_a: object, **_kw: object) -> None:
        return None

    def observe(self, *_a: object, **_kw: object) -> None:
        return None


class MetricsRegistry:
    """Namespaced Prometheus metrics for the whole pipeline."""

    def __init__(self, namespace: str = "book_recognition") -> None:
        self.namespace = namespace
        if _PROM_AVAILABLE:
            self.registry = CollectorRegistry()
            self.agent_calls_total = Counter(
                f"{namespace}_agent_calls_total",
                "Total agent invocations",
                ["agent", "status"],
                registry=self.registry,
            )
            self.agent_latency_seconds = Histogram(
                f"{namespace}_agent_latency_seconds",
                "Agent execution latency",
                ["agent"],
                registry=self.registry,
            )
            self.tool_calls_total = Counter(
                f"{namespace}_tool_calls_total",
                "Total tool invocations",
                ["tool", "status"],
                registry=self.registry,
            )
            self.tool_latency_seconds = Histogram(
                f"{namespace}_tool_latency_seconds",
                "Tool execution latency",
                ["tool"],
                registry=self.registry,
            )
            self.tool_retries_total = Counter(
                f"{namespace}_tool_retries_total",
                "Total tool retry attempts",
                ["tool"],
                registry=self.registry,
            )
            self.tokens_used_total = Counter(
                f"{namespace}_tokens_used_total",
                "Total LLM tokens consumed",
                ["model", "kind"],
                registry=self.registry,
            )
            self.estimated_cost_usd_total = Counter(
                f"{namespace}_estimated_cost_usd_total",
                "Total estimated USD cost of LLM calls",
                ["model"],
                registry=self.registry,
            )
            self.requests_total = Counter(
                f"{namespace}_requests_total",
                "Total API requests",
                ["route", "status_code"],
                registry=self.registry,
            )
        else:  # pragma: no cover
            noop = _NoOpMetric()
            self.registry = None
            self.agent_calls_total = noop
            self.agent_latency_seconds = noop
            self.tool_calls_total = noop
            self.tool_latency_seconds = noop
            self.tool_retries_total = noop
            self.tokens_used_total = noop
            self.estimated_cost_usd_total = noop
            self.requests_total = noop

    @contextmanager
    def time_agent(self, agent_name: str) -> Iterator[None]:
        start = perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.agent_latency_seconds.labels(agent=agent_name).observe(perf_counter() - start)
            self.agent_calls_total.labels(agent=agent_name, status=status).inc()

    @contextmanager
    def time_tool(self, tool_name: str) -> Iterator[None]:
        start = perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.tool_latency_seconds.labels(tool=tool_name).observe(perf_counter() - start)
            self.tool_calls_total.labels(tool=tool_name, status=status).inc()

    def render(self) -> bytes:
        if not _PROM_AVAILABLE or self.registry is None:  # pragma: no cover
            return b""
        from prometheus_client import generate_latest

        return generate_latest(self.registry)


_metrics_singleton: MetricsRegistry | None = None


def get_metrics(namespace: str = "book_recognition") -> MetricsRegistry:
    global _metrics_singleton
    if _metrics_singleton is None:
        _metrics_singleton = MetricsRegistry(namespace=namespace)
    return _metrics_singleton
