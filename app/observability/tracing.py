"""Optional OpenTelemetry tracing bootstrap.

Tracing is opt-in via `OTEL_ENABLED`. When disabled (default for local dev
and tests), `get_tracer()` returns a no-op tracer so instrumentation code
paths never branch on whether OTel is configured.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from app.config import get_settings


class _NoOpSpan:
    def set_attribute(self, *_a: object, **_kw: object) -> None:
        return None

    def record_exception(self, *_a: object, **_kw: object) -> None:
        return None


class _NoOpTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **_kw: object) -> Iterator[_NoOpSpan]:
        yield _NoOpSpan()


_tracer_singleton: object | None = None


def configure_tracing() -> None:
    settings = get_settings()
    if not settings.OTEL_ENABLED:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": settings.APP_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except ImportError:  # pragma: no cover
        return


def get_tracer(name: str) -> object:
    settings = get_settings()
    if not settings.OTEL_ENABLED:
        return _NoOpTracer()
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:  # pragma: no cover
        return _NoOpTracer()
