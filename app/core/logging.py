"""Structured, JSON-first logging configuration.

Uses stdlib `logging` with a custom JSON formatter so log aggregators
(Datadog, ELK, CloudWatch) can parse fields directly. A `contextvars`-backed
request id is injected into every record automatically.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
agent_name_ctx: ContextVar[str] = ContextVar("agent_name", default="-")


class JSONFormatter(logging.Formatter):
    """Render each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get(),
            "agent": agent_name_ctx.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | req=%(request_id)s | %(message)s"
        )

    def format(self, record: logging.LogRecord) -> str:
        record.request_id = request_id_ctx.get()  # type: ignore[attr-defined]
        return super().format(record)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if json_output else TextFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_extra(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit a log record carrying arbitrary structured fields."""
    logger.log(level, message, extra={"extra_fields": fields})
