"""Streaming recognition endpoint -- emits per-agent progress via SSE.

Useful for clients that want to show a live progress bar ("detecting
books... assessing quality... calling Gemini...") instead of waiting for
the full synchronous response.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, Depends, File, UploadFile
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.core.logging import get_logger
from app.graph.container import get_container
from app.graph.workflow import compile_pipeline
from app.models.state import new_pipeline_state
from app.security.auth import verify_api_key
from app.security.validation import sanitize_and_reencode, validate_upload

router = APIRouter(tags=["recognition"])
logger = get_logger(__name__)


@router.post(
    "/api/v1/books/recognize/stream",
    summary="Recognize books with streamed per-agent progress (SSE)",
    dependencies=[Depends(verify_api_key)],
)
async def recognize_books_stream(file: UploadFile = File(...)) -> EventSourceResponse:
    settings = get_settings()
    content = await file.read()
    validate_upload(content, file.content_type, settings)
    clean_bytes = sanitize_and_reencode(content)

    request_id = str(uuid.uuid4())
    upload_path = Path(settings.UPLOAD_DIR) / f"{request_id}.jpg"
    upload_path.write_bytes(clean_bytes)

    return EventSourceResponse(_stream_events(str(upload_path), request_id))


async def _stream_events(image_path: str, request_id: str) -> AsyncIterator[dict]:
    settings = get_settings()
    container = get_container()
    initial_state = new_pipeline_state(request_id=request_id, original_image_path=image_path)

    async with await compile_pipeline(container, settings) as app:
        config = {
            "configurable": {"thread_id": request_id},
            "recursion_limit": settings.GRAPH_RECURSION_LIMIT,
        }
        async for event in app.astream(initial_state, config=config, stream_mode="updates"):
            for node_name, delta in event.items():
                trace = delta.get("execution_trace", [])
                status = trace[-1].status if trace else "running"
                yield {
                    "event": "agent_update",
                    "data": json.dumps(
                        {"request_id": request_id, "agent": node_name, "status": status}
                    ),
                }
        yield {"event": "done", "data": json.dumps({"request_id": request_id})}
