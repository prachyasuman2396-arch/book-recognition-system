"""Book recognition endpoint -- the main pipeline entrypoint."""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from app.config import get_settings
from app.core.logging import get_logger, log_extra
from app.graph.container import get_container
from app.graph.runner import run_pipeline
from app.models.state import BookRecognitionResponse
from app.security.auth import verify_api_key
from app.security.validation import read_upload_capped, sanitize_and_reencode, validate_upload

router = APIRouter(tags=["recognition"])
logger = get_logger(__name__)


@router.post(
    "/api/v1/books/recognize",
    response_model=BookRecognitionResponse,
    summary="Recognize and recommend books from a photo",
    dependencies=[Depends(verify_api_key)],
)
async def recognize_books(file: UploadFile = File(...)) -> BookRecognitionResponse:
    """Upload an image containing one or more books.

    Runs the full LangGraph pipeline: detection -> quality assessment ->
    conditional enhancement -> Gemini Vision recognition -> Google Books
    validation -> recommendations -> aggregated response.
    """
    settings = get_settings()
    content = await read_upload_capped(file, settings)

    validate_upload(content, file.content_type, settings)
    # Re-encoding (PIL decode/encode) and the subsequent disk write are
    # both blocking, CPU/IO-bound operations; run them in a worker thread
    # so a large/slow image doesn't stall the event loop for other
    # concurrent requests.
    clean_bytes = await asyncio.to_thread(sanitize_and_reencode, content)

    request_id = str(uuid.uuid4())
    upload_path = Path(settings.UPLOAD_DIR) / f"{request_id}.jpg"
    await asyncio.to_thread(upload_path.write_bytes, clean_bytes)

    log_extra(logger, 20, "Upload accepted", request_id=request_id, filename=file.filename)

    container = get_container()
    result = await run_pipeline(
        image_path=str(upload_path), request_id=request_id, container=container, settings=settings
    )
    return result
