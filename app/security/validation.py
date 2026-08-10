"""Upload validation: file type, size, and basic image sanitization.

Called by the upload route before anything touches disk beyond a temp
buffer, so malformed or oversized payloads never reach the pipeline.
"""
from __future__ import annotations

import io

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import Settings, get_settings
from app.core.exceptions import FileTooLargeError, InvalidImageError, UnsupportedFileTypeError

_READ_CHUNK_BYTES = 1024 * 1024  # 1MB


async def read_upload_capped(file: UploadFile, settings: Settings | None = None) -> bytes:
    """Read `file` in bounded chunks, aborting as soon as the configured
    size cap is exceeded.

    `await file.read()` with no argument buffers the *entire* upload into
    memory before any size check ever runs -- a client can send an
    arbitrarily large body and exhaust worker memory well before
    `MAX_UPLOAD_SIZE_MB` is enforced. Reading in bounded chunks and
    checking the running total after each one means we never hold more
    than `max_bytes + one chunk` in memory, regardless of how much the
    client tries to send.
    """
    settings = settings or get_settings()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise FileTooLargeError(
                f"Upload exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB}MB",
                details={"size_bytes_at_abort": total},
            )
        chunks.append(chunk)
    return b"".join(chunks)


def validate_upload(content: bytes, content_type: str | None, settings: Settings | None = None) -> None:
    """Raise a typed error if the upload fails any security/validation check."""
    settings = settings or get_settings()

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise FileTooLargeError(
            f"Upload exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB}MB",
            details={"size_bytes": len(content)},
        )

    if content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise UnsupportedFileTypeError(
            f"Unsupported content type: {content_type}",
            details={"allowed": list(settings.ALLOWED_IMAGE_TYPES)},
        )

    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()  # raises if not a genuine, uncorrupted image
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError(f"Uploaded file is not a valid image: {exc}") from exc


def sanitize_and_reencode(content: bytes) -> bytes:
    """Strip EXIF/metadata and re-encode as clean JPEG to neutralize
    malformed-metadata exploits and embedded payloads before any tool
    touches the file.
    """
    with Image.open(io.BytesIO(content)) as img:
        img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
