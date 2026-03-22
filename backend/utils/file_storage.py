"""
File upload and temporary storage helpers for API workflows.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status


DEFAULT_CHUNK_SIZE = 1024 * 1024


def normalize_upload_filename(filename: str | None) -> str:
    candidate = Path(filename or "uploaded_video.mp4").name
    if not candidate:
        return "uploaded_video.mp4"
    return candidate


def validate_upload_metadata(
    filename: str,
    content_type: str | None,
    allowed_suffixes: set[str],
    allowed_mime_types: set[str],
) -> None:
    suffix = Path(filename).suffix.lower()
    if allowed_suffixes and suffix not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension. Allowed: {allowed}",
        )

    normalized_content_type = (content_type or "").strip().lower()
    if allowed_mime_types and normalized_content_type not in allowed_mime_types:
        allowed = ", ".join(sorted(allowed_mime_types))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type '{content_type}'. Allowed: {allowed}",
        )


def create_job_temp_dir(temp_root: Path, job_id: str | None = None) -> Path:
    suffix = job_id or str(uuid4())
    return Path(tempfile.mkdtemp(prefix=f"upload-{suffix}-", dir=str(temp_root)))


async def stream_upload_to_path(
    upload: UploadFile,
    target_path: Path,
    max_upload_size_bytes: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    bytes_written = 0
    with target_path.open("wb") as out_file:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > max_upload_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds max upload size ({max_upload_size_bytes} bytes).",
                )
            out_file.write(chunk)
    return bytes_written
