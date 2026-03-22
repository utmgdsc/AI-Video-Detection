"""
Analysis job API routes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from backend.api.schemas import AnalyzeResponse
from backend.utils.cleanup import cleanup_job_artifacts
from backend.utils.file_storage import (
    create_job_temp_dir,
    normalize_upload_filename,
    stream_upload_to_path,
    validate_upload_metadata,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])

CHUNK_SIZE = 1024 * 1024


def get_api_state(request: Request):
    return request.app.state.api_state


@router.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK)
async def analyze_video(
    video_file: UploadFile = File(...),
    api_state=Depends(get_api_state),
):
    if not api_state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=api_state.ready_error or "Service is not ready.",
        )

    filename = normalize_upload_filename(video_file.filename)
    validate_upload_metadata(
        filename=filename,
        content_type=video_file.content_type,
        allowed_suffixes=api_state.allowed_suffixes,
        allowed_mime_types=api_state.allowed_mime_types,
    )

    temp_root = api_state.temp_dir
    job_dir = create_job_temp_dir(temp_root=temp_root)
    target_path = job_dir / filename

    try:
        await stream_upload_to_path(
            upload=video_file,
            target_path=target_path,
            max_upload_size_bytes=api_state.max_upload_size_bytes,
            chunk_size=CHUNK_SIZE,
        )

        if api_state.detector_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=api_state.ready_error or "Detector service is not initialized.",
            )

        result = api_state.detector_service.analyze_video(str(target_path))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Synchronous analysis failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {exc}",
        ) from exc
    finally:
        await video_file.close()
        cleanup_job_artifacts(target_path, job_dir)

    return AnalyzeResponse(result=result)
