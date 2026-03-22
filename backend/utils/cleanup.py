"""
Temporary file cleanup helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil


logger = logging.getLogger(__name__)


def cleanup_job_artifacts(video_path: Path | None, job_dir: Path | None) -> None:
    if video_path is not None:
        try:
            if video_path.exists():
                video_path.unlink()
        except Exception:
            logger.exception("Failed to remove temp video: %s", video_path)

    if job_dir is not None:
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir, ignore_errors=True)
        except Exception:
            logger.exception("Failed to remove temp job directory: %s", job_dir)
