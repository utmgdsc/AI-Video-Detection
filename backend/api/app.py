"""
FastAPI application entrypoint for backend UI integration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.analyze import router as analyze_router
from backend.api.routes.health import router as health_router
from backend.core.config import ConfigValidationError, load_settings


logger = logging.getLogger(__name__)


class APIState:
    def __init__(self) -> None:
        self.ready = False
        self.ready_error: str | None = None
        self.settings: Any = None
        self.detector_service: Any = None
        self.max_upload_size_bytes = int(os.getenv("AIVD_MAX_UPLOAD_SIZE_BYTES", str(1024 * 1024 * 1024)))

        allowed = os.getenv("AIVD_ALLOWED_VIDEO_SUFFIXES", ".mp4,.mov,.avi,.mkv,.webm")
        self.allowed_suffixes = {item.strip().lower() for item in allowed.split(",") if item.strip()}
        mime_types = os.getenv(
            "AIVD_ALLOWED_VIDEO_MIME_TYPES",
            (
                "video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,"
                "video/webm,application/octet-stream"
            ),
        )
        self.allowed_mime_types = {
            item.strip().lower() for item in mime_types.split(",") if item.strip()
        }

        self.temp_dir = Path(os.getenv("AIVD_TEMP_DIR", "./outputs/tmp/")).resolve()
        self.temp_dir.mkdir(parents=True, exist_ok=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Video Detection Backend API",
        version="0.1.0",
    )
    app.state.api_state = APIState()

    allowed_origins = os.getenv("AIVD_CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(analyze_router)

    @app.on_event("startup")
    def startup_event() -> None:
        api_state: APIState = app.state.api_state
        config_path = os.getenv("AIVD_CONFIG_PATH", "./backend/config/ensemble.yaml")

        settings = None
        settings_error = None
        try:
            settings = load_settings(config_path=config_path, validate_paths=True)
        except ConfigValidationError as exc:
            settings_error = str(exc)
            logger.error("Settings validation failed: %s", settings_error)
            try:
                settings = load_settings(config_path=config_path, validate_paths=False)
            except Exception:
                settings = None

        if settings is None:
            api_state.ready = False
            api_state.ready_error = settings_error or "Failed to load settings."
            return

        api_state.settings = settings
        api_state.temp_dir = Path(settings.temp_dir).resolve()
        api_state.temp_dir.mkdir(parents=True, exist_ok=True)

        if settings_error is not None:
            api_state.ready = False
            api_state.ready_error = settings_error
            return

        try:
            from backend.services.detector_service import DetectorService

            api_state.detector_service = DetectorService(settings=settings)
            api_state.ready = True
            api_state.ready_error = None
        except Exception as exc:
            api_state.ready = False
            api_state.ready_error = str(exc)
            logger.exception("Detector service failed to initialize.")

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        api_state: APIState = app.state.api_state
        service = api_state.detector_service
        if service is not None:
            try:
                service.cleanup()
            except Exception:
                logger.exception("Failed to cleanup detector service during shutdown.")

    return app


app = create_app()
