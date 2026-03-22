"""
Service layer for deepfake detection orchestration.

This module contains reusable backend logic that can be called from
CLI entrypoints, APIs, or background workers.
"""

from __future__ import annotations

import logging
import os
import tempfile

import torch
from facenet_pytorch import MTCNN

from backend.core.config import AppSettings, load_settings
from backend.handlers.audio_handler import AudioHandler
from backend.handlers.video_handler import VideoHandler
from backend.preprocessing.video_processor import separate_audio


logger = logging.getLogger(__name__)


class DeepfakeDetector:
    """Core deepfake detector orchestrator."""

    def __init__(self, settings: AppSettings, device: str = "cuda"):
        self.settings = settings
        self.device = device
        aasist_cfg = self.settings.models.get("aasist") or {}
        self.audio_handler = AudioHandler(weights_path=aasist_cfg.get("weights_path"))
        self.video_handler = VideoHandler(device)

    def analyze(self, video_path: str, mtcnn: MTCNN, batch_size: int, frame_skip: int) -> dict[str, object]:
        """Analyze a video for audio/video deepfake signals."""
        results = {
            "audio_score": None,
            "video_score": None,
            "is_real": False,
            "confidence": 0.0,
            "individual_prediction": {},
            "details": "",
        }

        # 1. Extract audio (if present)
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_path = f.name
            separate_audio(video_path, audio_path)
            audio_result = self.audio_handler.process(audio_path)
            results["audio_score"] = audio_result["score"]
            results["individual_prediction"]["aasist_prediction"] = (
                False if audio_result["score"] > 0.5 else True
            )
            os.unlink(audio_path)
        except Exception as e:
            message = str(e)
            if "No audio stream found" in message:
                results["details"] += "Audio analysis skipped: no audio stream in input video.\n"
            else:
                results["details"] += f"Audio analysis skipped: {message}\n"

        # 2. Analyze video
        try:
            video_result = self.video_handler.process(
                self.settings.models,
                self.device,
                video_path,
                mtcnn,
                batch_size,
                sample_rate=frame_skip,
            )
            results["video_score"] = video_result["combined_score"]
            if results["video_score"] is not None:
                efficientnet_score = video_result["individual_scores"].get("efficientnet_score")
                mesonet_score = video_result["individual_scores"].get("mesonet_score")
                xceptionnet_score = video_result["individual_scores"].get("xceptionnet_score")

                results["individual_prediction"]["efficientnet_prediction"] = (
                    None
                    if efficientnet_score is None
                    else (False if efficientnet_score > 0.5 else True)
                )
                results["individual_prediction"]["mesonet_prediction"] = (
                    None
                    if mesonet_score is None
                    else (False if mesonet_score > 0.5 else True)
                )
                results["individual_prediction"]["xceptionnet_prediction"] = (
                    None
                    if xceptionnet_score is None
                    else (False if xceptionnet_score > 0.5 else True)
                )

                # Include raw numeric scores for UI display
                results["individual_scores"] = {
                    "efficientnet_score": float(efficientnet_score) if efficientnet_score is not None else None,
                    "mesonet_score": float(mesonet_score) if mesonet_score is not None else None,
                    "xceptionnet_score": float(xceptionnet_score) if xceptionnet_score is not None else None,
                }
        except Exception as e:
            results["details"] += f"Video analysis failed: {e}\n"
            raise

        # 3. Combine scores
        results["confidence"] = self._combine_scores(results["audio_score"], results["video_score"])
        if results["audio_score"] is not None and results["video_score"] is not None:
            results["audio_is_real"] = bool(results["audio_score"] < 0.5)
            results["video_is_real"] = bool(results["video_score"] < 0.5)
            results["is_real"] = bool(results["audio_is_real"] and results["video_is_real"])
        elif results["audio_score"] is None and results["video_score"] is not None:
            results["audio_is_real"] = None
            results["video_is_real"] = bool(results["video_score"] < 0.5)
            results["is_real"] = bool(results["video_is_real"])
        else:
            results["audio_is_real"] = None
            results["video_is_real"] = None
            results["is_real"] = None

        # Normalize scalar types for API JSON serialization.
        if results["audio_score"] is not None:
            results["audio_score"] = float(results["audio_score"])
        if results["video_score"] is not None:
            results["video_score"] = float(results["video_score"])
        if results["confidence"] is not None:
            results["confidence"] = float(results["confidence"])

        # Expose AASIST score alongside other individual scores
        if "individual_scores" not in results:
            results["individual_scores"] = {}
        results["individual_scores"]["aasist_score"] = results["audio_score"]

        return results

    def _combine_scores(self, audio_score: float | None, video_score: float | None) -> float | None:
        """Combine audio and video scores into one confidence score."""
        if audio_score is None:
            return video_score
        return (audio_score + video_score) / 2


class DetectorService:
    """Reusable service that owns model config, MTCNN, and detector lifecycle."""

    def __init__(self, settings: AppSettings, device: str | None = None):
        self.settings = settings
        self.device = self._resolve_device(device or self.settings.device)

        logger.info("Initializing MTCNN...")
        mtcnn_cfg = self.settings.mtcnn
        self.mtcnn = MTCNN(
            margin=mtcnn_cfg.margin,
            min_face_size=mtcnn_cfg.min_face_size,
            thresholds=mtcnn_cfg.thresholds,
            factor=mtcnn_cfg.factor,
            post_process=mtcnn_cfg.post_process,
            select_largest=mtcnn_cfg.select_largest,
            keep_all=mtcnn_cfg.keep_all,
            device=self.device,
        )
        self.detector = DeepfakeDetector(settings=self.settings, device=self.device)

    @classmethod
    def from_yaml(
        cls, config_path: str, device: str | None = None, validate_paths: bool = True
    ) -> "DetectorService":
        """Build service from YAML path."""
        settings = load_settings(config_path, validate_paths=validate_paths)
        return cls(settings=settings, device=device)

    def analyze_video(
        self, video_path: str, batch_size: int | None = None, frame_skip: int | None = None
    ) -> dict[str, object]:
        """Analyze one video path using service defaults unless overridden."""
        effective_batch_size = (
            batch_size if batch_size is not None else self.settings.batch_size
        )
        effective_frame_skip = (
            frame_skip if frame_skip is not None else self.settings.frame_skip
        )
        return self.detector.analyze(video_path, self.mtcnn, effective_batch_size, effective_frame_skip)

    def cleanup(self) -> None:
        """Release long-lived resources."""
        self.detector.video_handler.cleanup()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            return "cpu"
        return device
