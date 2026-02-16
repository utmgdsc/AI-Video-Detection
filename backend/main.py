"""
Main orchestrator for AI Video Detection pipeline.

Flow:
1. Receive video/link
2. Separate audio and video
3. Route to handlers (audio, video)
4. Combine scores
5. Return result
"""

import os
import tempfile
import sys
from facenet_pytorch import MTCNN
import torch
import argparse
import yaml

from backend.handlers.audio_handler import AudioHandler
from backend.handlers.video_handler import VideoHandler
from backend.preprocessing.video_processor import separate_audio

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("result.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Main orchestrator for deepfake detection pipeline."""

    def __init__(self, config=None, device="cuda"):
        """
        Initialize detector with handlers.

        Args:
            config: Optional configuration dict with paths to weights
        """
        self.config = config or {}
        self.audio_handler = AudioHandler(
            weights_path=self.config.get("aasist_weights")
        )
        self.video_handler = VideoHandler(device)

    def analyze(self, video_path, mtcnn, batch_size, frame_skip):
        """
        Analyze video for deepfake detection.

        Args:
            video_path: Path to video file

        Returns:
            dict: {
                'is_fake': bool,
                'confidence': float,
                'audio_score': float or None,
                'video_score': float,
                'details': str
            }
        """
        results = {
            "audio_score": None,
            "video_score": None,
            "is_fake": False,
            "confidence": 0.0,
            "details": "",
        }

        # 1. Extract audio (if present)
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_path = f.name
            separate_audio(video_path, audio_path)
            audio_result = self.audio_handler.process(audio_path)
            results["audio_score"] = audio_result["score"]
            os.unlink(audio_path)  # Clean up temp file
        except Exception as e:
            results["details"] += f"Audio analysis skipped: {e}\n"

        # 2. Analyze video
        try:
            video_result = self.video_handler.process(
                self.config["models"],
                self.config["device"],
                video_path,
                mtcnn,
                batch_size,
                sample_rate=frame_skip,
            )
            results["video_score"] = video_result["combined_score"]
        except Exception as e:
            results["details"] += f"Video analysis failed: {e}\n"
            raise

        # 3. Combine scores
        results["confidence"] = self._combine_scores(
            results["audio_score"], results["video_score"]
        )
        results["is_fake"] = results["confidence"] > 0.5

        return results

    def _combine_scores(self, audio_score, video_score):
        """
        Combine audio and video scores into final confidence.

        TODO: Define combination strategy based on experiments.
        """
        if audio_score is None:
            return video_score

        # Simple average for now
        # TODO: Experiment with weighted combinations
        return (audio_score + video_score) / 2


def main():
    parser = argparse.ArgumentParser(
        description="Extract faces from videos and images using MTCNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--input-dir",
        type=str,
        required=False,
        help="Input directory containing videos/images",
    )

    parser.add_argument(
        "--config", type=str, required=False, default="./backend/config/ensemble.yaml"
    )

    args = parser.parse_args()

    config_path = args.config

    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    device = cfg["device"]
    margin = cfg["margin"]
    min_face_size = cfg["min_face_size"]
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Initialize MTCNN
    logger.info("Initializing MTCNN...")
    mtcnn = MTCNN(
        margin=margin,
        min_face_size=min_face_size,
        device=device,
        keep_all=True,
    )

    if args.input_dir:
        video_path = args.input_dir
    else:
        video_path = cfg["datasets"]["faceforensic"]["example_video_path"]

    detector = DeepfakeDetector(config=cfg, device=device)
    result = detector.analyze(video_path, mtcnn, cfg["batch_size"], cfg["frame_skip"])

    logger.info(f"Is Fake: {result['is_fake']}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info(f"Audio Score: {result['audio_score']}")
    logger.info(f"Video Score: {result['video_score']}")
    logger.info(f"Details: {result['details']}")


if __name__ == "__main__":
    main()
