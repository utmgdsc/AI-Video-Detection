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
from facenet_pytorch import MTCNN
import torch
import argparse

from backend.handlers.audio_handler import AudioHandler
from backend.handlers.video_handler import VideoHandler
from backend.preprocessing.video_processor import separate_audio


class DeepfakeDetector:
    """Main orchestrator for deepfake detection pipeline."""

    def __init__(self, config=None):
        """
        Initialize detector with handlers.

        Args:
            config: Optional configuration dict with paths to weights
        """
        self.config = config or {}
        self.audio_handler = AudioHandler(
            weights_path=self.config.get("aasist_weights")
        )
        self.video_handler = VideoHandler()

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
                video_path, mtcnn, batch_size, sample_rate=frame_skip
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


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m backend.main <video_path>")
        sys.exit(1)

    # video_path = sys.argv[1]

    parser = argparse.ArgumentParser(
        description="Extract faces from videos and images using MTCNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing videos/images",
    )
    # parser.add_argument(
    #     "--output-dir",
    #     type=str,
    #     required=True,
    #     help="Output directory for extracted faces",
    # )

    # Processing options
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "image"],
        default="video",
        help="Processing mode",
    )
    parser.add_argument(
        "--batch-size", type=int, default=60, help="Batch size for processing"
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=30,
        help="Process every Nth frame (video only)",
    )

    # MTCNN parameters
    parser.add_argument("--stride", type=int, default=1, help="Detection stride")
    parser.add_argument(
        "--margin", type=int, default=50, help="Margin around detected face"
    )
    parser.add_argument(
        "--min-face-size", type=int, default=100, help="Minimum face size to detect"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    args = parser.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Initialize MTCNN
    print("Initializing MTCNN...")
    mtcnn = MTCNN(
        margin=args.margin,
        min_face_size=args.min_face_size,
        device=device,
        keep_all=True,
    )

    video_path = args.input_dir
    detector = DeepfakeDetector()
    result = detector.analyze(video_path, mtcnn, args.batch_size, args.frame_skip)

    print(f"Is Fake: {result['is_fake']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Audio Score: {result['audio_score']}")
    print(f"Video Score: {result['video_score']}")
    print(f"Details: {result['details']}")


if __name__ == "__main__":
    main()
