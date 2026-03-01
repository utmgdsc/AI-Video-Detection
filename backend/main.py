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
import pandas as pd
import re

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
                True
                if audio_result["score"] > 0.5
                else False
            )
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
            results["individual_prediction"]["efficientnet_prediction"] = (
                True
                if video_result["individual_scores"]["efficientnet_score"] > 0.5
                else False
            )
            results["individual_prediction"]["mesonet_prediction"] = (
                True
                if video_result["individual_scores"]["mesonet_score"] > 0.5
                else False
            )
            results["individual_prediction"]["xceptionnet_prediction"] = (
                True
                if video_result["individual_scores"]["xceptionnet_score"] > 0.5
                else False
            )
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
        # will do this once Mesonet + Xception are integrated
        return (audio_score + video_score) / 2


def get_video_ground_truth(df, full_video_path):

    # 1. Extract ONLY the filename (e.g., 'FvFa_00001_0_id06269_wavtolip.mp4')
    filename = os.path.basename(full_video_path)

    # 2. Strip custom prefixes like 'FvFa_', 'RvFa_', etc., if they exist
    # This turns 'FvFa_00004_fake.mp4' back into '00004_fake.mp4'
    clean_filename = re.sub(r"^(FvFa|FvRa|RvFa|RvRa)_", "", filename)

    # 3. Search for this clean filename in the CSV's 'path' column
    matches = df[df["path"] == clean_filename]

    if matches.empty:
        return {"error": f"Video {clean_filename} not found in metadata"}

    # Note: If the filename is '00109.mp4', matches might contain 5 rows
    # (because 5 different people have a real video named 00109.mp4).
    # Since they are ALL real videos, it's perfectly safe to just grab the first one.
    video_type = matches.iloc[0]["type"]

    return ("FakeVideo" in video_type, "FakeAudio" in video_type)


def print_output(result, video_idx):
    logger.info(f"==================================")
    logger.info(f"Video {video_idx}")
    logger.info(f"Is Fake: {result['is_fake']}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info(f"Audio Score: {result['audio_score']}")
    logger.info(f"Video Score: {result['video_score']}")
    logger.info(f"Details: {result['details']}")
    logger.info(f"==================================")

def print_accuracy(models_correct_prediction, total_videos):
    logger.info(f"==================================")
    logger.info(f"EfficientNet accuracy: {models_correct_prediction['efficientnet_correct_prediction']/total_videos}")
    logger.info(f"MesoNet accuracy: {models_correct_prediction['mesonet_correct_prediction']/total_videos}")
    logger.info(f"XeceptionNet accuracy: {models_correct_prediction['xeceptionnet_correct_prediction']/total_videos}")
    logger.info(f"AAsist accuracy: {models_correct_prediction['aasist_correct_prediction']/total_videos}")
    logger.info(f"Ensemble accuracy: {models_correct_prediction['ensemble_correct_prediction']/total_videos}")
    logger.info(f"==================================")
    
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
        input_path = args.input_dir
    else:
        input_path = cfg["datasets"]["FakeAVCeleb"]["example_video_set_path"]

    detector = DeepfakeDetector(config=cfg, device=device)
    if os.path.isfile(input_path):
        result = detector.analyze(
            input_path, mtcnn, cfg["batch_size"], cfg["frame_skip"]
        )
        print_output(result, 0)
    elif os.path.isdir(input_path):
        # load fakeavceleb metadata
        df = pd.read_csv(cfg["datasets"]["FakeAVCeleb"]["metadata"])

        # rename missing column
        df = df.rename(columns={"Unnamed: 9": "folder_path"})

        models_correct_prediction = {
            "efficientnet_correct_prediction": 0,
            "mesonet_correct_prediction": 0,
            "xeceptionnet_correct_prediction": 0,
            "aasist_correct_prediction": 0,
            "ensemble_correct_prediction": 0,
        }
        for idx, video_path in enumerate(os.listdir(input_path)):
            full_path = os.path.join(input_path, video_path)
            result = detector.analyze(
                full_path, mtcnn, cfg["batch_size"], cfg["frame_skip"]
            )
            # print result for one video
            print_output(result, idx)
            if cfg["compare_baseline_accuracy"]:
                ground_truth_video, ground_truth_audio = get_video_ground_truth(
                    df, full_path
                )
                if result["individual_prediction"]["efficientnet_prediction"] == ground_truth_video:
                    models_correct_prediction["efficientnet_correct_prediction"] += 1
                if result["individual_prediction"]["mesonet_prediction"] == ground_truth_video:
                    models_correct_prediction["mesonet_correct_prediction"] += 1
                if result["individual_prediction"]["xceptionnet_prediction"] == ground_truth_video:
                    models_correct_prediction["xeceptionnet_correct_prediction"] += 1
                if result["individual_prediction"]["aasist_prediction"] == ground_truth_audio:
                    models_correct_prediction["aasist_correct_prediction"] += 1
                if result["is_fake"] != ground_truth_audio and result["is_fake"] != ground_truth_video:
                    models_correct_prediction["ensemble_correct_prediction"] += 1
                                    
        print_accuracy(models_correct_prediction, len(os.listdir(input_path)))
if __name__ == "__main__":
    main()
