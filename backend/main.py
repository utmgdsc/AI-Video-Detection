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
                'is_real': bool,
                'confidence': float,
                'audio_score': float or None,
                'video_score': float,
                'details': str
            }
        """
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
                False
                if audio_result["score"] > 0.5
                else True
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
            if results["video_score"] is not None:
                results["individual_prediction"]["efficientnet_prediction"] = (
                    False
                    if video_result["individual_scores"]["efficientnet_score"] > 0.5
                    else True
                )
                results["individual_prediction"]["mesonet_prediction"] = (
                    False
                    if video_result["individual_scores"]["mesonet_score"] > 0.5
                    else True
                )
                results["individual_prediction"]["xceptionnet_prediction"] = (
                    False
                    if video_result["individual_scores"]["xceptionnet_score"] > 0.5
                    else True
                )
        except Exception as e:
            results["details"] += f"Video analysis failed: {e}\n"
            raise

        # 3. Combine scores
        results["confidence"] = self._combine_scores(
            results["audio_score"], results["video_score"]
        )
        if results["audio_score"] is not None and results["video_score"] is not None: 
            results["audio_is_real"] = results["audio_score"] < 0.5
            results["video_is_real"] = results["video_score"] < 0.5
            results["is_real"] = results["audio_is_real"] and results["video_is_real"]
        elif results["audio_score"] is None and results["video_score"] is not None:
            results["audio_is_real"] = None;
            results["video_is_real"] = results["video_score"] < 0.5
            results["is_real"] = results["video_is_real"]
        else: 
            results["audio_is_real"] = None
            results["video_is_real"] = None
            results["is_real"] = None
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
    try:
        # 1. Extract ONLY the filename (e.g., 'FvFa_00001_0_id06269_wavtolip.mp4')
        filename = os.path.basename(full_video_path)

        # 2. Strip custom prefixes like 'FvFa_', 'RvFa_', etc., if they exist
        clean_filename = re.sub(r"^(FvFa|FvRa|RvFa|RvRa)_", "", filename)

        # 3. Search for this clean filename in the CSV's 'path' column
        matches = df[df["path"] == clean_filename]

        # Return (None, None) immediately if no match is found
        if matches.empty:
            logger.warning(f"Video {clean_filename} not found in metadata. Skipping.")
            return (None, None)

        # 4. Grab the first one
        video_type = matches.iloc[0]["type"]

        return ("FakeVideo" not in video_type, "FakeAudio" not in video_type)

    except Exception as e:
        # Catch any unexpected errors (like missing columns or pandas issues)
        logger.error(f"Error processing ground truth for {full_video_path}: {e}")
        return (None, None)

def print_output(result, video_idx):
    logger.info(f"==================================")
    logger.info(f"Video {video_idx}")
    logger.info(f"Is Real: {result['is_real']}")
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
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    # Initialize MTCNN
    logger.info("Initializing MTCNN...")
    mtcnn_cfg = cfg["mtcnn"]
    mtcnn = MTCNN(
        margin=mtcnn_cfg["margin"],
        min_face_size=mtcnn_cfg["min_face_size"],
        thresholds=mtcnn_cfg["thresholds"],
        factor=mtcnn_cfg["factor"],
        post_process=mtcnn_cfg["post_process"],
        select_largest=mtcnn_cfg["select_largest"],
        keep_all=mtcnn_cfg["keep_all"],
        device=device
    )

    if args.input_dir:
        input_path = args.input_dir
    else:
        input_path = cfg["datasets"]["faceforensic"]["example_video_set_path"]

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
                if ground_truth_audio == None or ground_truth_video == None:
                    continue
                if result["audio_score"] is None or result["video_score"] is None: 
                    continue
                if result["individual_prediction"]["efficientnet_prediction"] == ground_truth_video:
                    models_correct_prediction["efficientnet_correct_prediction"] += 1
                if result["individual_prediction"]["mesonet_prediction"] == ground_truth_video:
                    models_correct_prediction["mesonet_correct_prediction"] += 1
                if result["individual_prediction"]["xceptionnet_prediction"] == ground_truth_video:
                    models_correct_prediction["xeceptionnet_correct_prediction"] += 1
                if result["individual_prediction"]["aasist_prediction"] == ground_truth_audio:
                    models_correct_prediction["aasist_correct_prediction"] += 1
                overall_truth = ground_truth_audio and ground_truth_video
                # If the model's prediction matches the overall truth, it is correct
                # Print the ground truths
                logger.info(f"Ground Truth Video: {ground_truth_video} | Ground Truth Audio: {ground_truth_audio} | Overall Truth: {overall_truth}")
                # Added the video_path to the print statement so you can verify the file
                logger.info(f"File: {video_path} | GT Video: {ground_truth_video} | GT Audio: {ground_truth_audio} | Overall Truth: {overall_truth}")
                
                if result["is_real"] == overall_truth:
                    models_correct_prediction["ensemble_correct_prediction"] += 1
        
        detector.video_handler.cleanup()
        print_accuracy(models_correct_prediction, len(os.listdir(input_path)))
if __name__ == "__main__":
    main()
