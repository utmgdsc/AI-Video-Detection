"""
CLI entrypoint for AI Video Detection.

This module keeps command-line behavior while delegating detector
orchestration to the service layer for reuse by APIs/workers.
"""

import argparse
import logging
import os
import re

import pandas as pd

from backend.core.config import load_settings
from backend.services.detector_service import DetectorService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("result.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


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
    logger.info("==================================")
    logger.info(f"Video {video_idx}")
    logger.info(f"Is Real: {result['is_real']}")
    logger.info(f"Confidence: {result['confidence']:.2%}")
    logger.info(f"Audio Score: {result['audio_score']}")
    logger.info(f"Video Score: {result['video_score']}")
    logger.info(f"Details: {result['details']}")
    logger.info("==================================")


def print_accuracy(models_correct_prediction, total_videos):
    logger.info("==================================")
    logger.info(
        f"EfficientNet accuracy: {models_correct_prediction['efficientnet_correct_prediction']/total_videos}"
    )
    logger.info(
        f"MesoNet accuracy: {models_correct_prediction['mesonet_correct_prediction']/total_videos}"
    )
    logger.info(
        f"XeceptionNet accuracy: {models_correct_prediction['xeceptionnet_correct_prediction']/total_videos}"
    )
    logger.info(
        f"AAsist accuracy: {models_correct_prediction['aasist_correct_prediction']/total_videos}"
    )
    logger.info(
        f"Ensemble accuracy: {models_correct_prediction['ensemble_correct_prediction']/total_videos}"
    )
    logger.info("==================================")


def main():
    parser = argparse.ArgumentParser(
        description="Extract faces from videos and images using MTCNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
    settings = load_settings(config_path, validate_paths=True)

    if args.input_dir:
        input_path = args.input_dir
    else:
        input_path = settings.datasets["faceforensic"]["example_video_set_path"]

    service = DetectorService(settings=settings)

    try:
        if os.path.isfile(input_path):
            result = service.analyze_video(input_path)
            print_output(result, 0)
        elif os.path.isdir(input_path):
            df = None
            if settings.compare_baseline_accuracy:
                metadata_path = settings.datasets.get("FakeAVCeleb", {}).get("metadata")
                if not metadata_path:
                    raise ValueError(
                        "compare_baseline_accuracy=true but datasets.FakeAVCeleb.metadata is not set."
                    )
                df = pd.read_csv(metadata_path)
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
                result = service.analyze_video(full_path)

                print_output(result, idx)
                if settings.compare_baseline_accuracy:
                    ground_truth_video, ground_truth_audio = get_video_ground_truth(df, full_path)
                    if ground_truth_audio is None or ground_truth_video is None:
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
                    logger.info(
                        f"Ground Truth Video: {ground_truth_video} | Ground Truth Audio: {ground_truth_audio} | Overall Truth: {overall_truth}"
                    )
                    logger.info(
                        f"File: {video_path} | GT Video: {ground_truth_video} | GT Audio: {ground_truth_audio} | Overall Truth: {overall_truth}"
                    )

                    if result["is_real"] == overall_truth:
                        models_correct_prediction["ensemble_correct_prediction"] += 1

            print_accuracy(models_correct_prediction, len(os.listdir(input_path)))
        else:
            raise ValueError(f"Input path does not exist: {input_path}")
    finally:
        service.cleanup()


if __name__ == "__main__":
    main()
