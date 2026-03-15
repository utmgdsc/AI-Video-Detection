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
import argparse
import yaml
import torch
from facenet_pytorch import MTCNN

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

        self.config = config or {}

        aasist_cfg = self.config.get("models", {}).get("aasist", {})

        self.audio_handler = AudioHandler(
            weights_path=aasist_cfg.get(
                "weights_path", self.config.get("aasist_weights")
            )
        )

        self.video_handler = VideoHandler(device)

    def analyze(
        self,
        video_path,
        mtcnn,
        batch_size,
        frame_skip,
        audio_decision_threshold,
        video_decision_threshold,
    ):

        results = {
            "audio_score": None,
            "video_score": None,
            "confidence": None,
            "is_real": None,
            "individual_scores": {},
            "details": "",
        }

        # -------------------------
        # AUDIO ANALYSIS
        # -------------------------

        audio_path = None

        try:

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                audio_path = f.name

            separate_audio(video_path, audio_path)

            audio_result = self.audio_handler.process(audio_path)

            results["audio_score"] = float(audio_result["score"])
            results["individual_scores"]["aasist_score"] = results["audio_score"]

        except Exception as e:

            results["details"] += f"Audio analysis skipped: {e}\n"

        finally:

            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

        # -------------------------
        # VIDEO ANALYSIS
        # -------------------------

        try:

            video_result = self.video_handler.process(
                self.config["models"],
                self.config["device"],
                video_path,
                mtcnn,
                batch_size,
                sample_rate=frame_skip,
            )

            results["video_score"] = video_result.get("combined_score")

            individual_scores = video_result.get("individual_scores", {})

            for key in ["efficientnet_score", "mesonet_score", "xceptionnet_score"]:

                score = individual_scores.get(key)

                if score is not None:
                    results["individual_scores"][key] = float(score)

        except Exception as e:

            results["details"] += f"Video analysis failed: {e}\n"

        # -------------------------
        # ENSEMBLE
        # -------------------------

        confidence = self._mean_fusion(
            [results["audio_score"], results["video_score"]]
        )

        results["confidence"] = confidence

        if confidence is not None:
            results["is_real"] = confidence < 0.5

        return results

    def _mean_fusion(self, scores):

        valid = [s for s in scores if s is not None]

        if not valid:
            return None

        return sum(valid) / len(valid)


def print_output(result, video_idx):

    confidence = result["confidence"]
    confidence_str = "N/A" if confidence is None else f"{confidence:.2%}"

    logger.info("==================================")
    logger.info(f"Video {video_idx}")
    logger.info(f"Is Real: {result['is_real']}")
    logger.info(f"Confidence: {confidence_str}")
    logger.info(f"Audio Score: {result['audio_score']}")
    logger.info(f"Video Score: {result['video_score']}")
    logger.info(f"Individual Scores: {result.get('individual_scores', {})}")
    logger.info(f"Details: {result['details']}")
    logger.info("==================================")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="./backend/config/ensemble.yaml",
    )

    args = parser.parse_args()

    # -------------------------
    # LOAD CONFIG
    # -------------------------

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    device = cfg["device"]

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cfg["device"] = device

    # -------------------------
    # INIT MTCNN
    # -------------------------

    mtcnn_cfg = cfg["mtcnn"]

    mtcnn = MTCNN(
        margin=mtcnn_cfg["margin"],
        min_face_size=mtcnn_cfg["min_face_size"],
        thresholds=mtcnn_cfg["thresholds"],
        factor=mtcnn_cfg["factor"],
        post_process=mtcnn_cfg["post_process"],
        select_largest=mtcnn_cfg["select_largest"],
        keep_all=mtcnn_cfg["keep_all"],
        device=device,
    )

    # -------------------------
    # INPUT PATH
    # -------------------------

    if args.input_dir:
        input_path = args.input_dir
    else:
        input_path = cfg["datasets"]["FakeAVCeleb"]["example_video_set_path"]

    if not os.path.isdir(input_path):

        logger.error(f"Invalid input path: {input_path}")
        return

    detector = DeepfakeDetector(config=cfg, device=device)

    # -------------------------
    # ACCURACY COUNTERS
    # -------------------------

    efficientnet_correct = 0
    efficientnet_total = 0

    mesonet_correct = 0
    mesonet_total = 0

    xceptionnet_correct = 0
    xceptionnet_total = 0

    aasist_correct = 0
    aasist_total = 0

    ensemble_correct = 0
    ensemble_total = 0

    # -------------------------
    # PROCESS DATASET
    # -------------------------

    videos = sorted(os.listdir(input_path))

    for idx, video in enumerate(videos):

        full_path = os.path.join(input_path, video)

        if not os.path.isfile(full_path):
            continue

        ground_truth = "real" in video.lower()

        try:

            result = detector.analyze(
                full_path,
                mtcnn,
                cfg["batch_size"],
                cfg["frame_skip"],
                cfg.get("audio_decision_threshold", 0.5),
                cfg.get("video_decision_threshold", 0.5),
            )

        except Exception as e:

            logger.error(f"Failed on video {video}: {e}")
            continue

        print_output(result, idx)

        scores = result.get("individual_scores", {})

        if "efficientnet_score" in scores:
            efficientnet_total += 1
            pred = scores["efficientnet_score"] < 0.5
            if pred == ground_truth:
                efficientnet_correct += 1

        if "mesonet_score" in scores:
            mesonet_total += 1
            pred = scores["mesonet_score"] < 0.5
            if pred == ground_truth:
                mesonet_correct += 1

        if "xceptionnet_score" in scores:
            xceptionnet_total += 1
            pred = scores["xceptionnet_score"] < 0.5
            if pred == ground_truth:
                xceptionnet_correct += 1

        if "aasist_score" in scores:
            aasist_total += 1
            pred = scores["aasist_score"] < 0.5
            if pred == ground_truth:
                aasist_correct += 1

        if result["is_real"] is not None:
            ensemble_total += 1
            if result["is_real"] == ground_truth:
                ensemble_correct += 1

    # -------------------------
    # FINAL RESULTS
    # -------------------------

    logger.info("==================================")

    if efficientnet_total:
        logger.info(
            f"Efficientnet accuracy: {efficientnet_correct/efficientnet_total:.4f}"
        )

    if mesonet_total:
        logger.info(
            f"Mesonet accuracy: {mesonet_correct/mesonet_total:.4f}"
        )

    if xceptionnet_total:
        logger.info(
            f"Xceptionnet accuracy: {xceptionnet_correct/xceptionnet_total:.4f}"
        )

    if aasist_total:
        logger.info(
            f"Aasist accuracy: {aasist_correct/aasist_total:.4f}"
        )

    if ensemble_total:
        logger.info(
            f"Ensemble accuracy: {ensemble_correct/ensemble_total:.4f}"
        )

    logger.info("==================================")


if __name__ == "__main__":
    main()