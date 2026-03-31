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
import pickle
import numpy as np
import pandas as pd

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

        # -------------------------
        # LOAD STACKING MODEL
        # -------------------------
        self.stacking_model = None
        fusion_method = self.config.get("ensemble_method", "mean")
        stacking_model_path = self.config.get("stacking_model_path")

        if fusion_method == "stacking":
            if stacking_model_path and os.path.exists(stacking_model_path):
                try:
                    with open(stacking_model_path, "rb") as f:
                        self.stacking_model = pickle.load(f)
                    logger.info(f"Loaded stacking model from {stacking_model_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to load stacking model: {e}. Falling back to mean fusion."
                    )
            else:
                logger.warning(
                    "Stacking selected but stacking model path is missing or invalid. "
                    "Falling back to mean fusion."
                )


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


        fusion_method = self.config.get("ensemble_method", "mean")

        if fusion_method == "stacking":
            confidence = self._stacking_fusion(results["individual_scores"])

        elif fusion_method == "weighted_average":
            confidence = self._weighted_average_fusion(results["individual_scores"])

        elif fusion_method == "majority_voting":
            confidence = self._majority_voting_fusion(results["individual_scores"])

        elif fusion_method == "weighted_voting":
            confidence = self._weighted_voting_fusion(results["individual_scores"])

        else:
            confidence = self._mean_fusion(results["individual_scores"].values())

        results["confidence"] = confidence

        if confidence is not None:
            results["is_real"] = confidence < 0.5

        return results

    def _mean_fusion(self, scores):

        valid = [s for s in scores if s is not None]

        if not valid:
            return None

        return sum(valid) / len(valid)
    
    def _weighted_average_fusion(self, individual_scores):
        """
        Weighted average of per-model fake probabilities.
        Higher score = more likely fake.
        """
        weights = self.config.get("ensemble_weights", {})
        

        valid_scores = []
        valid_weights = []

        for model_name, score in individual_scores.items():
            if score is None:
                continue

            base_name = model_name.replace("_score", "")
            weight = float(weights.get(model_name, weights.get(base_name, 1.0)))
            valid_scores.append(float(score))
            valid_weights.append(weight)

        if not valid_scores:
            return None

        total_weight = sum(valid_weights)
        if total_weight == 0:
            return None

        return sum(s * w for s, w in zip(valid_scores, valid_weights)) / total_weight


    def _majority_voting_fusion(self, individual_scores):
        """
        Unweighted majority vote across per-model predictions.
        Returns a pseudo-probability in [0,1]:
        fraction of models voting fake.
        """

        if not individual_scores:
            return None

        votes = []

        for model_name, score in individual_scores.items():
            if score is None:
                continue

            threshold = self._get_model_threshold(model_name)
            vote_fake = 1 if float(score) >= threshold else 0
            votes.append(vote_fake)

        if not votes:
            return None

        return sum(votes) / len(votes)


    def _weighted_voting_fusion(self, individual_scores):
        """
        Weighted majority vote across per-model predictions.
        Returns weighted fraction voting fake in [0,1].
        """

        weights = self.config.get("ensemble_weights", {})

        weighted_fake_votes = 0.0
        total_weight = 0.0

        for model_name, score in individual_scores.items():
            if score is None:
                continue

            threshold = self._get_model_threshold(model_name)
            vote_fake = 1 if float(score) >= threshold else 0
            base_name = model_name.replace("_score", "")
            weight = float(weights.get(model_name, weights.get(base_name, 1.0)))

            weighted_fake_votes += vote_fake * weight
            total_weight += weight

        if total_weight == 0:
            return None

        return weighted_fake_votes / total_weight


    def _get_model_threshold(self, model_name):
        """
        AASIST uses audio threshold.
        All video models use video threshold.
        """

        if model_name == "aasist_score":
            return float(self.config.get("audio_decision_threshold", 0.5))

        return float(self.config.get("video_decision_threshold", 0.5))

    def _stacking_fusion(self, individual_scores):
        """
        Perform stacking using a trained meta-model.

        Expected feature order:
        [aasist_score, efficientnet_score, mesonet_score, xceptionnet_score]
        """

        if self.stacking_model is None:
            logger.warning("Stacking model not loaded. Using mean fusion fallback.")
            return self._mean_fusion(list(individual_scores.values()))

        feature_order = [
            "aasist_score",
            "efficientnet_score",
            "mesonet_score",
            "xceptionnet_score",
        ]

        features = []
        for key in feature_order:
            score = individual_scores.get(key)
            if score is None:
                score = 0.5  # neutral fallback if one model failed
            features.append(float(score))

        X = pd.DataFrame([features], columns=feature_order)

        try:
            logger.info(f"Stacking features: {features}")

            if hasattr(self.stacking_model, "predict_proba"):
                prob_fake = self.stacking_model.predict_proba(X)[0][1]
                logger.info(f"Stacking output (prob_fake): {prob_fake}")
                return float(prob_fake)

            pred = self.stacking_model.predict(X)[0]
            logger.info(f"Stacking raw output: {pred}")
            return float(pred)

        except Exception as e:
            logger.warning(f"Stacking inference failed: {e}. Using mean fusion fallback.")
            return self._mean_fusion(list(individual_scores.values()))


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

def get_ground_truth_label(video_name):
    """
    FakeAVCeleb flattened filename prefixes:
    RvRa = RealVideo-RealAudio  -> real
    RvFa = RealVideo-FakeAudio  -> fake
    FvRa = FakeVideo-RealAudio  -> fake
    FvFa = FakeVideo-FakeAudio  -> fake
    """
    if video_name.startswith("RvRa_"):
        return True

    if video_name.startswith(("RvFa_", "FvRa_", "FvFa_")):
        return False

    return None

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
    # INPUT PATH (Both file and Directory)
    # -------------------------

    if args.input_dir:
        input_path = args.input_dir
    else:
        input_path = cfg["datasets"]["FakeAVCeleb"]["example_video_set_path"]

    if not os.path.exists(input_path):
        logger.error(f"Invalid input path: {input_path}")
        return

    if os.path.isfile(input_path):
        videos = [input_path]
    elif os.path.isdir(input_path):
        videos = [
            os.path.join(input_path, video)
            for video in sorted(os.listdir(input_path))
            if os.path.isfile(os.path.join(input_path, video))
        ]
    else:
        logger.error(f"Input path is neither a file nor a directory: {input_path}")
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

    for idx, full_path in enumerate(videos):

        video = os.path.basename(full_path)

        ground_truth = get_ground_truth_label(video)

        if ground_truth is None:
            logger.warning(
                f"Skipping {video}: could not determine ground truth from filename prefix."
            )
            continue

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
            f"Efficientnet accuracy: {efficientnet_correct/efficientnet_total:.4f} ({efficientnet_correct}/{efficientnet_total})"
        )

    if mesonet_total:
        logger.info(
            f"Mesonet accuracy: {mesonet_correct/mesonet_total:.4f}({mesonet_correct}/{mesonet_total})"
        )

    if xceptionnet_total:
        logger.info(
            f"Xceptionnet accuracy: {xceptionnet_correct/xceptionnet_total:.4f}({xceptionnet_correct}/{xceptionnet_total})"
        )

    if aasist_total:
        logger.info(
            f"Aasist accuracy: {aasist_correct/aasist_total:.4f}({aasist_correct}/{aasist_total})"
        )

    if ensemble_total:
        logger.info(
            f"Ensemble accuracy: {ensemble_correct/ensemble_total:.4f}({ensemble_correct}/{ensemble_total})"
        )

    logger.info("==================================")
    detector.video_handler.cleanup()


if __name__ == "__main__":
    main()