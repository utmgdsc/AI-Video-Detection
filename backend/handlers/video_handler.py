"""
Video Handler - routes video to appropriate analyzers.

Based on content type, routes to:
- Facial Analyzer (for face-based deepfakes)
- Image Analyzer (for general image manipulation)
"""

import sys
import random
import torch
from backend.handlers.facial_analyzer import EfficientNetFacialAnalyzer, MesoNetFacialAnalyzer, XceptionNetFacialAnalyzer
from backend.handlers.image_analyzer import ImageAnalyzer
from backend.preprocessing import video_processor
# from backend.handlers.utils.efficient_net_val_transform import transform_faces
import logging

logger = logging.getLogger(__name__)
class VideoHandler:

    def __init__(self, device):
        """Initialize video handler with analyzers."""
        self.xceptionnet_facial_analyzer = XceptionNetFacialAnalyzer(
            model_name="XceptionNet", device=device
        )
        self.efficientnet_facial_analyzer = EfficientNetFacialAnalyzer(
            model_name="EfficientNet", device=device
        )
        self.mesonet_facial_analyzer = MesoNetFacialAnalyzer(model_name="MesoNet")
        self.image_analyzer = ImageAnalyzer()

    def process(self, models_cfg, device, video_path, mtcnn, batch_size, sample_rate):
        """
        Process video file and return deepfake scores.

        Args:
            video_path: Path to video file

        Returns:
            dict: {
                'facial_score': float or None,
                'image_score': float or None,
                'combined_score': float,
                "individual_scores" : dict,
                'details': str
            }
        """
        # 1. Extract frames from video
        frames = video_processor.extract_frames(video_path, sample_rate)
        details = []
        if frames != []:
            logger.info(f"DEBUG: {len(frames)} frames extracted")
            
        # 2. Detect faces in frames
        faces = video_processor.detect_faces(frames, mtcnn, batch_size)

        individual_scores = {
            "efficientnet_score": None,
            "mesonet_score": None,
            "xceptionnet_score": None,
        }
        
        # 3. If faces found, run facial analyzer
        if faces is not None and len(faces) > 0:
            logger.info(f"{len(faces)} faces detected")
            logger.info("Start processing faces")

            def _run_model(model_key, analyzer, cfg):
                if cfg is None:
                    details.append(f"{model_key} config missing")
                    return None
                if cfg.get("active", True) is False:
                    details.append(f"{model_key} inactive")
                    return None
                try:
                    out = analyzer.process(faces, cfg)
                    if not isinstance(out, dict):
                        details.append(f"{model_key} output invalid")
                        return None
                    score = out.get("score")
                    if score is None:
                        details.append(f"{model_key} score missing")
                        return None
                    return float(score)
                except Exception as e:
                    details.append(f"{model_key} failed: {e}")
                    return None

            individual_scores["efficientnet_score"] = _run_model(
                "efficientnet", self.efficientnet_facial_analyzer, models_cfg.get("efficientnet_b1")
            )
            individual_scores["xceptionnet_score"] = _run_model(
                "xceptionnet", self.xceptionnet_facial_analyzer, models_cfg.get("xceptionnet")
            )
            individual_scores["mesonet_score"] = _run_model(
                "mesonet", self.mesonet_facial_analyzer, models_cfg.get("mesonet")
            )

            combined_score = self._combine_scores(list(individual_scores.values()))
            combined_score_dict = {
                "facial_score": combined_score,
                "image_score": None,
                "combined_score": combined_score,
                "individual_scores": individual_scores,
                "details": " | ".join(details) if details else "",
            }
        else:
            # no face detected
            combined_score_dict = {
                "facial_score": None,
                "image_score": None,
                "combined_score": None,
                "individual_scores": individual_scores,
                "details": "No faces detected",
            }

        return combined_score_dict

    def _combine_scores(self, scores):
        """Combine scores from different analyzers."""
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            return None
        return sum(valid_scores) / len(valid_scores)
    
    def cleanup(self):
        self.mesonet_facial_analyzer.cleanup()
        
    def __exit__(self, exc_type, exc, tb):
        # On exit, stop MesoNet environment
        self.cleanup()
