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
        if frames != []:
            logger.info(f"DEBUG: {len(frames)} frames extracted")
            
        # 2. Detect faces in frames
        faces = video_processor.detect_faces(frames, mtcnn, batch_size)
        
        # 3. If faces found, run facial analyzer
        if faces is not None and len(faces) > 0:
            logger.info(f"{len(faces)} faces detected")
            logger.info("Start processing faces")
            details = []

            efficientnet_score = None
            xceptionnet_score = None
            mesonet_score = None

            try:
                efficientnet_facial_score = self.efficientnet_facial_analyzer.process(
                    faces, models_cfg["efficientnet_b1"]
                )
                efficientnet_score = efficientnet_facial_score["score"]
            except Exception as e:
                logger.exception(f"EfficientNet inference failed: {e}")
                details.append(f"EfficientNet failed: {e}")

            try:
                xceptionnet_facial_score = self.xceptionnet_facial_analyzer.process(
                    faces, models_cfg["xceptionnet"]
                )
                xceptionnet_score = xceptionnet_facial_score["score"]
            except Exception as e:
                logger.exception(f"XceptionNet inference failed: {e}")
                details.append(f"XceptionNet failed: {e}")

            try:
                mesonet_facial_score = self.mesonet_facial_analyzer.process(
                    faces, models_cfg["mesonet"]
                )
                mesonet_score = mesonet_facial_score["score"]
            except Exception as e:
                logger.exception(f"MesoNet inference failed: {e}")
                details.append(f"MesoNet failed: {e}")

            individual_scores = {
                "efficientnet_score": efficientnet_score,
                "mesonet_score": mesonet_score,
                "xceptionnet_score": xceptionnet_score,
            }
            
            # logger.info(f"facial_score: {facial_score['score']}")

            # 4. Run image analyzer on frames
            # TO-DOs: implement general AI video detection
            # image_score = self.image_analyzer.process(frames)

            # 5. Combine scores
            combined_score = self._combine_scores(
                efficientnet_score,
                mesonet_score,
                xceptionnet_score,
            )
            combined_score_dict = {
                "facial_score": combined_score,
                "image_score": 0,
                "combined_score": combined_score,
                "individual_scores": individual_scores,
                "details": " | ".join(details) if details else "All facial models succeeded",
            }
        else:
            # no face detected
            combined_score_dict = {
                "facial_score": None,
                "image_score": None,
                "combined_score": None,
                "individual_scores": {
                    "efficientnet_score": None,
                    "mesonet_score": None,
                    "xceptionnet_score": None,
                },
                "details": "This is the dictionary for all scores",
            }

        return combined_score_dict

    def _combine_scores(
        self, efficientnet_facial_score, mesonet_facial_score, xceptionnet_facial_score
    ):
        """Combine scores from different analyzers."""
        available_scores = [
            s
            for s in [
                efficientnet_facial_score,
                mesonet_facial_score,
                xceptionnet_facial_score,
            ]
            if s is not None
        ]
        if not available_scores:
            return None
        return sum(available_scores) / len(available_scores)
    
    def cleanup(self):
        self.mesonet_facial_analyzer.cleanup()
        
    def __exit__(self, exc_type, exc, tb):
        # On exit, stop MesoNet environment
        self.cleanup()
