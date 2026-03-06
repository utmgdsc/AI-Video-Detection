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
        self.mesonet_facial_analyzer = MesoNetFacialAnalyzer(model_name="MesoNet", weights_path="weights/Meso4_DF.h5") # TODO: Change this to yaml? 
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
            
            efficientnet_facial_score = self.efficientnet_facial_analyzer.process(
                faces, models_cfg["efficientnet_b1"]
            )
            xceptionnet_facial_score = self.xceptionnet_facial_analyzer.process(
                faces, models_cfg["xceptionnet"]
            )
            mesonet_facial_score = self.mesonet_facial_analyzer.process(faces, models_cfg["mesonet"])
            individual_scores = {
                "efficientnet_score": efficientnet_facial_score['score'],
                "mesonet_score": mesonet_facial_score['score'],
                "xceptionnet_score": xceptionnet_facial_score['score'],
            }
            
            # logger.info(f"facial_score: {facial_score['score']}")

            # 4. Run image analyzer on frames
            # TO-DOs: implement general AI video detection
            # image_score = self.image_analyzer.process(frames)

            # 5. Combine scores
            combined_score = self._combine_scores(
                efficientnet_facial_score["score"],
                mesonet_facial_score["score"],
                xceptionnet_facial_score["score"],
            )
            combined_score_dict = {
                "facial_score": combined_score,
                "image_score": 0,
                "combined_score": combined_score,
                "individual_scores": individual_scores,
                "details": "This is the dictionary for all scores",
            }
        else:
            # no face detected
            combined_score_dict = {
                "facial_score": 0,
                "image_score": 0,
                "combined_score": 0,
                "individual_scores": {
                    "efficientnet_score": 0,
                    "mesonet_score": 0,
                    "xceptionnet_score": 0,
                },
                "details": "This is the dictionary for all scores",
            }

        return combined_score_dict

    def _combine_scores(
        self, efficientnet_facial_score, mesonet_facial_score, xceptionnet_facial_score
    ):
        """Combine scores from different analyzers."""
        # TODO: Define combination strategy
        # Could be: average, weighted average, max, etc.

        return (
            efficientnet_facial_score * (1 / 3)
            + mesonet_facial_score * (1 / 3)
            + xceptionnet_facial_score * (1 / 3)
        )
    
    def __exit__(self, exc_type, exc, tb):
        # On exit, stop MesoNet environment
        self.mesonet_facial_analyzer.cleanup()
