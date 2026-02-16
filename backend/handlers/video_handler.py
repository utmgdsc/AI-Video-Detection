"""
Video Handler - routes video to appropriate analyzers.

Based on content type, routes to:
- Facial Analyzer (for face-based deepfakes)
- Image Analyzer (for general image manipulation)
"""

import sys
from backend.handlers.facial_analyzer import EfficientNetFacialAnalyzer
from backend.handlers.image_analyzer import ImageAnalyzer
from backend.preprocessing import video_processor


class VideoHandler:
    def __init__(self):
        """Initialize video handler with analyzers."""
        # change to XceptionNet facial analyzer
        self.xceptionnet_facial_analyzer = FacialAnalyzer(model_name="XceptionNet")
        self.efficientnet_facial_analyzer = EfficientNetFacialAnalyzer(
            model_name="EfficientNet"
        )
        # change to mesonet facial analyzer
        self.mesonet_facial_analyzer = FacialAnalyzer(model_name="MesoNet")
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
                'details': str
            }
        """
        # TODO: Implement
        # 1. Extract frames from video
        frames = video_processor.extract_frames(video_path, sample_rate)
        print("FRAMES EXTRACTED!!!")
        # 2. Detect faces in frames
        faces = video_processor.detect_faces(frames, mtcnn, batch_size)
        if faces:
            print("FACES DETECTED!!!")

        # 3. If faces found, run facial analyzer
        if faces:
            facial_score = self.efficientnet_facial_analyzer.process(
                faces, models_cfg["efficientnet_b1"], device=device
            )
            print("facial_score is", facial_score)
        sys.exit()
        # 4. Run image analyzer on frames
        image_score = self.image_analyzer.process(frames)

        # 5. Combine scores
        combined = self._combine_scores(facial_score, image_score)

        raise NotImplementedError("Implement process()")

    def _combine_scores(self, facial_score, image_score):
        """Combine scores from different analyzers."""
        # TODO: Define combination strategy
        # Could be: average, weighted average, max, etc.
        raise NotImplementedError("Implement _combine_scores()")
