"""
Video Handler - routes video to appropriate analyzers.

Based on content type, routes to:
- Facial Analyzer (for face-based deepfakes)
- Image Analyzer (for general image manipulation)
"""

import sys
from backend.handlers.facial_analyzer import FacialAnalyzer
from backend.handlers.image_analyzer import ImageAnalyzer
from backend.preprocessing import video_processor


class VideoHandler:
    def __init__(self):
        """Initialize video handler with analyzers."""
        self.facial_analyzer = FacialAnalyzer()
        self.image_analyzer = ImageAnalyzer()

    def process(self, video_path, mtcnn, batch_size, sample_rate):
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
        print("BYE BYE!!!")

        # 3. If faces found, run facial analyzer
        if faces:
            facial_score = self.facial_analyzer.process(faces)

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
