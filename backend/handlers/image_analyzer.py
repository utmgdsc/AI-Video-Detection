"""
Image Analyzer - detects general image manipulation.

Uses EfficientNet for image-level deepfake/manipulation detection.
"""

from backend.models.wrappers import efficientnet
from backend.preprocessing import image_processor


class ImageAnalyzer:
    def __init__(self, weights_path=None):
        """Initialize image analyzer with EfficientNet model."""
        self.model = None
        self.weights_path = weights_path

    def load(self):
        """Load the EfficientNet model."""
        self.model = efficientnet.load_model(self.weights_path)

    def process(self, frames):
        """
        Analyze frames for image manipulation.

        Args:
            frames: List of video frames

        Returns:
            dict: {
                'score': float (0-1, higher = more likely fake),
                'per_frame_scores': list of floats,
                'details': str
            }
        """
        if self.model is None:
            self.load()

        # TODO: Implement
        # 1. Preprocess each frame
        # 2. Run inference on each frame
        # 3. Aggregate scores across frames

        raise NotImplementedError("Implement process()")
