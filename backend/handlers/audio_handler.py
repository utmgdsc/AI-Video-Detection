"""
Audio Handler - processes audio track from video.

Uses AASIST model for audio deepfake detection.
"""

from backend.models.wrappers import aasist
from backend.preprocessing import audio_processor


class AudioHandler:
    def __init__(self, weights_path=None):
        """Initialize audio handler with AASIST model."""
        self.model = None
        self.weights_path = weights_path

    def load(self):
        """Load the AASIST model."""
        self.model = aasist.load_model(self.weights_path)

    def process(self, audio_path):
        """
        Process audio file and return deepfake score.

        Args:
            audio_path: Path to audio file (extracted from video)

        Returns:
            dict: {
                'score': float (0-1, higher = more likely fake),
                'confidence': float,
                'details': str
            }
        """
        if self.model is None:
            self.load()

        # TODO: Implement
        # 1. Preprocess audio
        audio = audio_processor.preprocess(audio_path)
        # 2. Run inference
        score = aasist.predict(self.model, audio)
        # 3. Return result
        return {
        "score": score,
        "confidence": score,  # for now same as score
        "details": f"AASIST spoof probability: {score:.6f}",
        }
