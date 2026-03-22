"""
DeepFake Detection using EfficientNet
A robust, production-ready deepfake detection framework.
"""

__version__ = "2.0.0"
__author__ = "Umit Kacar"
__license__ = "MIT"

from deepfake_detector.models import DeepFakeDetector
# Optional modules are not always present in this vendored copy.
# Keep package importable for inference-only runtime.

__all__ = [
    "DeepFakeDetector",
]
