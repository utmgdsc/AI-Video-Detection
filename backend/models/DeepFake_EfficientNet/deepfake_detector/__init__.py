"""
DeepFake Detection using EfficientNet
A robust, production-ready deepfake detection framework.
"""

__version__ = "2.0.0"
__author__ = "Umit Kacar"
__license__ = "MIT"

from deepfake_detector.models import DeepFakeDetector
from deepfake_detector.data import DeepFakeDataset
from deepfake_detector.utils import setup_logger, calculate_metrics

__all__ = [
    "DeepFakeDetector",
    "DeepFakeDataset",
    "setup_logger",
    "calculate_metrics",
]
