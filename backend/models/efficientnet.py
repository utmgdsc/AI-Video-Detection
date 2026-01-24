"""
EfficientNet model for deepfake detection.

Team member: [YOUR NAME]
Docs: docs/models/efficientnet/
"""

import torch
import torch.nn as nn

# TODO: Implement model loading
# Reference your docs/models/efficientnet/02-source-and-setup.md for setup instructions


def load_model(weights_path=None):
    """
    Load EfficientNet model.

    Args:
        weights_path: Path to pretrained weights (optional).
                      If None, loads default pretrained weights.

    Returns:
        model: Loaded PyTorch model ready for inference.
    """
    # TODO: Implement
    raise NotImplementedError("Implement load_model() - see docs/models/efficientnet/")


def predict(model, image):
    """
    Run inference on a single image/frame.

    Args:
        model: Loaded model from load_model()
        image: Preprocessed image tensor

    Returns:
        score: Float between 0-1 (0=real, 1=fake)
    """
    # TODO: Implement
    raise NotImplementedError("Implement predict()")
