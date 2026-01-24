"""
AASIST model for audio deepfake detection.

Team member: [YOUR NAME]
Docs: docs/models/aasist/
"""

import torch
import torch.nn as nn

# TODO: Implement model loading
# Reference your docs/models/aasist/02-source-and-setup.md for setup instructions


def load_model(weights_path=None):
    """
    Load AASIST model.

    Args:
        weights_path: Path to pretrained weights (optional).
                      If None, loads default pretrained weights.

    Returns:
        model: Loaded PyTorch model ready for inference.
    """
    # TODO: Implement
    raise NotImplementedError("Implement load_model() - see docs/models/aasist/")


def predict(model, audio):
    """
    Run inference on audio sample.

    Args:
        model: Loaded model from load_model()
        audio: Preprocessed audio tensor

    Returns:
        score: Float between 0-1 (0=real, 1=fake)
    """
    # TODO: Implement
    raise NotImplementedError("Implement predict()")
