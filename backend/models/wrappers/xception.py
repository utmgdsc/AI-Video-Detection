"""
XceptionNet model for deepfake detection.

Team member: Lin Wei/Marco Lin
Docs: docs/models/xception/
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

path = (Path(__file__).resolve().parents[3]
              / "backend/models/XceptionNet-Detector/Deepfake-Detection")
sys.path.append(str(path))
from network.models import model_selection
from detect_from_video import predict_with_model

# TODO: Implement model loading
# Reference your docs/models/xception/02-source-and-setup.md for setup instructions


def load_model(weights_path=None, cuda=True):
    """
    Load XceptionNet model.

    Args:
        weights_path: Path to pretrained weights (optional).
                      If None, loads default pretrained weights.

    Returns:
        model: Loaded PyTorch model ready for inference.
    """
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if cuda:
        model = model.cuda()
    model.eval()
    return model


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
