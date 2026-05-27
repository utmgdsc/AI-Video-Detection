"""
XceptionNet model for deepfake detection.

Contributor: Lin Wei/Marco Lin
Docs: docs/models/xception/
"""

import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

path = (Path(__file__).resolve().parents[3]
              / "backend/models/XceptionNet-Detector/Deepfake-Detection")
sys.path.append(str(path))
try:
    from network.models import model_selection
    from detect_from_video import predict_with_model
    _XCEPTION_AVAILABLE = True
except ImportError as _e:
    logger.warning("XceptionNet dependencies not found (%s) — model will be skipped.", _e)
    _XCEPTION_AVAILABLE = False

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
    if not _XCEPTION_AVAILABLE:
        raise RuntimeError("XceptionNet is not available: repo files missing.")
    model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    if weights_path:
        raw = torch.load(weights_path, map_location=torch.device('cpu'))
        if isinstance(raw, dict):
            # Checkpoint dict — look for common state_dict keys
            state_dict = raw.get("state_dict", raw.get("model", raw))
        else:
            # Full model object saved with torch.save(model, path)
            state_dict = raw.state_dict()
        model.load_state_dict(state_dict)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if cuda and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def set_bn_adaptive_mode(model, enabled=True, momentum=None):
    """
    Configure Domain Adaptive Batch Normalization (DABN)-style behavior.
    Keeps non-BN layers in eval mode while optionally adapting BN running stats.
    """
    model.eval()

    if not enabled:
        return model

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()
            if momentum is not None:
                module.momentum = momentum

    logger.info("DABN mode enabled for Xception BN layers")
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
