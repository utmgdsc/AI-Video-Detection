"""
EfficientNet model wrapper for deepfake detection.

Team member: Wu Hung Mao/Marvin Wu
Docs: docs/models/efficientnet/

This is a THIN WRAPPER that provides a standard interface to the
DeepFake-EfficientNet repository added by the team member.
"""

import sys
import os
import torch

# 1. FIX PATH FIRST
# Get the path to: backend/models/DeepFake_EfficientNet
project_root = os.path.dirname(__file__)
deepfake_path = os.path.abspath(os.path.join(project_root, "../DeepFake_EfficientNet"))

# Add it to system path so internal imports like "from deepfake_detector..." work
if deepfake_path not in sys.path:
    sys.path.insert(0, deepfake_path)

# 2. THEN IMPORT MODEL
# Now Python can find the internal dependencies
from backend.models.DeepFake_EfficientNet.deepfake_detector.models.efficientnet import (
    DeepFakeDetector as EfficientNetModel,
)


def load_model(weights_path=None, model_name="efficientnet-b1", device="cuda"):
    """
    Load EfficientNet model.

    Args:
        weights_path: Path to pretrained weights.
                      If None, uses default: 'outputs/checkpoints/best_model.pth'
        model_name: Model variant (efficientnet-b0 to efficientnet-b7)

    Returns:
        model: Loaded PyTorch model ready for inference.
    """
    efficient_net_model = EfficientNetModel(model_name=model_name)
    checkpoint = torch.load(weights_path, map_location=device)

    if "model_state_dict" in checkpoint:
        efficient_net_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        efficient_net_model.load_state_dict(checkpoint)

    efficient_net_model = efficient_net_model.to(device)
    efficient_net_model.eval()
    return efficient_net_model


def predict(model, image):
    """
    Run inference on a single image/frame.

    Args:
        model: Loaded model from load_model()
        image: Preprocessed image (numpy array or tensor, depends on model API)

    Returns:
        score: Float between 0-1 (0=real, 1=fake)
    """
    # TODO: Implement based on the DeepFake-EfficientNet repo structure
    # Example (update based on actual repo API):
    # with torch.no_grad():
    #     score = model.predict(image)
    # return score

    raise NotImplementedError(
        "Implement predict() wrapper - see docs/models/efficientnet/02-source-and-setup.md"
    )
