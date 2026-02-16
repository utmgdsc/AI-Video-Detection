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
from backend.models.DeepFake_EfficientNet.deepfake_detector.models.efficientnet import (
    DeepFakeDetector as EfficientNetModel,
)

# Add the full repo to path (update path if repo name is different)
REPO_PATH = os.path.join(os.path.dirname(__file__), "../DeepFake_EfficientNet")
if os.path.exists(REPO_PATH):
    sys.path.insert(0, REPO_PATH)
else:
    print(f"Warning: DeepFake-EfficientNet repo not found at {REPO_PATH}")
    print("Add the full repo to backend/models/ first (see backend/models/README.md)")


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
