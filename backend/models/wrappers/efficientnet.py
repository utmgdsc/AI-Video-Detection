"""
EfficientNet model wrapper for deepfake detection.

Team member: [YOUR NAME - Update when implementing]
Docs: docs/models/efficientnet/

This is a THIN WRAPPER that provides a standard interface to the
DeepFake-EfficientNet repository added by the team member.
"""

import sys
import os
import torch

# Add the full repo to path (update path if repo name is different)
REPO_PATH = os.path.join(os.path.dirname(__file__), '../DeepFake-EfficientNet')
if os.path.exists(REPO_PATH):
    sys.path.insert(0, REPO_PATH)
else:
    print(f"Warning: DeepFake-EfficientNet repo not found at {REPO_PATH}")
    print("Add the full repo to backend/models/ first (see backend/models/README.md)")


def load_model(weights_path=None, model_name='efficientnet-b1'):
    """
    Load EfficientNet model.

    Args:
        weights_path: Path to pretrained weights.
                      If None, uses default: 'outputs/checkpoints/best_model.pth'
        model_name: Model variant (efficientnet-b0 to efficientnet-b7)

    Returns:
        model: Loaded PyTorch model ready for inference.
    """
    # TODO: Implement based on the DeepFake-EfficientNet repo structure
    # Example (update based on actual repo API):
    # from deepfake_detector.models import DeepFakeDetector
    # if weights_path is None:
    #     weights_path = 'outputs/checkpoints/best_model.pth'
    # model = DeepFakeDetector.from_pretrained(weights_path, model_name=model_name)
    # return model

    raise NotImplementedError(
        "Implement load_model() wrapper - see docs/models/efficientnet/02-source-and-setup.md"
    )


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
