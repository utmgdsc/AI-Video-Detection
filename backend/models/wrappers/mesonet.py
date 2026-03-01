"""
MesoNet model for deepfake detection.

Team member: Frank Bi
Docs: docs/models/mesonet/
"""
import sys
import os

project_root = os.path.dirname(__file__)
backend_path = os.path.abspath(os.path.join(project_root, ".."))

if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from backend.models.MesoNet.mesonet_interface import MesoNetClient as MesoNetModel
# Reference your docs/models/mesonet/02-source-and-setup.md for setup instructions


def load_model(weights_path=None):
    """
    Load MesoNet model.

    Args:
        weights_path: Path to pretrained weights (optional).
                      If None, loads default pretrained weights.

    Returns:
        model: Loaded PyTorch model ready for inference.
    """
    model = MesoNetModel()
    model.load_model(weights_path)
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


def process(faces, model_cfg):
    """
    Analyze faces for deepfake detection.

    Args:
        faces: List of face images (cropped from video frames)

    Returns:
        dict: {
            'score': float (0-1, higher = more likely fake),
            'per_frame_scores': list of floats,
            'details': str
        }
    """
    # TODO: Since MesoNetFacialAnalyzer calls self.model.process instead of mesonet.process(), this function never gets called
    raise NotImplementedError("This function is not expected to be called.")
    results = model.process(faces)
    # TODO: Stop server?
    return results
