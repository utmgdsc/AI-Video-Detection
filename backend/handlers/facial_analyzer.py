"""
Facial Analyzer - detects face-based deepfakes.

Uses XceptionNet and/or MesoNet for facial deepfake detection.
"""

import torch

# from scipy.special import softmax
import torch.nn.functional as F
from backend.models.wrappers import xception, mesonet, efficientnet
from backend.preprocessing import image_processor
from backend.models.DeepFake_EfficientNet.deepfake_detector.data import (
    get_val_transforms,
)

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class FacialAnalyzer:
    def __init__(self, model_name, weights_path=None):
        """
        Initialize facial analyzer.

        Args:
            model_name: 'xception' or 'mesonet' or 'efficientnet'
            weights_path: Path to pretrained weights
        """
        self.model_name = model_name
        self.weights_path = weights_path
        self.model = None

    def load_model(self, weights_path, device):
        """Load the selected model."""
        if self.model_name == "XceptionNet":
            self.model = xception.load_model(weights_path)
        elif self.model_name == "MesoNet":
            self.model = mesonet.load_model(weights_path)
        elif self.model_name == "EfficientNet":
            self.model = efficientnet.load_model(
                weights_path=weights_path, device=device
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def process(self, faces, model_cfg):
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
        raise NotImplementedError("Implement process()")


class EfficientNetFacialAnalyzer(FacialAnalyzer):

    def __init__(self, model_name, device, weights_path=None):
        """
        Initialize facial analyzer.

        Args:
            model_name: 'xception' or 'mesonet' or 'efficientnet'
            weights_path: Path to pretrained weights
        """
        super().__init__(model_name, weights_path)
        self.device = device

    def predict_single(self, model, image_tensor, device):
        """Run prediction on a single image."""
        model.eval()

        with torch.no_grad():
            # tensor is 4D (Batch, C, H, W)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)

        return probs.cpu().numpy()[0]

    def process(self, faces, model_cfg):
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
        if self.model is None:
            self.load_model(model_cfg["weights_path"], self.device)

        # Prepare transform
        image_size = model_cfg["image_size"]
        transform = get_val_transforms(image_size)
        face_pred_result = []
        logger.info("Start processing faces")
        for idx, face in enumerate(faces):
            try:
                if torch.is_tensor(face):
                    # Move channels from front to back: (3, 160, 160) -> (160, 160, 3)
                    face = face.permute(1, 2, 0).cpu().numpy()
                transformed = transform(image=face)
                image_tensor = transformed["image"]
                probs = self.predict_single(
                    self.model, image_tensor, self.device)

                fake_prob = probs[0]
                real_prob = probs[1]
                prediction = "real" if real_prob >= model_cfg["threshold"] else "fake"

                face_pred_result.append(
                    {
                        "face idx": idx,
                        "prediction": prediction,
                        "real_prob": real_prob,
                        "fake_prob": fake_prob,
                        "confidence": max(fake_prob, real_prob),
                    }
                )
            except Exception as e:
                logger.warning(f"Cannot process face {idx}: {e}")
                continue

        if len(face_pred_result) == 0:
            return {}

        real_count = sum(
            1 for r in face_pred_result if r["prediction"] == "real")
        fake_count = sum(
            1 for r in face_pred_result if r["prediction"] == "fake")
        summary = {
            "score": fake_count / (fake_count + real_count),
            "per_frame_score": [x["confidence"] for x in face_pred_result],
            "details": "This contain efficientnet result",
        }

        return summary


class MesoNetFacialAnalyzer(FacialAnalyzer):

    def process(self, faces, model_cfg):
        image_size = model_cfg["image_size"]

        # If no model is loaded, initialize one
        if self.model is None:
            weights_path = None
            if "weights_path" in model_cfg:
                weights_path = model_cfg["weights_path"]
            self.load_model(weights_path, None)

        if self.model is None:
            summary = {}

        processed_faces = []

        for face in faces:
            if torch.is_tensor(face):
                # Convert from Torch Tensor GPU to CPU, then from Tensor to Numpy for Keras
                face = face.cpu().numpy()

            # If (3, 256, 256), we convert to (256, 256, 3)
            if face.ndim == 3 and face.shape[-1] != 3:
                face = np.transpose(face, (1, 2, 0))

            # Resize to expected size
            face = cv2.resize(face, (image_size, image_size))

            # Convert to float32 for Keras
            face = face.astype(np.float32)

            # Normalize to [0,1] if needed
            if face.max() > 1.0:
                face = face / 255.0

            processed_faces.append(face)

        images = np.stack(processed_faces, axis=0)  # (N, 256, 256, 3)

        # We can set stop_server=False to keep the model active after the ensemble has terminated
        results = self.model.process(images, stop_server=False)
        results = 1 - results

        # TODO: Integrate mesonet results
        summary = {
            "score": None,
            "per_frame_score": None,
            "details": "MesoNet results not implemented.",
        }
        return summary

    def cleanup(self):
        """
        Stops running the model when it is no longer needed.
        """
        if self.model is not None:
            self.model.cleanup()
        self.model = None

    def __exit__(self, exc_type, exc, tb):
        if self.model is not None:
            self.model.cleanup()
