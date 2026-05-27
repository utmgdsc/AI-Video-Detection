"""
Facial Analyzer - detects face-based deepfakes.

Uses XceptionNet and/or MesoNet for facial deepfake detection.
"""

import torch
import numpy as np

# from scipy.special import softmax
import torch.nn.functional as F
from backend.models.wrappers import xception, mesonet, efficientnet
from backend.preprocessing import image_processor
from backend.models.wrappers.xception import _XCEPTION_AVAILABLE
if _XCEPTION_AVAILABLE:
    from backend.models.wrappers.xception import predict_with_model
else:
    predict_with_model = None
import logging
import numpy as np
import cv2
import traceback # Make sure to import this at the top of your file!
from albumentations import (
    Compose,
    Resize,
    Normalize,
)
from albumentations.pytorch import ToTensorV2

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

    def load_model(self, weights_path, device, model_cfg=None):
        """Load the selected model."""
        if self.model_name == "XceptionNet":
            cuda = str(device).startswith("cuda")
            self.model = xception.load_model(weights_path, cuda=cuda)
        elif self.model_name == "MesoNet":
            self.model = mesonet.load_model(model_cfg)
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

    @staticmethod
    def get_dabn_settings(model_cfg, default_enabled=False):
        dabn_cfg = (model_cfg or {}).get("dabn", {})
        enabled = dabn_cfg.get("enabled", default_enabled)
        momentum = dabn_cfg.get("momentum")
        return bool(enabled), momentum


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

    # this is extracted from efficientnet repo train.py
    def get_val_transforms(self, image_size: int = 224) -> Compose:
        """
        Get validation/test data preprocessing pipeline.

        Args:
            image_size: Target image size

        Returns:
            Albumentations Compose object with validation transforms

        Example:
            >>> transforms = get_val_transforms(224)
            >>> preprocessed = transforms(image=image)
        """
        return Compose(
            [
                Resize(image_size, image_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def transform_faces(self, faces):
        # EfficientNet-B1 expects 240x240
        eff_transforms = self.get_val_transforms(image_size=240) 
        eff_tensor_faces = []
        
        for face in faces:
            # Apply Albumentations transform (must extract "image" key)
            transformed_face = eff_transforms(image=face)["image"]
            eff_tensor_faces.append(transformed_face)
        # Stack list of tensors into a single batch (Shape: [Batch_Size, Channels, Height, Width])
        faces = torch.stack(eff_tensor_faces)
        
        return faces

    def process(self, faces, model_cfg):
        """
        Analyze faces for deepfake detection.
        Put 15 faces in a batch and process them simultaneously
        Args:
            faces_tensor: A properly formatted PyTorch tensor batch of faces.
        """
        if self.model is None:
            logger.info("loading model")
            self.load_model(model_cfg["weights_path"], self.device)
        
        faces_tensor = self.transform_faces(faces)
        face_pred_result = []
        
        # Move the batch to the GPU/CPU
        faces_tensor = faces_tensor.to(self.device)

        dabn_enabled, dabn_momentum = self.get_dabn_settings(
            model_cfg, default_enabled=True
        )
        efficientnet.set_bn_adaptive_mode(
            self.model, enabled=dabn_enabled, momentum=dabn_momentum
        )

        try:
            with torch.no_grad():
                # Evaluate the ENTIRE batch at once! No loops needed.
                outputs = self.model(faces_tensor)
                
                # DEBUG: Let's see exactly what shape the model is spitting out
                logger.info(f"DEBUG: Model outputs shape: {outputs.shape}")
                
                # Convert raw logits to probabilities (percentages)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()

            # Process the results
            for idx, prob in enumerate(probs):
                real_prob = prob[0] 
                fake_prob = prob[1] 
                
                prediction = "real" if real_prob >= model_cfg["threshold"] else "fake"

                face_pred_result.append(
                    {
                        "face idx": idx,
                        "prediction": prediction,
                        "real_prob": float(real_prob),
                        "fake_prob": float(fake_prob),
                        "confidence": float(max(fake_prob, real_prob)),
                    }
                )
        except Exception as e:
            # BUG FIX: Print the exact line number and code that caused the crash!
            logger.error(f"Failed to process face batch:\n{traceback.format_exc()}")
            return {}

        if len(face_pred_result) == 0:
            return {}

        avg_fake_score = sum(r["fake_prob"] for r in face_pred_result) / len(face_pred_result)
        summary = {
            "score": float(avg_fake_score),
            "per_frame_score": [float(x["confidence"]) for x in face_pred_result],
            "details": "This contain efficientnet result",
        }

        return summary

class XceptionNetFacialAnalyzer(FacialAnalyzer):

    def __init__(self, model_name, device, weights_path=None):
        """
        Initialize facial analyzer.

        Args:
            model_name: 'xception' or 'mesonet' or 'efficientnet'
            weights_path: Path to pretrained weights
        """
        super().__init__(model_name, weights_path)
        self.device = device


    def _to_bgr_numpy(self, face):
        face_np = face.detach().cpu().permute(1, 2, 0).numpy()
        face_np = (face_np * 255).clip(0, 255).astype("uint8") 
        return cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)

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
        if predict_with_model is None:
            raise RuntimeError("XceptionNet is unavailable: repo files missing.")
        if self.model is None:
            self.load_model(model_cfg["weights_path"], self.device)
        summary = {
            "score": 0.0,
            "per_frame_score": [],
            "details": "No face detected",
        }
        if len(faces) == 0:
            return summary

        dabn_enabled, dabn_momentum = self.get_dabn_settings(
            model_cfg, default_enabled=False
        )
        xception.set_bn_adaptive_mode(
            self.model, enabled=dabn_enabled, momentum=dabn_momentum
        )
        
        use_cuda = str(self.device).startswith("cuda")
        scores = []
        for face in faces: 
            # FIX: Face is already a NumPy array, just convert RGB to BGR
            # prediction, output = predict_with_model(self._to_bgr_numpy(face), self.model, cuda=use_cuda)
            face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            prediction, output = predict_with_model(face_bgr, self.model, cuda=use_cuda)

            scores.append(float(output.detach().cpu().numpy()[0][1])) # append fake score
        
        summary = {
            "score": sum(scores)/len(scores), #fake_count / (fake_count + real_count),
            "per_frame_score": scores,
            "details": "This contain xceptionnet result",
        }
        print("xceptionnet video score:"+ str(summary["score"]))
        return summary
class MesoNetFacialAnalyzer(FacialAnalyzer):

    def process(self, faces, model_cfg):
        # Extract config variables
        image_size = 256
        threshold = 0.5
        score_method = None
        if "image_size" in model_cfg:
            image_size = model_cfg["image_size"]
        if "threshold" in model_cfg:
            threshold = model_cfg["threshold"]
        if "scoring_method" in model_cfg:
            score_method = model_cfg["scoring_method"]

        # If no model is loaded, initialize one
        if self.model is None:
            self.load_model(None, None, model_cfg)
            if self.model is None:
                raise RuntimeError("MesoNet Model failed to load.")

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

        # stop_server is false so that we may continue using the same server for multiple videos
        results = self.model.process(images, stop_server=False)
        if results == None:
            self.cleanup()
            raise RuntimeError("Failed to process MesoNet results.")
        
        results = np.array(results)
        # MesoNet classes 1 as real and 0 as fake, so we flip the score
        results = 1.0 - results
        score = self.scoring_method(results, threshold, score_method)
        
        summary = {
            "score": score,
            "per_frame_score": results.tolist(),
            "details": "This contains MesoNet results.",
        }
        print("mesonet:"+ str(summary["score"]))
        return summary
    
    def scoring_method(self, results, threshold, option):
        match option:
            case "mean score":
                return np.mean(results)
            case "median score":
                return np.median(results)
            case _:
                return np.mean(results > threshold)

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
