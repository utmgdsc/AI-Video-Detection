"""
Image preprocessing utilities.

Prepares images/frames for deepfake detection models.
"""

import cv2
import numpy as np
import sys
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
path = (Path(__file__).resolve().parents[3]
              / "backend/models/XceptionNet-Detector/Deepfake-Detection")
sys.path.append(str(path))
from detect_from_video import preprocess_image
from dataset.transform import xception_default_data_transforms

class MyDataset(Dataset):
    def __init__(self, faces):
        self.faces = faces
        self.transform = xception_default_data_transforms['test']

    def __getitem__(self, index):
        face = self.faces[index]
        if isinstance(face, torch.Tensor):
            if face.ndim == 2:
                face = face.unsqueeze(0).repeat(3, 1, 1)  # (3,H,W)
            elif face.ndim == 3 and face.shape[0] == 1:
                face = face.repeat(3, 1, 1)

            face = transforms.ToPILImage()(face.cpu())
        return self.transform(face)

    def __len__(self):
        return len(self.faces)


def preprocess_for_xception(image, target_size=(299, 299)):
    """
    Preprocess image for XceptionNet.

    Args:
        image: Input image (numpy array, BGR)
        target_size: Target dimensions

    Returns:
        tensor: Preprocessed image tensor
    """
    # TODO: Implement based on XceptionNet requirements
    # 1. Resize
    # 2. Convert BGR to RGB
    # 3. Normalize (ImageNet mean/std)
    # 4. Convert to tensor
    face = image.detach().cpu().permute(1, 2, 0).numpy()
    face_np = (face * 255).clip(0, 255).astype("uint8") 
    face = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    return preprocess_image(face)


def preprocess_for_efficientnet(image, target_size=(224, 224)):
    """
    Preprocess image for EfficientNet.

    Args:
        image: Input image (numpy array, BGR)
        target_size: Target dimensions

    Returns:
        tensor: Preprocessed image tensor
    """
    # TODO: Implement based on EfficientNet requirements

    raise NotImplementedError("Implement preprocess_for_efficientnet()")


def preprocess_for_mesonet(image, target_size=(256, 256)):
    """
    Preprocess image for MesoNet.

    Args:
        image: Input image (numpy array, BGR)
        target_size: Target dimensions

    Returns:
        tensor: Preprocessed image tensor
    """
    # TODO: Implement based on MesoNet requirements

    raise NotImplementedError("Implement preprocess_for_mesonet()")
