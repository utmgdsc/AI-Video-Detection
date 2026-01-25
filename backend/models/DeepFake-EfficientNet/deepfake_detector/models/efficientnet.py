"""
EfficientNet-based DeepFake Detection Model
Provides a robust and efficient architecture for binary classification.
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DeepFakeDetector(nn.Module):
    """
    EfficientNet-based deepfake detector.

    This model uses a pre-trained EfficientNet backbone with a custom
    classification head for binary deepfake detection.

    Args:
        model_name: EfficientNet variant (e.g., 'efficientnet-b0', 'efficientnet-b1')
        num_classes: Number of output classes (default: 2 for binary classification)
        dropout_rate: Dropout probability in classification head (default: 0.5)
        pretrained: Whether to load ImageNet pre-trained weights (default: True)

    Example:
        >>> model = DeepFakeDetector('efficientnet-b1')
        >>> output = model(torch.randn(4, 3, 224, 224))
        >>> print(output.shape)  # torch.Size([4, 2])
    """

    def __init__(
        self,
        model_name: str = 'efficientnet-b1',
        num_classes: int = 2,
        dropout_rate: float = 0.5,
        pretrained: bool = True
    ):
        super(DeepFakeDetector, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pre-trained EfficientNet
        if pretrained:
            logger.info(f"Loading pre-trained {model_name}")
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            logger.info(f"Initializing {model_name} from scratch")
            self.backbone = EfficientNet.from_name(model_name)

        # Get number of features from backbone
        num_ftrs = self.backbone._fc.in_features

        # Custom classification head
        self.backbone._fc = nn.Sequential(
            nn.Linear(num_ftrs, 1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(1000, num_classes, bias=True)
        )

        logger.info(f"Model initialized with {num_ftrs} features -> {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width)

        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        return self.backbone(x)

    def get_image_size(self) -> int:
        """Get the expected input image size for this model variant."""
        return EfficientNet.get_image_size(self.model_name)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze the backbone parameters.

        Args:
            freeze: If True, freeze backbone; if False, unfreeze
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

        # Always keep classification head trainable
        for param in self.backbone._fc.parameters():
            param.requires_grad = True

        status = "frozen" if freeze else "unfrozen"
        logger.info(f"Backbone {status}, classification head trainable")

    def load_checkpoint(self, checkpoint_path: str, device: str = 'cpu') -> None:
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load the model on
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            self.load_state_dict(checkpoint)
            logger.info("Loaded model weights")

    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: Optional[int] = None,
        optimizer_state: Optional[dict] = None,
        metrics: Optional[dict] = None
    ) -> None:
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch number
            optimizer_state: Optimizer state dict
            metrics: Dictionary of metrics to save
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
        }

        if epoch is not None:
            checkpoint['epoch'] = epoch
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def __repr__(self) -> str:
        total, trainable = self.count_parameters()
        return (
            f"DeepFakeDetector(\n"
            f"  model={self.model_name},\n"
            f"  num_classes={self.num_classes},\n"
            f"  total_params={total:,},\n"
            f"  trainable_params={trainable:,}\n"
            f")"
        )
