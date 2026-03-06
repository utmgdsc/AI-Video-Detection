
from albumentations import (
    Compose,
    Resize,
    Normalize,
)
from albumentations.pytorch import ToTensorV2
# this is extracted from efficientnet repo train.py
def get_val_transforms(image_size: int = 224) -> Compose:
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
