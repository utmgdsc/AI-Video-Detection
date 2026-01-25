# EfficientNet — Source & Setup

## Source location(s)

- GitHub repo link: https://github.com/umitkacar/multimodal-deepfake-detector
- Paper link: https://arxiv.org/abs/1905.11946
- Any pretrained weights link: 
- License notes (if known):

## What I verified

- Does the repo match the paper description? Partially
- Pretrained model available? Yes(Yes/No)
- Dataset expected by repo: FaceForensic++ Dataset (C23), AIGVDBench

## Environment / dependencies

List exactly what you installed (with versions if possible):

- OS: Linux
- Python: Python 3.12
- PyTorch: 2.5.1
- CUDA (if used):
- OpenCV: 4.13.0.90
- Other deps:
Checkout requirements.txt
## Setup steps (copy/paste friendly)

### 1) Clone / install

```Ubuntu
    1. git clone https://github.com/umitkacar/DeepFake-EfficientNet.git
    2. cd DeepFake-EfficientNet
    3. python -m venv venv
    4. source venv/bin/activate 
    5. pip install -r requirements.txt
```

### 2) Any downloads needed (weights/datasets)

- What to download:
Assuming you want to train
Download kaggle FaceForensics++ dataset

- Where to place it:
Example paths (adjust to your setup):
Move real videos to: <repo_root>/temp/real/
Move fake videos to: <repo_root>/temp/fake/

### 3) How to run

0.    Make following changes to the repo
# (venv) wu@Wu-Hung-Mao:~/CSC492/DeepFake-EfficientNet$ git diff cdddceb77758b04610dbfa7bdab9dee51d1d430d
# diff --git a/deepfake_detector/data/transforms.py b/deepfake_detector/data/transforms.py
# index 136d273..42bd261 100644
# --- a/deepfake_detector/data/transforms.py
# +++ b/deepfake_detector/data/transforms.py
# @@ -7,8 +7,8 @@ from albumentations import (
#      Compose, HorizontalFlip, RandomResizedCrop, Resize,
#      Normalize, VerticalFlip, Rotate, ShiftScaleRotate,
#      OpticalDistortion, GridDistortion, ElasticTransform,
# -    JpegCompression, HueSaturationValue, RGBShift,
# -    RandomBrightness, RandomContrast, Blur, MotionBlur,
# +    ImageCompression, HueSaturationValue, RGBShift,
# +    RandomBrightnessContrast, RandomBrightnessContrast, Blur, MotionBlur,
#      MedianBlur, GaussNoise, CLAHE, RandomGamma, CoarseDropout
#  )
#  from albumentations.pytorch import ToTensorV2
# @@ -63,7 +63,7 @@ def get_train_transforms(
#              ),
 
#              # Compression and noise
# -            JpegCompression(quality_lower=75, quality_upper=100, p=0.3),
# +            ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
#              GaussNoise(var_limit=(10.0, 50.0), p=0.3),
 
#              # Color augmentations
# @@ -74,8 +74,8 @@ def get_train_transforms(
#                  p=0.3
#              ),
#              RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
# -            RandomBrightness(limit=0.2, p=0.3),
# -            RandomContrast(limit=0.2, p=0.3),
# +            RandomBrightnessContrast(limit=0.2, p=0.3),
# +            RandomBrightnessContrast(limit=0.2, p=0.3),
#              RandomGamma(gamma_limit=(80, 120), p=0.3),
#              CLAHE(p=0.2),
 
# @@ -106,7 +106,7 @@ def get_train_transforms(
#          return Compose([
#              HorizontalFlip(p=0.5),
#              RandomResizedCrop(
# -                image_size, image_size,
# +                size=(image_size, image_size),  # One single tuple
#                  scale=(0.5, 1.0),
#                  p=0.5
#              ),
# @@ -177,7 +177,7 @@ def get_test_time_augmentation_transforms(image_size: int = 224) -> list:
 
#          # Slight brightness adjustment
#          Compose([
# -            RandomBrightness(limit=0.1, p=1.0),
# +            RandomBrightnessContrast(limit=0.1, p=1.0),
#              Resize(image_size, image_size, always_apply=True),
#              Normalize(
#                  mean=[0.485, 0.456, 0.406],
# @@ -189,7 +189,7 @@ def get_test_time_augmentation_transforms(image_size: int = 224) -> list:
 
#          # Slight contrast adjustment
#          Compose([
# -            RandomContrast(limit=0.1, p=1.0),
# +            RandomBrightnessContrast(limit=0.1, p=1.0),
#              Resize(image_size, image_size, always_apply=True),
#              Normalize(
#                  mean=[0.485, 0.456, 0.406],
# diff --git a/deepfake_detector/utils/metrics.py b/deepfake_detector/utils/metrics.py
# index e2cc10a..3ccfacb 100644
# --- a/deepfake_detector/utils/metrics.py
# +++ b/deepfake_detector/utils/metrics.py
# @@ -8,6 +8,7 @@ import math
#  from typing import Tuple, List, Dict
#  from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
#  import logging
# +from typing import Optional, List, Any, Union, Dict
 
#  logger = logging.getLogger(__name__)
 
# diff --git a/scripts/train.py b/scripts/train.py
# index e8ca242..398be6c 100755
# --- a/scripts/train.py
# +++ b/scripts/train.py
# @@ -21,11 +21,12 @@ from sklearn.metrics import accuracy_score, confusion_matrix
 
#  # Import from our package
#  from deepfake_detector.models import DeepFakeDetector
# -from deepfake_detector.data import create_combined_dataset, get_train_transforms, get_val_transforms, create_dataloaders
# -from deepfake_detector.utils import setup_logger, calculate_comprehensive_metrics, plot_confusion_matrix, plot_training_history
# +from deepfake_detector.data import DeepFakeDataset, get_train_transforms, get_val_transforms, create_dataloaders
# +from deepfake_detector.utils import setup_logger, calculate_metrics, plot_confusion_matrix, plot_training_history
#  from deepfake_detector.config import Config, load_config
 
# -from transformers import AdamW, get_cosine_schedule_with_warmup
# +from torch.optim import AdamW
# +from transformers import get_cosine_schedule_with_warmup
 
 
#  def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch, logger):
# @@ -165,8 +166,8 @@ def main():
#      val_real_config = [(path, -1) for path in args.val_real]
#      val_fake_config = [(path, -1) for path in args.val_fake]
 
# -    train_dataset = create_combined_dataset(train_real_config, train_fake_config, train_transforms)
# -    val_dataset = create_combined_dataset(val_real_config, val_fake_config, val_transforms)
# +    train_dataset = DeepFakeDataset(train_real_config, train_fake_config, train_transforms)
# +    val_dataset = DeepFakeDataset(val_real_config, val_fake_config, val_transforms)
 
#      logger.info(f"Train dataset: {len(train_dataset)} samples")
#      logger.info(f"Val dataset: {len(val_dataset)} samples")

1.  python3 scripts/extract_faces.py \
    --input-dir temp/real \
    --output-dir ff_extracted/train/real \
    --mode video --batch-size 60 --frame-skip 30

2.  python3 scripts/extract_faces.py \
    --input-dir temp/real \
    --output-dir ff_extracted/val/real \
    --mode video --batch-size 60 --frame-skip 30

3.  Run above two steps for fake(replace real with fake in above commands)

4.  python3 scripts/train.py \
    --train-real ff_extracted/train/real \
    --train-fake ff_extracted/train/fake \
    --val-real ff_extracted/val/real \
    --val-fake ff_extracted/val/fake \
    --output-dir outputs \
    --batch-size 16 \
    --epochs 5 \
    --lr 8e-4 \
    --model efficientnet-b1

5.  python3 scripts/inference.py \
    --input ff_extracted/train/real/001-1769132497.906613.jpg \
    --checkpoint outputs/checkpoints/best_model.pth \
    --model efficientnet-b1

## Output / results

- What output you saw (logs, metrics, saved files):
Train Loss: 0.1285, Train Acc: 1.0000
Val Loss: 0.1510, Val Acc: 1.0000
loss=0.0002
Best model saved with val_acc: 1.0000
- Add screenshots in assets/ if helpful.

## Issues encountered + fixes

- Issue: Needed to make some changes 
- Cause: Some packages are updated
- Fix (steps/commands):
See above how to run session

## Notes for teammates

- If someone else sets this up from scratch, what do they need to know?
If you have any issue, ask me on discord.
