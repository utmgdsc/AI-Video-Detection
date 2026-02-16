#!/usr/bin/env python3
"""
Inference Script for DeepFake Detection
Run inference on single images or directories.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import cv2
import numpy as np
from scipy.special import softmax
import glob
from tqdm import tqdm

from deepfake_detector.models import DeepFakeDetector
from deepfake_detector.data import get_val_transforms
from deepfake_detector.utils import setup_logger


def load_image(image_path, transform):
    """Load and preprocess an image."""
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Apply transforms
    transformed = transform(image=image_rgb)
    image_tensor = transformed["image"].unsqueeze(0)

    return image_tensor


def predict_single(model, image_tensor, device):
    """Run prediction on a single image."""
    model.eval()

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probs = softmax(output.cpu().numpy(), axis=1)[0]

    return probs


def main():
    parser = argparse.ArgumentParser(
        description="DeepFake Detection Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Input image or directory"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model", type=str, default="efficientnet-b1", help="Model architecture"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Classification threshold"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for batch inference results",
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(name="inference", level="INFO")

    logger.info("=" * 60)
    logger.info("DeepFake Detection Inference")
    logger.info("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    model = DeepFakeDetector(model_name=args.model, pretrained=False)
    model.load_checkpoint(args.checkpoint, device=str(device))
    model = model.to(device)
    model.eval()

    # Prepare transform
    image_size = 240 if "b1" in args.model else 224
    transform = get_val_transforms(image_size)

    # Check if input is file or directory
    input_path = Path(args.input)

    if input_path.is_file():
        # Single image inference
        logger.info(f"Running inference on: {input_path}")

        image_tensor = load_image(str(input_path), transform)
        probs = predict_single(model, image_tensor, device)

        fake_prob = probs[0]
        real_prob = probs[1]
        prediction = "REAL" if real_prob >= args.threshold else "FAKE"

        print("\n" + "=" * 60)
        print(f"Image: {input_path.name}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max(fake_prob, real_prob):.2%}")
        print(f"Real probability: {real_prob:.4f}")
        print(f"Fake probability: {fake_prob:.4f}")
        print("=" * 60 + "\n")

    elif input_path.is_dir():
        # Batch inference on directory
        logger.info(f"Running batch inference on directory: {input_path}")

        # Get all images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(str(input_path / ext)))

        if not image_files:
            logger.error(f"No images found in {input_path}")
            return

        logger.info(f"Found {len(image_files)} images")

        results = []

        for image_path in tqdm(image_files, desc="Processing"):
            try:
                image_tensor = load_image(image_path, transform)
                probs = predict_single(model, image_tensor, device)

                fake_prob = probs[0]
                real_prob = probs[1]
                prediction = "REAL" if real_prob >= args.threshold else "FAKE"

                results.append(
                    {
                        "image": Path(image_path).name,
                        "prediction": prediction,
                        "real_prob": real_prob,
                        "fake_prob": fake_prob,
                        "confidence": max(fake_prob, real_prob),
                    }
                )

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue

        # Print summary
        print("\n" + "=" * 60)
        print("Batch Inference Results")
        print("=" * 60)
        print(f"Total images: {len(results)}")
        real_count = sum(1 for r in results if r["prediction"] == "REAL")
        fake_count = sum(1 for r in results if r["prediction"] == "FAKE")
        print(f"Predicted REAL: {real_count}")
        print(f"Predicted FAKE: {fake_count}")
        print("=" * 60 + "\n")

        # Save results if output specified
        if args.output:
            import pandas as pd

            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            logger.info(f"Results saved to: {args.output}")

    else:
        logger.error(f"Invalid input path: {input_path}")
        return

    logger.info("Inference complete!")


if __name__ == "__main__":
    main()
