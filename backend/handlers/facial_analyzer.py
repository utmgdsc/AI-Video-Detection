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


class FacialAnalyzer:
    def __init__(self, model_name, weights_path=None):
        """
        Initialize facial analyzer.

        Args:
            model_name: 'xception' or 'mesonet' or 'efficientnet'
            weights_path: Path to pretrained weights
        """
        self.model_name = model_name
        # maybe we don't need this
        # self.weights_path = weights_path
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
    def __init__(self, model_name, weights_path=None):
        """
        Initialize facial analyzer.

        Args:
            model_name: 'xception' or 'mesonet' or 'efficientnet'
            weights_path: Path to pretrained weights
        """
        self.model_name = model_name
        # maybe we don't need this
        # self.weights_path = weights_path
        self.model = None

    def load_model(self, weights_path, device):
        """Load the selected model."""
        self.model = efficientnet.load_model(weights_path=weights_path, device=device)

    def predict_single(model, image_tensor, device):
        """Run prediction on a single image."""
        model.eval()

        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)

        return probs.cpu().numpy()[0]

    def process(self, faces, model_cfg, device):
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
            self.load_model(model_cfg["weights_path"], device)

        # Prepare transform
        image_size = model_cfg["image_size"]
        transform = get_val_transforms(image_size)
        face_pred_result = []
        for idx, face in enumerate(faces):
            print("Start processing faces")
            image_tensor = transform(face)
            probs = self.predict_single(self.model, image_tensor, device)

            fake_prob = probs[0]
            real_prob = probs[1]
            prediction = "REAL" if real_prob >= model_cfg["threshold"] else "FAKE"

            face_pred_result.append(
                {
                    "face idx": idx,
                    "prediction": prediction,
                    "real_prob": real_prob,
                    "fake_prob": fake_prob,
                    "confidence": max(fake_prob, real_prob),
                }
            )
            # print("\n" + "=" * 60)
            # print(f"Image: {input_path.name}")
            # print(f"Prediction: {prediction}")
            # print(f"Confidence: {max(fake_prob, real_prob):.2%}")
            # print(f"Real probability: {real_prob:.4f}")
            # print(f"Fake probability: {fake_prob:.4f}")
            # print("=" * 60 + "\n")

            # Get all images
            # image_files = []
            # for ext in ["*.jpg", "*.jpeg", "*.png"]:
            #     image_files.extend(glob.glob(str(input_path / ext)))

            # if not image_files:
            #     logger.error(f"No images found in {input_path}")
            #     return

            # logger.info(f"Found {len(image_files)} images")

            # results = []

            # for image_path in tqdm(image_files, desc="Processing"):
            #     try:
            #         image_tensor = load_image(image_path, transform)
            #         probs = predict_single(model, image_tensor, device)

            #         fake_prob = probs[0]
            #         real_prob = probs[1]
            #         prediction = "REAL" if real_prob >= args.threshold else "FAKE"

            #     except Exception as e:
            #         logger.error(f"Error processing {image_path}: {e}")
            #         continue

        # Print summary
        print("\n" + "=" * 60)
        print("Batch Inference Results")
        print("=" * 60)
        print(f"Total images: {len(face_pred_result)}")
        real_count = sum(1 for r in face_pred_result if r["prediction"] == "REAL")
        fake_count = sum(1 for r in face_pred_result if r["prediction"] == "FAKE")
        print(f"Predicted REAL: {real_count}")
        print(f"Predicted FAKE: {fake_count}")
        print("=" * 60 + "\n")

        return fake_count / (fake_count + real_count)
        # Save results if output specified
        # if args.output:
        #     import pandas as pd

        #     df = pd.DataFrame(results)
        #     df.to_csv(args.output, index=False)
        #     logger.info(f"Results saved to: {args.output}")

        # TODO: Implement
        # 1. Preprocess each face
        # 2. Run inference on each face
        # 3. Aggregate scores across frames
