import os
import csv
import yaml
import torch
from facenet_pytorch import MTCNN

from backend.main import DeepfakeDetector


import os


def get_label_from_path(video_path: str) -> int:
    """
    Binary label:
    0 = real  -> RvRa
    1 = fake  -> everything else
    """

    filename = os.path.basename(video_path)

    if filename.startswith("RvRa_"):
        return 0

    if filename.startswith(("RvFa_", "FvRa_", "FvFa_")):
        return 1

    raise ValueError(f"Could not determine label from filename: {video_path}")


def collect_video_files(root_dir: str):
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    collected = []

    for current_root, _, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in video_extensions:
                collected.append(os.path.join(current_root, file))

    return sorted(collected)


def main():
    config_path = "./backend/config/ensemble.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device = cfg["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    cfg["device"] = device

    mtcnn_cfg = cfg["mtcnn"]
    mtcnn = MTCNN(
        margin=mtcnn_cfg["margin"],
        min_face_size=mtcnn_cfg["min_face_size"],
        thresholds=mtcnn_cfg["thresholds"],
        factor=mtcnn_cfg["factor"],
        post_process=mtcnn_cfg["post_process"],
        select_largest=mtcnn_cfg["select_largest"],
        keep_all=mtcnn_cfg["keep_all"],
        device=device,
    )

    detector = DeepfakeDetector(config=cfg, device=device)

    input_path = cfg["datasets"]["FakeAVCeleb"]["example_video_set_path"]
    videos = collect_video_files(input_path)

    output_csv = "stacking_train_val.csv"

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video",
            "relative_path",
            "aasist_score",
            "efficientnet_score",
            "mesonet_score",
            "xceptionnet_score",
            "label",
        ])

        for idx, full_path in enumerate(videos):
            rel_path = os.path.relpath(full_path, input_path)
            label = get_label_from_path(full_path)

            try:
                result = detector.analyze(
                    full_path,
                    mtcnn,
                    cfg["batch_size"],
                    cfg["frame_skip"],
                    cfg.get("audio_decision_threshold", 0.5),
                    cfg.get("video_decision_threshold", 0.5),
                )

                scores = result.get("individual_scores", {})

                writer.writerow([
                    os.path.basename(full_path),
                    rel_path,
                    scores.get("aasist_score", ""),
                    scores.get("efficientnet_score", ""),
                    scores.get("mesonet_score", ""),
                    scores.get("xceptionnet_score", ""),
                    label,
                ])

                print(f"[{idx + 1}/{len(videos)}] Saved: {rel_path}")

            except Exception as e:
                print(f"Failed on {rel_path}: {e}")

    print(f"Done. Saved stacking data to {output_csv}")


if __name__ == "__main__":
    main()