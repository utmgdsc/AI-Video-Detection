"""
AASIST model for audio deepfake detection.

Team member: Laiba Khan
Docs: docs/models/aasist/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from backend.models.AASIST.aasist_detector.detector import AASISTDetector


def load_model(
    weights_path: Optional[Union[str, Path]] = None,
    conf_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> AASISTDetector:
    """
    Load AASIST model.

    Args:
        weights_path: Path to pretrained weights checkpoint (.pth/.pt).
                      If None, uses backend/models/AASIST/aasist_detector/weights/AASIST.pth
        conf_path: Path to config file (JSON content; your file is AASIST.conf).
                   If None, uses backend/models/AASIST/aasist_detector/config/AASIST.conf
        device: "cpu" or "cuda". If None, auto-selects CUDA if available.

    Returns:
        detector: AASISTDetector ready for inference.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    conf_path = Path(conf_path) if conf_path is not None else Path(
        "backend/models/AASIST/aasist_detector/config/AASIST.conf"
    )
    weights_path = Path(weights_path) if weights_path is not None else Path(
        "backend/models/AASIST/aasist_detector/weights/AASIST.pth"
    )

    if not conf_path.exists():
        raise FileNotFoundError(f"AASIST config not found: {conf_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"AASIST weights not found: {weights_path}")

    detector = AASISTDetector(
        conf_path=str(conf_path),
        ckpt_path=str(weights_path),
        device=device,
    )
    return detector


@torch.no_grad()
def predict(detector: AASISTDetector, audio: torch.Tensor) -> float:
    """
    Run inference on audio sample.

    Args:
        detector: Loaded detector from load_model()
        audio: Preprocessed audio tensor (waveform).
               Shapes accepted: [T] or [B, T]

    Returns:
        score: Float between 0-1 (0=real/bonafide, 1=fake/spoof)
    """
    if not isinstance(audio, torch.Tensor):
        raise TypeError("audio must be a torch.Tensor")

    x = audio
    if x.dim() == 1:
        x = x.unsqueeze(0)  # [1, T]
    elif x.dim() != 2:
        raise ValueError(f"Expected audio shape [T] or [B,T], got {tuple(x.shape)}")

    x = x.to(detector.device).float()

    # detector.model returns (embedding, logits)
    _, logits = detector.model(x)

    probs = torch.softmax(logits, dim=1)
    spoof_prob = probs[:, 1]  # index 1 = spoof/fake
    return float(spoof_prob.mean().item())


@torch.no_grad()
def predict_wav(detector: AASISTDetector, wav_path: Union[str, Path]) -> float:
    """
    Convenience: let the detector load wav from disk.
    Returns spoof probability 0..1.
    """
    return float(detector.predict_wav(str(wav_path)))


# -------------------------------------------------------------------
# Optional smoke test (toggle OR comment out)
# -------------------------------------------------------------------
RUN_SMOKE_TEST = True  # <-- set True when you want to run a quick test

if __name__ == "__main__" and RUN_SMOKE_TEST:
    det = load_model(device="cpu")  # or "cuda"
    score = predict_wav(det, "test_audio.wav")  # ensure a real path here
    print("Spoof probability:", score)

