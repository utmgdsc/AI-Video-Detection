# AASIST — Source & Setup

## Source location(s)

- GitHub repo link or openCV link:
  https://github.com/clovaai/aasist  
  (Official implementation by NAVER / CLOVA AI)

- Paper link: https://arxiv.org/abs/2110.01200

- License notes (if known):MIT License (as stated in the original repository)

## What I verified

- Does the repo match the paper description?
  Yes. The architecture and outputs align with the paper’s description of a raw waveform–based audio anti-spoofing model using spectro-temporal graph attention.

- Pretrained model available?
  Yes. Pretrained checkpoints (`AASIST.pth`, `AASIST-L.pth`) are provided and load correctly.

- Dataset expected by repo: 
  The original repo is designed for ASVspoof datasets, but for this project we only verified inference on arbitrary WAV files (dataset not required for Phase 1).



## Environment / dependencies

List exactly what you installed (with versions if possible):

- OS: Linux (remote machine)
- Python: 3.10 (Conda environment)
- PyTorch: 2.1+ (CPU inference)
- CUDA: Not used (CPU only)
- OpenCV: Not required for audio-only inference
- Other deps:
  - numpy
  - soundfile
  - torch
  - json (standard library)
  - librosa
  - torchaudio

---

## Setup steps (copy/paste friendly)

### 1) Clone / install

```bash

git clone https://github.com/utmgdsc/AI-Video-Detection.git
cd AI-Video-Detection

conda create -n audio-df python=3.10 -y
conda activate audio-df

pip install -r backend/requirements.txt


```

### 2) Any downloads needed (weights/datasets)

Weights: 
- What to download: Pretrained AASIST checkpoint (AASIST.pth) 
- Link: https://github.com/clovaai/aasist/tree/main/models/weights
- Where to place it: backend/models/AASIST/aasist_detector/weights/AASIST.pth

Dataset: 
- Name: ASVspoof 2019 — Logical Access (LA)
- Source: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset

Storage Location:
- Path: /home/gdgteam1/AI-Video-Detection/backend/dataset/ASVspoof2019_LA/ASVspoof2019_LA/
- Access Method: Dataset is downloaded manually from Kaggle and unzipped into backend/dataset/


### 3) How to run (Test)

```bash
# Test Dummy wav
python - <<'PY'
import numpy as np, soundfile as sf
from backend.models.AASIST.aasist_detector.detector import AASISTDetector

# Create a short dummy WAV (for pipeline validation only)
sr = 16000
x = (np.random.randn(sr * 2) * 0.01).astype("float32")
sf.write("dummy.wav", x, sr)

det = AASISTDetector(
    conf_path="backend/models/AASIST/aasist_detector/config/AASIST.conf",
    ckpt_path="backend/models/AASIST/aasist_detector/weights/AASIST.pth",
    device="cpu",
)

print("spoof score:", det.predict_wav("dummy.wav"))
PY

```

# Test real AVspoof file (.flac file) 

```bash
# Test real ASVspoof .flac file
python - <<'PY'
from backend.models.AASIST.aasist_detector.detector import AASISTDetector

det = AASISTDetector(
    conf_path="backend/models/AASIST/aasist_detector/config/AASIST.conf",
    ckpt_path="backend/models/AASIST/aasist_detector/weights/AASIST.pth",
    device="cpu",
)

# Replace with actual path to a real .flac file
flac_path = "backend/dataset/ASVspoof2019_LA/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac/LA_D_1000137.flac"

print("spoof score:", det.predict_wav(flac_path))
PY


```

## Output / results

- What output you saw (logs, metrics, saved files):
    The model successfully loads configuration and weights with:
    - Missing keys: 0
    - Unexpected keys: 0
    Inference returns a single floating-point value:
    - spoof_prob ∈ [0,1]
    - Higher values indicate higher likelihood of spoofed / AI-generated audio

    Example Output: spoof score: 0.63

- Add screenshots in assets/ if helpful.

## Issues encountered + fixes

- Issue: Confusion over model output format.
- Cause: AASIST returns (embedding, logits) instead of just logits.
- Fix (steps/commands): Extracted logits explicitly and applied softmax to compute spoof probability.

## Notes for teammates

- If someone else sets this up from scratch, what do they need to know?
- Model expects mono, 16kHz audio; script will average stereo to mono; resampling is not done automatically (sample rate mismatch raises error).
