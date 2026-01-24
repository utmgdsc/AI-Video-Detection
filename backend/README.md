# Backend - AI Video Detection Pipeline

## Overview

This is the backend pipeline for deepfake detection. It processes videos through multiple models and returns a combined score.

## Architecture

```
Input Video
    │
    ├── Audio Handler (AASIST)
    │   └── Audio deepfake score
    │
    └── Video Handler
        ├── Facial Analyzer (XceptionNet / MesoNet)
        │   └── Face-based deepfake score
        │
        └── Image Analyzer (EfficientNet)
            └── Image manipulation score
    │
    ▼
Combined Score + Result
```

## Setup

### 1. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 2. Download model weights

Each team member downloads their model's pretrained weights:

| Model | Download from | Place in |
|-------|--------------|----------|
| XceptionNet | [TBD] | `weights/xception.pth` |
| EfficientNet | [TBD] | `weights/efficientnet.pth` |
| MesoNet | [TBD] | `weights/mesonet.pth` |
| AASIST | [TBD] | `weights/aasist.pth` |

### 3. Run

```bash
python -m backend.main <video_path>
```

## Folder Structure

```
backend/
├── models/           # Model architecture code
│   ├── xception.py
│   ├── efficientnet.py
│   ├── mesonet.py
│   └── aasist.py
├── handlers/         # Pipeline handlers
│   ├── audio_handler.py
│   ├── video_handler.py
│   ├── facial_analyzer.py
│   └── image_analyzer.py
├── preprocessing/    # Data preprocessing
│   ├── video_processor.py
│   ├── audio_processor.py
│   └── image_processor.py
├── main.py          # Orchestrator
└── requirements.txt
```

## For Team Members

1. **Document** your model in `docs/models/<your-model>/`
2. **Implement** your model's `load_model()` and `predict()` in `backend/models/<your-model>.py`
3. **Test** locally with sample videos
4. **Commit** your code (NOT the weights)
