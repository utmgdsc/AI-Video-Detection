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

### 2. Add your model code

Each team member adds their full model repository:

```bash
cd backend/models/
# Clone or copy your model repo here
git clone https://github.com/your-username/your-model-repo.git
```

See `backend/models/README.md` for detailed instructions.

### 3. Download model weights

Download pretrained weights and place in `weights/`:

| Model | Download from | Place in |
|-------|--------------|----------|
| XceptionNet | [TBD] | `weights/xception.pth` |
| EfficientNet | [TBD] | `weights/efficientnet.pth` |
| MesoNet | [TBD] | `weights/mesonet.pth` |
| AASIST | [TBD] | `weights/aasist.pth` |

### 4. Run

```bash
python -m backend.main <video_path>
```

## Folder Structure

```
backend/
├── models/                    # Full model repositories
│   ├── DeepFake-EfficientNet/ # Full repo (example from PR #1)
│   ├── XceptionNet-Detector/  # Add your full repo here
│   ├── MesoNet/               # Add your full repo here
│   ├── AASIST/                # Add your full repo here
│   ├── wrappers/              # Thin wrappers for standard interface
│   │   ├── xception.py
│   │   ├── efficientnet.py
│   │   ├── mesonet.py
│   │   └── aasist.py
│   └── README.md              # Instructions for adding models
├── handlers/                  # Pipeline handlers
│   ├── audio_handler.py
│   ├── video_handler.py
│   ├── facial_analyzer.py
│   └── image_analyzer.py
├── preprocessing/             # Data preprocessing
│   ├── video_processor.py
│   ├── audio_processor.py
│   └── image_processor.py
├── main.py                    # Orchestrator
└── requirements.txt
```

## For Team Members

### Adding Your Model

1. **Add your full repo** to `backend/models/YourModelName/`
2. **Document** in `docs/models/<your-model>/`
3. **Create a wrapper** in `backend/models/wrappers/<your-model>.py`
4. **Test** locally with sample videos
5. **Commit** wrappers and docs (NOT large repos or weights)

### Timeline

- **Week 1-6:** Focus on documentation and getting models working
- **Week 7-8:** Create wrappers and integrate into pipeline
- **Week 9:** Test end-to-end pipeline
- **Week 10:** Final testing and documentation

## Why Hybrid Structure?

We use a **hybrid approach** where each team member:
- Keeps their full working model code
- Adds thin wrappers for the unified pipeline

This means:
- ✅ Lower risk - working code stays working
- ✅ Independent work - no merge conflicts
- ✅ Flexibility - use whatever structure your model needs
- ✅ Integration later - wrappers provide standard interface

See `backend/models/README.md` for details.
