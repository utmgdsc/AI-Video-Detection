# Models Directory

## Structure (Hybrid Approach)

This directory uses a hybrid approach for managing models:

```
backend/models/
├── DeepFake-EfficientNet/     # Full EfficientNet repo (cloned/added by team member)
├── XceptionNet-Detector/      # Full XceptionNet repo (add your full repo here)
├── MesoNet/                   # Full MesoNet repo (add your full repo here)
├── AASIST/                    # Full AASIST repo (add your full repo here)
└── wrappers/
    ├── efficientnet.py        # Thin wrapper - standardized interface
    ├── xception.py            # Thin wrapper - standardized interface
    ├── mesonet.py             # Thin wrapper - standardized interface
    └── aasist.py              # Thin wrapper - standardized interface
```

## Why This Approach?

1. **Team members keep full working code** - Clone/add your entire model repo
2. **Wrappers provide standard interface** - For the unified pipeline (later)
3. **Low risk** - If integration doesn't work, we still have documented working models
4. **Independent work** - Each person can work without breaking others' code

## For Team Members

### Step 1: Add Your Model Repo

Clone or copy your full model repository into this directory:

```bash
cd backend/models/

# Option A: Clone a repo
git clone https://github.com/umitkacar/DeepFake-EfficientNet.git

# Option B: Copy your local code
cp -r ~/path/to/your/model ./YourModelName/
```

### Step 2: Update Your Wrapper

Edit `wrappers/<your-model>.py` to provide a standardized interface:

```python
# Example: wrappers/efficientnet.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../DeepFake-EfficientNet'))

from deepfake_detector.models import DeepFakeDetector

def load_model(weights_path=None):
    """Load EfficientNet model."""
    if weights_path is None:
        weights_path = 'outputs/checkpoints/best_model.pth'
    return DeepFakeDetector.from_pretrained(weights_path, model_name='efficientnet-b1')

def predict(model, image):
    """Run inference on image."""
    return model.predict(image)
```

### Step 3: Test Your Wrapper

```python
from backend.models.wrappers import efficientnet

model = efficientnet.load_model()
score = efficientnet.predict(model, image)
print(f"Fake probability: {score}")
```

## Integration Timeline

- **Week 1-6:** Add full repos, document in `docs/models/`
- **Week 7-8:** Create wrappers for unified pipeline
- **Week 9:** Test integration in `backend/main.py`

## Notes

- Full repos are NOT committed if they're too large (use `.gitignore`)
- Wrappers ARE committed (they're small Python files)
- Document your repo's location in `docs/models/<your-model>/02-source-and-setup.md`
