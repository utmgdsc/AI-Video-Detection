import sys
from pathlib import Path

# Tell Python to look in the parent directory for modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from deepfake_detector.models import DeepFakeDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your exact model and weights
model = DeepFakeDetector(model_name="efficientnet-b1", pretrained=False)
model.load_checkpoint("/home/gdgteam1/AI-Video-Detection/backend/models/DeepFake_EfficientNet/faceforensic_output/outputs/checkpoints/best_model.pth", device=str(device)) # UPDATE THIS PATH
model.to(device)
model.eval()

# Load the saved validation tensors
t_eff = torch.load("/home/gdgteam1/AI-Video-Detection/efficientnet_tensor.pt").to(device)
t_ens = torch.load("/home/gdgteam1/AI-Video-Detection/ensemble_tensor.pt").to(device)

with torch.no_grad():
    out_eff = model(t_eff)
    out_ens = model(t_ens)

print("=== EFFICIENTNET TENSOR (From Training Script) ===")
print("Raw Logits (First 2):")
print(out_eff[:2])
print("Probabilities (Fake vs Real):")
print(torch.softmax(out_eff[:2], dim=1))

print("\n=== ENSEMBLE TENSOR (From Video Handler) ===")
print("Raw Logits (First 2):")
print(out_ens[:2])
print("Probabilities (Fake vs Real):")
print(torch.softmax(out_ens[:2], dim=1))