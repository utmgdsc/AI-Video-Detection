import json
import torch
import soundfile as sf

from .model import Model


class AASISTDetector:
    def __init__(
        self,
        conf_path: str,
        ckpt_path: str,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        with open(conf_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.model = Model(cfg["model_config"]).to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]

        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()

    def predict_wav(self, wav_path: str) -> float:
        x, _ = sf.read(wav_path)
        if x.ndim == 2:
            x = x.mean(axis=1)

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding, logits = self.model(x)

        probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].item())  # spoof probability
