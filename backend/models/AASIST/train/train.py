import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from backend.models.AASIST.aasist_detector.model import Model
from backend.models.AASIST.preprocessing.audionorm import pad_or_crop


def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if len(x) == target_len:
        return x
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)), mode="constant")

    # crop
    if train:
        start = np.random.randint(0, len(x) - target_len + 1)
    else:
        start = (len(x) - target_len) // 2
    return x[start : start + target_len]


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    scores: higher => more spoof
    labels: 1=spoof, 0=bonafide
    """
    idx = np.argsort(scores)
    scores = scores[idx]
    labels = labels[idx]

    P = np.sum(labels == 1)  # spoof
    N = np.sum(labels == 0)  # bonafide
    if P == 0 or N == 0:
        return float("nan"), float("nan")

    thr = scores
    spoof_suffix = np.cumsum((labels[::-1] == 1).astype(np.int64))[::-1]
    bona_suffix  = np.cumsum((labels[::-1] == 0).astype(np.int64))[::-1]

    TP = spoof_suffix
    FP = bona_suffix
    FN = P - TP

    FPR = FP / N
    FNR = FN / P

    k = int(np.argmin(np.abs(FPR - FNR)))
    eer = float((FPR[k] + FNR[k]) / 2.0)
    return eer, float(thr[k])


@dataclass
class ProtocolLine:
    utt_id: str
    label: int  # 1=spoof, 0=bonafide


def parse_asvspoof_protocol(protocol_path: str) -> List[ProtocolLine]:
    out: List[ProtocolLine] = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            utt_id = parts[1]              # 2nd token is utterance id
            lab_str = parts[-1].lower()    # last token is bonafide/spoof
            label = 1 if lab_str == "spoof" else 0
            out.append(ProtocolLine(utt_id, label))
    return out


class ASVspoofWavDataset(Dataset):
    def __init__(self, wav_dir: str, protocol_path: str, target_len: int, train: bool):
        self.wav_dir = wav_dir
        self.items = parse_asvspoof_protocol(protocol_path)
        self.target_len = target_len
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]

        # ASVspoof LA is usually .flac
        flac_path = os.path.join(self.wav_dir, item.utt_id + ".flac")
        wav_path = os.path.join(self.wav_dir, item.utt_id + ".wav")
        path = flac_path if os.path.exists(flac_path) else wav_path

        x, sr = sf.read(path)
        if x.ndim == 2:
            x = x.mean(axis=1)

        # AASIST LA is typically 16kHz; keep baseline strict
        if sr != 16000:
            raise ValueError(f"Sample rate mismatch in {path}: {sr} != 16000")

        x = pad_or_crop(x.astype(np.float32), self.target_len, train=self.train)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(item.label, dtype=torch.long)
        return x, y


@torch.no_grad()
def eval_dev(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_scores = []
    all_labels = []
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        _, logits = model(x)
        probs = torch.softmax(logits, dim=1)
        spoof_prob = probs[:, 1].detach().cpu().numpy()

        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

        all_scores.append(spoof_prob)
        all_labels.append(y.detach().cpu().numpy())

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    acc = correct / max(total, 1)
    eer, thr = compute_eer(scores, labels)
    return acc, eer, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_wav_dir", required=True)
    ap.add_argument("--train_protocol", required=True)
    ap.add_argument("--dev_wav_dir", required=True)
    ap.add_argument("--dev_protocol", required=True)
    ap.add_argument("--config_path", required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_ckpt", default="backend/models/AASIST/runs/aasist_baseline.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    import json
    with open(args.config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = cfg["model_config"]
    target_len = int(model_cfg.get("nb_samp", 64600))  # match your AASIST.conf

    model = Model(model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    train_ds = ASVspoofWavDataset(args.train_wav_dir, args.train_protocol, target_len, train=True)
    dev_ds   = ASVspoofWavDataset(args.dev_wav_dir, args.dev_protocol, target_len, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_eer = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            _, logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(len(train_loader), 1)
        acc, eer, thr = eval_dev(model, dev_loader, device)
        print(f"[epoch {ep:02d}] train_loss={train_loss:.4f} | dev_acc={acc:.4f} | dev_eer={eer:.4f} @thr={thr:.4f}")

        if eer < best_eer:
            best_eer = eer
            os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
            torch.save({"model": model.state_dict(), "config": cfg, "best_dev_eer": best_eer}, args.out_ckpt)
            print(f"  -> saved: {args.out_ckpt} (best_dev_eer={best_eer:.4f})")


if __name__ == "__main__":
    main()
