import re
from pathlib import Path

import matplotlib.pyplot as plt

LOG_PATH = Path("backend/models/AASIST/runs/baseline_run.log")
OUT_CSV  = Path("backend/models/AASIST/runs/baseline_metrics.csv")
OUT_PNG  = Path("backend/models/AASIST/runs/baseline_curves.png")
OUT_MD   = Path("backend/models/AASIST/runs/baseline_report.md")

pattern = re.compile(
    r"\[epoch\s+(\d+)\]\s+train_loss=([0-9.]+)\s+\|\s+dev_acc=([0-9.]+)\s+\|\s+dev_eer=([0-9.]+)\s+@thr=([0-9.]+)"
)

epochs, train_loss, dev_acc, dev_eer, thr = [], [], [], [], []

text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
for m in pattern.finditer(text):
    epochs.append(int(m.group(1)))
    train_loss.append(float(m.group(2)))
    dev_acc.append(float(m.group(3)))
    dev_eer.append(float(m.group(4)))
    thr.append(float(m.group(5)))

if not epochs:
    raise SystemExit(f"No epochs parsed from {LOG_PATH}. Make sure the log contains lines like: [epoch 01] ...")

# Best epoch by EER
best_i = min(range(len(dev_eer)), key=lambda i: dev_eer[i])
best_epoch = epochs[best_i]
best_eer = dev_eer[best_i]
best_acc = dev_acc[best_i]
best_thr = thr[best_i]

# Save CSV
OUT_CSV.write_text("epoch,train_loss,dev_acc,dev_eer,thr\n", encoding="utf-8")
with OUT_CSV.open("a", encoding="utf-8") as f:
    for i in range(len(epochs)):
        f.write(f"{epochs[i]},{train_loss[i]},{dev_acc[i]},{dev_eer[i]},{thr[i]}\n")

# Plot curves (no seaborn)
plt.figure()
plt.plot(epochs, dev_eer, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Dev EER")
plt.title("AASIST Baseline: Dev EER vs Epoch")
plt.grid(True)
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close()

# Write short markdown report
OUT_MD.write_text(
    f"""# AASIST Baseline Report (ASVspoof2019 LA)

**Setup**
- Dataset: ASVspoof2019 LA (train → dev)
- Config: `backend/models/AASIST/aasist_detector/config/AASIST.conf`
- Batch size: 24
- Learning rate: 1e-4
- Epochs run: {max(epochs)}

**Results (Development Set)**
- **Best EER:** {best_eer:.4f} (**{best_eer*100:.2f}%**) at **epoch {best_epoch}**
- **Dev Accuracy at best EER epoch:** {best_acc:.4f} (**{best_acc*100:.2f}%**)
- Threshold at EER: {best_thr:.4f}

**Artifacts**
- Metrics CSV: `{OUT_CSV}`
- Plot: `{OUT_PNG}`
- Best checkpoint: `backend/models/AASIST/runs/aasist_baseline.pt`
""",
    encoding="utf-8",
)

print("Wrote:")
print(" -", OUT_CSV)
print(" -", OUT_PNG)
print(" -", OUT_MD)
print(f"Best dev EER = {best_eer:.4f} ({best_eer*100:.2f}%) at epoch {best_epoch}")
