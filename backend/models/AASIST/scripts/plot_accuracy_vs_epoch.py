import csv
from pathlib import Path
import matplotlib.pyplot as plt

CSV_PATH = Path("backend/models/AASIST/runs/baseline_metrics.csv")
OUT_PNG  = Path("backend/models/AASIST/runs/accuracy_vs_epoch.png")

epochs = []
dev_acc = []

with CSV_PATH.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch"]))
        dev_acc.append(float(row["dev_acc"]) * 100.0)  # convert to %

# find best epoch (highest accuracy)
best_i = max(range(len(dev_acc)), key=lambda i: dev_acc[i])
best_epoch = epochs[best_i]
best_acc = dev_acc[best_i]

plt.figure()
plt.plot(epochs, dev_acc, marker="o", label="Accuracy")
plt.scatter([best_epoch], [best_acc], zorder=3, label=f"Best epoch ({best_epoch})")

plt.xlabel("Epoch")
plt.ylabel("Dev Accuracy (%)")
plt.title("AASIST Baseline: Dev Accuracy vs Epoch")
plt.grid(True)
plt.legend()

plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved {OUT_PNG}")
print(f"Best dev accuracy: {best_acc:.2f}% at epoch {best_epoch}")
