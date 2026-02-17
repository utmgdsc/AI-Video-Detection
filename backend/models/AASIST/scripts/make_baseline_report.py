import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse AASIST training log and generate baseline CSV, plot, and markdown report."
    )

    # Inputs
    p.add_argument(
        "--log_path",
        type=Path,
        default=Path("backend/models/AASIST/runs/baseline_run.log"),
        help="Path to the training log file to parse.",
    )

    # Output directory (optional convenience)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("backend/models/AASIST/runs"),
        help="Directory to write outputs to (used if out_csv/out_png/out_md are not provided).",
    )

    # Outputs (override out_dir defaults)
    p.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Output CSV path. If not set, uses <out_dir>/baseline_metrics.csv",
    )
    p.add_argument(
        "--out_png",
        type=Path,
        default=None,
        help="Output PNG path. If not set, uses <out_dir>/baseline_curves.png",
    )
    p.add_argument(
        "--out_md",
        type=Path,
        default=None,
        help="Output Markdown report path. If not set, uses <out_dir>/baseline_report.md",
    )

    # Optional metadata for the report (so it's not hardcoded either)
    p.add_argument(
        "--config_path",
        type=str,
        default="backend/models/AASIST/aasist_detector/config/AASIST.conf",
        help="Config path to mention in the generated report.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=24,
        help="Batch size to mention in the generated report.",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate to mention in the generated report.",
    )
    p.add_argument(
        "--dataset_name",
        type=str,
        default="ASVspoof2019 LA (train → dev)",
        help="Dataset description to mention in the generated report.",
    )
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="backend/models/AASIST/runs/aasist_baseline.pt",
        help="Checkpoint path to mention in the generated report.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    log_path: Path = args.log_path

    # Resolve output paths
    out_dir: Path = args.out_dir
    out_csv: Path = args.out_csv if args.out_csv else (out_dir / "baseline_metrics.csv")
    out_png: Path = args.out_png if args.out_png else (out_dir / "baseline_curves.png")
    out_md: Path = args.out_md if args.out_md else (out_dir / "baseline_report.md")

    # Ensure output directories exist
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    pattern = re.compile(
        r"\[epoch\s+(\d+)\]\s+train_loss=([0-9.]+)\s+\|\s+dev_acc=([0-9.]+)\s+\|\s+dev_eer=([0-9.]+)\s+@thr=([0-9.]+)"
    )

    epochs, train_loss, dev_acc, dev_eer, thr = [], [], [], [], []

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for m in pattern.finditer(text):
        epochs.append(int(m.group(1)))
        train_loss.append(float(m.group(2)))
        dev_acc.append(float(m.group(3)))
        dev_eer.append(float(m.group(4)))
        thr.append(float(m.group(5)))

    if not epochs:
        raise SystemExit(
            f"No epochs parsed from {log_path}. "
            f"Make sure the log contains lines like: [epoch 01] train_loss=... | dev_acc=... | dev_eer=... @thr=..."
        )

    # Best epoch by EER
    best_i = min(range(len(dev_eer)), key=lambda i: dev_eer[i])
    best_epoch = epochs[best_i]
    best_eer = dev_eer[best_i]
    best_acc = dev_acc[best_i]
    best_thr = thr[best_i]

    # Save CSV
    out_csv.write_text("epoch,train_loss,dev_acc,dev_eer,thr\n", encoding="utf-8")
    with out_csv.open("a", encoding="utf-8") as f:
        for i in range(len(epochs)):
            f.write(f"{epochs[i]},{train_loss[i]},{dev_acc[i]},{dev_eer[i]},{thr[i]}\n")

    # Plot curves
    plt.figure()
    plt.plot(epochs, dev_eer, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Dev EER")
    plt.title("AASIST Baseline: Dev EER vs Epoch")
    plt.grid(True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    # Write markdown report
    out_md.write_text(
        f"""# AASIST Baseline Report

**Setup**
- Dataset: {args.dataset_name}
- Config: `{args.config_path}`
- Batch size: {args.batch_size}
- Learning rate: {args.lr}
- Epochs run: {max(epochs)}

**Results (Development Set)**
- **Best EER:** {best_eer:.4f} (**{best_eer*100:.2f}%**) at **epoch {best_epoch}**
- **Dev Accuracy at best EER epoch:** {best_acc:.4f} (**{best_acc*100:.2f}%**)
- Threshold at EER: {best_thr:.4f}

**Artifacts**
- Metrics CSV: `{out_csv}`
- Plot: `{out_png}`
- Best checkpoint (reference): `{args.checkpoint_path}`
""",
        encoding="utf-8",
    )

    print("Wrote:")
    print(" -", out_csv)
    print(" -", out_png)
    print(" -", out_md)
    print(f"Best dev EER = {best_eer:.4f} ({best_eer*100:.2f}%) at epoch {best_epoch}")


if __name__ == "__main__":
    main()
