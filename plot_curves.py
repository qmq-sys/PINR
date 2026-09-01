"""Training / adaptation curve plots for Phase 3."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def plot_training_curve(history_csv: Path, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    rows: list[dict[str, Any]] = []
    with open(history_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty training history: {history_csv}")

    epochs = [int(float(r["epoch"])) for r in rows]
    # Support both naming conventions
    def _col(r: dict, *names: str) -> float:
        for n in names:
            if n in r and r[n] not in (None, ""):
                return float(r[n])
        return float("nan")

    train_l = [_col(r, "train_signal_loss", "train_mse") for r in rows]
    val_l = [_col(r, "val_signal_loss", "val_mse") for r in rows]
    wall = [_col(r, "wall_time_sec", "wall_clock_time_sec") for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_l, label="train signal loss", marker="o", markersize=2)
    axes[0].plot(epochs, val_l, label="val signal loss (zero-shot)", marker="o", markersize=2)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("signal loss (S0-norm MSE)")
    axes[0].set_title("Population training curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, wall, color="gray")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("wall-clock time (sec)")
    axes[1].set_title("Wall-clock time")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[plot] training curve → {out_png}")


def plot_adaptation_curve(csv_path: Path, out_png: Path, subject_id: str | None = None) -> None:
    import matplotlib.pyplot as plt

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if subject_id:
        rows = [r for r in rows if r.get("subject_id") == subject_id]
    adapt = [r for r in rows if r.get("mode") == "latent_adaptation"]
    if not adapt:
        adapt = [r for r in rows if r.get("adapt_iter") not in (None, "")]
    if not adapt:
        raise SystemExit(f"no adaptation rows in {csv_path}")

    adapt = sorted(adapt, key=lambda r: int(float(r.get("adapt_iter", 0))))
    xs = [int(float(r["adapt_iter"])) for r in adapt]
    fa = [float(r["FA_MAE"]) for r in adapt]
    md = [float(r["MD_MAE"]) for r in adapt]
    psnr = [float(r["signal_PSNR"]) for r in adapt]
    rel = [float(r.get("signal_RelMSE", "nan")) for r in adapt]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(xs, fa, marker="o")
    axes[0, 0].set_title("FA MAE")
    axes[0, 0].set_xlabel("iteration")
    axes[0, 1].plot(xs, md, marker="o")
    axes[0, 1].set_title("MD MAE")
    axes[0, 1].set_xlabel("iteration")
    axes[1, 0].plot(xs, psnr, marker="o")
    axes[1, 0].set_title("Signal PSNR")
    axes[1, 0].set_xlabel("iteration")
    axes[1, 1].plot(xs, rel, marker="o")
    axes[1, 1].set_title("Signal RelMSE")
    axes[1, 1].set_xlabel("iteration")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Latent adaptation curve ({subject_id or 'all'})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    out_csv = out_png.with_suffix(".csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["adapt_iter", "FA_MAE", "MD_MAE", "signal_PSNR", "signal_RelMSE"]
        )
        w.writeheader()
        for x, a, b, c, d in zip(xs, fa, md, psnr, rel):
            w.writerow(
                {
                    "adapt_iter": x,
                    "FA_MAE": a,
                    "MD_MAE": b,
                    "signal_PSNR": c,
                    "signal_RelMSE": d,
                }
            )
    print(f"[plot] adaptation curve → {out_png} / {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-csv", default="")
    ap.add_argument("--adaptation-csv", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subject-id", default="")
    args = ap.parse_args()
    out = Path(args.out_dir)
    if args.training_csv:
        plot_training_curve(Path(args.training_csv), out / "training_curve.png")
    if args.adaptation_csv:
        plot_adaptation_curve(
            Path(args.adaptation_csv),
            out / "adaptation_curve.png",
            args.subject_id or None,
        )


if __name__ == "__main__":
    main()
