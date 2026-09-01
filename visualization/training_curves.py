"""Training / validation curve plots from experiment logs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _read_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _to_float(x: Any) -> float | None:
    try:
        v = float(x)
        return v
    except Exception:
        return None


def find_training_history(exp_dir: Path) -> Path | None:
    candidates = [
        exp_dir / "checkpoints" / "best" / "training_history.csv",
        exp_dir / "checkpoints" / "last" / "training_history.csv",
        exp_dir / "logs" / "training_history.csv",
        exp_dir / "metrics" / "training_history.csv",
    ]
    # phase4a-style epoch folders
    ckpt = exp_dir / "checkpoints"
    if ckpt.is_dir():
        for p in sorted(ckpt.glob("epoch_*/training_history.csv")):
            candidates.append(p)
        for p in sorted(ckpt.glob("*/training_history.csv")):
            candidates.append(p)
    for c in candidates:
        if c.is_file():
            return c
    return None


def plot_training_curves(exp_dir: Path, out_dir: Path) -> dict[str, str]:
    """
    Write loss_curve.png and optionally validation_curve.png.
    Returns paths dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path = find_training_history(Path(exp_dir))
    written: dict[str, str] = {}
    if hist_path is None:
        return written

    rows = _read_history(hist_path)
    if not rows:
        return written

    # flexible column names
    epoch_key = next((k for k in ("epoch", "iter", "step") if k in rows[0]), None)
    train_key = next(
        (k for k in ("train_mse", "train_loss", "loss", "train_RelMSE") if k in rows[0]),
        None,
    )
    val_key = next(
        (k for k in ("val_mse", "val_loss", "val_RelMSE", "val_FA_MAE") if k in rows[0]),
        None,
    )
    if epoch_key is None or train_key is None:
        return written

    xs, ys = [], []
    for r in rows:
        x = _to_float(r.get(epoch_key))
        y = _to_float(r.get(train_key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, marker="o", markersize=3, lw=1.5, label=train_key)
    ax.set_xlabel(epoch_key)
    ax.set_ylabel(train_key)
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    loss_path = out_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written["loss_curve"] = str(loss_path)

    if val_key is not None:
        vx, vy = [], []
        for r in rows:
            x = _to_float(r.get(epoch_key))
            y = _to_float(r.get(val_key))
            if x is not None and y is not None:
                vx.append(x)
                vy.append(y)
        if vx:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(vx, vy, marker="o", markersize=3, lw=1.5, color="C1", label=val_key)
            ax.set_xlabel(epoch_key)
            ax.set_ylabel(val_key)
            ax.set_title("Validation metric")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            val_path = out_dir / "validation_curve.png"
            fig.savefig(val_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written["validation_curve"] = str(val_path)

    return written
