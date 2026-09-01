"""Scatter plots: reference vs PINR prediction with Pearson r."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics.evaluator import pearson_corr


def plot_scalar_scatter(
    ref: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    max_points: int = 80000,
    seed: int = 42,
) -> float:
    m = np.asarray(mask, dtype=bool)
    r = np.asarray(ref, dtype=np.float64)[m]
    p = np.asarray(pred, dtype=np.float64)[m]
    cc = pearson_corr(p, r)

    if r.size > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(r.size, size=max_points, replace=False)
        r_s, p_s = r[sel], p[sel]
    else:
        r_s, p_s = r, p

    lo = float(np.nanmin([r_s.min(), p_s.min()])) if r_s.size else 0.0
    hi = float(np.nanmax([r_s.max(), p_s.max()])) if r_s.size else 1.0
    pad = 0.02 * (hi - lo + 1e-8)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(r_s, p_s, s=2, alpha=0.25, c="C0", rasterized=True)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nPearson r = {cc:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return float(cc)
