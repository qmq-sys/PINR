"""FA / MD map comparison figures: GT | Prediction | Error."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _mid_slice(vol: np.ndarray, axis: int = 2) -> np.ndarray:
    vol = np.asarray(vol)
    idx = int(vol.shape[axis] // 2)
    return np.take(vol, idx, axis=axis)


def _masked_display(vol: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    out = np.asarray(vol, dtype=np.float64).copy()
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        out = np.where(m, out, np.nan)
    return out


def plot_scalar_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    title: str = "",
    cmap: str = "gray",
    err_cmap: str = "hot",
    vmax: float | None = None,
    err_vmax: float | None = None,
    out_path: Path | None = None,
    also_save_parts: Path | None = None,
    part_prefix: str = "FA",
) -> Path | None:
    """
    One figure with three panels: GT | Prediction | |Error|.
    Optionally also write {prefix}_gt.png / _pred.png / _error.png under also_save_parts.
    """
    gt_s = _mid_slice(_masked_display(gt, mask))
    pr_s = _mid_slice(_masked_display(pred, mask))
    err_s = np.abs(pr_s - gt_s)

    if vmax is None:
        finite = np.concatenate([gt_s[np.isfinite(gt_s)], pr_s[np.isfinite(pr_s)]])
        vmax = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
        vmax = max(vmax, 1e-8)
    if err_vmax is None:
        err_vmax = float(np.nanpercentile(err_s[np.isfinite(err_s)], 99.0)) if np.isfinite(err_s).any() else 1.0
        err_vmax = max(err_vmax, 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(np.rot90(gt_s), cmap=cmap, vmin=0.0, vmax=vmax)
    axes[0].set_title("GT")
    im1 = axes[1].imshow(np.rot90(pr_s), cmap=cmap, vmin=0.0, vmax=vmax)
    axes[1].set_title("Prediction")
    im2 = axes[2].imshow(np.rot90(err_s), cmap=err_cmap, vmin=0.0, vmax=err_vmax)
    axes[2].set_title("|Error|")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    saved: Path | None = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        saved = out_path

    if also_save_parts is not None:
        part_dir = Path(also_save_parts)
        part_dir.mkdir(parents=True, exist_ok=True)
        for name, data, cm, vmax_i in (
            (f"{part_prefix}_gt.png", gt_s, cmap, vmax),
            (f"{part_prefix}_pred.png", pr_s, cmap, vmax),
            (f"{part_prefix}_error.png", err_s, err_cmap, err_vmax),
        ):
            fig_i, ax_i = plt.subplots(figsize=(4, 4))
            ax_i.imshow(np.rot90(data), cmap=cm, vmin=0.0, vmax=vmax_i)
            ax_i.axis("off")
            ax_i.set_title(name.replace(".png", "").replace("_", " "))
            fig_i.tight_layout()
            fig_i.savefig(part_dir / name, dpi=150, bbox_inches="tight")
            plt.close(fig_i)

    plt.close(fig)
    return saved


def save_subject_map_figures(
    *,
    subject_id: str,
    mode: str,
    pred_maps: dict[str, np.ndarray],
    ref_maps: dict[str, np.ndarray],
    mask: np.ndarray,
    maps_dir: Path,
) -> dict[str, Any]:
    """Write FA/MD comparison panels + per-panel PNGs for one subject/mode."""
    maps_dir = Path(maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{subject_id}_{mode}"
    out: dict[str, Any] = {"subject_id": subject_id, "mode": mode}

    for key, vmax in (("FA", 1.0), ("MD", None)):
        panel = maps_dir / f"{tag}_{key}_comparison.png"
        parts = maps_dir / tag
        plot_scalar_comparison(
            ref_maps[key],
            pred_maps[key],
            mask=mask,
            title=f"{subject_id} | {mode} | {key}",
            cmap="gray",
            vmax=vmax,
            out_path=panel,
            also_save_parts=parts,
            part_prefix=key,
        )
        out[f"{key}_panel"] = str(panel)
        out[f"{key}_gt"] = str(parts / f"{key}_gt.png")
        out[f"{key}_pred"] = str(parts / f"{key}_pred.png")
        out[f"{key}_error"] = str(parts / f"{key}_error.png")
    return out
