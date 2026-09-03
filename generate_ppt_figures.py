"""
Generate 5 core PPT figures for Population-DTI-INR proposal talk.

Uses real experiment outputs:
  - INR experiment3_low_sampling pct50 independent (Single-QINR)
  - WLS-DTI reference NIfTI
  - Population-DTI-INR Phase 7/9 drift & lambda ablation

Style: deep blue + white, one conclusion per figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    raise SystemExit("nibabel required") from e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(r"e:\BaiduNetdiskDownload\Population-DTI-INR")
INR_ROOT = Path(r"e:\BaiduNetdiskDownload\INR")
OUT = ROOT / "ppt_figures"
OUT.mkdir(parents=True, exist_ok=True)

SINGLE_PCT50 = INR_ROOT / "outputs" / "experiment3_low_sampling" / "pct50" / "independent_inr"
WLS_ROOT = INR_ROOT / "outputs" / "step1_traditional_dti"
PHASE9 = ROOT / "experiments" / "population_dti_phase9" / "20260830_210238"
PHASE7 = ROOT / "experiments" / "population_dti_phase7" / "20260830_151400"

# Focus subjects for Single-QINR story (matched to later Phase 7–9 cohort)
FOCUS = ["106319", "120717", "121618"]
FOCUS_LABELS = ["S1\n106319", "S2\n120717", "S3\n121618"]
DEMO_SUBJ = "106319"

# Colors
BLUE = "#1B3A6B"
BLUE_L = "#3A6EA5"
BLUE_LL = "#7BA3C9"
ACCENT = "#C0392B"
GRAY = "#5A5A5A"
BG = "#FFFFFF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "axes.edgecolor": BLUE,
        "axes.labelcolor": BLUE,
        "xtick.color": BLUE,
        "ytick.color": BLUE,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": BG,
    }
)


def _load_metrics(sid: str) -> dict:
    with open(SINGLE_PCT50 / sid / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


def _load_fa_pair(sid: str):
    maps = np.load(SINGLE_PCT50 / sid / "maps.npz")
    fa_pred = np.asarray(maps["FA"], dtype=np.float64)
    fa_ref = np.asarray(nib.load(str(WLS_ROOT / sid / "FA.nii.gz")).get_fdata(), dtype=np.float64)
    mask = np.asarray(nib.load(str(WLS_ROOT / sid / "valid_mask.nii.gz")).get_fdata()) > 0
    brain = WLS_ROOT / sid / "brain_mask.nii.gz"
    if brain.exists():
        mask = mask & (np.asarray(nib.load(str(brain)).get_fdata()) > 0)
    return fa_ref, fa_pred, mask


def _mid_axial(vol: np.ndarray) -> np.ndarray:
    return np.take(vol, vol.shape[2] // 2, axis=2)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) - a.mean()
    b = b.astype(np.float64) - b.mean()
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / den) if den > 1e-20 else float("nan")


# ===========================================================================
# Fig 1 — Signal fitting vs FA agreement (core paradox)
# ===========================================================================
def fig1_signal_vs_fa():
    """
    Dual panel proving: Single-QINR fits signal well, but FA vs WLS is limited.
    (No stored WLS signal RelMSE — WLS is the parameter reference.)
    """
    relmse, fa_r, fa_mae = [], [], []
    for sid in FOCUS:
        m = _load_metrics(sid)
        relmse.append(m["dwi"]["relative_mse"])
        fa_r.append(m["parameter_metrics"]["FA"]["Pearson"])
        fa_mae.append(m["parameter_metrics"]["FA"]["MAE"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    x = np.arange(len(FOCUS))
    w = 0.55

    ax = axes[0]
    bars = ax.bar(x, relmse, width=w, color=BLUE_L, edgecolor=BLUE, linewidth=1.2)
    for b, v in zip(bars, relmse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(FOCUS_LABELS)
    ax.set_ylabel("Signal RelMSE  ↓ better")
    ax.set_title("Signal reconstruction\n(Single-QINR, 50% sampling)")
    ax.set_ylim(0, max(relmse) * 1.35)
    ax.axhline(0.1, color=GRAY, ls="--", lw=1, alpha=0.7)
    ax.text(len(FOCUS) - 0.5, 0.102, "RelMSE < 0.1", fontsize=8, color=GRAY, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    bars = ax.bar(x, fa_r, width=w, color=ACCENT, edgecolor="#8B1E1E", linewidth=1.2, alpha=0.9)
    for b, v in zip(bars, fa_r):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(FOCUS_LABELS)
    ax.set_ylabel("FA Pearson r vs WLS  ↑ better")
    ax.set_title("Parameter agreement\n(FA vs WLS reference)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.85, color=GRAY, ls="--", lw=1, alpha=0.7)
    ax.text(len(FOCUS) - 0.5, 0.87, "target r > 0.85", fontsize=8, color=GRAY, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Single-QINR fits signal well, but FA agreement remains far below target",
        fontsize=13,
        fontweight="bold",
        color=BLUE,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Signal fitting ≠ Parameter accuracy   |   HCP-YA · b=1000 · 50% DWI sampling · Independent INR",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    fig.tight_layout()
    path = OUT / "fig1_signal_vs_fa_paradox.png"
    fig.savefig(path)
    plt.close(fig)

    # Also a simpler single-panel RelMSE bar (user request style)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    bars = ax.bar(x, relmse, width=w, color=BLUE_L, edgecolor=BLUE, linewidth=1.2)
    for b, v in zip(bars, relmse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(FOCUS_LABELS)
    ax.set_ylabel("Signal RelMSE")
    ax.set_title("Single-QINR signal reconstruction\n(50% sampling)")
    ax.set_ylim(0, max(relmse) * 1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path2 = OUT / "fig1b_signal_relmse_bars.png"
    fig.savefig(path2)
    plt.close(fig)

    return path, path2, {"relmse": relmse, "fa_r": fa_r, "fa_mae": fa_mae}


# ===========================================================================
# Fig 2 — FA maps: Reference | Single-QINR | |Error|
# ===========================================================================
def fig2_fa_maps():
    fa_ref, fa_pred, mask = _load_fa_pair(DEMO_SUBJ)
    gt = np.where(mask, fa_ref, np.nan)
    pr = np.where(mask, fa_pred, np.nan)
    gt_s = np.rot90(_mid_axial(gt))
    pr_s = np.rot90(_mid_axial(pr))
    err_s = np.abs(pr_s - gt_s)

    vmax = float(np.nanpercentile(np.concatenate([gt_s[np.isfinite(gt_s)], pr_s[np.isfinite(pr_s)]]), 99))
    vmax = max(vmax, 0.8)
    err_vmax = float(np.nanpercentile(err_s[np.isfinite(err_s)], 95))
    err_vmax = max(err_vmax, 0.05)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    panels = [
        (gt_s, "Reference FA\n(WLS-DTI)", "gray", 0.0, vmax),
        (pr_s, "Single-QINR FA\n(50% sampling)", "gray", 0.0, vmax),
        (err_s, "|Error|\n|QINR − WLS|", "hot", 0.0, err_vmax),
    ]
    for ax, (data, title, cmap, vmin, vmax_i) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.set_title(title, fontsize=11, color=BLUE)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    r = _pearson(fa_pred[mask], fa_ref[mask])
    mae = float(np.mean(np.abs(fa_pred[mask] - fa_ref[mask])))
    fig.suptitle(
        f"Subject {DEMO_SUBJ}: FA maps reveal spatial parameter error (r={r:.2f}, MAE={mae:.3f})",
        fontsize=12,
        fontweight="bold",
        color=BLUE,
        y=1.05,
    )
    fig.tight_layout()
    path = OUT / "fig2_fa_maps_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path, {"FA_r": r, "FA_MAE": mae}


# ===========================================================================
# Fig 3 — FA scatter: Reference vs Predicted
# ===========================================================================
def fig3_fa_scatter():
    fa_ref, fa_pred, mask = _load_fa_pair(DEMO_SUBJ)
    r = fa_ref[mask]
    p = fa_pred[mask]
    cc = _pearson(p, r)

    rng = np.random.default_rng(42)
    n = min(60000, r.size)
    sel = rng.choice(r.size, size=n, replace=False)
    r_s, p_s = r[sel], p[sel]

    # Multi-subject FA Pearson bars for WLS-reference comparison story
    fa_rs = [_load_metrics(s)["parameter_metrics"]["FA"]["Pearson"] for s in FOCUS]

    fig = plt.figure(figsize=(11.0, 4.5))
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)

    ax0.scatter(r_s, p_s, s=2.5, alpha=0.18, c=BLUE_L, rasterized=True, edgecolors="none")
    lo, hi = 0.0, 1.0
    ax0.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x")
    ax0.set_xlim(lo, hi)
    ax0.set_ylim(lo, hi)
    ax0.set_aspect("equal", adjustable="box")
    ax0.set_xlabel("Reference FA (WLS)")
    ax0.set_ylabel("Predicted FA (Single-QINR)")
    ax0.set_title(f"Subject {DEMO_SUBJ}\nPearson r = {cc:.3f}")
    ax0.legend(loc="upper left", fontsize=8, frameon=False)
    ax0.grid(True, alpha=0.25)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    x = np.arange(len(FOCUS))
    bars = ax1.bar(x, fa_rs, width=0.55, color=BLUE_L, edgecolor=BLUE, linewidth=1.2)
    ax1.axhline(0.85, color=ACCENT, ls="--", lw=1.3, label="target r = 0.85")
    for b, v in zip(bars, fa_rs):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=10, color=BLUE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(FOCUS_LABELS)
    ax1.set_ylabel("FA Pearson r vs WLS")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("FA correlation across subjects\n(Single-QINR, 50% sampling)")
    ax1.legend(loc="upper right", fontsize=8, frameon=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.suptitle(
        "FA correlation confirms parameter instability under sparse sampling",
        fontsize=13,
        fontweight="bold",
        color=BLUE,
        y=1.02,
    )
    fig.tight_layout()
    path = OUT / "fig3_fa_scatter_correlation.png"
    fig.savefig(path)
    plt.close(fig)
    return path, {"FA_r_demo": cc, "FA_r_all": fa_rs}


# ===========================================================================
# Fig 4 — Tensor drift boxplot (Phase 9, λ_dis=0 signal adaptation)
# ===========================================================================
def fig4_tensor_drift():
    df = pd.read_csv(PHASE9 / "metrics" / "phase9_final_metrics.csv")
    # Signal-only adaptation (lambda_dis=0): shows unconstrained D drift
    sub = df[np.isclose(df["lambda_dis"], 0.0)].copy()
    # Prefer Phase7 absolute Frobenius delta for box-like multi-subject view
    p7 = pd.read_csv(PHASE7 / "metrics" / "phase7_subject_summary.csv")
    p7 = p7[np.isclose(p7["sampling_fraction"], 1.0)]  # full volumes split obs/holdout

    subjects = list(sub["subject"].astype(str))
    # Voxel-level distributions not stored — use subject-level mean_rel_D_drift
    # and show as bar + swarm; also include abs drift for scale.
    rel = sub["mean_rel_D_drift"].to_numpy()
    abs_d = sub["mean_abs_D_drift"].to_numpy() * 1e4  # scale for readability
    delta_fa = sub["delta_FA_MAE"].to_numpy()
    d_psnr = sub["delta_PSNR_holdout"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    # Phase7 per-subject mean Frobenius tensor change during signal adaptation
    drift_vals = p7["mean_delta_D_fro_vs_z0"].to_numpy() * 1e4
    labels = [f"S{i+1}\n{s}" for i, s in enumerate(p7["subject_id"].astype(str))]
    xpos = np.arange(len(drift_vals))
    ax.bar(xpos, drift_vals, width=0.55, color=BLUE_LL, edgecolor=BLUE, linewidth=1.2)
    ax.scatter(xpos, drift_vals, c=ACCENT, s=55, zorder=5)
    for xi, v in zip(xpos, drift_vals):
        ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=BLUE)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mean ‖ΔD‖$_F$  (×10$^{-4}$)")
    ax.set_title("Tensor drift during signal adaptation\n(z = 0 → adapted z)")
    ax.set_ylim(0, max(drift_vals) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    x = np.arange(len(subjects))
    w = 0.35
    b1 = ax.bar(x - w / 2, d_psnr, width=w, color=BLUE_L, edgecolor=BLUE, label="Δ holdout PSNR ↑")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, delta_fa, width=w, color=ACCENT, edgecolor="#8B1E1E", alpha=0.85, label="Δ FA MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i+1}\n{s}" for i, s in enumerate(subjects)])
    ax.set_ylabel("Δ holdout PSNR (dB)", color=BLUE)
    ax2.set_ylabel("Δ FA MAE (vs z=0)", color=ACCENT)
    ax.set_title("Signal ↑ but FA does not improve\n(Phase 9, λ_dis = 0)")
    ax.spines["top"].set_visible(False)
    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False)

    fig.suptitle(
        "Signal-driven adaptation induces tensor drift without FA recovery",
        fontsize=13,
        fontweight="bold",
        color=BLUE,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.03,
        "Evidence of parameter ambiguity: many D tensors can fit the same sparse signals",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    fig.tight_layout()
    path = OUT / "fig4_tensor_drift.png"
    fig.savefig(path)
    plt.close(fig)

    # Compact single-panel version for PPT (bar + overall box of subject means)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    xpos = np.arange(len(drift_vals))
    ax.bar(xpos, drift_vals, width=0.55, color=BLUE_LL, edgecolor=BLUE, linewidth=1.2)
    ax.scatter(xpos, drift_vals, c=ACCENT, s=70, zorder=5)
    for xi, v in zip(xpos, drift_vals):
        ax.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=BLUE)
    # side box summarizing cross-subject distribution
    bp = ax.boxplot(
        [drift_vals],
        positions=[len(drift_vals) + 0.8],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=ACCENT, linewidth=1.5),
        boxprops=dict(facecolor="#F5E6E6", edgecolor=ACCENT),
        whiskerprops=dict(color=ACCENT),
        capprops=dict(color=ACCENT),
    )
    ax.set_xticks(list(xpos) + [len(drift_vals) + 0.8])
    ax.set_xticklabels([f"S{i}\n{s}" for i, s in enumerate(p7["subject_id"].astype(str), 1)] + ["all"])
    ax.set_ylabel(r"Tensor drift  mean ‖ΔD‖$_F$  (×10$^{-4}$)")
    ax.set_title("Tensor drift across subjects\n(signal adaptation)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path_b = OUT / "fig4b_tensor_drift_boxplot.png"
    fig.savefig(path_b)
    plt.close(fig)

    return path, path_b, {
        "drift": drift_vals.tolist(),
        "delta_psnr": d_psnr.tolist(),
        "delta_fa_mae": delta_fa.tolist(),
        "mean_rel_D_drift": rel.tolist(),
    }


# ===========================================================================
# Fig 5 — Phase 9 λ_dis ablation
# ===========================================================================
def fig5_lambda_ablation():
    df = pd.read_csv(PHASE9 / "metrics" / "phase9_final_metrics.csv")
    lambdas = sorted(df["lambda_dis"].unique())
    fa_means, fa_stds = [], []
    abs_means, abs_stds = [], []
    dpsnr_means = []
    for lam in lambdas:
        g = df[np.isclose(df["lambda_dis"], lam)]
        fa_means.append(g["FA_MAE"].mean())
        fa_stds.append(g["FA_MAE"].std(ddof=0))
        # absolute drift ×1e4 — avoids relative-drift outlier dominating scale
        a = g["mean_abs_D_drift"].to_numpy() * 1e4
        abs_means.append(float(np.mean(a)))
        abs_stds.append(float(np.std(a)))
        dpsnr_means.append(g["delta_PSNR_holdout"].mean())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    ax = axes[0]
    for sid, grp in df.groupby("subject"):
        g = grp.sort_values("lambda_dis")
        ax.plot(g["lambda_dis"], g["FA_MAE"], "o-", color=BLUE_LL, alpha=0.5, lw=1.2, ms=5)
    ax.errorbar(
        lambdas,
        fa_means,
        yerr=fa_stds,
        fmt="o-",
        color=ACCENT,
        lw=2.2,
        ms=9,
        capsize=4,
        zorder=5,
        label="mean ± std",
    )
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.set_xlabel(r"Regularization strength  $\lambda_{\mathrm{dis}}$")
    ax.set_ylabel("FA MAE vs WLS")
    ax.set_title("FA error vs λ  (almost flat)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    for sid, grp in df.groupby("subject"):
        g = grp.sort_values("lambda_dis")
        ax.plot(
            g["lambda_dis"],
            g["mean_abs_D_drift"] * 1e4,
            "s-",
            color=BLUE_LL,
            alpha=0.5,
            lw=1.2,
            ms=5,
        )
    ax.errorbar(
        lambdas,
        abs_means,
        yerr=abs_stds,
        fmt="s-",
        color=BLUE,
        lw=2.2,
        ms=9,
        capsize=4,
        zorder=5,
        label="mean ± std",
    )
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.set_xlabel(r"Regularization strength  $\lambda_{\mathrm{dis}}$")
    ax.set_ylabel(r"Mean ‖ΔD‖$_F$  (×10$^{-4}$)")
    ax.set_title("Tensor drift vs λ  (almost flat)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Regularization reduces fluctuation little — cannot remove parameter ambiguity",
        fontsize=12,
        fontweight="bold",
        color=BLUE,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        r"Phase 9 · $\lambda_{\mathrm{dis}}\in\{0, 0.01, 0.1\}$ · 4 subjects · decision: NO_GO",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    fig.tight_layout()
    path = OUT / "fig5_phase9_lambda_ablation.png"
    fig.savefig(path)
    plt.close(fig)

    # Simple FA-error-only line (user sketch style)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for sid, grp in df.groupby("subject"):
        g = grp.sort_values("lambda_dis")
        ax.plot(g["lambda_dis"], g["FA_MAE"], "o-", ms=7, lw=1.5, label=str(sid))
    ax.plot(lambdas, fa_means, "k--", lw=2, label="mean")
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("FA error (MAE)")
    ax.set_title("λ ablation: FA error barely changes")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path_b = OUT / "fig5b_lambda_fa_error.png"
    fig.savefig(path_b)
    plt.close(fig)

    return path, path_b, {
        "lambdas": lambdas,
        "fa_mae_mean": fa_means,
        "abs_drift_mean_x1e4": abs_means,
        "delta_psnr_mean": dpsnr_means,
    }


def main():
    summary = {}
    print("Generating Fig1...")
    p1, p1b, s1 = fig1_signal_vs_fa()
    summary["fig1"] = {"paths": [str(p1), str(p1b)], **s1}
    print("  ->", p1)
    print("  ->", p1b)

    print("Generating Fig2...")
    p2, s2 = fig2_fa_maps()
    summary["fig2"] = {"path": str(p2), **s2}
    print("  ->", p2)

    print("Generating Fig3...")
    p3, s3 = fig3_fa_scatter()
    summary["fig3"] = {"path": str(p3), **s3}
    print("  ->", p3)

    print("Generating Fig4...")
    p4, p4b, s4 = fig4_tensor_drift()
    summary["fig4"] = {"paths": [str(p4), str(p4b)], **s4}
    print("  ->", p4)
    print("  ->", p4b)

    print("Generating Fig5...")
    p5, p5b, s5 = fig5_lambda_ablation()
    summary["fig5"] = {"paths": [str(p5), str(p5b)], **{k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v) for k, v in s5.items()}}
    print("  ->", p5)
    print("  ->", p5b)

    meta_path = OUT / "figure_summary.json"
    # make JSON-serializable
    def _ser(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: _ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_ser(x) for x in o]
        return o

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_ser(summary), f, indent=2)
    print("Wrote", meta_path)
    print("Done. All figures in:", OUT)


if __name__ == "__main__":
    main()
