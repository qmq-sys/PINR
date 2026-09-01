#!/usr/bin/env python
"""Phase 9: Latent disentanglement / DTI-drift regularization (diagnostic only).

Fixed theta = Phase4-A epoch_0150. No architecture / physics / baseline changes.
WLS is evaluation-only — never in the adaptation objective.

Protocol (same as Phase 7/8 @ 50%):
  - load full DTI shell
  - all b0 → observed
  - 50% non-b0 → observed; remainder → holdout
  - optimize z_new only from z=0

Objective:
  L = L_signal + lambda_dis * L_dis + lambda_z * ||z||^2
  L_dis = mean_x ||D(x,z)-D(x,0)||_F^2 / (||D(x,0)||_F^2 + eps)
  with D(x,0) detached (no grad through z=0 reference).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import SubjectBundle, load_subject_bundle, s0_obs_from_batch
from data.split import split_from_config
from metrics.evaluator import compute_dti_scalars_from_D, dti_parameter_metrics, signal_metrics
from models.population_dti_inr import PopulationDTIINR
from physics.dti_forward import predict_signal
from predict import predict_maps
from utils_io import (
    _write_csv,
    load_theta,
    load_yaml,
    make_experiment_dir,
    resolve_device,
    save_json,
    save_yaml,
)

DEFAULT_PHASE4A = ROOT / "experiments" / "population_dti_phase4a" / "20260829_192714"
EPS = 1e-8
KEY_ITERS = (0, 10, 50, 100, 200, 500, 1000)
SUBJECTS = ("106319", "120717", "121618", "116726")
OBS_RATIO = 0.5
LAMBDA_DIS_GRID = (0.0, 0.01, 0.1)
# Phase 8 protocol default: lambda_z = 0 unless explicitly set in cfg
DEFAULT_LAMBDA_Z = 0.0


def _build_model(cfg: dict[str, Any], train_ids: list[str], device: torch.device) -> PopulationDTIINR:
    return PopulationDTIINR(
        train_subject_ids=train_ids,
        latent_dim=int(cfg.get("latent_dim", 16)),
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 4)),
        pe_freqs=int(cfg.get("pe_freqs", 8)),
    ).to(device)


def _fro(D: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(D.astype(np.float64) ** 2, axis=(-2, -1)))


def latent_norm(z: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(z.detach().float()).cpu())


def split_by_observed_ratio(
    bvals: np.ndarray,
    *,
    b0_threshold: float,
    observed_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Identical to Phase 8: all b0 observed; observed_ratio of non-b0 observed."""
    bvals = np.asarray(bvals).reshape(-1)
    b0 = bvals < float(b0_threshold)
    b0_idx = np.flatnonzero(b0)
    dwi_idx = np.flatnonzero(~b0).copy()
    rng = np.random.default_rng(int(seed))
    rng.shuffle(dwi_idx)

    ratio = float(observed_ratio)
    if ratio >= 1.0 - 1e-12:
        n_obs_dwi = len(dwi_idx)
    else:
        n_obs_dwi = int(round(len(dwi_idx) * ratio))
        n_obs_dwi = min(max(n_obs_dwi, 1), max(len(dwi_idx) - 1, 1))

    obs_dwi = dwi_idx[:n_obs_dwi]
    hold = dwi_idx[n_obs_dwi:]
    obs = np.sort(np.concatenate([b0_idx, obs_dwi])).astype(np.int64)
    hold = np.sort(hold).astype(np.int64)
    if obs.size == 0:
        raise RuntimeError("observed empty")
    return obs, hold


def loss_signal(pred: torch.Tensor, target: torch.Tensor, bvals: torch.Tensor, cfg: dict) -> torch.Tensor:
    b0 = float(cfg["b0_threshold"])
    s0 = s0_obs_from_batch(target, bvals, b0)
    return F.mse_loss(pred / s0, target / s0)


def loss_dis_relative(D_z: torch.Tensor, D0_det: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    L_dis = mean ||D(z)-D(0)||_F^2 / (||D(0)||_F^2 + eps)
    D0_det must be detached; D_z keeps grad w.r.t. z.
    """
    diff = D_z - D0_det
    num = (diff * diff).sum(dim=(-2, -1))
    den = (D0_det * D0_det).sum(dim=(-2, -1)) + float(eps)
    return (num / den).mean()


@torch.no_grad()
def eval_signal_channels(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    z: torch.Tensor,
    device: torch.device,
    cfg: dict[str, Any],
    channel_idx: np.ndarray,
    *,
    seed: int = 42,
    max_voxels: int = 65536,
) -> dict[str, float]:
    if channel_idx.size == 0:
        return {"PSNR": float("nan"), "RelMSE": float("nan"), "MSE": float("nan")}

    model.eval()
    b0_thr = float(cfg["b0_threshold"])
    b_scale = float(cfg.get("b_scale", 1.0))
    max_signal = float(cfg.get("max_signal", 1.0))
    rng = np.random.default_rng(seed)
    n = int(subj.eval_coords.shape[0])
    sel = np.arange(n) if n <= max_voxels else rng.choice(n, size=max_voxels, replace=False)

    xyz = torch.from_numpy(subj.eval_coords[sel]).to(device)
    flat = subj.eval_flat_idx[sel]
    target_full = torch.from_numpy(subj.dwi_flat[flat]).to(device)
    bvals_all = torch.from_numpy(subj.bvals).to(device)
    bvecs_all = torch.from_numpy(subj.bvecs).to(device)
    S0, D = model(xyz, z=z.to(device))
    pred_full = predict_signal(S0, D, bvals_all, bvecs_all, b_scale=b_scale)
    s0 = s0_obs_from_batch(target_full, bvals_all, b0_thr)
    ch = torch.as_tensor(channel_idx, device=device, dtype=torch.long)
    pred_n = (pred_full.index_select(-1, ch) / s0).detach().cpu().numpy()
    obs_n = (target_full.index_select(-1, ch) / s0).detach().cpu().numpy()
    return signal_metrics(pred_n, obs_n, max_signal=max_signal, compute_ssim=False)


@torch.no_grad()
def snapshot(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    z: torch.Tensor,
    D0_vol: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    iteration: int,
    obs_idx: np.ndarray,
    hold_idx: np.ndarray,
    *,
    loss_parts: dict[str, float] | None = None,
) -> dict[str, Any]:
    model.eval()
    model.freeze_theta()
    z = z.detach()
    sig_obs = eval_signal_channels(model, subj, z, device, cfg, obs_idx, seed=int(cfg.get("seed", 42)))
    sig_hold = eval_signal_channels(model, subj, z, device, cfg, hold_idx, seed=int(cfg.get("seed", 42)))

    maps = predict_maps(
        model, subj.train_coords, subj.train_flat_idx, subj.shape_xyz, z.to(device), device, want_D=True
    )
    dti = dti_parameter_metrics(
        {"FA": maps["FA"], "MD": maps["MD"], "AD": maps["AD"], "RD": maps["RD"]},
        {"FA": subj.ref["FA"], "MD": subj.ref["MD"], "AD": subj.ref["AD"], "RD": subj.ref["RD"]},
        subj.common_mask,
    )
    mask = subj.common_mask
    Dv = maps["D"][mask].astype(np.float64)
    D0v = D0_vol[mask].astype(np.float64)
    abs_drift = _fro(Dv - D0v)
    rel_drift = abs_drift / (_fro(D0v) + EPS)
    scalars = compute_dti_scalars_from_D(Dv)

    # eigenvalues descending for optional logging
    Df = 0.5 * (Dv + np.swapaxes(Dv, -1, -2))
    ev = np.linalg.eigvalsh(np.nan_to_num(Df, nan=0.0) + 1e-12 * np.eye(3))
    ev = np.clip(ev[..., ::-1], 0.0, None)

    row = {
        "iteration": int(iteration),
        "z_norm": latent_norm(z),
        "PSNR_obs": float(sig_obs["PSNR"]),
        "RelMSE_obs": float(sig_obs["RelMSE"]),
        "PSNR_holdout": float(sig_hold["PSNR"]),
        "RelMSE_holdout": float(sig_hold["RelMSE"]),
        "FA_MAE": float(dti["FA"]["MAE"]),
        "MD_MAE": float(dti["MD"]["MAE"]),
        "AD_MAE": float(dti["AD"]["MAE"]),
        "RD_MAE": float(dti["RD"]["MAE"]),
        "mean_abs_D_drift": float(np.mean(abs_drift)),
        "mean_rel_D_drift": float(np.mean(rel_drift)),
        "mean_lambda1": float(np.mean(ev[..., 0])),
        "mean_lambda2": float(np.mean(ev[..., 1])),
        "mean_lambda3": float(np.mean(ev[..., 2])),
        "mean_FA_pred": float(np.mean(scalars["FA"])),
        "n_observed_vols": int(obs_idx.size),
        "n_holdout_vols": int(hold_idx.size),
    }
    if loss_parts:
        row.update({f"loss_{k}": v for k, v in loss_parts.items()})
    return row


def run_adaptation(
    *,
    model: PopulationDTIINR,
    subj: SubjectBundle,
    cfg: dict[str, Any],
    device: torch.device,
    obs_idx: np.ndarray,
    hold_idx: np.ndarray,
    D0_vol: np.ndarray,
    lambda_dis: float,
    lambda_z: float,
    max_iter: int,
    key_iters: tuple[int, ...],
    seed: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model.freeze_theta()
    z = model.new_z(trainable=True, device=device, init="zeros")
    z0 = model.zero_z(device=device)  # fixed zero reference (not a Parameter)
    opt = torch.optim.Adam([z], lr=float(cfg.get("adapt_lr", cfg.get("lr", 1e-3))))
    batch = int(cfg.get("batch_voxels", 4096))
    b_scale = float(cfg.get("b_scale", 1.0))

    bvals_obs = torch.from_numpy(subj.bvals[obs_idx]).to(device)
    bvecs_obs = torch.from_numpy(subj.bvecs[obs_idx]).to(device)
    coords_t = torch.from_numpy(subj.train_coords)
    obs_np = np.asarray(obs_idx, dtype=np.int64)

    max_iter = int(max_iter)
    key_set = sorted({int(x) for x in key_iters if int(x) <= max_iter})
    if max_iter not in key_set:
        key_set.append(max_iter)
    key_lookup = set(key_set)
    rows: list[dict[str, Any]] = []
    last_parts: dict[str, float] = {}

    def record(it: int) -> None:
        m = snapshot(
            model, subj, z, D0_vol, cfg, device, it, obs_idx, hold_idx, loss_parts=last_parts or None
        )
        rows.append(m)
        print(
            f"    [λ_dis={lambda_dis} λz={lambda_z}] it={it:4d} "
            f"||z||={m['z_norm']:.4f} hold={m['PSNR_holdout']:.3f} "
            f"FA={m['FA_MAE']:.4f} relDrift={m['mean_rel_D_drift']:.4e}",
            flush=True,
        )

    if 0 in key_lookup:
        record(0)

    for it in range(1, max_iter + 1):
        model.train()
        model.freeze_theta()
        opt.zero_grad(set_to_none=True)

        sel = rng.integers(0, int(coords_t.shape[0]), size=batch, endpoint=False)
        xyz = coords_t[sel].to(device)
        flat = subj.train_flat_idx[sel]
        target = torch.from_numpy(subj.dwi_flat[flat][:, obs_np]).to(device)

        # D(x,0) reference — no grad
        with torch.no_grad():
            _, D0_b = model(xyz, z=z0)
            D0_b = D0_b.detach()

        S0, D = model(xyz, z=z)
        pred = predict_signal(S0, D, bvals_obs, bvecs_obs, b_scale=b_scale)
        Ls = loss_signal(pred, target, bvals_obs, cfg)
        Ld = loss_dis_relative(D, D0_b, eps=EPS)
        Lz = torch.sum(z.float() ** 2)
        loss = Ls + float(lambda_dis) * Ld + float(lambda_z) * Lz

        loss.backward()
        for p in model.theta_parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                raise RuntimeError("theta received gradients — forbidden in Phase 9")
        opt.step()

        last_parts = {
            "signal": float(Ls.detach().cpu()),
            "dis": float(Ld.detach().cpu()),
            "z": float(Lz.detach().cpu()),
            "total": float(loss.detach().cpu()),
        }
        if it in key_lookup:
            record(it)

    return rows, z.detach().clone()


def classify_run(r0: dict[str, Any], rf: dict[str, Any]) -> dict[str, Any]:
    # Spec: PSNR_holdout_final > PSNR_holdout_z0 (raw, no extra margin)
    hold_improve = bool(rf["PSNR_holdout"] > r0["PSNR_holdout"])
    dti_stable = bool(rf["FA_MAE"] <= r0["FA_MAE"] + 0.005)
    dti_improved = bool(rf["FA_MAE"] < r0["FA_MAE"])
    obs_improve = bool(rf["PSNR_obs"] > r0["PSNR_obs"])
    hold_worse = bool(rf["PSNR_holdout"] < r0["PSNR_holdout"])
    overfit = bool(obs_improve and hold_worse)
    # λ_dis kills both drift and signal gain → over-regularized
    over_reg = bool(rf["mean_rel_D_drift"] < 1e-4) and (not hold_improve)

    return {
        "signal_holdout_improve": hold_improve,
        "dti_stable": dti_stable,
        "dti_improved": dti_improved,
        "overfit_obs_up_hold_down": overfit,
        "over_regularized": over_reg,
        "delta_PSNR_holdout": float(rf["PSNR_holdout"] - r0["PSNR_holdout"]),
        "delta_FA_MAE": float(rf["FA_MAE"] - r0["FA_MAE"]),
        "delta_rel_D_drift": float(rf["mean_rel_D_drift"] - r0["mean_rel_D_drift"]),
    }


def run_one(
    *,
    model: PopulationDTIINR,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt: Path,
    trad: Path,
    subject_id: str,
    lambda_dis: float,
    lambda_z: float,
    max_iter: int,
    seed: int,
    exp_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()
    subj = load_subject_bundle(
        subject_id=subject_id, cfg=cfg, trad_dir=trad / subject_id, sampling_fraction=1.0
    )
    obs_idx, hold_idx = split_by_observed_ratio(
        subj.bvals,
        b0_threshold=float(cfg["b0_threshold"]),
        observed_ratio=OBS_RATIO,
        seed=int(seed) + int(subject_id) % 10000,
    )
    print(
        f"\n=== {subject_id} λ_dis={lambda_dis} λz={lambda_z} | "
        f"vols={subj.n_volumes} obs={obs_idx.size} hold={hold_idx.size} ===",
        flush=True,
    )

    maps0 = predict_maps(
        model,
        subj.train_coords,
        subj.train_flat_idx,
        subj.shape_xyz,
        model.zero_z(device=device),
        device,
        want_D=True,
    )
    D0_vol = maps0["D"]

    rows, zf = run_adaptation(
        model=model,
        subj=subj,
        cfg=cfg,
        device=device,
        obs_idx=obs_idx,
        hold_idx=hold_idx,
        D0_vol=D0_vol,
        lambda_dis=float(lambda_dis),
        lambda_z=float(lambda_z),
        max_iter=max_iter,
        key_iters=KEY_ITERS,
        seed=seed,
    )
    meta = {
        "subject": subject_id,
        "sampling_ratio": OBS_RATIO,
        "lambda_dis": float(lambda_dis),
        "lambda_z": float(lambda_z),
    }
    for r in rows:
        r.update(meta)

    r0, rf = rows[0], rows[-1]
    flags = classify_run(r0, rf)
    final = {
        **meta,
        **{k: rf[k] for k in (
            "z_norm", "PSNR_obs", "PSNR_holdout", "RelMSE_obs", "RelMSE_holdout",
            "FA_MAE", "MD_MAE", "AD_MAE", "RD_MAE",
            "mean_abs_D_drift", "mean_rel_D_drift",
            "mean_lambda1", "mean_lambda2", "mean_lambda3",
        )},
        "PSNR_obs_z0": r0["PSNR_obs"],
        "PSNR_holdout_z0": r0["PSNR_holdout"],
        "FA_MAE_z0": r0["FA_MAE"],
        "MD_MAE_z0": r0["MD_MAE"],
        "mean_rel_D_drift_z0": r0["mean_rel_D_drift"],
        "mean_abs_D_drift_z0": r0["mean_abs_D_drift"],
        **flags,
    }

    tag = f"{subject_id}_ldis{lambda_dis}_lz{lambda_z}"
    _write_csv(exp_dir / "runs" / f"{tag}_trajectory.csv", rows)
    torch.save({"z": zf.cpu(), **meta}, exp_dir / "runs" / f"{tag}_z.pt")
    save_json(exp_dir / "runs" / f"{tag}_final.json", final)
    return rows, final


def make_plots(
    all_traj: list[dict[str, Any]],
    finals: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {0.0: "C0", 0.01: "C1", 0.1: "C2"}
    subjects = sorted({str(f["subject"]) for f in finals}, key=lambda s: SUBJECTS.index(s) if s in SUBJECTS else s)

    # per-subject trajectories: holdout PSNR + FA
    for sid in subjects:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plotted = False
        for ld in LAMBDA_DIS_GRID:
            rows = [
                r
                for r in all_traj
                if r["subject"] == sid and abs(float(r["lambda_dis"]) - ld) < 1e-12
            ]
            if not rows:
                continue
            plotted = True
            xs = [r["iteration"] for r in rows]
            axes[0].plot(xs, [r["PSNR_holdout"] for r in rows], marker="o", color=colors[ld], label=f"λ_dis={ld}")
            axes[1].plot(xs, [r["FA_MAE"] for r in rows], marker="o", color=colors[ld], label=f"λ_dis={ld}")
        axes[0].set_title(f"{sid}: PSNR holdout")
        axes[1].set_title(f"{sid}: FA MAE")
        for ax in axes:
            ax.set_xlabel("iteration")
            ax.grid(True, alpha=0.3)
            if plotted:
                ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"subject_{sid}_trajectory.png", dpi=150)
        plt.close(fig)

    # Figure aggregate: PSNR holdout vs FA across finals
    fig, ax = plt.subplots(figsize=(6, 5))
    for ld in LAMBDA_DIS_GRID:
        pts = [f for f in finals if abs(float(f["lambda_dis"]) - ld) < 1e-12]
        if not pts:
            continue
        ax.scatter(
            [f["PSNR_holdout"] for f in pts],
            [f["FA_MAE"] for f in pts],
            s=70,
            label=f"λ_dis={ld}",
            color=colors[ld],
        )
        for f in pts:
            ax.annotate(f["subject"][-3:], (f["PSNR_holdout"], f["FA_MAE"]), fontsize=7)
    ax.set_xlabel("PSNR holdout @final")
    ax.set_ylabel("FA MAE @final")
    ax.set_title("Phase9: holdout PSNR vs FA")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase9_psnr_vs_fa.png", dpi=150)
    plt.close(fig)

    # drift vs FA
    fig, ax = plt.subplots(figsize=(6, 5))
    for ld in LAMBDA_DIS_GRID:
        pts = [f for f in finals if abs(float(f["lambda_dis"]) - ld) < 1e-12]
        if not pts:
            continue
        ax.scatter(
            [f["mean_rel_D_drift"] for f in pts],
            [f["FA_MAE"] for f in pts],
            s=70,
            label=f"λ_dis={ld}",
            color=colors[ld],
        )
    ax.set_xlabel("mean relative D drift vs z=0")
    ax.set_ylabel("FA MAE")
    ax.set_title("Phase9: DTI drift vs FA")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase9_drift_vs_fa.png", dpi=150)
    plt.close(fig)

    # lambda comparison bars
    if not subjects:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("delta_PSNR_holdout", "Δ holdout PSNR"),
        ("delta_FA_MAE", "Δ FA MAE"),
        ("mean_rel_D_drift", "rel D drift"),
    ]
    x = np.arange(len(subjects))
    w = 0.25
    for ax, (key, title) in zip(axes, metrics):
        for i, ld in enumerate(LAMBDA_DIS_GRID):
            vals = []
            for sid in subjects:
                f = next(
                    (f for f in finals if f["subject"] == sid and abs(float(f["lambda_dis"]) - ld) < 1e-12),
                    None,
                )
                vals.append(float(f[key]) if f is not None else float("nan"))
            ax.bar(x + (i - 1) * w, vals, w, label=f"λ={ld}", color=colors[ld])
        ax.set_xticks(x)
        ax.set_xticklabels([s[-3:] for s in subjects])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Phase9 lambda comparison @final")
    fig.tight_layout()
    fig.savefig(out_dir / "phase9_lambda_comparison.png", dpi=150)
    plt.close(fig)


def diagnose(finals: list[dict[str, Any]]) -> dict[str, Any]:
    by_ld: dict[float, list[dict]] = {ld: [] for ld in LAMBDA_DIS_GRID}
    for f in finals:
        by_ld[float(f["lambda_dis"])].append(f)

    n_subj = max(1, len({f["subject"] for f in finals}))
    # Q1: does λ_dis reduce drift vs λ=0?
    mean_drift = {
        ld: float(np.mean([f["mean_rel_D_drift"] for f in xs])) if xs else float("nan")
        for ld, xs in by_ld.items()
    }
    q1 = bool(
        (by_ld[0.01] and mean_drift[0.01] < mean_drift[0.0] * 0.9)
        or (by_ld[0.1] and mean_drift[0.1] < mean_drift[0.0] * 0.9)
    )

    # Q2: FA degradation reduced when drift reduced?
    mean_dfa = {
        ld: float(np.mean([f["delta_FA_MAE"] for f in xs])) if xs else float("nan")
        for ld, xs in by_ld.items()
    }
    q2 = bool(
        (by_ld[0.01] and mean_dfa[0.01] < mean_dfa[0.0] - 1e-4)
        or (by_ld[0.1] and mean_dfa[0.1] < mean_dfa[0.0] - 1e-4)
    )

    # Q3: holdout still improves under some λ_dis>0?
    q3 = any(f["signal_holdout_improve"] for ld in (0.01, 0.1) for f in by_ld[ld])

    # Spec success: holdout↑ AND DTI stable (eval vs z=0)
    def subject_ok(f: dict) -> bool:
        return bool(f["signal_holdout_improve"] and f["dti_stable"])

    def subject_fa_imp(f: dict) -> bool:
        return bool(f["dti_improved"])

    per_lambda = {}
    for ld, xs in by_ld.items():
        oks = [f for f in xs if subject_ok(f)]
        fa_imps = [f for f in xs if subject_fa_imp(f)]
        per_lambda[str(ld)] = {
            "n_subjects": len(xs),
            "n_holdout_improve": sum(1 for f in xs if f["signal_holdout_improve"]),
            "n_dti_stable": sum(1 for f in xs if f["dti_stable"]),
            "n_ok_holdout_and_stable": len(oks),
            "subjects_ok": [f["subject"] for f in oks],
            "n_fa_improved": len(fa_imps),
            "subjects_fa_improved": [f["subject"] for f in fa_imps],
            "n_over_regularized": sum(1 for f in xs if f["over_regularized"]),
            "n_overfit": sum(1 for f in xs if f["overfit_obs_up_hold_down"]),
            "mean_rel_D_drift": mean_drift[ld],
            "mean_delta_FA_MAE": mean_dfa[ld],
            "mean_delta_PSNR_holdout": float(np.mean([f["delta_PSNR_holdout"] for f in xs])) if xs else float("nan"),
            "mean_z_norm": float(np.mean([f["z_norm"] for f in xs])) if xs else float("nan"),
        }

    nonempty = [ld for ld in LAMBDA_DIS_GRID if by_ld[ld]]
    best_ld = max(
        nonempty,
        key=lambda ld: (
            per_lambda[str(ld)]["n_ok_holdout_and_stable"],
            per_lambda[str(ld)]["n_fa_improved"],
            -abs(per_lambda[str(ld)]["mean_delta_FA_MAE"]),
        ),
    )
    best = per_lambda[str(best_ld)]
    # thresholds use 3/4 and 2/4 of the planned subject set when full; scale if partial smoke
    need_ok = 3 if n_subj >= 4 else max(1, int(np.ceil(0.75 * n_subj)))
    need_fa = 2 if n_subj >= 4 else max(1, int(np.ceil(0.5 * n_subj)))

    strong = bool(best["n_ok_holdout_and_stable"] >= need_ok and best["n_fa_improved"] >= need_fa)
    conditional = bool(best["n_ok_holdout_and_stable"] >= need_ok and best["n_fa_improved"] < need_fa)

    no_go = False
    if best["n_ok_holdout_and_stable"] < need_ok:
        if all(per_lambda[str(ld)]["n_over_regularized"] >= need_ok for ld in (0.01, 0.1) if by_ld[ld]):
            no_go = True
        elif (not by_ld[0.0] or per_lambda["0.0"]["n_ok_holdout_and_stable"] == 0) and best[
            "n_ok_holdout_and_stable"
        ] == 0:
            no_go = True

    if strong:
        decision = "STRONG_GO"
    elif conditional:
        decision = "CONDITIONAL_GO"
    elif no_go:
        decision = "NO_GO"
    else:
        if best["n_ok_holdout_and_stable"] >= 1 and q1 and q2:
            decision = "CONDITIONAL_GO"
        else:
            decision = "NO_GO"

    # mechanism + Q7
    drift0 = mean_drift.get(0.0, float("nan"))
    near_null_lambda = bool(
        by_ld[0.01]
        and by_ld[0.1]
        and drift0 == drift0
        and abs(mean_drift[0.01] - drift0) / max(abs(drift0), 1e-12) < 0.05
        and abs(mean_drift[0.1] - drift0) / max(abs(drift0), 1e-12) < 0.05
    )

    if decision == "STRONG_GO":
        mechanism = "disentanglement_regularization_enables_signal_gain_with_DTI_stability"
        q7 = "adaptation_objective_problem_partially_mitigated_by_disentanglement"
    elif decision == "CONDITIONAL_GO":
        mechanism = "partial_disentanglement_helps_DTI_stability_but_not_systematic_FA_gain"
        q7 = "adaptation_objective_problem_partially_mitigated_by_disentanglement"
    else:
        if all(
            per_lambda[str(ld)]["n_over_regularized"] >= max(1, need_ok - 1)
            for ld in (0.01, 0.1)
            if by_ld[ld]
        ):
            mechanism = "adaptation_objective_over_regularization_kills_signal_with_DTI_stability"
            q7 = "adaptation_objective_and_latent_identifiability_problem"
        elif near_null_lambda:
            # prescribed λ too weak vs L_signal; Phase5 oracle implies capacity exists
            mechanism = (
                "prescribed_lambda_dis_insufficient_vs_L_signal__"
                "signal_adaptation_still_mismatches_DTI"
            )
            q7 = "adaptation_objective_and_latent_identifiability_problem"
        elif not q1:
            mechanism = "representation_or_pathway_issue_L_dis_cannot_control_drift"
            q7 = "latent_identifiability_and_adaptation_objective_problem"
        elif q1 and not q2:
            mechanism = "FA_degradation_not_driven_solely_by_Frobenius_D_drift"
            q7 = "latent_identifiability_and_adaptation_objective_problem"
        else:
            mechanism = "latent_identifiability_adaptation_objective_mismatch_persists"
            q7 = "latent_identifiability_and_adaptation_objective_problem"

    q4_subjects = sorted({f["subject"] for f in finals if subject_ok(f)})
    q5 = bool(best["n_ok_holdout_and_stable"] >= need_ok)
    trend_01 = (
        by_ld[0.01]
        and mean_drift[0.01] < mean_drift[0.0]
        and mean_dfa[0.01] <= mean_dfa[0.0] + 1e-4
    )
    trend_1 = (
        by_ld[0.1]
        and mean_drift[0.1] < mean_drift[0.0]
        and mean_dfa[0.1] <= mean_dfa[0.0] + 1e-4
    )
    q6 = bool(trend_01 and trend_1)

    return {
        "Q1_disentanglement_reduces_DTI_drift": q1,
        "Q2_FA_degradation_reduced_with_drift": q2,
        "Q3_signal_adaptation_still_improves_holdout": q3,
        "Q4_subjects_signal_up_holdout_up_FA_stable": q4_subjects,
        "Q5_reproducible_on_ge_3_of_4": q5,
        "Q6_lambda_0p01_and_0p1_consistent_trend": q6,
        "Q7_failure_type": q7,
        "Q8_decision": decision,
        "best_lambda_dis": best_ld,
        "per_lambda": per_lambda,
        "mean_drift_by_lambda": mean_drift,
        "mean_delta_FA_by_lambda": mean_dfa,
        "mechanism": mechanism,
        "near_null_lambda_effect": near_null_lambda,
        "baseline_lambda0": per_lambda.get("0.0", {}),
        "thresholds": {"need_ok": need_ok, "need_fa": need_fa, "n_subjects": n_subj},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 9 latent disentanglement diagnostic")
    ap.add_argument("--phase4a-dir", default=str(DEFAULT_PHASE4A))
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--lambda-z", type=float, default=None, help="override; default=Phase8 (0)")
    args = ap.parse_args()

    phase4a = Path(args.phase4a_dir)
    cfg = dict(load_yaml(phase4a / "config" / "run_config.yaml"))
    cfg["adapt_lr"] = float(cfg.get("adapt_lr", cfg.get("lr", 1e-3)))
    split = split_from_config(cfg)
    device = resolve_device(str(cfg.get("device", "auto")))
    ckpt = phase4a / "checkpoints" / "epoch_0150"
    trad = Path(cfg["trad_root"])
    lambda_z = float(args.lambda_z) if args.lambda_z is not None else float(cfg.get("adapt_lambda_z", DEFAULT_LAMBDA_Z))

    exp_dir = make_experiment_dir(tag="population_dti_phase9")
    for sub in ("metrics", "plots", "runs", "diagnostics", "logs"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    save_yaml(
        exp_dir / "phase9_config.yaml",
        {
            **cfg,
            "max_iter": args.max_iter,
            "observed_ratio": OBS_RATIO,
            "lambda_dis_grid": list(LAMBDA_DIS_GRID),
            "lambda_z": lambda_z,
            "L_dis": "mean ||D(z)-D(0)||_F^2 / (||D(0)||_F^2 + eps); D(0) detached",
            "WLS": "evaluation only",
            "key_iters": list(KEY_ITERS),
        },
    )
    save_json(
        exp_dir / "config" / "sources.json",
        {"theta": str(ckpt / "theta.pt"), "phase4a": str(phase4a), "note": "no architecture change"},
    )

    print("=" * 72)
    print("Phase 9 — Latent Disentanglement Diagnostic")
    print(f"  theta={ckpt / 'theta.pt'}")
    print(f"  λ_dis ∈ {LAMBDA_DIS_GRID}  λ_z={lambda_z}  ratio={OBS_RATIO}")
    print(f"  out={exp_dir}")
    print("=" * 72)

    if not (ckpt / "theta.pt").is_file():
        raise FileNotFoundError(ckpt / "theta.pt")

    model = _build_model(cfg, split["train"], device)
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()

    subjects = args.subjects or list(SUBJECTS)
    all_traj: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []

    for sid in subjects:
        for ld in LAMBDA_DIS_GRID:
            rows, final = run_one(
                model=model,
                cfg=cfg,
                device=device,
                ckpt=ckpt,
                trad=trad,
                subject_id=sid,
                lambda_dis=float(ld),
                lambda_z=lambda_z,
                max_iter=int(args.max_iter),
                seed=int(args.seed),
                exp_dir=exp_dir,
            )
            all_traj.extend(rows)
            finals.append(final)
            _write_csv(exp_dir / "metrics" / "phase9_trajectory.csv", all_traj)
            _write_csv(exp_dir / "metrics" / "phase9_final_metrics.csv", finals)

    _write_csv(exp_dir / "metrics" / "phase9_trajectory.csv", all_traj)
    _write_csv(exp_dir / "metrics" / "phase9_final_metrics.csv", finals)

    # summary / go-no-go table
    summary_rows = []
    go_rows = []
    for f in finals:
        summary_rows.append(
            {
                "subject": f["subject"],
                "lambda_dis": f["lambda_dis"],
                "lambda_z": f["lambda_z"],
                "z_norm": f["z_norm"],
                "PSNR_holdout_z0": f["PSNR_holdout_z0"],
                "PSNR_holdout": f["PSNR_holdout"],
                "delta_PSNR_holdout": f["delta_PSNR_holdout"],
                "FA_MAE_z0": f["FA_MAE_z0"],
                "FA_MAE": f["FA_MAE"],
                "delta_FA_MAE": f["delta_FA_MAE"],
                "mean_rel_D_drift": f["mean_rel_D_drift"],
                "signal_holdout_improve": f["signal_holdout_improve"],
                "dti_stable": f["dti_stable"],
                "dti_improved": f["dti_improved"],
                "overfit": f["overfit_obs_up_hold_down"],
                "over_regularized": f["over_regularized"],
            }
        )
        go_rows.append(
            {
                "subject": f["subject"],
                "lambda_dis": f["lambda_dis"],
                "holdout_improve": f["signal_holdout_improve"],
                "dti_stable": f["dti_stable"],
                "dti_improved": f["dti_improved"],
                "ok_holdout_and_stable": bool(f["signal_holdout_improve"] and f["dti_stable"]),
            }
        )
    _write_csv(exp_dir / "metrics" / "phase9_summary.csv", summary_rows)
    _write_csv(exp_dir / "metrics" / "phase9_go_no_go.csv", go_rows)

    make_plots(all_traj, finals, exp_dir / "plots")
    diag = diagnose(finals)
    save_json(exp_dir / "diagnostics" / "phase9_diagnosis.json", diag)

    conclusion = {
        "decision": diag["Q8_decision"],
        "best_lambda_dis": diag["best_lambda_dis"],
        "answers": {
            "Q1_disentanglement_reduces_DTI_drift": diag["Q1_disentanglement_reduces_DTI_drift"],
            "Q2_FA_degradation_reduced_with_drift": diag["Q2_FA_degradation_reduced_with_drift"],
            "Q3_signal_still_improves_holdout": diag["Q3_signal_adaptation_still_improves_holdout"],
            "Q4_subjects_signal_holdout_FA_stable": diag["Q4_subjects_signal_up_holdout_up_FA_stable"],
            "Q5_reproducible_3of4": diag["Q5_reproducible_on_ge_3_of_4"],
            "Q6_lambdas_consistent": diag["Q6_lambda_0p01_and_0p1_consistent_trend"],
            "Q7_failure_type": diag["Q7_failure_type"],
            "Q8_decision": diag["Q8_decision"],
        },
        "per_lambda": diag["per_lambda"],
        "mechanism": diag["mechanism"],
        "next": (
            "Do not auto-enter Phase 10 / v2. Review phase9_final_conclusion.json; "
            "human decides whether Population-DTI-INR v1 remains research-worthy."
        ),
        "experiment_dir": str(exp_dir),
        "constraints_honored": {
            "no_architecture_change": True,
            "no_WLS_in_adaptation": True,
            "no_DKI": True,
            "phase4_8_untouched": True,
        },
    }
    save_json(exp_dir / "diagnostics" / "phase9_final_conclusion.json", conclusion)

    print("\n===== PHASE 9 FINAL =====")
    print(f"  decision: {diag['Q8_decision']}")
    print(f"  best λ_dis: {diag['best_lambda_dis']}")
    print(f"  Q1 drift↓: {diag['Q1_disentanglement_reduces_DTI_drift']}")
    print(f"  Q2 FA deg↓: {diag['Q2_FA_degradation_reduced_with_drift']}")
    print(f"  Q3 holdout↑ under λ>0: {diag['Q3_signal_adaptation_still_improves_holdout']}")
    print(f"  Q4 OK subjects: {diag['Q4_subjects_signal_up_holdout_up_FA_stable']}")
    print(f"  Q5 ≥3/4: {diag['Q5_reproducible_on_ge_3_of_4']}")
    print(f"  Q7 type: {diag['Q7_failure_type']}")
    print(f"  mechanism: {diag['mechanism']}")
    for ld in LAMBDA_DIS_GRID:
        pl = diag["per_lambda"][str(ld)]
        print(
            f"  λ={ld}: ok={pl['n_ok_holdout_and_stable']}/4 "
            f"FA↑={pl['n_fa_improved']} drift={pl['mean_rel_D_drift']:.4e} "
            f"ΔFA={pl['mean_delta_FA_MAE']:.4f} Δhold={pl['mean_delta_PSNR_holdout']:.2f}"
        )
    print(f"\n[Phase9] done → {exp_dir}")


if __name__ == "__main__":
    main()
