#!/usr/bin/env python
"""Phase 8: signal-observable DTI-aware latent adaptation (diagnostic only).

Fixed theta = Phase4-A epoch_0150. No architecture / physics / training changes.
WLS is evaluation-only — never in adaptation objective.

Protocol (Phase 8):
  - load full DTI shell (sampling_fraction=1.0)
  - all b0 → observed
  - non-b0 observed_ratio ∈ {0.25, 0.5, 1.0}; remainder → holdout
  - optimize z_new only from z=0
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
from metrics.evaluator import dti_parameter_metrics, signal_metrics
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
DEFAULT_PHASE5 = ROOT / "experiments" / "population_dti_phase5" / "20260830_133430"
EPS = 1e-12
KEY_ITERS = (0, 10, 50, 100, 200, 500, 1000)

SUBJECTS = ("106319", "120717", "121618", "116726")
OBS_RATIOS = (0.25, 0.5, 1.0)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Phase-8 split on full shell:
      observed = all b0 + observed_ratio of non-b0
      holdout  = remaining non-b0
    Returns (obs_idx, hold_idx, nonb0_obs_idx_within_full).
    """
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
    return obs, hold, np.sort(obs_dwi).astype(np.int64)


def loss_signal(pred: torch.Tensor, target: torch.Tensor, bvals: torch.Tensor, cfg: dict) -> torch.Tensor:
    """L_signal = MSE(pred/S0, obs/S0) on provided channels (must include b0)."""
    b0 = float(cfg["b0_threshold"])
    s0 = s0_obs_from_batch(target, bvals, b0)
    return F.mse_loss(pred / s0, target / s0)


def loss_directional(
    pred: torch.Tensor,
    target: torch.Tensor,
    bvals: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """
    B1: direction-wise standardized residual on non-b0 S/S0.
    Reduces pure amplitude gaming by per-direction z-scoring using observed stats.
    """
    b0_thr = float(cfg["b0_threshold"])
    s0 = s0_obs_from_batch(target, bvals, b0_thr)
    r_pred = pred / s0
    r_obs = target / s0
    nb0 = bvals.reshape(-1) >= b0_thr
    if not bool(nb0.any()):
        return pred.new_zeros(())
    rp = r_pred[:, nb0]
    ro = r_obs[:, nb0]
    mu = ro.mean(dim=0, keepdim=True)
    std = ro.std(dim=0, keepdim=True).clamp_min(1e-4)
    return F.mse_loss((rp - mu) / std, (ro - mu) / std)


def loss_angular(pred: torch.Tensor, target: torch.Tensor, bvals: torch.Tensor, cfg: dict) -> torch.Tensor:
    """
    B2: voxel-wise angular-shape MSE on non-b0.
    r_i = S_i / mean_j(S_j)  (independent for pred and obs).
    """
    b0_thr = float(cfg["b0_threshold"])
    nb0 = bvals.reshape(-1) >= b0_thr
    if not bool(nb0.any()):
        return pred.new_zeros(())
    p = pred[:, nb0].clamp_min(EPS)
    o = target[:, nb0].clamp_min(EPS)
    rp = p / p.mean(dim=-1, keepdim=True).clamp_min(EPS)
    ro = o / o.mean(dim=-1, keepdim=True).clamp_min(EPS)
    return F.mse_loss(rp, ro)


def compute_adapt_loss(
    *,
    objective: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    bvals: torch.Tensor,
    cfg: dict,
    z: torch.Tensor,
    lambda_shape: float,
    lambda_z: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    objective:
      signal | directional | angular | combined
    """
    Ls = loss_signal(pred, target, bvals, cfg)
    Ld = loss_directional(pred, target, bvals, cfg)
    La = loss_angular(pred, target, bvals, cfg)
    Lz = torch.sum(z.float() ** 2)

    if objective == "signal":
        loss = Ls
    elif objective == "directional":
        loss = Ld
    elif objective == "angular":
        loss = La
    elif objective == "combined":
        # scale-normalize angular by detach scale of Ls so λ≈1 is meaningful
        scale = (Ls.detach() / (La.detach() + 1e-8)).clamp(0.1, 100.0)
        loss = Ls + float(lambda_shape) * scale * La
    else:
        raise ValueError(objective)

    loss = loss + float(lambda_z) * Lz
    parts = {
        "L_signal": float(Ls.detach().cpu()),
        "L_directional": float(Ld.detach().cpu()),
        "L_angular": float(La.detach().cpu()),
        "L_z": float(Lz.detach().cpu()),
        "L_total": float(loss.detach().cpu()),
    }
    return loss, parts


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
        return {"PSNR": float("nan"), "RelMSE": float("nan"), "MSE": float("nan"), "n_channels": 0}

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
    sig = signal_metrics(pred_n, obs_n, max_signal=max_signal, compute_ssim=False)
    sig["n_channels"] = int(channel_idx.size)
    return sig


@torch.no_grad()
def snapshot(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    z: torch.Tensor,
    D0: np.ndarray,
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
    D0v = D0[mask].astype(np.float64)
    Dref = np.asarray(subj.ref["D"], dtype=np.float64)[mask]
    row = {
        "iteration": int(iteration),
        "z_norm": latent_norm(z),
        "PSNR_obs": float(sig_obs["PSNR"]),
        "RelMSE_obs": float(sig_obs["RelMSE"]),
        "MSE_obs": float(sig_obs["MSE"]),
        "PSNR_holdout": float(sig_hold["PSNR"]),
        "RelMSE_holdout": float(sig_hold["RelMSE"]),
        "MSE_holdout": float(sig_hold["MSE"]),
        "FA_MAE": float(dti["FA"]["MAE"]),
        "MD_MAE": float(dti["MD"]["MAE"]),
        "AD_MAE": float(dti["AD"]["MAE"]),
        "RD_MAE": float(dti["RD"]["MAE"]),
        "mean_D_error": float(np.mean(_fro(Dv - Dref))),
        "mean_delta_D": float(np.mean(_fro(Dv - D0v))),
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
    D0: np.ndarray,
    objective: str,
    lambda_shape: float,
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
        m = snapshot(model, subj, z, D0, cfg, device, it, obs_idx, hold_idx, loss_parts=last_parts or None)
        rows.append(m)
        print(
            f"    [{objective} λs={lambda_shape} λz={lambda_z}] it={it:4d} "
            f"||z||={m['z_norm']:.4f} obs={m['PSNR_obs']:.3f} hold={m['PSNR_holdout']:.3f} "
            f"FA={m['FA_MAE']:.4f}",
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
        S0, D = model(xyz, z=z)
        pred = predict_signal(S0, D, bvals_obs, bvecs_obs, b_scale=b_scale)
        loss, parts = compute_adapt_loss(
            objective=objective,
            pred=pred,
            target=target,
            bvals=bvals_obs,
            cfg=cfg,
            z=z,
            lambda_shape=lambda_shape,
            lambda_z=lambda_z,
        )
        loss.backward()
        for p in model.theta_parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                raise RuntimeError("theta received gradients — forbidden in Phase 8")
        opt.step()
        last_parts = parts
        if it in key_lookup:
            record(it)

    return rows, z.detach().clone()


def status_flags(r0: dict[str, Any], rf: dict[str, Any]) -> dict[str, Any]:
    hold0, holdf = r0["PSNR_holdout"], rf["PSNR_holdout"]
    fa0, faf = r0["FA_MAE"], rf["FA_MAE"]
    md0, mdf = r0["MD_MAE"], rf["MD_MAE"]
    n_hold = int(r0.get("n_holdout_vols", 0) or 0)
    no_holdout = n_hold == 0 or (isinstance(holdf, float) and not np.isfinite(holdf))
    if no_holdout:
        signal_success = bool(rf["PSNR_obs"] > r0["PSNR_obs"] + 0.3)
        delta_hold = float(rf["PSNR_obs"] - r0["PSNR_obs"])
    else:
        signal_success = bool(holdf > hold0 + 0.3)
        delta_hold = float(holdf - hold0)
    dti_success = bool(faf < fa0 - 1e-4) and bool(mdf <= md0 + 1e-7)
    fa_not_worse = bool(faf <= fa0 + 0.002)
    if signal_success and (faf < fa0 - 0.005):
        overall = "STRONG_WIN"
    elif signal_success and fa_not_worse:
        overall = "WIN"
    elif signal_success and faf > fa0 + 0.002:
        overall = "FAIL"
    else:
        overall = "MIXED"
    return {
        "delta_PSNR_holdout_vs_z0": delta_hold,
        "delta_FA_MAE_vs_z0": float(faf - fa0),
        "delta_MD_MAE_vs_z0": float(mdf - md0),
        "signal_success": signal_success,
        "dti_success": dti_success,
        "overall_status": overall,
    }


def pick_best_fa_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Among key iters, pick lowest FA_MAE (ties → higher holdout PSNR)."""
    def key(r):
        hold = r["PSNR_holdout"]
        if not np.isfinite(hold):
            hold = r["PSNR_obs"]
        return (float(r["FA_MAE"]), -float(hold))

    return min(rows, key=key)


def run_one(
    *,
    model: PopulationDTIINR,
    cfg: dict[str, Any],
    device: torch.device,
    ckpt: Path,
    trad: Path,
    subject_id: str,
    observed_ratio: float,
    objective: str,
    lambda_shape: float,
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
    obs_idx, hold_idx, _ = split_by_observed_ratio(
        subj.bvals,
        b0_threshold=float(cfg["b0_threshold"]),
        observed_ratio=float(observed_ratio),
        seed=int(seed) + int(subject_id) % 10000,
    )
    print(
        f"\n=== {subject_id} obs_ratio={observed_ratio:.0%} obj={objective} "
        f"λs={lambda_shape} λz={lambda_z} | vols={subj.n_volumes} "
        f"obs={obs_idx.size} hold={hold_idx.size} ===",
        flush=True,
    )
    maps0 = predict_maps(
        model, subj.train_coords, subj.train_flat_idx, subj.shape_xyz, model.zero_z(device=device), device, want_D=True
    )
    D0 = maps0["D"]
    rows, zf = run_adaptation(
        model=model,
        subj=subj,
        cfg=cfg,
        device=device,
        obs_idx=obs_idx,
        hold_idx=hold_idx,
        D0=D0,
        objective=objective,
        lambda_shape=lambda_shape,
        lambda_z=lambda_z,
        max_iter=max_iter,
        key_iters=KEY_ITERS,
        seed=seed,
    )
    meta = {
        "subject": subject_id,
        "sampling_ratio": float(observed_ratio),
        "objective": objective,
        "lambda_shape": float(lambda_shape),
        "lambda_z": float(lambda_z),
    }
    for r in rows:
        r.update(meta)

    r0, rf = rows[0], rows[-1]
    rb = pick_best_fa_row(rows)
    flags_final = status_flags(r0, rf)
    flags_best = status_flags(r0, rb)
    flags_best = {f"bestFA_{k}": v for k, v in flags_best.items()}

    final = {
        **meta,
        **{k: rf[k] for k in (
            "z_norm", "PSNR_obs", "PSNR_holdout", "RelMSE_obs", "RelMSE_holdout",
            "FA_MAE", "MD_MAE", "AD_MAE", "RD_MAE", "mean_D_error", "mean_delta_D",
        )},
        **flags_final,
        "FA_MAE_z0": r0["FA_MAE"],
        "MD_MAE_z0": r0["MD_MAE"],
        "PSNR_holdout_z0": r0["PSNR_holdout"],
        "PSNR_obs_z0": r0["PSNR_obs"],
        "n_observed_vols": rf["n_observed_vols"],
        "n_holdout_vols": rf["n_holdout_vols"],
        "bestFA_iteration": int(rb["iteration"]),
        "bestFA_FA_MAE": float(rb["FA_MAE"]),
        "bestFA_MD_MAE": float(rb["MD_MAE"]),
        "bestFA_PSNR_holdout": float(rb["PSNR_holdout"]),
        "bestFA_PSNR_obs": float(rb["PSNR_obs"]),
        "bestFA_z_norm": float(rb["z_norm"]),
        "bestFA_mean_delta_D": float(rb["mean_delta_D"]),
        **flags_best,
    }
    tag = f"{subject_id}_r{observed_ratio}_{objective}_ls{lambda_shape}_lz{lambda_z}"
    _write_csv(exp_dir / "runs" / f"{tag}_trajectory.csv", rows)
    torch.save({"z": zf.cpu(), **meta}, exp_dir / "runs" / f"{tag}_z.pt")
    save_json(exp_dir / "runs" / f"{tag}_final.json", final)
    return rows, final


def smoke_plots(trajs: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for name, rows in trajs.items():
        xs = [r["iteration"] for r in rows]
        axes[0, 0].plot(xs, [r["PSNR_obs"] for r in rows], marker="o", label=name)
        axes[0, 1].plot(xs, [r["PSNR_holdout"] for r in rows], marker="o", label=name)
        axes[1, 0].plot(xs, [r["FA_MAE"] for r in rows], marker="o", label=name)
        axes[1, 1].plot(xs, [r["z_norm"] for r in rows], marker="o", label=name)
    axes[0, 0].set_title("PSNR obs")
    axes[0, 1].set_title("PSNR holdout")
    axes[1, 0].set_title("FA MAE")
    axes[1, 1].set_title("||z||")
    for ax in axes.ravel():
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Phase8 smoke: 106319 @ 50% non-b0 observed")
    fig.tight_layout()
    fig.savefig(out_dir / "smoke_106319_r50.png", dpi=150)
    plt.close(fig)


def make_full_plots(finals: list[dict[str, Any]], all_traj: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # signal vs dti scatter @1000 and bestFA
    fig, ax = plt.subplots(figsize=(7, 5))
    for f in finals:
        if int(f.get("n_holdout_vols", 0) or 0) == 0:
            continue
        ax.scatter(f["delta_PSNR_holdout_vs_z0"], f["delta_FA_MAE_vs_z0"], s=40, alpha=0.7)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Δ holdout PSNR vs z0 (↑ better)")
    ax.set_ylabel("Δ FA MAE vs z0 (↓ better)")
    ax.set_title("Phase8 @1000: signal vs DTI")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "signal_vs_dti.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for f in finals:
        if int(f.get("n_holdout_vols", 0) or 0) == 0:
            continue
        ax.scatter(
            f["bestFA_delta_PSNR_holdout_vs_z0"],
            f["bestFA_delta_FA_MAE_vs_z0"],
            s=40,
            alpha=0.7,
        )
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Δ holdout PSNR vs z0 @ best-FA iter")
    ax.set_ylabel("Δ FA MAE vs z0 @ best-FA iter")
    ax.set_title("Phase8 best-FA checkpoint: signal vs DTI")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "psnr_holdout_vs_fa.png", dpi=150)
    plt.close(fig)

    # objective comparison: mean ΔFA and Δhold PSNR by objective (ratio=0.5)
    objs = sorted({f["objective"] for f in finals})
    sub = [f for f in finals if abs(float(f["sampling_ratio"]) - 0.5) < 1e-9]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(
        axes,
        ("delta_FA_MAE_vs_z0", "delta_PSNR_holdout_vs_z0"),
        ("mean ΔFA @1000 (r=50%)", "mean Δhold PSNR @1000 (r=50%)"),
    ):
        means = []
        for o in objs:
            xs = [f[key] for f in sub if f["objective"] == o and np.isfinite(f[key])]
            means.append(float(np.mean(xs)) if xs else float("nan"))
        ax.bar(range(len(objs)), means)
        ax.set_xticks(range(len(objs)))
        ax.set_xticklabels(objs, rotation=20)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "objective_comparison.png", dpi=150)
    plt.close(fig)

    # sampling comparison for signal vs angular at @1000
    fig, ax = plt.subplots(figsize=(7, 4))
    for obj, marker in (("signal", "o"), ("angular", "s"), ("combined", "^")):
        pts = [f for f in finals if f["objective"] == obj and f["subject"] == "106319"]
        if not pts:
            continue
        pts = sorted(pts, key=lambda x: float(x["sampling_ratio"]))
        # for combined take λ=1 if available else first
        if obj == "combined":
            by_r: dict[float, list] = {}
            for p in pts:
                by_r.setdefault(float(p["sampling_ratio"]), []).append(p)
            pts = []
            for r, lst in sorted(by_r.items()):
                prefer = [x for x in lst if abs(float(x["lambda_shape"]) - 1.0) < 1e-9]
                pts.append(prefer[0] if prefer else lst[0])
        ax.plot(
            [p["sampling_ratio"] for p in pts],
            [p["FA_MAE"] for p in pts],
            marker=marker,
            label=obj,
        )
    ax.set_xlabel("non-b0 observed ratio")
    ax.set_ylabel("FA MAE @1000")
    ax.set_title("106319 sampling comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sampling_comparison.png", dpi=150)
    plt.close(fig)

    # latent / deltaD trajectories for 106319 r=0.5
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for obj in ("signal", "angular", "directional"):
        rows = [
            r
            for r in all_traj
            if r["subject"] == "106319"
            and abs(float(r["sampling_ratio"]) - 0.5) < 1e-9
            and r["objective"] == obj
            and float(r.get("lambda_shape", 0) or 0) == 0
            and float(r.get("lambda_z", 0) or 0) == 0
        ]
        if not rows:
            continue
        xs = [r["iteration"] for r in rows]
        axes[0].plot(xs, [r["z_norm"] for r in rows], marker="o", label=obj)
        axes[1].plot(xs, [r["mean_delta_D"] for r in rows], marker="o", label=obj)
    axes[0].set_title("||z|| vs iter (106319 r50)")
    axes[1].set_title("||ΔD|| vs iter (106319 r50)")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "latent_norm_vs_iteration.png", dpi=150)
    fig.savefig(out_dir / "deltaD_vs_iteration.png", dpi=150)
    plt.close(fig)


def diagnose_full(finals: list[dict[str, Any]], oracle_ref: dict[str, Any] | None) -> dict[str, Any]:
    # Prefer holdout-available settings (ratio < 1)
    eval_set = [f for f in finals if int(f.get("n_holdout_vols", 0) or 0) > 0]

    def group_key(f):
        return (f["objective"], float(f["lambda_shape"]), float(f["lambda_z"]))

    # Count WIN/STRONG_WIN by objective using bestFA flags (more informative given angular mid-traj)
    from collections import defaultdict

    stats: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "win": 0, "strong": 0, "fail": 0, "subjects_win": set()})
    for f in eval_set:
        k = group_key(f)
        st = stats[k]
        st["n"] += 1
        status = f.get("bestFA_overall_status", f["overall_status"])
        sid = f["subject"]
        if status in ("WIN", "STRONG_WIN"):
            st["win"] += 1
            st["subjects_win"].add(sid)
        if status == "STRONG_WIN":
            st["strong"] += 1
        if status == "FAIL" or (
            f.get("bestFA_signal_success") and f.get("bestFA_delta_FA_MAE_vs_z0", 0) > 0.002
        ):
            st["fail"] += 1

    ranked = []
    for k, st in stats.items():
        ranked.append(
            {
                "objective": k[0],
                "lambda_shape": k[1],
                "lambda_z": k[2],
                "n_evals": st["n"],
                "n_win": st["win"],
                "n_strong": st["strong"],
                "n_fail": st["fail"],
                "n_subjects_with_win": len(st["subjects_win"]),
                "subjects_with_win": sorted(st["subjects_win"]),
            }
        )
    ranked.sort(key=lambda x: (x["n_subjects_with_win"], x["n_win"], x["n_strong"]), reverse=True)
    best = ranked[0] if ranked else None

    # Signal-only mismatch persistence
    sig_only = [f for f in eval_set if f["objective"] == "signal"]
    sig_fail = sum(1 for f in sig_only if f["overall_status"] == "FAIL")
    signal_mismatch = bool(sig_fail >= max(1, len(sig_only) // 2))

    # Angular improves mismatch?
    ang = [f for f in eval_set if f["objective"] == "angular"]
    ang_best_fa_improve = sum(1 for f in ang if f.get("bestFA_delta_FA_MAE_vs_z0", 1) < -0.002)
    ang_helps = bool(ang_best_fa_improve >= max(1, len(ang) // 3))

    # Simultaneous improve at bestFA for some objective
    any_simultaneous = bool(best and best["n_subjects_with_win"] >= 1)
    go_criterion = bool(best and best["n_subjects_with_win"] >= 3)

    # sampling dependence: for best obj, compare win rates across ratios
    sampling_dep = False
    if best:
        by_r: dict[float, list] = defaultdict(list)
        for f in eval_set:
            if (
                f["objective"] == best["objective"]
                and abs(float(f["lambda_shape"]) - best["lambda_shape"]) < 1e-12
                and abs(float(f["lambda_z"]) - best["lambda_z"]) < 1e-12
            ):
                by_r[float(f["sampling_ratio"])].append(f.get("bestFA_overall_status"))
        rates = []
        for r, sts in by_r.items():
            rates.append(sum(1 for s in sts if s in ("WIN", "STRONG_WIN")) / max(len(sts), 1))
        if len(rates) >= 2:
            sampling_dep = bool(max(rates) - min(rates) > 0.34)

    # z_norm / deltaD anomalies
    z_norms = [f["z_norm"] for f in finals]
    mean_z = float(np.mean(z_norms)) if z_norms else float("nan")
    z_runaway = bool(mean_z > 1.0)
    d_drifts = [f["mean_delta_D"] for f in finals]
    mean_drift = float(np.mean(d_drifts)) if d_drifts else float("nan")

    if go_criterion:
        decision = "GO"
        mechanism = "signal_observable_angular_aware_objective_partially_identifies_DTI"
        next_step = (
            "Continue v1: refine angular/combined adaptation (early-stop / schedule), "
            "validate on more subjects; still do not use WLS in formal inference."
        )
    elif any_simultaneous and ang_helps:
        decision = "CONDITIONAL_GO"
        mechanism = "partial_identifiability_angular_helps_but_not_stable_3of4"
        next_step = (
            "Keep architecture; focus on adaptation objective / early-stop / regularization. "
            "Do not jump to v2 yet."
        )
    elif signal_mismatch and not ang_helps:
        decision = "HOLD"
        mechanism = "persistent_signal_DTI_identifiability_limitation_under_v1_latent"
        next_step = (
            "HOLD for current v1 inference story. Consider limited v2 exploration "
            "(stronger z→D pathway) only after documenting objective failure."
        )
    else:
        decision = "NO_GO"
        mechanism = "adaptation_fails_and_or_zeroshot_unstable"
        next_step = "Do not continue Population-DTI-INR v1 adaptation line without redesign."

    # zero-shot check
    z0_fas = [f["FA_MAE_z0"] for f in finals]
    zeroshot_ok = bool(float(np.mean(z0_fas)) < 0.22) if z0_fas else False
    if not zeroshot_ok:
        decision = "NO_GO"
        mechanism = "zeroshot_unstable"

    return {
        "Q1_signal_only_still_mismatch": signal_mismatch,
        "Q2_angular_improves_mismatch": ang_helps,
        "Q3_exists_objective_both_holdout_and_DTI": any_simultaneous,
        "Q4_reproducible_on_ge_3_of_4_subjects": go_criterion,
        "Q5_depends_on_sampling_ratio": sampling_dep,
        "Q6_z_norm_abnormally_large": z_runaway,
        "Q7_abnormal_D_drift": bool(mean_drift > 5e-4),
        "Q8_worth_continuing": decision in ("GO", "CONDITIONAL_GO"),
        "Q9_if_GO_next_direction": next_step if decision in ("GO", "CONDITIONAL_GO") else "N/A",
        "Q10_if_HOLD_NOGO_failure_mechanism": mechanism if decision in ("HOLD", "NO_GO") else "N/A",
        "decision": decision,
        "best_objective_config": best,
        "ranked_objectives": ranked[:15],
        "oracle_reference_phase5": oracle_ref,
        "aggregates": {
            "mean_z_norm_final": mean_z,
            "mean_delta_D_final": mean_drift,
            "mean_FA_z0": float(np.mean(z0_fas)) if z0_fas else float("nan"),
            "n_eval_with_holdout": len(eval_set),
            "signal_fail_count": sig_fail,
            "angular_bestFA_improve_count": ang_best_fa_improve,
        },
        "zeroshot_reasonable": zeroshot_ok,
        "next_step": next_step,
        "failure_or_success_mechanism": mechanism,
    }


def load_phase5_oracle_ref() -> dict[str, Any] | None:
    p = DEFAULT_PHASE5 / "metrics" / "phase5b_objective_comparison.csv"
    if not p.exists():
        return None
    import csv

    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    out = {}
    for r in rows:
        out[r["objective"]] = {
            "PSNR": float(r["PSNR"]),
            "FA_MAE": float(r["FA_MAE"]),
            "latent_norm": float(r["latent_norm"]),
            "D_frobenius_error_vs_wls": float(r["D_frobenius_error_vs_wls"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 8 signal-observable DTI-aware adaptation")
    ap.add_argument("--phase4a-dir", default=str(DEFAULT_PHASE4A))
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="Step2 smoke only")
    ap.add_argument("--skip-reg", action="store_true", help="Skip Phase8-C λz stage")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--ratios", nargs="*", type=float, default=None)
    ap.add_argument("--objectives", nargs="*", default=None)
    args = ap.parse_args()

    phase4a = Path(args.phase4a_dir)
    cfg = dict(load_yaml(phase4a / "config" / "run_config.yaml"))
    cfg["adapt_lr"] = float(cfg.get("adapt_lr", cfg.get("lr", 1e-3)))
    split = split_from_config(cfg)
    device = resolve_device(str(cfg.get("device", "auto")))
    ckpt = phase4a / "checkpoints" / "epoch_0150"
    trad = Path(cfg["trad_root"])

    exp_dir = make_experiment_dir(tag="population_dti_phase8")
    for sub in ("metrics", "plots", "runs", "diagnosis", "logs"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)

    save_yaml(
        exp_dir / "phase8_config.yaml",
        {
            **cfg,
            "max_iter": args.max_iter,
            "smoke": bool(args.smoke),
            "protocol": "non-b0 observed_ratio; all b0 observed; WLS eval-only",
            "key_iters": list(KEY_ITERS),
            "combined_lambdas": [0.01, 0.1, 1.0, 10.0],
            "reg_lambdas_z": [0.0, 0.001, 0.01, 0.1],
        },
    )
    save_json(
        exp_dir / "config" / "sources.json",
        {
            "theta": str(ckpt / "theta.pt"),
            "phase4a": str(phase4a),
            "phase5_oracle_ref": str(DEFAULT_PHASE5),
            "note": "WLS never in adaptation objective",
        },
    )

    print("=" * 72)
    print("Phase 8 Diagnostic")
    print(f"  theta={ckpt / 'theta.pt'}")
    print(f"  out={exp_dir}")
    print(f"  smoke={args.smoke}")
    print("=" * 72)

    if not (ckpt / "theta.pt").is_file():
        raise FileNotFoundError(ckpt / "theta.pt")

    model = _build_model(cfg, split["train"], device)
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()

    if args.smoke:
        jobs = [
            ("106319", 0.5, "signal", 0.0, 0.0),
            ("106319", 0.5, "angular", 0.0, 0.0),
            ("106319", 0.5, "directional", 0.0, 0.0),
        ]
    else:
        subjects = args.subjects or list(SUBJECTS)
        ratios = args.ratios or list(OBS_RATIOS)
        objectives = args.objectives or ["signal", "angular", "directional", "combined"]
        jobs = []
        for sid in subjects:
            for ratio in ratios:
                for obj in objectives:
                    if obj == "combined":
                        # combined needs holdout for fair scoring; still run at 1.0 for FA-only view
                        for ls in (0.01, 0.1, 1.0, 10.0):
                            jobs.append((sid, float(ratio), obj, float(ls), 0.0))
                    else:
                        jobs.append((sid, float(ratio), obj, 0.0, 0.0))

    all_traj: list[dict[str, Any]] = []
    finals: list[dict[str, Any]] = []
    smoke_trajs: dict[str, list[dict[str, Any]]] = {}

    for sid, ratio, obj, ls, lz in jobs:
        rows, final = run_one(
            model=model,
            cfg=cfg,
            device=device,
            ckpt=ckpt,
            trad=trad,
            subject_id=sid,
            observed_ratio=ratio,
            objective=obj,
            lambda_shape=ls,
            lambda_z=lz,
            max_iter=int(args.max_iter),
            seed=int(args.seed),
            exp_dir=exp_dir,
        )
        all_traj.extend(rows)
        finals.append(final)
        if args.smoke:
            smoke_trajs[obj] = rows
        # incremental save
        _write_csv(exp_dir / "metrics" / "phase8_summary.csv", finals)

    def filt(obj_name: str) -> list[dict]:
        return [r for r in all_traj if r["objective"] == obj_name]

    _write_csv(exp_dir / "metrics" / "phase8a_signal_baseline.csv", filt("signal"))
    _write_csv(exp_dir / "metrics" / "phase8b_angular.csv", filt("angular") + filt("directional"))
    _write_csv(exp_dir / "metrics" / "phase8c_combined.csv", filt("combined"))
    _write_csv(exp_dir / "metrics" / "phase8_summary.csv", finals)

    if args.smoke:
        smoke_plots(smoke_trajs, exp_dir / "plots")
        by = {f["objective"]: f for f in finals}
        save_json(
            exp_dir / "diagnosis" / "phase8_smoke_diagnosis.json",
            {"signal": by.get("signal"), "angular": by.get("angular"), "directional": by.get("directional")},
        )
        print("\n===== PHASE 8 SMOKE =====")
        for obj, f in by.items():
            print(
                f"  {obj}: hold {f['PSNR_holdout_z0']:.3f}→{f['PSNR_holdout']:.3f} "
                f"FA {f['FA_MAE_z0']:.4f}→{f['FA_MAE']:.4f} "
                f"bestFA@{f['bestFA_iteration']}={f['bestFA_FA_MAE']:.4f} status={f['overall_status']}"
            )
        print(f"\n[Phase8 smoke] done → {exp_dir}")
        return

    # Phase 8-C: λz on best combined (or angular) at ratio=0.5
    if not args.skip_reg:
        print("\n===== Phase 8-C: latent norm regularization =====", flush=True)
        # choose best λ_shape for combined on r=0.5 by n_subjects with bestFA WIN
        comb = [f for f in finals if f["objective"] == "combined" and abs(float(f["sampling_ratio"]) - 0.5) < 1e-9]
        best_ls = 1.0
        if comb:
            from collections import defaultdict

            score = defaultdict(int)
            for f in comb:
                if f.get("bestFA_overall_status") in ("WIN", "STRONG_WIN"):
                    score[float(f["lambda_shape"])] += 1
            if score:
                best_ls = max(score.items(), key=lambda kv: kv[1])[0]
            else:
                # fallback: minimize mean bestFA FA
                by_ls = defaultdict(list)
                for f in comb:
                    by_ls[float(f["lambda_shape"])].append(f["bestFA_FA_MAE"])
                best_ls = min(by_ls.items(), key=lambda kv: float(np.mean(kv[1])))[0]
        print(f"  selected combined λ_shape={best_ls} for λz sweep @ r=50%", flush=True)

        reg_finals = []
        for sid in (args.subjects or list(SUBJECTS)):
            for lz in (0.0, 0.001, 0.01, 0.1):
                # skip duplicate λz=0 already in mains if combined λs=best_ls exists
                if abs(lz) < 1e-15:
                    exist = [
                        f
                        for f in finals
                        if f["subject"] == sid
                        and f["objective"] == "combined"
                        and abs(float(f["sampling_ratio"]) - 0.5) < 1e-9
                        and abs(float(f["lambda_shape"]) - best_ls) < 1e-12
                        and abs(float(f["lambda_z"])) < 1e-15
                    ]
                    if exist:
                        reg_finals.append(exist[0])
                        continue
                rows, final = run_one(
                    model=model,
                    cfg=cfg,
                    device=device,
                    ckpt=ckpt,
                    trad=trad,
                    subject_id=sid,
                    observed_ratio=0.5,
                    objective="combined",
                    lambda_shape=float(best_ls),
                    lambda_z=float(lz),
                    max_iter=int(args.max_iter),
                    seed=int(args.seed),
                    exp_dir=exp_dir,
                )
                all_traj.extend(rows)
                finals.append(final)
                reg_finals.append(final)
        _write_csv(exp_dir / "metrics" / "phase8c_regularization.csv", reg_finals)
        _write_csv(exp_dir / "metrics" / "phase8_summary.csv", finals)
        _write_csv(exp_dir / "metrics" / "phase8c_combined.csv", filt("combined"))

    make_full_plots(finals, all_traj, exp_dir / "plots")
    oracle = load_phase5_oracle_ref()
    diag = diagnose_full(finals, oracle)
    save_json(exp_dir / "diagnosis" / "phase8_diagnosis.json", diag)

    conclusion = {
        "Go_NoGo": diag["decision"],
        "answers": {
            "1_signal_only_mismatch": diag["Q1_signal_only_still_mismatch"],
            "2_angular_improves_mismatch": diag["Q2_angular_improves_mismatch"],
            "3_exists_simultaneous_objective": diag["Q3_exists_objective_both_holdout_and_DTI"],
            "4_reproducible_3of4": diag["Q4_reproducible_on_ge_3_of_4_subjects"],
            "5_sampling_dependent": diag["Q5_depends_on_sampling_ratio"],
            "6_z_norm_abnormal": diag["Q6_z_norm_abnormally_large"],
            "7_D_drift_abnormal": diag["Q7_abnormal_D_drift"],
            "8_worth_continuing": diag["Q8_worth_continuing"],
            "9_next_if_GO": diag["Q9_if_GO_next_direction"],
            "10_failure_if_HOLD_NOGO": diag["Q10_if_HOLD_NOGO_failure_mechanism"],
        },
        "best_objective_config": diag["best_objective_config"],
        "ranked_objectives": diag["ranked_objectives"],
        "oracle_reference_phase5": oracle,
        "aggregates": diag["aggregates"],
        "next_step": diag["next_step"],
        "mechanism": diag["failure_or_success_mechanism"],
        "experiment_dir": str(exp_dir),
    }
    save_json(exp_dir / "diagnosis" / "phase8_final_conclusion.json", conclusion)

    print("\n===== PHASE 8 FINAL =====")
    print(f"  decision: {diag['decision']}")
    print(f"  best: {diag['best_objective_config']}")
    print(f"  Q1 mismatch: {diag['Q1_signal_only_still_mismatch']}")
    print(f"  Q2 angular helps: {diag['Q2_angular_improves_mismatch']}")
    print(f"  Q4 3/4 subjects: {diag['Q4_reproducible_on_ge_3_of_4_subjects']}")
    print(f"  next: {diag['next_step']}")
    print(f"\n[Phase8] done → {exp_dir}")


if __name__ == "__main__":
    main()
