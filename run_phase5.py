#!/usr/bin/env python
"""Phase 5 diagnostic: latent adaptation trajectory + signal vs DTI-guided oracle.

Inference / diagnostic only. Does NOT modify PopulationDTIINR, training, or baselines.
theta fixed = Phase4-A epoch_0150. Unseen subject = 106319.
DTI-guided adaptation is ORACLE diagnostic — not a formal test method.
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
from evaluate import evaluate_subject
from models.population_dti_inr import PopulationDTIINR
from physics.dti_forward import predict_signal
from predict import predict_maps
from utils_io import (
    _write_csv,
    load_latents,
    load_theta,
    load_yaml,
    make_experiment_dir,
    resolve_device,
    save_json,
    save_yaml,
)

DEFAULT_PHASE4A = ROOT / "experiments" / "population_dti_phase4a" / "20260829_192714"
EPS = 1e-12
KEY_ITERS = (0, 10, 50, 100, 200, 500, 1000)


def _build_model(cfg: dict[str, Any], train_ids: list[str], device: torch.device) -> PopulationDTIINR:
    return PopulationDTIINR(
        train_subject_ids=train_ids,
        latent_dim=int(cfg.get("latent_dim", 16)),
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 4)),
        pe_freqs=int(cfg.get("pe_freqs", 8)),
    ).to(device)


def signal_mse_loss(pred, target, bvals, cfg) -> torch.Tensor:
    """Exact Population inference signal loss: MSE(pred/S0, obs/S0)."""
    mode = str(cfg.get("signal_normalization", "s0")).lower()
    b0 = float(cfg["b0_threshold"])
    if mode in {"s0", "s0_norm", "normalized"}:
        s0 = s0_obs_from_batch(target, bvals, b0)
        return F.mse_loss(pred / s0, target / s0)
    return F.mse_loss(pred, target)


def dti_frobenius_loss(D_pred: torch.Tensor, D_ref: torch.Tensor) -> torch.Tensor:
    """Diagnostic only: mean ||D_pred - D_ref||_F^2."""
    diff = D_pred - D_ref
    return (diff * diff).sum(dim=(-2, -1)).mean()


def _eigs_desc(D: np.ndarray) -> np.ndarray:
    Df = 0.5 * (D + np.swapaxes(D, -1, -2))
    Df = np.nan_to_num(Df, nan=0.0, posinf=0.0, neginf=0.0) + EPS * np.eye(3)
    ev = np.linalg.eigvalsh(Df.astype(np.float64))
    return np.clip(ev, 0.0, None)[..., ::-1].copy()


def _fa_md_ad_rd(l123: np.ndarray) -> dict[str, np.ndarray]:
    l1, l2, l3 = l123[..., 0], l123[..., 1], l123[..., 2]
    md = (l1 + l2 + l3) / 3.0
    ad, rd = l1, 0.5 * (l2 + l3)
    num = (l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2
    den = l1**2 + l2**2 + l3**2
    fa = np.clip(np.sqrt(1.5 * num / np.maximum(den, EPS)), 0.0, 1.0)
    return {"FA": fa, "MD": md, "AD": ad, "RD": rd}


def _fro(D: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(D.astype(np.float64) ** 2, axis=(-2, -1)))


def latent_norm(z: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(z.detach().float()).cpu())


@torch.no_grad()
def snapshot_metrics(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    z: torch.Tensor,
    D0: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    iteration: int,
) -> dict[str, Any]:
    """Full eval at one adaptation iteration (signal + DTI vs WLS + D vs z=0)."""
    model.eval()
    model.freeze_theta()
    z = z.detach()
    res = evaluate_subject(
        model, subj, z, device=device, cfg=cfg, mode="phase5", adapt_iter=iteration, seed=int(cfg.get("seed", 42))
    )
    row_sig = res["row"]

    maps = predict_maps(
        model,
        subj.train_coords,
        subj.train_flat_idx,
        subj.shape_xyz,
        z.to(device),
        device,
        want_D=True,
    )
    mask = subj.common_mask  # agreement with WLS metrics
    Dv = maps["D"][mask].astype(np.float64)
    D0v = D0[mask].astype(np.float64)
    Dref = np.asarray(subj.ref["D"], dtype=np.float64)[mask]

    l = _eigs_desc(Dv)
    l0 = _eigs_desc(D0v)
    lref = _eigs_desc(Dref)
    sc = _fa_md_ad_rd(l)
    sc0 = _fa_md_ad_rd(l0)

    dD = Dv - D0v
    fro_d = _fro(dD)
    fro0 = _fro(D0v)
    rel = fro_d / (fro0 + EPS)

    err_wls = _fro(Dv - Dref)
    fro_ref = _fro(Dref)
    rel_wls = err_wls / (fro_ref + EPS)
    dlam_ref = np.abs(l - lref)

    return {
        "iteration": int(iteration),
        "latent_norm": latent_norm(z),
        "PSNR": float(row_sig["signal_PSNR"]),
        "RelMSE": float(row_sig["signal_RelMSE"]),
        "MSE": float(row_sig["signal_MSE"]),
        "FA_MAE": float(row_sig["FA_MAE"]),
        "MD_MAE": float(row_sig["MD_MAE"]),
        "AD_MAE": float(row_sig["AD_MAE"]),
        "RD_MAE": float(row_sig["RD_MAE"]),
        "mean_delta_D_fro": float(np.mean(fro_d)),
        "mean_relative_delta_D": float(np.mean(rel)),
        "mean_delta_lambda1": float(np.mean(l[:, 0] - l0[:, 0])),
        "mean_delta_lambda2": float(np.mean(l[:, 1] - l0[:, 1])),
        "mean_delta_lambda3": float(np.mean(l[:, 2] - l0[:, 2])),
        "mean_signed_delta_FA": float(np.mean(sc["FA"] - sc0["FA"])),
        "mean_abs_delta_FA": float(np.mean(np.abs(sc["FA"] - sc0["FA"]))),
        "mean_signed_delta_MD": float(np.mean(sc["MD"] - sc0["MD"])),
        "mean_abs_delta_MD": float(np.mean(np.abs(sc["MD"] - sc0["MD"]))),
        # vs WLS (diagnostic)
        "mean_D_fro_error_vs_wls": float(np.mean(err_wls)),
        "mean_relative_D_error_vs_wls": float(np.mean(rel_wls)),
        "mean_abs_delta_lambda1_vs_wls": float(np.mean(dlam_ref[:, 0])),
        "mean_abs_delta_lambda2_vs_wls": float(np.mean(dlam_ref[:, 1])),
        "mean_abs_delta_lambda3_vs_wls": float(np.mean(dlam_ref[:, 2])),
    }


def run_adaptation(
    *,
    model: PopulationDTIINR,
    subj: SubjectBundle,
    cfg: dict[str, Any],
    device: torch.device,
    objective: str,
    max_iter: int,
    key_iters: tuple[int, ...],
    D0_vol: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """
    objective: 'signal' | 'dti'
    Returns trajectory rows at key_iters and final z.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model.freeze_theta()
    z = model.new_z(trainable=True, device=device, init="zeros")
    opt = torch.optim.Adam([z], lr=float(cfg.get("adapt_lr", cfg.get("lr", 1e-3))))
    batch = int(cfg.get("batch_voxels", 4096))
    b_scale = float(cfg.get("b_scale", 1.0))
    bvals_t = torch.from_numpy(subj.bvals).to(device)
    bvecs_t = torch.from_numpy(subj.bvecs).to(device)
    coords_t = torch.from_numpy(subj.train_coords)

    # WLS D flat for DTI-guided
    X, Y, Z = subj.shape_xyz
    D_wls_flat = np.asarray(subj.ref["D"], dtype=np.float32).reshape(X * Y * Z, 3, 3)
    # Prefer common_mask voxels for DTI objective (valid WLS)
    common_flat = np.flatnonzero(subj.common_mask.reshape(-1))
    # Map train_flat_idx positions that are in common mask
    train_in_common = np.intersect1d(subj.train_flat_idx, common_flat, assume_unique=False)
    if train_in_common.size == 0:
        train_in_common = subj.train_flat_idx
    # index into train_coords: positions where train_flat_idx is in common
    flat_to_trainpos = {int(f): i for i, f in enumerate(subj.train_flat_idx.tolist())}
    common_train_pos = np.asarray(
        [flat_to_trainpos[int(f)] for f in train_in_common if int(f) in flat_to_trainpos],
        dtype=np.int64,
    )
    if common_train_pos.size == 0:
        common_train_pos = np.arange(subj.train_coords.shape[0], dtype=np.int64)

    max_iter = int(max_iter)
    key_set = sorted({int(x) for x in key_iters if int(x) <= max_iter})
    if max_iter not in key_set:
        key_set.append(max_iter)
    key_lookup = set(key_set)
    rows: list[dict[str, Any]] = []

    def record(it: int) -> None:
        m = snapshot_metrics(model, subj, z, D0_vol, cfg, device, it)
        m["objective"] = objective
        rows.append(m)
        print(
            f"  [{objective}] it={it:4d} ||z||={m['latent_norm']:.4f} "
            f"PSNR={m['PSNR']:.3f} FA_MAE={m['FA_MAE']:.4f} "
            f"relΔD={m['mean_relative_delta_D']:.4e} D_err_WLS={m['mean_D_fro_error_vs_wls']:.4e}",
            flush=True,
        )

    if 0 in key_lookup:
        record(0)

    for it in range(1, max_iter + 1):
        model.train()
        model.freeze_theta()
        opt.zero_grad(set_to_none=True)

        if objective == "signal":
            sel = rng.integers(0, int(coords_t.shape[0]), size=batch, endpoint=False)
            xyz = coords_t[sel].to(device)
            target = torch.from_numpy(subj.dwi_flat[subj.train_flat_idx[sel]]).to(device)
            S0, D = model(xyz, z=z)
            pred = predict_signal(S0, D, bvals_t, bvecs_t, b_scale=b_scale)
            loss = signal_mse_loss(pred, target, bvals_t, cfg)
        elif objective == "dti":
            # Sample common-mask voxels (valid WLS reference)
            n_c = int(common_train_pos.shape[0])
            sel_local = rng.integers(0, n_c, size=min(batch, n_c), endpoint=False)
            pos = common_train_pos[sel_local]
            xyz = coords_t[pos].to(device)
            flat_idx = subj.train_flat_idx[pos]
            D_ref = torch.from_numpy(D_wls_flat[flat_idx]).to(device=device, dtype=torch.float32)
            _, D = model(xyz, z=z)
            loss = dti_frobenius_loss(D, D_ref)
        else:
            raise ValueError(objective)

        loss.backward()
        for p in model.theta_parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                raise RuntimeError("theta received gradients — forbidden in Phase 5")
        opt.step()

        if it in key_lookup:
            record(it)

    return rows, z.detach().clone()

def plot_phase5a(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = [r["iteration"] for r in rows]

    def save_line(path, ys, title, ylabel):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    save_line(out_dir / "phase5a_latent_norm.png", [r["latent_norm"] for r in rows], "||z|| vs iter", "latent_norm")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, [r["PSNR"] for r in rows], marker="o")
    axes[0].set_title("PSNR")
    axes[1].plot(xs, [r["RelMSE"] for r in rows], marker="o")
    axes[1].set_title("RelMSE")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "phase5a_signal_metrics.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, [r["FA_MAE"] for r in rows], marker="o")
    axes[0].set_title("FA MAE")
    axes[1].plot(xs, [r["MD_MAE"] for r in rows], marker="o")
    axes[1].set_title("MD MAE")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "phase5a_dti_metrics.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, [r["mean_delta_D_fro"] for r in rows], marker="o")
    axes[0].set_title("mean ||ΔD||_F vs z=0")
    axes[1].plot(xs, [r["mean_relative_delta_D"] for r in rows], marker="o")
    axes[1].set_title("mean relative ΔD vs z=0")
    for ax in axes:
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "phase5a_d_sensitivity.png", dpi=150)
    plt.close(fig)


def diagnose_5a(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {int(r["iteration"]): r for r in rows}
    r0, r200, r1000 = by[0], by.get(200, by[max(by)]), by[max(by)]

    norms = [r["latent_norm"] for r in rows]
    psnrs = [r["PSNR"] for r in rows]
    fas = [r["FA_MAE"] for r in rows]
    rels = [r["mean_relative_delta_D"] for r in rows]

    z_increasing = bool(norms[-1] > norms[0] + 1e-3 and norms[-1] >= max(norms[: max(1, len(norms) // 2)]) - 1e-3)
    # continued growth after 200?
    cont_after_200 = bool(200 in by and norms[-1] > by[200]["latent_norm"] * 1.1)
    psnr_improving = bool(psnrs[-1] > psnrs[0] + 0.5)
    psnr_after_200 = bool(200 in by and psnrs[-1] > by[200]["PSNR"] + 0.3)
    d_increasing = bool(rels[-1] > rels[0] + 1e-3)
    d_after_200 = bool(200 in by and rels[-1] > by[200]["mean_relative_delta_D"] * 1.1)
    fa_improving = bool(fas[-1] < fas[0] - 0.01)
    fa_after_200 = bool(200 in by and fas[-1] < by[200]["FA_MAE"] - 0.005)
    mismatch = bool(psnr_improving and not fa_improving)

    # Case classification
    if z_increasing and d_increasing and fa_improving:
        case = "A_more_iterations_helps_DTI"
    elif z_increasing and (psnr_improving or d_increasing) and not fa_improving:
        case = "B_signal_DTI_objective_mismatch"
    elif (not cont_after_200) and (abs(norms[-1] - by.get(200, r0)["latent_norm"]) < 0.05 * max(norms[-1], 1e-6)):
        case = "C_early_saturation_small_latent"
    else:
        case = "MIXED_or_partial"

    return {
        "case": case,
        "z_norm_continues_increasing": z_increasing,
        "z_norm_continues_after_200": cont_after_200,
        "PSNR_continues_improving": psnr_improving,
        "PSNR_improves_after_200": psnr_after_200,
        "D_field_perturbation_increases": d_increasing,
        "D_field_increases_after_200": d_after_200,
        "FA_MD_eventually_improves": fa_improving,
        "FA_improves_after_200": fa_after_200,
        "only_200_iters_insufficient": bool(cont_after_200 or psnr_after_200 or d_after_200 or fa_after_200),
        "signal_DTI_mismatch_persists_at_1000": mismatch and int(max(by)) >= 1000,
        "trajectory_endpoints": {
            "iter0": {k: r0[k] for k in ("latent_norm", "PSNR", "RelMSE", "FA_MAE", "MD_MAE", "mean_relative_delta_D")},
            "iter200": {k: r200[k] for k in ("latent_norm", "PSNR", "RelMSE", "FA_MAE", "MD_MAE", "mean_relative_delta_D")},
            "iter_final": {k: r1000[k] for k in ("latent_norm", "PSNR", "RelMSE", "FA_MAE", "MD_MAE", "mean_relative_delta_D")},
        },
        "answers": {
            "1_z_norm_keeps_increasing": z_increasing,
            "2_PSNR_keeps_improving": psnr_improving,
            "3_D_perturbation_keeps_increasing": d_increasing,
            "4_FA_MD_eventually_improves": fa_improving,
            "5_200_iters_insufficient": bool(cont_after_200 or psnr_after_200 or fa_after_200),
            "6_mismatch_at_1000": mismatch and int(max(by)) >= 1000,
        },
    }


def diagnose_5b(
    sig_rows: list[dict[str, Any]],
    dti_rows: list[dict[str, Any]],
    z_sig: torch.Tensor,
    z_dti: torch.Tensor,
    train_latents: dict[str, torch.Tensor],
) -> dict[str, Any]:
    s_final = sig_rows[-1]
    d_final = dti_rows[-1]
    s0 = sig_rows[0]
    d0 = dti_rows[0]

    dti_fa_gain = float(d0["FA_MAE"] - d_final["FA_MAE"])
    sig_fa_gain = float(s0["FA_MAE"] - s_final["FA_MAE"])
    dti_d_err_drop = float(d0["mean_D_fro_error_vs_wls"] - d_final["mean_D_fro_error_vs_wls"])
    sig_d_err_drop = float(s0["mean_D_fro_error_vs_wls"] - s_final["mean_D_fro_error_vs_wls"])
    dti_psnr_gain = float(d_final["PSNR"] - d0["PSNR"])
    sig_psnr_gain = float(s_final["PSNR"] - s0["PSNR"])

    dti_helps_geometry = bool(dti_fa_gain > 0.01 or dti_d_err_drop > 1e-5)
    sig_helps_geometry = bool(sig_fa_gain > 0.01 or sig_d_err_drop > 1e-5)
    sig_helps_signal = bool(sig_psnr_gain > 0.5)
    dti_continues = bool(
        len(dti_rows) >= 2
        and (
            dti_rows[-1]["FA_MAE"] < dti_rows[-2]["FA_MAE"] - 1e-4
            or dti_rows[-1]["mean_D_fro_error_vs_wls"]
            < dti_rows[-2]["mean_D_fro_error_vs_wls"] * 0.98
        )
    )

    # Latent geometry
    def nn_dist(z: torch.Tensor) -> tuple[float, str]:
        best, name = float("inf"), ""
        for sid, zt in train_latents.items():
            dist = float(torch.linalg.vector_norm(z.cpu().float() - zt.cpu().float()))
            if dist < best:
                best, name = dist, sid
        return best, name

    nn_sig, nn_sig_id = nn_dist(z_sig)
    nn_dti, nn_dti_id = nn_dist(z_dti)
    train_norms = {sid: float(torch.linalg.vector_norm(zt.float())) for sid, zt in train_latents.items()}
    mean_train_norm = float(np.mean(list(train_norms.values())))

    # Case logic for Phase 5 overall
    if dti_helps_geometry and (not sig_helps_geometry) and sig_helps_signal:
        arch_case = "Case1_architecture_OK_inference_objective_problem"
    elif not dti_helps_geometry and float(d_final["latent_norm"]) > 0.5 * mean_train_norm:
        arch_case = "Case2_architecture_limitation"
    elif dti_helps_geometry and dti_continues and d_final["iteration"] >= 500:
        # still improving at end
        last_half = dti_rows[len(dti_rows) // 2 :]
        still = last_half[-1]["FA_MAE"] < last_half[0]["FA_MAE"] - 0.005
        arch_case = "Case3_optimization_limitation" if still else "Case1_architecture_OK_inference_objective_problem"
    elif dti_helps_geometry:
        arch_case = "Case1_architecture_OK_inference_objective_problem"
    else:
        arch_case = "Case2_or_inconclusive_need_more_budget"

    return {
        "arch_case": arch_case,
        "signal_guided_final": s_final,
        "dti_guided_final": d_final,
        "comparison": {
            "FA_MAE_signal": float(s_final["FA_MAE"]),
            "FA_MAE_dti": float(d_final["FA_MAE"]),
            "FA_gain_signal": sig_fa_gain,
            "FA_gain_dti": dti_fa_gain,
            "PSNR_signal": float(s_final["PSNR"]),
            "PSNR_dti": float(d_final["PSNR"]),
            "D_fro_err_signal": float(s_final["mean_D_fro_error_vs_wls"]),
            "D_fro_err_dti": float(d_final["mean_D_fro_error_vs_wls"]),
            "D_err_drop_signal": sig_d_err_drop,
            "D_err_drop_dti": dti_d_err_drop,
            "latent_norm_signal": float(s_final["latent_norm"]),
            "latent_norm_dti": float(d_final["latent_norm"]),
            "mean_train_latent_norm": mean_train_norm,
        },
        "latent_geometry": {
            "||z_signal||": float(s_final["latent_norm"]),
            "||z_DTI||": float(d_final["latent_norm"]),
            "train_norms": train_norms,
            "nn_signal": {"dist": nn_sig, "subject": nn_sig_id},
            "nn_dti": {"dist": nn_dti, "subject": nn_dti_id},
        },
        "flags": {
            "dti_guided_improves_geometry": dti_helps_geometry,
            "signal_guided_improves_geometry": sig_helps_geometry,
            "signal_guided_improves_signal": sig_helps_signal,
            "dti_guided_still_improving_at_end": dti_continues,
        },
    }


def plot_5b_comparison(sig_rows, dti_rows, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    xs_s = [r["iteration"] for r in sig_rows]
    xs_d = [r["iteration"] for r in dti_rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(xs_s, [r["latent_norm"] for r in sig_rows], marker="o", label="signal")
    axes[0, 0].plot(xs_d, [r["latent_norm"] for r in dti_rows], marker="s", label="DTI")
    axes[0, 0].set_title("||z||")
    axes[0, 0].legend()

    axes[0, 1].plot(xs_s, [r["PSNR"] for r in sig_rows], marker="o", label="signal")
    axes[0, 1].plot(xs_d, [r["PSNR"] for r in dti_rows], marker="s", label="DTI")
    axes[0, 1].set_title("PSNR")
    axes[0, 1].legend()

    axes[1, 0].plot(xs_s, [r["FA_MAE"] for r in sig_rows], marker="o", label="signal")
    axes[1, 0].plot(xs_d, [r["FA_MAE"] for r in dti_rows], marker="s", label="DTI")
    axes[1, 0].set_title("FA MAE")
    axes[1, 0].legend()

    axes[1, 1].plot(xs_s, [r["mean_D_fro_error_vs_wls"] for r in sig_rows], marker="o", label="signal")
    axes[1, 1].plot(xs_d, [r["mean_D_fro_error_vs_wls"] for r in dti_rows], marker="s", label="DTI")
    axes[1, 1].set_title("||D-D_WLS||_F")
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phase 5-B: Signal-guided vs DTI-guided (oracle)")
    fig.tight_layout()
    fig.savefig(out_dir / "objective_comparison.png", dpi=150)
    plt.close(fig)


def final_conclusion(diag_a: dict, diag_b: dict) -> dict[str, Any]:
    arch = diag_b["arch_case"]
    go_v1 = arch == "Case1_architecture_OK_inference_objective_problem"
    go_v2 = arch == "Case2_architecture_limitation"
    go_opt = arch == "Case3_optimization_limitation"
    inconclusive = "inconclusive" in arch.lower() or arch.startswith("Case2_or")

    if go_v1:
        q4 = "inference_objective_identifiability"
        q5 = "N/A — architecture appears capable under DTI-guided oracle; prioritize inference objective."
        go = "CONDITIONAL_GO — keep architecture; fix latent adaptation objective / protocol"
    elif go_opt:
        q4 = "optimization_budget"
        q5 = "N/A until DTI-guided adaptation saturates; extend budget before claiming architecture failure."
        go = "HOLD — optimization may be incomplete; extend DTI-guided budget before v2"
    elif go_v2:
        q4 = "architecture_limitation"
        q5 = (
            "Increase subject-specific DTI expressivity of latent pathway "
            "(e.g. D_pop(x)+ΔD(x,z) or stronger z→D coupling); not just longer signal adapt."
        )
        go = "NO-GO for v1 inference story — evidence for architecture limit; consider v2"
    else:
        q4 = "architecture_or_inconclusive"
        q5 = "Insufficient evidence; re-check DTI-guided trajectory length and latent scale vs train."
        go = "HOLD — inconclusive; do not enter v2 yet"

    return {
        "arch_case": arch,
        "Q1_200_iters_insufficient": bool(diag_a["answers"]["5_200_iters_insufficient"]),
        "Q2_signal_only_has_signal_DTI_mismatch": bool(
            diag_a["answers"]["6_mismatch_at_1000"] or diag_a["case"] == "B_signal_DTI_objective_mismatch"
        ),
        "Q3_latent_space_can_express_subject_DTI": bool(
            diag_b["flags"]["dti_guided_improves_geometry"]
        ),
        "Q4_if_capable_problem_is": q4,
        "Q5_if_not_capable_v2_should_address": q5,
        "Go_NoGo_Population_DTI_INR_v1": go,
        "evidence_for_v2": bool(go_v2),
        "inconclusive": bool(inconclusive),
        "phase5a_case": diag_a["case"],
        "phase5b_flags": diag_b["flags"],
        "comparison_snapshot": diag_b["comparison"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 Latent Adaptation Diagnostics")
    ap.add_argument("--phase4a-dir", default=str(DEFAULT_PHASE4A))
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    phase4a = Path(args.phase4a_dir)
    cfg = load_yaml(phase4a / "config" / "run_config.yaml")
    cfg = dict(cfg)
    cfg["adapt_lr"] = float(cfg.get("adapt_lr", cfg.get("lr", 1e-3)))
    split = split_from_config(cfg)
    device = resolve_device(str(cfg.get("device", "auto")))
    ckpt = phase4a / "checkpoints" / "epoch_0150"

    exp_dir = make_experiment_dir(tag="population_dti_phase5")
    for sub in ("signal_guided", "dti_guided", "metrics", "plots"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    save_yaml(exp_dir / "phase5_config.yaml", {**cfg, "max_iter": args.max_iter, "key_iters": list(KEY_ITERS)})
    save_json(exp_dir / "config" / "sources.json", {"theta": str(ckpt / "theta.pt"), "phase4a": str(phase4a)})

    print("=" * 72)
    print("Phase 5 Diagnostic (inference only)")
    print(f"  theta={ckpt / 'theta.pt'}")
    print(f"  out={exp_dir}")
    print("=" * 72)

    model = _build_model(cfg, split["train"], device)
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()
    train_latents = {k: v.float() for k, v in load_latents(ckpt / "latents.pt").items()}

    trad = Path(cfg["trad_root"])
    test_id = split["test"][0]
    subj = load_subject_bundle(subject_id=test_id, cfg=cfg, trad_dir=trad / test_id)
    print(f"[Phase5] subject={test_id} common={int(subj.common_mask.sum())}", flush=True)

    # D(x, z=0) reference volume for DeltaD
    print("[Phase5] computing D(x,z=0) baseline...", flush=True)
    maps0 = predict_maps(
        model,
        subj.train_coords,
        subj.train_flat_idx,
        subj.shape_xyz,
        model.zero_z(device=device),
        device,
        want_D=True,
    )
    D0 = maps0["D"]

    # ----- Phase 5-A = signal-guided extended trajectory -----
    print("\n===== Phase 5-A: Signal-only latent trajectory =====", flush=True)
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()
    sig_rows, z_sig = run_adaptation(
        model=model,
        subj=subj,
        cfg=cfg,
        device=device,
        objective="signal",
        max_iter=int(args.max_iter),
        key_iters=KEY_ITERS,
        D0_vol=D0,
        seed=int(args.seed),
    )
    _write_csv(exp_dir / "metrics" / "phase5a_latent_trajectory.csv", sig_rows)
    _write_csv(exp_dir / "signal_guided" / "trajectory.csv", sig_rows)
    save_json(exp_dir / "signal_guided" / "final_metrics.json", sig_rows[-1])
    torch.save({"z_signal": z_sig.cpu()}, exp_dir / "signal_guided" / "z_final.pt")
    plot_phase5a(sig_rows, exp_dir / "plots")
    # copy requested aliases
    import shutil

    shutil.copy(exp_dir / "plots" / "phase5a_latent_norm.png", exp_dir / "plots" / "latent_norm.png")
    shutil.copy(exp_dir / "plots" / "phase5a_signal_metrics.png", exp_dir / "plots" / "signal_vs_iteration.png")
    shutil.copy(exp_dir / "plots" / "phase5a_dti_metrics.png", exp_dir / "plots" / "dti_vs_iteration.png")
    shutil.copy(exp_dir / "plots" / "phase5a_d_sensitivity.png", exp_dir / "plots" / "d_sensitivity.png")

    diag_a = diagnose_5a(sig_rows)
    save_json(exp_dir / "phase5a_diagnosis.json", diag_a)

    # ----- Phase 5-B DTI-guided (oracle) -----
    print("\n===== Phase 5-B: DTI-guided oracle adaptation =====", flush=True)
    # Fresh z; same theta
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()
    dti_rows, z_dti = run_adaptation(
        model=model,
        subj=subj,
        cfg=cfg,
        device=device,
        objective="dti",
        max_iter=int(args.max_iter),
        key_iters=KEY_ITERS,
        D0_vol=D0,
        seed=int(args.seed),
    )
    _write_csv(exp_dir / "dti_guided" / "trajectory.csv", dti_rows)
    save_json(exp_dir / "dti_guided" / "final_metrics.json", dti_rows[-1])
    torch.save({"z_dti": z_dti.cpu()}, exp_dir / "dti_guided" / "z_final.pt")

    # Comparison table
    comp_rows = []
    for tag, rows in (("signal", sig_rows), ("dti", dti_rows)):
        f = rows[-1]
        comp_rows.append(
            {
                "objective": tag,
                "iteration": f["iteration"],
                "latent_norm": f["latent_norm"],
                "PSNR": f["PSNR"],
                "RelMSE": f["RelMSE"],
                "FA_MAE": f["FA_MAE"],
                "MD_MAE": f["MD_MAE"],
                "AD_MAE": f["AD_MAE"],
                "RD_MAE": f["RD_MAE"],
                "D_frobenius_error_vs_wls": f["mean_D_fro_error_vs_wls"],
                "relative_D_error_vs_wls": f["mean_relative_D_error_vs_wls"],
                "mean_abs_delta_lambda1_vs_wls": f["mean_abs_delta_lambda1_vs_wls"],
                "mean_abs_delta_lambda2_vs_wls": f["mean_abs_delta_lambda2_vs_wls"],
                "mean_abs_delta_lambda3_vs_wls": f["mean_abs_delta_lambda3_vs_wls"],
                "mean_relative_delta_D_vs_z0": f["mean_relative_delta_D"],
            }
        )
    _write_csv(exp_dir / "metrics" / "phase5b_objective_comparison.csv", comp_rows)

    # Latent geometry CSV
    geom = []
    for name, z in [
        ("z_0", torch.zeros_like(z_sig.cpu())),
        ("z_signal", z_sig.cpu()),
        ("z_DTI", z_dti.cpu()),
        ("z_101309", train_latents["101309"]),
        ("z_102715", train_latents["102715"]),
        ("z_103515", train_latents["103515"]),
    ]:
        a = z.detach().float().numpy().ravel()
        geom.append(
            {
                "name": name,
                "l2_norm": float(np.linalg.norm(a)),
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
            }
        )
    # pairwise distances from z_signal / z_DTI to train
    for zname, z in (("z_signal", z_sig.cpu()), ("z_DTI", z_dti.cpu())):
        for sid, zt in train_latents.items():
            dist = float(torch.linalg.vector_norm(z.float() - zt.float()))
            geom.append({"name": f"dist_{zname}_to_{sid}", "l2_norm": dist, "mean": "", "std": "", "min": "", "max": ""})
    _write_csv(exp_dir / "metrics" / "phase5b_latent_geometry.csv", geom)

    plot_5b_comparison(sig_rows, dti_rows, exp_dir / "plots")
    # latent geometry bar
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["z_0", "z_signal", "z_DTI", "z_101309", "z_102715", "z_103515"]
    norms = [next(g["l2_norm"] for g in geom if g["name"] == n) for n in names]
    ax.bar(names, norms)
    ax.set_ylabel("||z||_2")
    ax.set_title("Latent norms")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(exp_dir / "plots" / "latent_geometry.png", dpi=150)
    plt.close(fig)

    diag_b = diagnose_5b(sig_rows, dti_rows, z_sig, z_dti, train_latents)
    save_json(exp_dir / "phase5b_diagnosis.json", diag_b)

    conclusion = final_conclusion(diag_a, diag_b)
    save_json(exp_dir / "phase5_final_conclusion.json", conclusion)

    print("\n===== PHASE 5 FINAL =====")
    print(f"  5A case: {diag_a['case']}")
    print(f"  5B arch_case: {diag_b['arch_case']}")
    print(f"  Signal final: FA={sig_rows[-1]['FA_MAE']:.4f} PSNR={sig_rows[-1]['PSNR']:.3f} ||z||={sig_rows[-1]['latent_norm']:.4f}")
    print(f"  DTI    final: FA={dti_rows[-1]['FA_MAE']:.4f} PSNR={dti_rows[-1]['PSNR']:.3f} ||z||={dti_rows[-1]['latent_norm']:.4f}")
    print(f"  Go/No-Go: {conclusion['Go_NoGo_Population_DTI_INR_v1']}")
    for q in ("Q1_200_iters_insufficient", "Q2_signal_only_has_signal_DTI_mismatch", "Q3_latent_space_can_express_subject_DTI", "Q4_if_capable_problem_is", "Q5_if_not_capable_v2_should_address"):
        print(f"  {q}: {conclusion[q]}")
    print(f"\n[Phase5] done → {exp_dir}")


if __name__ == "__main__":
    main()
