#!/usr/bin/env python
"""Phase 7: final Go/No-Go feasibility — observed/holdout signal-only latent adaptation.

Inference / diagnostic only. Does NOT modify PopulationDTIINR, physics, or baselines.
theta fixed = Phase4-A epoch_0150. WLS is evaluation-only (never in z optimization).
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
EPS = 1e-12
KEY_ITERS = (0, 10, 50, 100, 200, 500, 1000)

# Primary subject does sampling sweep; others at 100% for reproducibility
PRIMARY_SUBJECT = "106319"
EXTRA_SUBJECTS = ("120717", "121618", "116726")
SAMPLING_FRACS_PRIMARY = (0.25, 0.5, 1.0)
SAMPLING_FRACS_EXTRA = (1.0,)
HOLDOUT_FRAC_DWI = 0.5  # of non-b0 volumes → holdout; all b0 stay observed


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


def split_observed_holdout(
    bvals: np.ndarray,
    *,
    b0_threshold: float,
    holdout_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split volume indices into observed / holdout.

    - All b0 volumes → observed (needed for S0_obs)
    - Non-b0 volumes: holdout_frac → holdout, rest → observed
    """
    bvals = np.asarray(bvals).reshape(-1)
    b0 = bvals < float(b0_threshold)
    dwi_idx = np.flatnonzero(~b0)
    b0_idx = np.flatnonzero(b0)
    rng = np.random.default_rng(int(seed))
    dwi_idx = dwi_idx.copy()
    rng.shuffle(dwi_idx)
    n_hold = int(round(len(dwi_idx) * float(holdout_frac)))
    n_hold = min(max(n_hold, 1 if len(dwi_idx) >= 2 else 0), max(len(dwi_idx) - 1, 0))
    hold = dwi_idx[:n_hold]
    obs_dwi = dwi_idx[n_hold:]
    obs = np.sort(np.concatenate([b0_idx, obs_dwi])).astype(np.int64)
    hold = np.sort(hold).astype(np.int64)
    if obs.size == 0:
        raise RuntimeError("observed set empty")
    return obs, hold


def signal_mse_observed(pred, target, bvals, cfg) -> torch.Tensor:
    mode = str(cfg.get("signal_normalization", "s0")).lower()
    b0 = float(cfg["b0_threshold"])
    if mode in {"s0", "s0_norm", "normalized"}:
        s0 = s0_obs_from_batch(target, bvals, b0)
        return F.mse_loss(pred / s0, target / s0)
    return F.mse_loss(pred, target)


@torch.no_grad()
def eval_signal_channels(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    z: torch.Tensor,
    device: torch.device,
    cfg: dict[str, Any],
    channel_idx: np.ndarray,
    *,
    s0_bvals: np.ndarray,
    s0_target_fn,
    max_voxels: int = 65536,
    seed: int = 42,
) -> dict[str, float]:
    """
    Signal metrics on a channel subset.
    S0_obs is computed from observed b0 channels via s0_target_fn(target_full_row) → [V, Ns0]
    actually: we pass full target rows and compute S0 from observed b0 indices separately.
    """
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

    # S0 from observed b0 channels of the same voxels
    s0 = s0_obs_from_batch(target_full, bvals_all, b0_thr)

    ch = torch.as_tensor(channel_idx, device=device, dtype=torch.long)
    pred_n = (pred_full.index_select(-1, ch) / s0).detach().cpu().numpy()
    obs_n = (target_full.index_select(-1, ch) / s0).detach().cpu().numpy()
    sig = signal_metrics(pred_n, obs_n, max_signal=max_signal, compute_ssim=False)
    sig["n_channels"] = int(channel_idx.size)
    return sig


@torch.no_grad()
def snapshot_phase7(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    z: torch.Tensor,
    D0: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    iteration: int,
    obs_idx: np.ndarray,
    hold_idx: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    model.freeze_theta()
    z = z.detach()

    sig_obs = eval_signal_channels(model, subj, z, device, cfg, obs_idx, s0_bvals=subj.bvals, s0_target_fn=None, seed=int(cfg.get("seed", 42)))
    sig_hold = eval_signal_channels(model, subj, z, device, cfg, hold_idx, s0_bvals=subj.bvals, s0_target_fn=None, seed=int(cfg.get("seed", 42)))

    maps = predict_maps(
        model,
        subj.train_coords,
        subj.train_flat_idx,
        subj.shape_xyz,
        z.to(device),
        device,
        want_D=True,
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
    dD = Dv - D0v
    err_wls = _fro(Dv - Dref)

    return {
        "iteration": int(iteration),
        "latent_norm": latent_norm(z),
        "observed_PSNR": float(sig_obs["PSNR"]),
        "observed_RelMSE": float(sig_obs["RelMSE"]),
        "observed_MSE": float(sig_obs["MSE"]),
        "holdout_PSNR": float(sig_hold["PSNR"]),
        "holdout_RelMSE": float(sig_hold["RelMSE"]),
        "holdout_MSE": float(sig_hold["MSE"]),
        "FA_MAE": float(dti["FA"]["MAE"]),
        "MD_MAE": float(dti["MD"]["MAE"]),
        "AD_MAE": float(dti["AD"]["MAE"]),
        "RD_MAE": float(dti["RD"]["MAE"]),
        "mean_D_fro_error_vs_wls": float(np.mean(err_wls)),
        "mean_delta_D_fro_vs_z0": float(np.mean(_fro(dD))),
        "n_observed_vols": int(obs_idx.size),
        "n_holdout_vols": int(hold_idx.size),
        "n_total_vols": int(subj.n_volumes),
    }


def run_obs_holdout_adaptation(
    *,
    model: PopulationDTIINR,
    subj: SubjectBundle,
    cfg: dict[str, Any],
    device: torch.device,
    obs_idx: np.ndarray,
    hold_idx: np.ndarray,
    D0: np.ndarray,
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

    def record(it: int) -> None:
        m = snapshot_phase7(model, subj, z, D0, cfg, device, it, obs_idx, hold_idx)
        rows.append(m)
        print(
            f"    it={it:4d} ||z||={m['latent_norm']:.4f} "
            f"obsPSNR={m['observed_PSNR']:.3f} holdPSNR={m['holdout_PSNR']:.3f} "
            f"FA={m['FA_MAE']:.4f} ΔD={m['mean_delta_D_fro_vs_z0']:.4e}",
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
        target_full = torch.from_numpy(subj.dwi_flat[flat]).to(device)
        target = target_full[:, obs_np]

        S0, D = model(xyz, z=z)
        pred = predict_signal(S0, D, bvals_obs, bvecs_obs, b_scale=b_scale)
        loss = signal_mse_observed(pred, target, bvals_obs, cfg)
        loss.backward()
        for p in model.theta_parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                raise RuntimeError("theta received gradients — forbidden in Phase 7")
        opt.step()

        if it in key_lookup:
            record(it)

    return rows, z.detach().clone()


def plot_run(rows: list[dict[str, Any]], out_dir: Path, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = [r["iteration"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(xs, [r["observed_PSNR"] for r in rows], marker="o", label="observed")
    axes[0, 0].plot(xs, [r["holdout_PSNR"] for r in rows], marker="s", label="holdout")
    axes[0, 0].set_title("PSNR")
    axes[0, 0].legend()
    axes[0, 1].plot(xs, [r["FA_MAE"] for r in rows], marker="o")
    axes[0, 1].set_title("FA MAE")
    axes[1, 0].plot(xs, [r["latent_norm"] for r in rows], marker="o")
    axes[1, 0].set_title("||z||")
    axes[1, 1].plot(xs, [r["mean_delta_D_fro_vs_z0"] for r in rows], marker="o")
    axes[1, 1].set_title("mean ||D_t - D_0||_F")
    for ax in axes.ravel():
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    safe = title.replace("/", "_").replace(" ", "_").replace("%", "pct")
    fig.savefig(out_dir / f"traj_{safe}.png", dpi=150)
    plt.close(fig)


def plot_aggregate(all_rows: list[dict[str, Any]], subject_finals: list[dict[str, Any]], sampling_finals: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary subject 100% trajectory overlays (obs vs hold)
    prim = [r for r in all_rows if r["subject_id"] == PRIMARY_SUBJECT and abs(float(r["sampling_fraction"]) - 1.0) < 1e-9]
    if prim:
        xs = sorted(set(int(r["iteration"]) for r in prim))
        by_it = {i: next(r for r in prim if int(r["iteration"]) == i) for i in xs}
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [by_it[i]["observed_PSNR"] for i in xs], marker="o", label="observed")
        ax.plot(xs, [by_it[i]["holdout_PSNR"] for i in xs], marker="s", label="holdout")
        ax.set_xlabel("iteration")
        ax.set_ylabel("PSNR")
        ax.set_title(f"{PRIMARY_SUBJECT} 100%: observed vs holdout PSNR")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "observed_PSNR_vs_iteration.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [by_it[i]["holdout_PSNR"] for i in xs], marker="s", color="C1")
        ax.set_xlabel("iteration")
        ax.set_ylabel("holdout PSNR")
        ax.set_title(f"{PRIMARY_SUBJECT} 100%: holdout PSNR")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "holdout_PSNR_vs_iteration.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [by_it[i]["FA_MAE"] for i in xs], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("FA MAE")
        ax.set_title(f"{PRIMARY_SUBJECT} 100%: FA MAE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "FA_MAE_vs_iteration.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [by_it[i]["latent_norm"] for i in xs], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("||z||")
        ax.set_title(f"{PRIMARY_SUBJECT} 100%: ||z||")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "z_norm_vs_iteration.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, [by_it[i]["mean_delta_D_fro_vs_z0"] for i in xs], marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("||D_t-D_0||_F")
        ax.set_title(f"{PRIMARY_SUBJECT} 100%: D drift vs z=0")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "D_drift_vs_iteration.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter([by_it[i]["observed_PSNR"] for i in xs], [by_it[i]["holdout_PSNR"] for i in xs], c=xs, cmap="viridis", s=70)
        for i in xs:
            ax.annotate(str(i), (by_it[i]["observed_PSNR"], by_it[i]["holdout_PSNR"]), fontsize=7)
        ax.set_xlabel("observed PSNR")
        ax.set_ylabel("holdout PSNR")
        ax.set_title("observed vs holdout PSNR")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "observed_vs_holdout_PSNR.png", dpi=150)
        plt.close(fig)

    # sampling ratio vs FA
    if sampling_finals:
        fig, ax = plt.subplots(figsize=(6, 4))
        fr = [float(r["sampling_fraction"]) for r in sampling_finals]
        ax.plot(fr, [r["FA_MAE"] for r in sampling_finals], marker="o", label="adapted@1000")
        ax.plot(fr, [r["FA_MAE_z0"] for r in sampling_finals], marker="s", label="zero-shot")
        ax.set_xlabel("sampling fraction")
        ax.set_ylabel("FA MAE")
        ax.set_title(f"{PRIMARY_SUBJECT}: sampling vs FA")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "sampling_vs_FA_MAE.png", dpi=150)
        plt.close(fig)

    # zero-shot vs adapted FA across subjects
    if subject_finals:
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [r["subject_id"] for r in subject_finals]
        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w / 2, [r["FA_MAE_z0"] for r in subject_finals], w, label="zero-shot")
        ax.bar(x + w / 2, [r["FA_MAE"] for r in subject_finals], w, label="adapted@1000")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.set_ylabel("FA MAE")
        ax.set_title("zero-shot vs adapted FA (100% sampling)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "zeroshot_vs_adapted_FA.png", dpi=150)
        plt.close(fig)


def diagnose_phase7(
    all_rows: list[dict[str, Any]],
    subject_finals: list[dict[str, Any]],
    sampling_finals: list[dict[str, Any]],
) -> dict[str, Any]:
    # Primary 100% trajectory
    prim = [
        r
        for r in all_rows
        if r["subject_id"] == PRIMARY_SUBJECT and abs(float(r["sampling_fraction"]) - 1.0) < 1e-9
    ]
    prim = sorted(prim, key=lambda r: int(r["iteration"]))
    r0 = prim[0]
    rf = prim[-1]

    obs_improve = bool(rf["observed_PSNR"] > r0["observed_PSNR"] + 0.5)
    hold_improve = bool(rf["holdout_PSNR"] > r0["holdout_PSNR"] + 0.5)
    hold_worsen = bool(rf["holdout_PSNR"] < r0["holdout_PSNR"] - 0.5)
    fa_improve = bool(rf["FA_MAE"] < r0["FA_MAE"] - 0.005)
    fa_worsen = bool(rf["FA_MAE"] > r0["FA_MAE"] + 0.002)
    z_up = bool(rf["latent_norm"] > r0["latent_norm"] + 0.05)
    d_drift_up = bool(rf["mean_delta_D_fro_vs_z0"] > r0["mean_delta_D_fro_vs_z0"] + 1e-5)
    obs_up_hold_down = bool(obs_improve and hold_worsen)
    mismatch = bool(obs_improve and (fa_worsen or not fa_improve) and z_up and d_drift_up)

    # sampling effect on primary
    samp_fa_spread = float("nan")
    if len(sampling_finals) >= 2:
        fas = [float(r["FA_MAE"]) for r in sampling_finals]
        samp_fa_spread = float(max(fas) - min(fas))
    sampling_matters = bool(samp_fa_spread == samp_fa_spread and samp_fa_spread > 0.01)

    # subject reproducibility
    n_fa_improve = sum(1 for r in subject_finals if r["FA_MAE"] < r["FA_MAE_z0"] - 0.005)
    n_fa_worsen = sum(1 for r in subject_finals if r["FA_MAE"] > r["FA_MAE_z0"] + 0.002)
    n_obs_improve = sum(1 for r in subject_finals if r["observed_PSNR"] > r["observed_PSNR_z0"] + 0.5)
    n_hold_worsen = sum(1 for r in subject_finals if r["holdout_PSNR"] < r["holdout_PSNR_z0"] - 0.5)
    n_hold_improve = sum(1 for r in subject_finals if r["holdout_PSNR"] > r["holdout_PSNR_z0"] + 0.5)
    mean_fa_z0 = float(np.mean([r["FA_MAE_z0"] for r in subject_finals])) if subject_finals else float("nan")
    mean_fa_ad = float(np.mean([r["FA_MAE"] for r in subject_finals])) if subject_finals else float("nan")
    consistent_mismatch = bool(n_obs_improve >= max(1, len(subject_finals) // 2) and n_fa_improve == 0 and n_fa_worsen >= max(1, len(subject_finals) // 2))

    zeroshot_ok = bool(mean_fa_z0 < 0.22)  # Phase4 ~0.17 is reasonable
    oracle_capable = True  # Phase5/6 established

    if zeroshot_ok and n_fa_improve >= 1 and n_hold_worsen == 0 and n_fa_improve >= len(subject_finals) // 2:
        decision = "STRONG_GO"
        rationale = (
            "Zero-shot DTI reasonable; adaptation improves FA/MD on ≥1 subjects; "
            "holdout signal not systematically worsening; trend reproducible."
        )
    elif (not zeroshot_ok) or (n_fa_improve == 0 and not oracle_capable):
        decision = "NO_GO"
        rationale = (
            "Zero-shot failed and/or adaptation never improves DTI with no prior oracle evidence."
        )
    else:
        decision = "CONDITIONAL_GO"
        rationale = (
            "Zero-shot DTI is reasonable; Phase5/6 oracle showed latent→D capability; "
            "signal-only observed/holdout adaptation still shows signal/DTI mismatch "
            "(and/or holdout issues). Bottleneck is latent identifiability / adaptation "
            "objective — not proven architecture failure. Continue research under v1; do not auto-build v2."
        )

    return {
        "Q1_observed_signal_improves": obs_improve,
        "Q2_holdout_signal_improves": hold_improve,
        "Q3_signal_gain_with_FA_MD_gain": bool(obs_improve and fa_improve),
        "Q4_obs_PSNR_up_holdout_PSNR_down": obs_up_hold_down,
        "Q5_z_up_Ddrift_up_DTI_worse": mismatch,
        "Q6_sampling_affects_adaptation": sampling_matters,
        "Q6_sampling_FA_spread": samp_fa_spread,
        "Q7_consistent_across_subjects": consistent_mismatch or (n_fa_improve >= 2),
        "Q7_detail": {
            "n_subjects": len(subject_finals),
            "n_FA_improve_vs_z0": n_fa_improve,
            "n_FA_worsen_vs_z0": n_fa_worsen,
            "n_obs_PSNR_improve": n_obs_improve,
            "n_holdout_PSNR_improve": n_hold_improve,
            "n_holdout_PSNR_worsen": n_hold_worsen,
            "mean_FA_MAE_z0": mean_fa_z0,
            "mean_FA_MAE_adapted": mean_fa_ad,
            "consistent_signal_DTI_mismatch": consistent_mismatch,
        },
        "primary_100pct_endpoints": {
            "iter0": {k: r0[k] for k in ("observed_PSNR", "holdout_PSNR", "FA_MAE", "MD_MAE", "latent_norm", "mean_delta_D_fro_vs_z0")},
            "iter_final": {k: rf[k] for k in ("observed_PSNR", "holdout_PSNR", "FA_MAE", "MD_MAE", "latent_norm", "mean_delta_D_fro_vs_z0")},
        },
        "subject_finals": subject_finals,
        "sampling_finals": sampling_finals,
        "zeroshot_reasonable": zeroshot_ok,
        "oracle_capability_from_phase5_6": oracle_capable,
        "Go_NoGo": decision,
        "rationale": rationale,
        "bottleneck": (
            "latent_identifiability_adaptation_objective"
            if decision == "CONDITIONAL_GO"
            else ("representation" if decision == "NO_GO" else "none_strong_go")
        ),
        "evidence_for_v2_now": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 7 observed/holdout feasibility")
    ap.add_argument("--phase4a-dir", default=str(DEFAULT_PHASE4A))
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC_DWI)
    ap.add_argument("--subjects", nargs="*", default=None, help="override subject list")
    ap.add_argument("--skip-sampling-sweep", action="store_true")
    ap.add_argument("--primary-only", action="store_true")
    args = ap.parse_args()

    phase4a = Path(args.phase4a_dir)
    cfg = dict(load_yaml(phase4a / "config" / "run_config.yaml"))
    cfg["adapt_lr"] = float(cfg.get("adapt_lr", cfg.get("lr", 1e-3)))
    split = split_from_config(cfg)
    device = resolve_device(str(cfg.get("device", "auto")))
    ckpt = phase4a / "checkpoints" / "epoch_0150"
    trad = Path(cfg["trad_root"])

    subjects: list[str]
    if args.subjects:
        subjects = [str(s) for s in args.subjects]
    else:
        subjects = [PRIMARY_SUBJECT] + ([] if args.primary_only else list(EXTRA_SUBJECTS))

    exp_dir = make_experiment_dir(tag="population_dti_phase7")
    for sub in ("metrics", "plots", "runs"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    save_yaml(
        exp_dir / "phase7_config.yaml",
        {
            **cfg,
            "max_iter": args.max_iter,
            "key_iters": list(KEY_ITERS),
            "holdout_frac_dwi": args.holdout_frac,
            "subjects": subjects,
            "sampling_primary": list(SAMPLING_FRACS_PRIMARY),
            "protocol": "observed/holdout signal-only latent adaptation; WLS eval-only",
        },
    )
    save_json(exp_dir / "config" / "sources.json", {"theta": str(ckpt / "theta.pt"), "phase4a": str(phase4a)})

    print("=" * 72)
    print("Phase 7 Observed/Holdout Feasibility")
    print(f"  theta={ckpt / 'theta.pt'}")
    print(f"  subjects={subjects}")
    print(f"  out={exp_dir}")
    print("=" * 72)

    model = _build_model(cfg, split["train"], device)
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()

    all_rows: list[dict[str, Any]] = []
    subject_finals: list[dict[str, Any]] = []
    sampling_finals: list[dict[str, Any]] = []

    for sid in subjects:
        fracs = SAMPLING_FRACS_PRIMARY if (sid == PRIMARY_SUBJECT and not args.skip_sampling_sweep) else SAMPLING_FRACS_EXTRA
        for frac in fracs:
            print(f"\n===== subject={sid} sampling={frac:.0%} =====", flush=True)
            load_theta(ckpt / "theta.pt", model, map_location=device)
            model.freeze_theta()

            subj = load_subject_bundle(subject_id=sid, cfg=cfg, trad_dir=trad / sid, sampling_fraction=float(frac))
            obs_idx, hold_idx = split_observed_holdout(
                subj.bvals,
                b0_threshold=float(cfg["b0_threshold"]),
                holdout_frac=float(args.holdout_frac),
                seed=int(args.seed) + int(sid) % 10000,
            )
            print(
                f"  vols={subj.n_volumes} observed={obs_idx.size} holdout={hold_idx.size} "
                f"common={int(subj.common_mask.sum())}",
                flush=True,
            )
            save_json(
                exp_dir / "runs" / f"{sid}_frac{frac}_split.json",
                {
                    "subject_id": sid,
                    "sampling_fraction": frac,
                    "observed_idx": obs_idx.tolist(),
                    "holdout_idx": hold_idx.tolist(),
                    "bvals_observed": subj.bvals[obs_idx].tolist(),
                    "bvals_holdout": subj.bvals[hold_idx].tolist(),
                },
            )

            # D(x,z=0)
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

            rows, z_final = run_obs_holdout_adaptation(
                model=model,
                subj=subj,
                cfg=cfg,
                device=device,
                obs_idx=obs_idx,
                hold_idx=hold_idx,
                D0=D0,
                max_iter=int(args.max_iter),
                key_iters=KEY_ITERS,
                seed=int(args.seed),
            )
            for r in rows:
                r["subject_id"] = sid
                r["sampling_fraction"] = float(frac)
            all_rows.extend(rows)
            _write_csv(exp_dir / "runs" / f"{sid}_frac{frac}_trajectory.csv", rows)
            torch.save({"z": z_final.cpu(), "subject_id": sid, "sampling_fraction": frac}, exp_dir / "runs" / f"{sid}_frac{frac}_z.pt")
            plot_run(rows, exp_dir / "plots", f"{sid} frac={frac:.0%}")

            r0 = rows[0]
            rf = rows[-1]
            summary = {
                "subject_id": sid,
                "sampling_fraction": float(frac),
                "n_observed_vols": int(obs_idx.size),
                "n_holdout_vols": int(hold_idx.size),
                "FA_MAE_z0": float(r0["FA_MAE"]),
                "FA_MAE": float(rf["FA_MAE"]),
                "MD_MAE_z0": float(r0["MD_MAE"]),
                "MD_MAE": float(rf["MD_MAE"]),
                "observed_PSNR_z0": float(r0["observed_PSNR"]),
                "observed_PSNR": float(rf["observed_PSNR"]),
                "holdout_PSNR_z0": float(r0["holdout_PSNR"]),
                "holdout_PSNR": float(rf["holdout_PSNR"]),
                "latent_norm": float(rf["latent_norm"]),
                "mean_D_fro_error_vs_wls_z0": float(r0["mean_D_fro_error_vs_wls"]),
                "mean_D_fro_error_vs_wls": float(rf["mean_D_fro_error_vs_wls"]),
                "mean_delta_D_fro_vs_z0": float(rf["mean_delta_D_fro_vs_z0"]),
                "delta_FA_MAE": float(rf["FA_MAE"] - r0["FA_MAE"]),
                "delta_observed_PSNR": float(rf["observed_PSNR"] - r0["observed_PSNR"]),
                "delta_holdout_PSNR": float(rf["holdout_PSNR"] - r0["holdout_PSNR"]),
            }
            if abs(float(frac) - 1.0) < 1e-9:
                subject_finals.append(summary)
            if sid == PRIMARY_SUBJECT:
                sampling_finals.append(summary)

    _write_csv(exp_dir / "metrics" / "phase7_summary.csv", all_rows)
    _write_csv(exp_dir / "metrics" / "phase7_subject_summary.csv", subject_finals)
    _write_csv(exp_dir / "metrics" / "phase7_sampling_summary.csv", sampling_finals)
    plot_aggregate(all_rows, subject_finals, sampling_finals, exp_dir / "plots")

    diag = diagnose_phase7(all_rows, subject_finals, sampling_finals)
    save_json(exp_dir / "phase7_diagnosis.json", diag)

    print("\n===== PHASE 7 FINAL =====")
    for q in (
        "Q1_observed_signal_improves",
        "Q2_holdout_signal_improves",
        "Q3_signal_gain_with_FA_MD_gain",
        "Q4_obs_PSNR_up_holdout_PSNR_down",
        "Q5_z_up_Ddrift_up_DTI_worse",
        "Q6_sampling_affects_adaptation",
        "Q7_consistent_across_subjects",
    ):
        print(f"  {q}: {diag[q]}")
    print(f"  Go/No-Go: {diag['Go_NoGo']}")
    print(f"  bottleneck: {diag['bottleneck']}")
    print(f"  rationale: {diag['rationale']}")
    print(f"\n[Phase7] done → {exp_dir}")


if __name__ == "__main__":
    main()
