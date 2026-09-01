#!/usr/bin/env python
"""Phase 6 diagnostic: constrained latent adaptation / Pareto trade-off.

Inference / diagnostic only. Does NOT modify PopulationDTIINR, training, physics,
or Independent/Shared baselines.

Fixed:
  theta = Phase4-A epoch_0150
  subject = 106319
  z init = 0
  optimize z_new only (theta frozen)

WLS-DTI is ORACLE diagnostic reference only — not a formal test protocol.
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
DEFAULT_PHASE5 = ROOT / "experiments" / "population_dti_phase5" / "20260830_133430"
EPS = 1e-12
KEY_ITERS = (0, 10, 50, 100, 200, 500, 1000)

# Phase 6-A: user λ plus high-λ bridge (signal grads dominate value-balanced mix).
LAMBDA_DTI_USER = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0)
LAMBDA_DTI_HIGH = (50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0)
LAMBDA_DTI_SWEEP = LAMBDA_DTI_USER + LAMBDA_DTI_HIGH

# Phase 6-B
LAMBDA_DTI_NORM = (0.01, 0.1, 1.0, 100.0, 1000.0)
LAMBDA_Z_NORM = (0.0, 0.001, 0.01, 0.1)


def _build_model(cfg: dict[str, Any], train_ids: list[str], device: torch.device) -> PopulationDTIINR:
    return PopulationDTIINR(
        train_subject_ids=train_ids,
        latent_dim=int(cfg.get("latent_dim", 16)),
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 4)),
        pe_freqs=int(cfg.get("pe_freqs", 8)),
    ).to(device)


def signal_mse_loss(pred, target, bvals, cfg) -> torch.Tensor:
    """Formal Population signal loss: MSE(pred/S0, obs/S0)."""
    mode = str(cfg.get("signal_normalization", "s0")).lower()
    b0 = float(cfg["b0_threshold"])
    if mode in {"s0", "s0_norm", "normalized"}:
        s0 = s0_obs_from_batch(target, bvals, b0)
        return F.mse_loss(pred / s0, target / s0)
    return F.mse_loss(pred, target)


def dti_mean_frobenius_loss(D_pred: torch.Tensor, D_ref: torch.Tensor) -> torch.Tensor:
    """Diagnostic only: mean ||D_pred - D_ref||_F  (NOT used in formal training)."""
    diff = D_pred - D_ref
    return torch.linalg.matrix_norm(diff, ord="fro").mean()


def latent_norm_sq(z: torch.Tensor) -> torch.Tensor:
    return torch.sum(z.float() ** 2)


def _fro(D: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(D.astype(np.float64) ** 2, axis=(-2, -1)))


def latent_norm(z: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(z.detach().float()).cpu())


def _prepare_common_index(subj: SubjectBundle) -> tuple[np.ndarray, np.ndarray]:
    """Returns (D_wls_flat [N,3,3], common_train_pos into train_coords)."""
    X, Y, Z = subj.shape_xyz
    D_wls_flat = np.asarray(subj.ref["D"], dtype=np.float32).reshape(X * Y * Z, 3, 3)
    common_flat = np.flatnonzero(subj.common_mask.reshape(-1))
    train_in_common = np.intersect1d(subj.train_flat_idx, common_flat, assume_unique=False)
    if train_in_common.size == 0:
        train_in_common = subj.train_flat_idx
    flat_to_trainpos = {int(f): i for i, f in enumerate(subj.train_flat_idx.tolist())}
    common_train_pos = np.asarray(
        [flat_to_trainpos[int(f)] for f in train_in_common if int(f) in flat_to_trainpos],
        dtype=np.int64,
    )
    if common_train_pos.size == 0:
        common_train_pos = np.arange(subj.train_coords.shape[0], dtype=np.int64)
    return D_wls_flat, common_train_pos


@torch.no_grad()
def snapshot_metrics(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    z: torch.Tensor,
    D0: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    iteration: int,
    *,
    L_signal_batch: float | None = None,
    L_DTI_batch: float | None = None,
    L_z_batch: float | None = None,
) -> dict[str, Any]:
    model.eval()
    model.freeze_theta()
    z = z.detach()
    res = evaluate_subject(
        model,
        subj,
        z,
        device=device,
        cfg=cfg,
        mode="phase6",
        adapt_iter=iteration,
        seed=int(cfg.get("seed", 42)),
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
    mask = subj.common_mask
    Dv = maps["D"][mask].astype(np.float64)
    D0v = D0[mask].astype(np.float64)
    Dref = np.asarray(subj.ref["D"], dtype=np.float64)[mask]

    dD = Dv - D0v
    fro_d = _fro(dD)
    fro0 = _fro(D0v)
    err_wls = _fro(Dv - Dref)

    return {
        "iteration": int(iteration),
        "latent_norm": latent_norm(z),
        "L_signal": float(L_signal_batch) if L_signal_batch is not None else float(row_sig["signal_MSE"]),
        "L_DTI": float(L_DTI_batch) if L_DTI_batch is not None else float(np.mean(err_wls)),
        "L_z": float(L_z_batch) if L_z_batch is not None else float(latent_norm(z) ** 2),
        "PSNR": float(row_sig["signal_PSNR"]),
        "RelMSE": float(row_sig["signal_RelMSE"]),
        "MSE": float(row_sig["signal_MSE"]),
        "FA_MAE": float(row_sig["FA_MAE"]),
        "MD_MAE": float(row_sig["MD_MAE"]),
        "AD_MAE": float(row_sig["AD_MAE"]),
        "RD_MAE": float(row_sig["RD_MAE"]),
        "mean_D_fro_error_vs_wls": float(np.mean(err_wls)),
        "mean_relative_delta_D": float(np.mean(fro_d / (fro0 + EPS))),
        "mean_delta_D_fro": float(np.mean(fro_d)),
    }


def run_mixed_adaptation(
    *,
    model: PopulationDTIINR,
    subj: SubjectBundle,
    cfg: dict[str, Any],
    device: torch.device,
    lambda_DTI: float,
    lambda_z: float,
    max_iter: int,
    key_iters: tuple[int, ...],
    D0_vol: np.ndarray,
    seed: int,
    tag: str,
    dti_ref_scale: float,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    """
    L = L_signal + lambda_DTI * (L_DTI / dti_ref_scale) + lambda_z * ||z||^2

    L_DTI = mean ||D_pred - D_WLS||_F   (oracle diagnostic)
    dti_ref_scale = mean ||D(x,0)-D_WLS||_F so that λ≈1 makes DTI term O(1)
      comparable to late-stage L_signal (documented scale bridge; raw L_DTI still logged).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    dti_ref = max(float(dti_ref_scale), EPS)

    model.freeze_theta()
    z = model.new_z(trainable=True, device=device, init="zeros")
    opt = torch.optim.Adam([z], lr=float(cfg.get("adapt_lr", cfg.get("lr", 1e-3))))
    batch = int(cfg.get("batch_voxels", 4096))
    b_scale = float(cfg.get("b_scale", 1.0))
    bvals_t = torch.from_numpy(subj.bvals).to(device)
    bvecs_t = torch.from_numpy(subj.bvecs).to(device)
    coords_t = torch.from_numpy(subj.train_coords)
    D_wls_flat, common_train_pos = _prepare_common_index(subj)

    max_iter = int(max_iter)
    key_set = sorted({int(x) for x in key_iters if int(x) <= max_iter})
    if max_iter not in key_set:
        key_set.append(max_iter)
    key_lookup = set(key_set)
    rows: list[dict[str, Any]] = []

    last_Ls = last_Ld = last_Lz = 0.0

    def record(it: int) -> None:
        m = snapshot_metrics(
            model,
            subj,
            z,
            D0_vol,
            cfg,
            device,
            it,
            L_signal_batch=last_Ls,
            L_DTI_batch=last_Ld,
            L_z_batch=last_Lz,
        )
        m["lambda_DTI"] = float(lambda_DTI)
        m["lambda_z"] = float(lambda_z)
        m["tag"] = tag
        m["dti_ref_scale"] = float(dti_ref)
        m["L_DTI_scaled"] = float(m["L_DTI"]) / dti_ref
        m["weighted_DTI"] = float(lambda_DTI) * float(m["L_DTI"]) / dti_ref
        m["weighted_z"] = float(lambda_z) * float(m["L_z"])
        rows.append(m)
        print(
            f"  [{tag}] it={it:4d} ||z||={m['latent_norm']:.4f} "
            f"PSNR={m['PSNR']:.3f} FA={m['FA_MAE']:.4f} "
            f"Derr={m['mean_D_fro_error_vs_wls']:.4e} "
            f"Ls={m['L_signal']:.4e} Ld={m['L_DTI']:.4e} wD={m['weighted_DTI']:.4e}",
            flush=True,
        )

    if 0 in key_lookup:
        last_Ls = last_Ld = last_Lz = float("nan")
        record(0)

    for it in range(1, max_iter + 1):
        model.train()
        model.freeze_theta()
        opt.zero_grad(set_to_none=True)

        n_c = int(common_train_pos.shape[0])
        sel_local = rng.integers(0, n_c, size=min(batch, n_c), endpoint=False)
        pos = common_train_pos[sel_local]
        xyz = coords_t[pos].to(device)
        flat_idx = subj.train_flat_idx[pos]
        target = torch.from_numpy(subj.dwi_flat[flat_idx]).to(device)
        D_ref = torch.from_numpy(D_wls_flat[flat_idx]).to(device=device, dtype=torch.float32)

        S0, D = model(xyz, z=z)
        pred = predict_signal(S0, D, bvals_t, bvecs_t, b_scale=b_scale)
        Ls = signal_mse_loss(pred, target, bvals_t, cfg)
        Ld = dti_mean_frobenius_loss(D, D_ref)
        Lz = latent_norm_sq(z)
        loss = Ls + float(lambda_DTI) * (Ld / dti_ref) + float(lambda_z) * Lz

        loss.backward()
        for p in model.theta_parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                raise RuntimeError("theta received gradients — forbidden in Phase 6")
        opt.step()

        last_Ls = float(Ls.detach().cpu())
        last_Ld = float(Ld.detach().cpu())
        last_Lz = float(Lz.detach().cpu())

        if it in key_lookup:
            record(it)

    return rows, z.detach().clone()


def plot_phase6a(final_rows: list[dict[str, Any]], traj: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lams = [r["lambda_DTI"] for r in final_rows]
    xlabels = [str(x) for x in lams]

    def bar(path, ys, title, ylabel):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(ys)), ys)
        ax.set_xticks(range(len(ys)))
        ax.set_xticklabels(xlabels, rotation=30)
        ax.set_xlabel("lambda_DTI")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    bar(out_dir / "lambda_vs_PSNR.png", [r["PSNR"] for r in final_rows], "lambda vs PSNR (@1000)", "PSNR")
    bar(out_dir / "lambda_vs_FA_MAE.png", [r["FA_MAE"] for r in final_rows], "lambda vs FA MAE (@1000)", "FA MAE")
    bar(
        out_dir / "lambda_vs_D_error.png",
        [r["mean_D_fro_error_vs_wls"] for r in final_rows],
        "lambda vs mean ||D-D_WLS||_F (@1000)",
        "D Frobenius error",
    )
    bar(out_dir / "lambda_vs_z_norm.png", [r["latent_norm"] for r in final_rows], "||z|| vs lambda (@1000)", "||z||_2")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        [r["PSNR"] for r in final_rows],
        [r["FA_MAE"] for r in final_rows],
        c=np.log10(np.asarray(lams) + 1e-12),
        s=80,
        cmap="viridis",
    )
    for r in final_rows:
        ax.annotate(f"λ={r['lambda_DTI']}", (r["PSNR"], r["FA_MAE"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("PSNR (higher better)")
    ax.set_ylabel("FA MAE (lower better)")
    ax.set_title("Phase 6-A Pareto: PSNR vs FA MAE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_PSNR_vs_FA.png", dpi=150)
    plt.close(fig)

    # trajectories by lambda
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    by_lam: dict[float, list] = {}
    for r in traj:
        by_lam.setdefault(float(r["lambda_DTI"]), []).append(r)
    for lam, rows in sorted(by_lam.items()):
        xs = [r["iteration"] for r in rows]
        axes[0, 0].plot(xs, [r["PSNR"] for r in rows], marker="o", label=f"λ={lam}")
        axes[0, 1].plot(xs, [r["FA_MAE"] for r in rows], marker="o", label=f"λ={lam}")
        axes[1, 0].plot(xs, [r["mean_D_fro_error_vs_wls"] for r in rows], marker="o", label=f"λ={lam}")
        axes[1, 1].plot(xs, [r["latent_norm"] for r in rows], marker="o", label=f"λ={lam}")
    axes[0, 0].set_title("PSNR")
    axes[0, 1].set_title("FA MAE")
    axes[1, 0].set_title("D error vs WLS")
    axes[1, 1].set_title("||z||")
    for ax in axes.ravel():
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Phase 6-A trajectories")
    fig.tight_layout()
    fig.savefig(out_dir / "phase6a_trajectories.png", dpi=150)
    plt.close(fig)


def plot_phase6b(final_rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # heatmap-like scatter for (lambda_DTI, lambda_z)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key, title in zip(
        axes,
        ("PSNR", "FA_MAE", "latent_norm"),
        ("PSNR", "FA MAE", "||z||"),
    ):
        sc = ax.scatter(
            [r["lambda_DTI"] for r in final_rows],
            [r["lambda_z"] for r in final_rows],
            c=[r[key] for r in final_rows],
            s=120,
            cmap="viridis",
        )
        ax.set_xscale("symlog", linthresh=0.001)
        ax.set_yscale("symlog", linthresh=0.0001)
        ax.set_xlabel("lambda_DTI")
        ax.set_ylabel("lambda_z")
        ax.set_title(title)
        fig.colorbar(sc, ax=ax, fraction=0.046)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Phase 6-B norm regularization")
    fig.tight_layout()
    fig.savefig(out_dir / "norm_regularization_grid.png", dpi=150)
    plt.close(fig)


def plot_pareto_c(
    points: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    def scatter(ax, xk, yk, title, xlab, ylab):
        for p in points:
            ax.scatter(p[xk], p[yk], s=90, label=p["name"])
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    scatter(axes[0], "PSNR", "FA_MAE", "PSNR ↔ FA MAE", "PSNR ↑", "FA MAE ↓")
    scatter(axes[1], "PSNR", "mean_D_fro_error_vs_wls", "PSNR ↔ D error", "PSNR ↑", "||D-D_WLS||_F ↓")
    scatter(axes[2], "latent_norm", "mean_D_fro_error_vs_wls", "||z|| ↔ D error", "||z||", "||D-D_WLS||_F ↓")
    fig.suptitle("Phase 6-C Pareto analysis")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_analysis.png", dpi=150)
    plt.close(fig)


def _is_dominated(p: dict, others: list[dict], better_psnr: bool = True) -> bool:
    """Dominated if some other has ≥ PSNR and ≤ FA and ≤ Derr (strict in at least one)."""
    for o in others:
        if o is p:
            continue
        ge_psnr = o["PSNR"] >= p["PSNR"] - 1e-9
        le_fa = o["FA_MAE"] <= p["FA_MAE"] + 1e-12
        le_d = o["mean_D_fro_error_vs_wls"] <= p["mean_D_fro_error_vs_wls"] + 1e-15
        strict = (
            o["PSNR"] > p["PSNR"] + 1e-9
            or o["FA_MAE"] < p["FA_MAE"] - 1e-12
            or o["mean_D_fro_error_vs_wls"] < p["mean_D_fro_error_vs_wls"] - 1e-15
        )
        if ge_psnr and le_fa and le_d and strict:
            return True
    return False


def diagnose_phase6(
    sweep_final: list[dict[str, Any]],
    norm_final: list[dict[str, Any]],
    refs: dict[str, dict[str, Any]],
    mean_train_norm: float,
) -> dict[str, Any]:
    z0 = refs["z0"]
    sig5 = refs["phase5_signal"]
    dti5 = refs["phase5_dti"]

    # Q2: any lambda that improves BOTH PSNR and FA vs z=0
    both_better = []
    for r in sweep_final:
        if r["PSNR"] > z0["PSNR"] + 0.3 and r["FA_MAE"] < z0["FA_MAE"] - 0.001:
            both_better.append(
                {
                    "lambda_DTI": r["lambda_DTI"],
                    "PSNR": r["PSNR"],
                    "FA_MAE": r["FA_MAE"],
                    "D_err": r["mean_D_fro_error_vs_wls"],
                    "latent_norm": r["latent_norm"],
                }
            )

    # best compromise: maximize PSNR - α*FA (α chosen so scales comparable)
    fa_scale = max(abs(z0["FA_MAE"]), 1e-3)
    psnr_span = max(abs(sig5["PSNR"] - dti5["PSNR"]), 1.0)

    def score(r):
        # higher better: normalized PSNR gain vs FA penalty vs D err gain
        return (r["PSNR"] - z0["PSNR"]) / psnr_span - (r["FA_MAE"] - z0["FA_MAE"]) / fa_scale

    candidates = sweep_final + norm_final
    best = max(candidates, key=score)

    # latent runaway?
    runaway = bool(dti5["latent_norm"] > 1.5 * mean_train_norm)

    # norm reg helps?
    dti_like = [r for r in norm_final if r["lambda_DTI"] >= 0.1]
    with_z = [r for r in dti_like if r["lambda_z"] > 0]
    without_z = [r for r in dti_like if r["lambda_z"] == 0]
    norm_helps = False
    norm_detail = {}
    if with_z and without_z:
        # compare mean ||z|| and whether PSNR recovers while FA stays better than z0
        mean_z_with = float(np.mean([r["latent_norm"] for r in with_z]))
        mean_z_wo = float(np.mean([r["latent_norm"] for r in without_z]))
        best_with = max(with_z, key=score)
        best_wo = max(without_z, key=score)
        norm_helps = bool(
            mean_z_with < mean_z_wo * 0.9
            and (
                best_with["PSNR"] > best_wo["PSNR"] + 0.2
                or (best_with["latent_norm"] < 1.2 * mean_train_norm and best_with["FA_MAE"] < z0["FA_MAE"])
            )
        )
        norm_detail = {
            "mean_z_with_reg": mean_z_with,
            "mean_z_without_reg": mean_z_wo,
            "best_with_reg": {
                "lambda_DTI": best_with["lambda_DTI"],
                "lambda_z": best_with["lambda_z"],
                "PSNR": best_with["PSNR"],
                "FA_MAE": best_with["FA_MAE"],
                "latent_norm": best_with["latent_norm"],
            },
            "best_without_reg": {
                "lambda_DTI": best_wo["lambda_DTI"],
                "lambda_z": best_wo["lambda_z"],
                "PSNR": best_wo["PSNR"],
                "FA_MAE": best_wo["FA_MAE"],
                "latent_norm": best_wo["latent_norm"],
            },
        }

    conflict_relieved = bool(len(both_better) > 0) or bool(
        best["PSNR"] > z0["PSNR"] + 0.3 and best["FA_MAE"] < z0["FA_MAE"] - 0.001
    )

    # scale check: is DTI term active in sweep?
    scale_notes = []
    for r in sweep_final:
        if r["iteration"] == 0:
            continue
        w = float(r.get("weighted_DTI", r["lambda_DTI"] * r["L_DTI"]))
        ratio = w / (abs(r["L_signal"]) + EPS)
        if r["lambda_DTI"] > 0 and ratio < 0.01:
            scale_notes.append(
                f"λ={r['lambda_DTI']}: weighted_DTI / L_signal ≈ {ratio:.3e} (DTI term may be weak)"
            )
        elif r["lambda_DTI"] > 0:
            scale_notes.append(
                f"λ={r['lambda_DTI']}: weighted_DTI / L_signal ≈ {ratio:.3e}"
            )

    pareto_pts = [
        {"name": "z0", **z0},
        {"name": "phase5_signal", **sig5},
        {"name": "phase5_dti_oracle", **dti5},
        {
            "name": "phase6_best_compromise",
            "PSNR": best["PSNR"],
            "FA_MAE": best["FA_MAE"],
            "mean_D_fro_error_vs_wls": best["mean_D_fro_error_vs_wls"],
            "latent_norm": best["latent_norm"],
            "lambda_DTI": best.get("lambda_DTI"),
            "lambda_z": best.get("lambda_z", 0.0),
        },
    ]
    # add all finals for frontier
    all_named = []
    for r in candidates:
        all_named.append(
            {
                "name": f"λD={r['lambda_DTI']}_λz={r.get('lambda_z', 0)}",
                "PSNR": r["PSNR"],
                "FA_MAE": r["FA_MAE"],
                "mean_D_fro_error_vs_wls": r["mean_D_fro_error_vs_wls"],
                "latent_norm": r["latent_norm"],
            }
        )
    frontier = [p for p in all_named if not _is_dominated(p, all_named)]

    go_v1 = conflict_relieved or len(both_better) > 0 or (
        best["FA_MAE"] < z0["FA_MAE"] - 0.002 and best["PSNR"] > sig5["PSNR"] - 5
    )
    # If no compromise and oracle shows capability → hold for unsupervised objective, not immediate v2
    enter_v2 = False  # Phase6 policy: do not jump to v2; discuss only if no compromise AND oracle capable
    if not conflict_relieved and dti5["FA_MAE"] < z0["FA_MAE"] - 0.005:
        # discuss v2 later; still CONDITIONAL hold on unsupervised objective first
        go_msg = (
            "HOLD_v1 — mixed objective did not yield simultaneous signal+DTI gain vs z=0; "
            "oracle still shows DTI capability → prioritize unsupervised/identifiable adaptation, "
            "not immediate architecture v2"
        )
    elif conflict_relieved:
        go_msg = (
            "CONDITIONAL_GO_v1 — mixed/constrained adaptation finds a Pareto compromise; "
            "next: unsupervised / signal-observable identifiable adaptation objective"
        )
    else:
        go_msg = (
            "HOLD_v1 — partial trade-offs only; continue objective research under v1 architecture"
        )

    return {
        "Q1_mixed_objective_relieves_conflict": conflict_relieved,
        "Q2_exists_lambda_both_better_than_z0": len(both_better) > 0,
        "Q2_both_better_configs": both_better,
        "Q3_phase5_dti_z_is_runaway": runaway,
        "Q3_detail": {
            "z_DTI": dti5["latent_norm"],
            "mean_train_norm": mean_train_norm,
            "threshold_1p5x_train": 1.5 * mean_train_norm,
        },
        "Q4_norm_regularization_helps": norm_helps,
        "Q4_detail": norm_detail,
        "Q5_pareto_compromise_exists": conflict_relieved or len(frontier) >= 2,
        "Q6_discuss_v2_now": enter_v2,
        "best_compromise": {
            "tag": best.get("tag"),
            "lambda_DTI": best.get("lambda_DTI"),
            "lambda_z": best.get("lambda_z", 0.0),
            "PSNR": best["PSNR"],
            "FA_MAE": best["FA_MAE"],
            "MD_MAE": best.get("MD_MAE"),
            "mean_D_fro_error_vs_wls": best["mean_D_fro_error_vs_wls"],
            "latent_norm": best["latent_norm"],
            "score": float(score(best)),
        },
        "references": {
            "z0": z0,
            "phase5_signal": sig5,
            "phase5_dti_oracle": dti5,
        },
        "pareto_frontier_names": [p["name"] for p in frontier],
        "pareto_points": pareto_pts,
        "scale_warnings": scale_notes,
        "Go_NoGo": go_msg,
        "evidence_for_v2": False,
        "next_step": (
            "Keep v1 architecture. Research unsupervised / signal-observable adaptation "
            "objectives that improve DTI without WLS oracle. Use Phase6 Pareto points as diagnostics only."
        ),
    }


def _load_phase5_refs(phase5_dir: Path) -> dict[str, dict[str, Any]]:
    """Load Phase5 final metrics for Pareto anchors."""
    import json

    sig = json.loads((phase5_dir / "signal_guided" / "final_metrics.json").read_text(encoding="utf-8"))
    dti = json.loads((phase5_dir / "dti_guided" / "final_metrics.json").read_text(encoding="utf-8"))
    # z0 from either trajectory iter 0
    import csv

    z0 = None
    with open(phase5_dir / "metrics" / "phase5a_latent_trajectory.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row["iteration"])) == 0:
                z0 = {
                    "PSNR": float(row["PSNR"]),
                    "FA_MAE": float(row["FA_MAE"]),
                    "MD_MAE": float(row["MD_MAE"]),
                    "mean_D_fro_error_vs_wls": float(row["mean_D_fro_error_vs_wls"]),
                    "latent_norm": float(row["latent_norm"]),
                    "RelMSE": float(row["RelMSE"]),
                }
                break
    if z0 is None:
        raise RuntimeError("Phase5 trajectory missing iter 0")

    def pick(d: dict) -> dict:
        return {
            "PSNR": float(d["PSNR"]),
            "FA_MAE": float(d["FA_MAE"]),
            "MD_MAE": float(d.get("MD_MAE", float("nan"))),
            "mean_D_fro_error_vs_wls": float(d["mean_D_fro_error_vs_wls"]),
            "latent_norm": float(d["latent_norm"]),
            "RelMSE": float(d.get("RelMSE", float("nan"))),
        }

    return {"z0": z0, "phase5_signal": pick(sig), "phase5_dti": pick(dti)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 6 constrained latent adaptation diagnostics")
    ap.add_argument("--phase4a-dir", default=str(DEFAULT_PHASE4A))
    ap.add_argument("--phase5-dir", default=str(DEFAULT_PHASE5))
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-6b", action="store_true")
    args = ap.parse_args()

    phase4a = Path(args.phase4a_dir)
    phase5 = Path(args.phase5_dir)
    cfg = dict(load_yaml(phase4a / "config" / "run_config.yaml"))
    cfg["adapt_lr"] = float(cfg.get("adapt_lr", cfg.get("lr", 1e-3)))
    split = split_from_config(cfg)
    device = resolve_device(str(cfg.get("device", "auto")))
    ckpt = phase4a / "checkpoints" / "epoch_0150"

    exp_dir = make_experiment_dir(tag="population_dti_phase6")
    for sub in ("metrics", "plots", "lambda_sweep", "norm_reg"):
        (exp_dir / sub).mkdir(parents=True, exist_ok=True)
    save_yaml(
        exp_dir / "phase6_config.yaml",
        {
            **cfg,
            "max_iter": args.max_iter,
            "key_iters": list(KEY_ITERS),
            "lambda_DTI_sweep": list(LAMBDA_DTI_SWEEP),
            "lambda_DTI_norm": list(LAMBDA_DTI_NORM),
            "lambda_z_norm": list(LAMBDA_Z_NORM),
            "L_DTI_definition": "mean ||D_pred - D_WLS||_F (oracle diagnostic)",
            "L_adapt": "L_signal + lambda_DTI * (L_DTI / dti_ref_scale) + lambda_z * ||z||^2",
            "dti_ref_scale": "mean ||D(x,z=0)-D_WLS||_F so user lambdas have O(1) weight",
            "note": "WLS is diagnostic only — not formal test protocol",
        },
    )
    save_json(
        exp_dir / "config" / "sources.json",
        {"theta": str(ckpt / "theta.pt"), "phase4a": str(phase4a), "phase5": str(phase5)},
    )

    print("=" * 72)
    print("Phase 6 Diagnostic (constrained latent adaptation)")
    print(f"  theta={ckpt / 'theta.pt'}")
    print(f"  out={exp_dir}")
    print("=" * 72)

    model = _build_model(cfg, split["train"], device)
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()
    train_latents = {k: v.float() for k, v in load_latents(ckpt / "latents.pt").items()}
    mean_train_norm = float(np.mean([float(torch.linalg.vector_norm(v)) for v in train_latents.values()]))

    trad = Path(cfg["trad_root"])
    test_id = split["test"][0]
    subj = load_subject_bundle(subject_id=test_id, cfg=cfg, trad_dir=trad / test_id)
    print(f"[Phase6] subject={test_id} common={int(subj.common_mask.sum())} mean_train||z||={mean_train_norm:.4f}", flush=True)

    print("[Phase6] computing D(x,z=0) baseline...", flush=True)
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
    mask = subj.common_mask
    dti_ref_scale = float(np.mean(_fro(D0[mask].astype(np.float64) - np.asarray(subj.ref["D"], dtype=np.float64)[mask])))
    print(f"[Phase6] dti_ref_scale (z=0 mean ||D-D_WLS||_F) = {dti_ref_scale:.6e}", flush=True)
    save_json(exp_dir / "config" / "dti_ref_scale.json", {"dti_ref_scale": dti_ref_scale})

    # ----- Phase 6-A -----
    print("\n===== Phase 6-A: lambda_DTI sweep =====", flush=True)
    sweep_traj: list[dict[str, Any]] = []
    sweep_final: list[dict[str, Any]] = []
    for lam in LAMBDA_DTI_SWEEP:
        load_theta(ckpt / "theta.pt", model, map_location=device)
        model.freeze_theta()
        tag = f"lambdaDTI={lam}"
        print(f"\n--- {tag} ---", flush=True)
        rows, z_final = run_mixed_adaptation(
            model=model,
            subj=subj,
            cfg=cfg,
            device=device,
            lambda_DTI=float(lam),
            lambda_z=0.0,
            max_iter=int(args.max_iter),
            key_iters=KEY_ITERS,
            D0_vol=D0,
            seed=int(args.seed),
            tag=tag,
            dti_ref_scale=dti_ref_scale,
        )
        sweep_traj.extend(rows)
        sweep_final.append(rows[-1])
        torch.save({"z": z_final.cpu(), "lambda_DTI": lam, "lambda_z": 0.0}, exp_dir / "lambda_sweep" / f"z_lambda_{lam}.pt")
        _write_csv(exp_dir / "lambda_sweep" / f"trajectory_lambda_{lam}.csv", rows)

    _write_csv(exp_dir / "metrics" / "lambda_sweep.csv", sweep_traj)
    _write_csv(exp_dir / "metrics" / "lambda_sweep_final.csv", sweep_final)
    plot_phase6a(sweep_final, sweep_traj, exp_dir / "plots")

    # ----- Phase 6-B -----
    norm_traj: list[dict[str, Any]] = []
    norm_final: list[dict[str, Any]] = []
    if not args.skip_6b:
        print("\n===== Phase 6-B: latent norm regularization =====", flush=True)
        for lam_d in LAMBDA_DTI_NORM:
            for lam_z in LAMBDA_Z_NORM:
                load_theta(ckpt / "theta.pt", model, map_location=device)
                model.freeze_theta()
                tag = f"lambdaDTI={lam_d}_lambdaz={lam_z}"
                print(f"\n--- {tag} ---", flush=True)
                rows, z_final = run_mixed_adaptation(
                    model=model,
                    subj=subj,
                    cfg=cfg,
                    device=device,
                    lambda_DTI=float(lam_d),
                    lambda_z=float(lam_z),
                    max_iter=int(args.max_iter),
                    key_iters=KEY_ITERS,
                    D0_vol=D0,
                    seed=int(args.seed),
                    tag=tag,
                    dti_ref_scale=dti_ref_scale,
                )
                norm_traj.extend(rows)
                norm_final.append(rows[-1])
                torch.save(
                    {"z": z_final.cpu(), "lambda_DTI": lam_d, "lambda_z": lam_z},
                    exp_dir / "norm_reg" / f"z_ld{lam_d}_lz{lam_z}.pt",
                )
        _write_csv(exp_dir / "metrics" / "norm_regularization.csv", norm_traj)
        _write_csv(exp_dir / "metrics" / "norm_regularization_final.csv", norm_final)
        plot_phase6b(norm_final, exp_dir / "plots")
    else:
        print("[Phase6] skip 6-B", flush=True)

    # ----- Phase 6-C -----
    print("\n===== Phase 6-C: Pareto analysis =====", flush=True)
    refs = _load_phase5_refs(phase5)
    # use live z0 from sweep if available
    z0_live = next((r for r in sweep_traj if r["iteration"] == 0 and r["lambda_DTI"] == 0.0), None)
    if z0_live is not None:
        refs["z0"] = {
            "PSNR": z0_live["PSNR"],
            "FA_MAE": z0_live["FA_MAE"],
            "MD_MAE": z0_live["MD_MAE"],
            "mean_D_fro_error_vs_wls": z0_live["mean_D_fro_error_vs_wls"],
            "latent_norm": z0_live["latent_norm"],
            "RelMSE": z0_live["RelMSE"],
        }

    diag = diagnose_phase6(sweep_final, norm_final, refs, mean_train_norm)
    save_json(exp_dir / "phase6_diagnosis.json", diag)

    pareto_rows = list(diag["pareto_points"])
    # add all finals
    for r in sweep_final + norm_final:
        pareto_rows.append(
            {
                "name": r.get("tag", f"λD={r['lambda_DTI']}"),
                "PSNR": r["PSNR"],
                "FA_MAE": r["FA_MAE"],
                "mean_D_fro_error_vs_wls": r["mean_D_fro_error_vs_wls"],
                "latent_norm": r["latent_norm"],
                "lambda_DTI": r.get("lambda_DTI"),
                "lambda_z": r.get("lambda_z", 0.0),
            }
        )
    _write_csv(exp_dir / "metrics" / "pareto_points.csv", pareto_rows)
    plot_pareto_c(diag["pareto_points"], exp_dir / "plots")

    # also overlay all candidates on a fuller plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(refs["z0"]["PSNR"], refs["z0"]["FA_MAE"], s=120, marker="*", label="z=0", zorder=5)
    ax.scatter(refs["phase5_signal"]["PSNR"], refs["phase5_signal"]["FA_MAE"], s=100, marker="s", label="P5 signal", zorder=5)
    ax.scatter(refs["phase5_dti"]["PSNR"], refs["phase5_dti"]["FA_MAE"], s=100, marker="D", label="P5 DTI oracle", zorder=5)
    for r in sweep_final:
        ax.scatter(r["PSNR"], r["FA_MAE"], s=60, label=f"6A λ={r['lambda_DTI']}")
    for r in norm_final:
        if r["lambda_z"] > 0:
            ax.scatter(r["PSNR"], r["FA_MAE"], s=40, marker="^", alpha=0.8, label=f"6B λD={r['lambda_DTI']} λz={r['lambda_z']}")
    ax.set_xlabel("PSNR ↑")
    ax.set_ylabel("FA MAE ↓")
    ax.set_title("Phase 6 full Pareto (PSNR vs FA)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(exp_dir / "plots" / "pareto_full_PSNR_FA.png", dpi=150)
    plt.close(fig)

    print("\n===== PHASE 6 FINAL =====")
    print(f"  Q1 mixed relieves conflict: {diag['Q1_mixed_objective_relieves_conflict']}")
    print(f"  Q2 both better than z0: {diag['Q2_exists_lambda_both_better_than_z0']} → {diag['Q2_both_better_configs']}")
    print(f"  Q3 DTI z runaway: {diag['Q3_phase5_dti_z_is_runaway']} (||z||={diag['Q3_detail']['z_DTI']:.3f}, train_mean={mean_train_norm:.3f})")
    print(f"  Q4 norm reg helps: {diag['Q4_norm_regularization_helps']}")
    print(f"  best compromise: {diag['best_compromise']}")
    print(f"  Go/No-Go: {diag['Go_NoGo']}")
    if diag["scale_warnings"]:
        print("  scale warnings (first 3):")
        for w in diag["scale_warnings"][:3]:
            print(f"    - {w}")
    print(f"\n[Phase6] done → {exp_dir}")


if __name__ == "__main__":
    main()
