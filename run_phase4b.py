#!/usr/bin/env python
"""Phase 4-B: Latent Mechanism Diagnostic (inference only).

Fixed theta = Phase4-A epoch_0150. No architecture / loss / physics changes.
"""
from __future__ import annotations

import argparse
import csv
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

DEFAULT_PHASE4A = (
    ROOT
    / "experiments"
    / "population_dti_phase4a"
    / "20260829_192714"
)


def _build_model(cfg: dict[str, Any], train_ids: list[str], device: torch.device) -> PopulationDTIINR:
    return PopulationDTIINR(
        train_subject_ids=train_ids,
        latent_dim=int(cfg.get("latent_dim", 16)),
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 4)),
        pe_freqs=int(cfg.get("pe_freqs", 8)),
    ).to(device)


def _mse_loss(pred, target, bvals, cfg):
    mode = str(cfg.get("signal_normalization", "s0")).lower()
    b0 = float(cfg["b0_threshold"])
    if mode in {"s0", "s0_norm", "normalized"}:
        s0 = s0_obs_from_batch(target, bvals, b0)
        return F.mse_loss(pred / s0, target / s0)
    return F.mse_loss(pred, target)


def adapt_z_new(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    cfg: dict[str, Any],
    device: torch.device,
    n_iters: int = 200,
) -> torch.Tensor:
    """Freeze theta; optimize z_new for n_iters; return detached z."""
    model.freeze_theta()
    z_new = model.new_z(trainable=True, device=device, init="zeros")
    opt = torch.optim.Adam([z_new], lr=float(cfg.get("adapt_lr", 1e-3)))
    batch = int(cfg.get("batch_voxels", 4096))
    b_scale = float(cfg.get("b_scale", 1.0))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    coords_t = torch.from_numpy(subj.train_coords)
    bvals_t = torch.from_numpy(subj.bvals).to(device)
    bvecs_t = torch.from_numpy(subj.bvecs).to(device)
    n_vox = int(coords_t.shape[0])
    for it in range(1, int(n_iters) + 1):
        model.freeze_theta()
        sel = rng.integers(0, n_vox, size=batch, endpoint=False)
        xyz = coords_t[sel].to(device)
        target = torch.from_numpy(subj.dwi_flat[subj.train_flat_idx[sel]]).to(device)
        S0, D = model(xyz, z=z_new)
        pred = predict_signal(S0, D, bvals_t, bvecs_t, b_scale=b_scale)
        loss = _mse_loss(pred, target, bvals_t, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it in (1, 50, 100, 200):
            print(f"  [adapt] iter={it} loss={float(loss.detach()):.6e}", flush=True)
    return z_new.detach().clone()


def metrics_row_from_eval(tag: str, result: dict[str, Any], **extra: Any) -> dict[str, Any]:
    r = result["row"]
    out = {
        "tag": tag,
        "signal_relmse": r.get("signal_RelMSE"),
        "psnr": r.get("signal_PSNR"),
        "ssim": r.get("signal_SSIM"),
        "fa_mae": r.get("FA_MAE"),
        "md_mae": r.get("MD_MAE"),
        "ad_mae": r.get("AD_MAE"),
        "rd_mae": r.get("RD_MAE"),
    }
    out.update(extra)
    return out


@torch.no_grad()
def eval_z(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    z: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
    mode: str = "diag",
) -> dict[str, Any]:
    model.eval()
    model.freeze_theta()
    return evaluate_subject(
        model,
        subj,
        z.to(device),
        device=device,
        cfg=cfg,
        mode=mode,
        seed=int(cfg.get("seed", 42)),
    )


@torch.no_grad()
def map_stats(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    z: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """Mean S0/FA/MD/AD/RD on common_mask (representation stats, not vs WLS)."""
    maps = predict_maps(
        model,
        subj.train_coords,
        subj.train_flat_idx,
        subj.shape_xyz,
        z.to(device),
        device,
        want_D=False,
    )
    m = subj.common_mask
    out = {}
    for k in ("S0", "FA", "MD", "AD", "RD"):
        v = np.asarray(maps[k], dtype=np.float64)[m]
        out[f"mean_{k}"] = float(np.mean(v)) if v.size else float("nan")
        out[f"std_{k}"] = float(np.std(v)) if v.size else float("nan")
    return out


def latent_stats(name: str, z: torch.Tensor) -> dict[str, Any]:
    a = z.detach().float().cpu().numpy().ravel()
    return {
        "name": name,
        "dim": int(a.size),
        "l2_norm": float(np.linalg.norm(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "abs_mean": float(np.mean(np.abs(a))),
    }


def run_b1_alpha(
    model, subj, z_new, cfg, device, out_dir: Path
) -> list[dict[str, Any]]:
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    rows = []
    for a in alphas:
        z = (float(a) * z_new).to(device)
        res = eval_z(model, subj, z, cfg, device, mode="alpha_sensitivity")
        row = metrics_row_from_eval(f"alpha={a}", res, alpha=a)
        rows.append(row)
        print(
            f"  [B1] alpha={a:.2f} PSNR={row['psnr']:.3f} FA={row['fa_mae']:.4f} RelMSE={row['signal_relmse']:.4f}",
            flush=True,
        )
    _write_csv(out_dir / "metrics" / "latent_alpha_sensitivity.csv", rows)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    xs = [r["alpha"] for r in rows]
    axes[0].plot(xs, [r["psnr"] for r in rows], marker="o")
    axes[0].set_title("alpha vs PSNR")
    axes[0].set_xlabel("alpha")
    axes[1].plot(xs, [r["fa_mae"] for r in rows], marker="o")
    axes[1].set_title("alpha vs FA MAE")
    axes[1].set_xlabel("alpha")
    axes[2].plot(xs, [r["md_mae"] for r in rows], marker="o")
    axes[2].set_title("alpha vs MD MAE")
    axes[2].set_xlabel("alpha")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "latent_alpha_sensitivity.png", dpi=150)
    plt.close(fig)
    return rows


def run_b2_interp(
    model,
    train_bundles: dict[str, SubjectBundle],
    latents: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    device,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Interpolate latents; compute representation stats on each train subject's own anatomy."""
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    pairs = [("101309", "102715"), ("101309", "103515")]
    rows = []
    for a_id, b_id in pairs:
        z_a = latents[a_id].to(device)
        z_b = latents[b_id].to(device)
        # Evaluate representation on subject A's anatomy (fixed spatial grid)
        subj = train_bundles[a_id]
        for a in alphas:
            z = (1.0 - float(a)) * z_a + float(a) * z_b
            stats = map_stats(model, subj, z, device)
            # Also signal metrics on subject A observations with mixed latent
            res = eval_z(model, subj, z, cfg, device, mode="interp")
            row = {
                "pair": f"{a_id}->{b_id}",
                "anchor_subject": a_id,
                "alpha": a,
                **stats,
                "signal_relmse": res["row"]["signal_RelMSE"],
                "psnr": res["row"]["signal_PSNR"],
                "fa_mae_vs_wls": res["row"]["FA_MAE"],
                "md_mae_vs_wls": res["row"]["MD_MAE"],
            }
            rows.append(row)
            print(
                f"  [B2] {a_id}->{b_id} a={a:.2f} meanFA={stats['mean_FA']:.4f} "
                f"meanMD={stats['mean_MD']:.6f} PSNR={row['psnr']:.3f}",
                flush=True,
            )
    _write_csv(out_dir / "metrics" / "latent_interpolation.csv", rows)

    # Plot mean FA / MD vs alpha for each pair
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i, (a_id, b_id) in enumerate(pairs):
        sub = [r for r in rows if r["pair"] == f"{a_id}->{b_id}"]
        xs = [r["alpha"] for r in sub]
        axes[i, 0].plot(xs, [r["mean_FA"] for r in sub], marker="o")
        axes[i, 0].set_title(f"mean FA ({a_id}→{b_id})")
        axes[i, 1].plot(xs, [r["mean_MD"] for r in sub], marker="o")
        axes[i, 1].set_title(f"mean MD ({a_id}→{b_id})")
        for ax in axes[i]:
            ax.set_xlabel("alpha")
            ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "latent_interpolation.png", dpi=150)
    plt.close(fig)
    return rows


def run_b3_replacement(
    model,
    test_subj: SubjectBundle,
    latents: dict[str, torch.Tensor],
    z_new: torch.Tensor,
    cfg,
    device,
    out_dir: Path,
) -> list[dict[str, Any]]:
    specs = [
        ("z=0", torch.zeros_like(z_new)),
        ("z_101309", latents["101309"]),
        ("z_102715", latents["102715"]),
        ("z_103515", latents["103515"]),
        ("z_new_adapted", z_new),
    ]
    rows = []
    for name, z in specs:
        res = eval_z(model, test_subj, z.to(device), cfg, device, mode="replacement")
        row = metrics_row_from_eval(name, res, latent_source=name)
        rows.append(row)
        print(
            f"  [B3] {name}: FA={row['fa_mae']:.4f} PSNR={row['psnr']:.3f} RelMSE={row['signal_relmse']:.4f}",
            flush=True,
        )
    _write_csv(out_dir / "metrics" / "latent_replacement.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = [r["latent_source"] for r in rows]
    axes[0].bar(range(len(names)), [r["psnr"] for r in rows])
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_title("Replacement: PSNR on 106319")
    axes[1].bar(range(len(names)), [r["fa_mae"] for r in rows])
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_title("Replacement: FA MAE on 106319")
    for ax in axes:
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "plots" / "latent_replacement.png", dpi=150)
    plt.close(fig)
    return rows


def run_b4_stats(
    latents: dict[str, torch.Tensor], z_new: torch.Tensor, out_dir: Path
) -> list[dict[str, Any]]:
    rows = [
        latent_stats("z_101309", latents["101309"]),
        latent_stats("z_102715", latents["102715"]),
        latent_stats("z_103515", latents["103515"]),
        latent_stats("z_new", z_new),
    ]
    _write_csv(out_dir / "metrics" / "latent_statistics.csv", rows)
    for r in rows:
        print(
            f"  [B4] {r['name']}: ||z||={r['l2_norm']:.4f} mean={r['mean']:.4f} std={r['std']:.4f}",
            flush=True,
        )
    return rows


def diagnose(
    b1: list[dict],
    b2: list[dict],
    b3: list[dict],
    b4: list[dict],
) -> dict[str, Any]:
    # A/B: alpha sensitivity
    b1s = sorted(b1, key=lambda r: float(r["alpha"]))
    psnrs = [float(r["psnr"]) for r in b1s]
    fas = [float(r["fa_mae"]) for r in b1s]
    mds = [float(r["md_mae"]) for r in b1s]
    # Exclude alpha=0 for signal "effect" if needed; compare range
    psnr_range = float(max(psnrs) - min(psnrs))
    fa_range = float(max(fas) - min(fas))
    md_range = float(max(mds) - min(mds))
    # Relative: FA change vs baseline at alpha=1
    fa_at1 = next(float(r["fa_mae"]) for r in b1s if abs(float(r["alpha"]) - 1.0) < 1e-9)
    fa_at0 = next(float(r["fa_mae"]) for r in b1s if abs(float(r["alpha"]) - 0.0) < 1e-9)
    psnr_at1 = next(float(r["psnr"]) for r in b1s if abs(float(r["alpha"]) - 1.0) < 1e-9)
    psnr_at0 = next(float(r["psnr"]) for r in b1s if abs(float(r["alpha"]) - 0.0) < 1e-9)

    A_latent_affects_signal = bool(psnr_range > 1.0 or abs(psnr_at1 - psnr_at0) > 1.0)
    B_latent_affects_fa_md = bool(fa_range > 0.01 or md_range > 1e-5)

    # C: interpolation smoothness — second differences of mean_FA small
    smooth_flags = []
    for pair in sorted({r["pair"] for r in b2}):
        sub = sorted([r for r in b2 if r["pair"] == pair], key=lambda r: float(r["alpha"]))
        fa_m = [float(r["mean_FA"]) for r in sub]
        if len(fa_m) >= 3:
            d2 = [abs(fa_m[i + 1] - 2 * fa_m[i] + fa_m[i - 1]) for i in range(1, len(fa_m) - 1)]
            # smooth if second diffs small relative to total span
            span = max(fa_m) - min(fa_m) + 1e-12
            smooth_flags.append(float(np.mean(d2)) / span < 0.5)
        else:
            smooth_flags.append(True)
    C_interp_smooth = bool(all(smooth_flags)) if smooth_flags else False

    # D: z_new near train distribution?
    train_norms = [float(r["l2_norm"]) for r in b4 if r["name"] != "z_new"]
    znew = next(r for r in b4 if r["name"] == "z_new")
    train_mean_norm = float(np.mean(train_norms))
    train_std_norm = float(np.std(train_norms)) if len(train_norms) > 1 else 0.0
    # also centroid distance in R^16 — approximate via comparing to each train z norms of difference
    # Use stats already: if z_new norm within [min_train, max_train] * 1.5 or within 2 std of mean
    n_new = float(znew["l2_norm"])
    lo, hi = min(train_norms), max(train_norms)
    D_near_distribution = bool(lo * 0.5 <= n_new <= hi * 2.0) or (
        abs(n_new - train_mean_norm) <= 2.0 * max(train_std_norm, 1e-6) + 0.5 * train_mean_norm
    )

    # E: replacement effects
    by_src = {r["latent_source"]: r for r in b3}
    E_summary = {
        k: {
            "psnr": float(by_src[k]["psnr"]),
            "fa_mae": float(by_src[k]["fa_mae"]),
            "signal_relmse": float(by_src[k]["signal_relmse"]),
        }
        for k in by_src
    }

    return {
        "A_latent_magnitude_mainly_affects_signal": A_latent_affects_signal,
        "A_detail": {
            "psnr_alpha0": psnr_at0,
            "psnr_alpha1": psnr_at1,
            "psnr_range": psnr_range,
        },
        "B_latent_magnitude_changes_FA_MD": B_latent_affects_fa_md,
        "B_detail": {
            "fa_alpha0": fa_at0,
            "fa_alpha1": fa_at1,
            "fa_range": fa_range,
            "md_range": md_range,
        },
        "C_latent_interpolation_smooth": C_interp_smooth,
        "D_adapted_latent_near_train_distribution": D_near_distribution,
        "D_detail": {
            "train_l2_norms": {r["name"]: r["l2_norm"] for r in b4 if r["name"] != "z_new"},
            "z_new_l2_norm": n_new,
            "train_mean_l2": train_mean_norm,
            "train_std_l2": train_std_norm,
        },
        "E_latent_replacement_on_unseen": E_summary,
        "answers": {
            "A": "YES — latent magnitude strongly modulates signal"
            if A_latent_affects_signal
            else "NO — signal weakly sensitive to latent magnitude",
            "B": "YES — FA/MD change with latent magnitude"
            if B_latent_affects_fa_md
            else "NO — FA/MD largely insensitive to latent magnitude",
            "C": "YES — interpolation of representation stats is roughly smooth"
            if C_interp_smooth
            else "NO — interpolation shows non-smooth jumps",
            "D": "YES — ||z_new|| lies near training latent norms"
            if D_near_distribution
            else "NO — ||z_new|| is outside typical training latent scale",
            "E": E_summary,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4-B Latent Mechanism Diagnostic")
    ap.add_argument(
        "--phase4a-dir",
        default=str(DEFAULT_PHASE4A),
        help="Phase4-A experiment with epoch_0150 checkpoints",
    )
    ap.add_argument("--config", default="")
    ap.add_argument("--adapt-iters", type=int, default=200)
    args = ap.parse_args()

    phase4a = Path(args.phase4a_dir)
    cfg_path = Path(args.config) if args.config else phase4a / "config" / "run_config.yaml"
    cfg = load_yaml(cfg_path)
    split = split_from_config(cfg)
    device = resolve_device(str(cfg.get("device", "auto")))

    ckpt = phase4a / "checkpoints" / "epoch_0150"
    if not (ckpt / "theta.pt").is_file():
        raise FileNotFoundError(f"missing {ckpt / 'theta.pt'}")

    exp_dir = make_experiment_dir(tag="population_dti_phase4b")
    save_yaml(exp_dir / "config" / "run_config.yaml", cfg)
    save_json(
        exp_dir / "config" / "sources.json",
        {"phase4a_dir": str(phase4a), "theta": str(ckpt / "theta.pt"), "latents": str(ckpt / "latents.pt")},
    )

    print("=" * 72)
    print("Phase 4-B Latent Mechanism Diagnostic")
    print(f"  theta={ckpt / 'theta.pt'}")
    print(f"  out={exp_dir}")
    print("=" * 72)

    model = _build_model(cfg, split["train"], device)
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()
    latents = load_latents(ckpt / "latents.pt")
    for k, v in latents.items():
        latents[k] = v.float()

    trad = Path(cfg["trad_root"])
    test_id = split["test"][0]
    test_subj = load_subject_bundle(subject_id=test_id, cfg=cfg, trad_dir=trad / test_id)
    train_bundles = {
        sid: load_subject_bundle(subject_id=sid, cfg=cfg, trad_dir=trad / sid)
        for sid in split["train"]
    }

    print("\n[B0] adapting z_new on unseen (freeze theta)...", flush=True)
    z_new = adapt_z_new(model, test_subj, cfg, device, n_iters=int(args.adapt_iters))
    torch.save({"z_new": z_new.cpu(), "adapt_iters": int(args.adapt_iters)}, exp_dir / "checkpoints" / "z_new.pt")

    print("\n[B1] latent magnitude sensitivity...", flush=True)
    b1 = run_b1_alpha(model, test_subj, z_new, cfg, device, exp_dir)

    print("\n[B2] latent interpolation...", flush=True)
    b2 = run_b2_interp(model, train_bundles, latents, cfg, device, exp_dir)

    print("\n[B3] latent replacement on unseen...", flush=True)
    b3 = run_b3_replacement(model, test_subj, latents, z_new, cfg, device, exp_dir)

    print("\n[B4] latent statistics...", flush=True)
    b4 = run_b4_stats(latents, z_new, exp_dir)

    print("\n[B5] diagnosis...", flush=True)
    diag = diagnose(b1, b2, b3, b4)
    save_json(exp_dir / "metrics" / "phase4b_diagnosis.json", diag)

    print("\n===== PHASE 4-B DIAGNOSIS =====")
    for k, v in diag["answers"].items():
        if k != "E":
            print(f"  {k}: {v}")
    print("  E: latent replacement on 106319:")
    for src, m in diag["answers"]["E"].items():
        print(f"     {src}: FA={m['fa_mae']:.4f} PSNR={m['psnr']:.3f} RelMSE={m['signal_relmse']:.4f}")
    print(f"\n[Phase4B] done → {exp_dir}")


if __name__ == "__main__":
    main()
