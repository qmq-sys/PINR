#!/usr/bin/env python
"""Phase 4-A: Checkpoint Selection Diagnostic.

Retrain Population with epoch snapshots (model unchanged) and evaluate
unseen test under zero-shot / latent adaptation for each checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import SubjectBundle, load_subject_bundle, s0_obs_from_batch
from data.split import split_from_config
from evaluate import evaluate_subject
from models.population_dti_inr import PopulationDTIINR
from physics.dti_forward import predict_signal
from train_population import train_population
from utils_io import (
    load_theta,
    load_yaml,
    make_experiment_dir,
    resolve_device,
    save_json,
    save_yaml,
    _write_csv,
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


def _row_from_eval(checkpoint_epoch: int, mode: str, adapt_iter: int, result: dict[str, Any]) -> dict[str, Any]:
    r = result["row"]
    return {
        "checkpoint_epoch": int(checkpoint_epoch),
        "mode": mode,
        "adapt_iteration": int(adapt_iter),
        "signal_relmse": r.get("signal_RelMSE"),
        "psnr": r.get("signal_PSNR"),
        "ssim": r.get("signal_SSIM"),
        "fa_mae": r.get("FA_MAE"),
        "md_mae": r.get("MD_MAE"),
        "ad_mae": r.get("AD_MAE"),
        "rd_mae": r.get("RD_MAE"),
        "signal_mse": r.get("signal_MSE"),
        "signal_rmse": r.get("signal_RMSE"),
        "fa_rmse": r.get("FA_RMSE"),
        "md_rmse": r.get("MD_RMSE"),
    }


def eval_zero_shot(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    cfg: dict[str, Any],
    device: torch.device,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    model.freeze_theta()
    model.eval()
    z = model.zero_z(device=device)
    result = evaluate_subject(
        model,
        subj,
        z,
        device=device,
        cfg=cfg,
        mode="zero_shot",
        adapt_iter=0,
        seed=int(cfg.get("seed", 42)),
    )
    return _row_from_eval(checkpoint_epoch, "zero_shot", 0, result)


def eval_latent_adaptation(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    cfg: dict[str, Any],
    device: torch.device,
    checkpoint_epoch: int,
    adapt_iters: list[int],
) -> list[dict[str, Any]]:
    model.freeze_theta()
    z_new = model.new_z(trainable=True, device=device, init="zeros")
    lr = float(cfg.get("adapt_lr", cfg.get("lr", 1e-3)))
    opt = torch.optim.Adam([z_new], lr=lr)
    batch = int(cfg.get("batch_voxels", 4096))
    b_scale = float(cfg.get("b_scale", 1.0))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    coords_t = torch.from_numpy(subj.train_coords)
    bvals_t = torch.from_numpy(subj.bvals).to(device)
    bvecs_t = torch.from_numpy(subj.bvecs).to(device)
    n_vox = int(coords_t.shape[0])
    ckpts = sorted(set(int(x) for x in adapt_iters))
    max_iter = max(ckpts) if ckpts else 0
    rows: list[dict[str, Any]] = []

    def _eval_at(it: int) -> dict[str, Any]:
        with torch.no_grad():
            z_eval = z_new.detach().clone()
        result = evaluate_subject(
            model,
            subj,
            z_eval,
            device=device,
            cfg=cfg,
            mode="latent_adaptation",
            adapt_iter=it,
            seed=seed,
        )
        return _row_from_eval(checkpoint_epoch, "latent_adaptation", it, result)

    if 0 in ckpts:
        rows.append(_eval_at(0))

    for it in range(1, max_iter + 1):
        model.freeze_theta()
        sel = rng.integers(0, n_vox, size=batch, endpoint=False)
        xyz = coords_t[sel].to(device)
        idx = subj.train_flat_idx[sel]
        target = torch.from_numpy(subj.dwi_flat[idx]).to(device)
        S0, D = model(xyz, z=z_new)
        pred = predict_signal(S0, D, bvals_t, bvecs_t, b_scale=b_scale)
        loss = _mse_loss(pred, target, bvals_t, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p in model.theta_parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                raise RuntimeError("theta received gradients during latent adaptation")
        opt.step()
        if it in ckpts:
            rows.append(_eval_at(it))
            print(f"  [ep{checkpoint_epoch}] adapt@{it} FA={rows[-1]['fa_mae']:.4f} PSNR={rows[-1]['psnr']:.3f}", flush=True)
    return rows


def plot_phase4a(metrics_csv: Path, history_csv: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(metrics_csv, newline="", encoding="utf-8")))

    def fnum(r, k):
        return float(r[k])

    zs = [r for r in rows if r["mode"] == "zero_shot"]
    zs = sorted(zs, key=lambda r: int(r["checkpoint_epoch"]))
    ad200 = [
        r
        for r in rows
        if r["mode"] == "latent_adaptation" and int(float(r["adapt_iteration"])) == 200
    ]
    ad200 = sorted(ad200, key=lambda r: int(r["checkpoint_epoch"]))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ex = [int(r["checkpoint_epoch"]) for r in zs]
    axes[0, 0].plot(ex, [fnum(r, "fa_mae") for r in zs], marker="o")
    axes[0, 0].set_title("Zero-shot FA MAE vs checkpoint")
    axes[0, 0].set_xlabel("checkpoint_epoch")
    axes[0, 0].set_ylabel("FA MAE")

    axes[0, 1].plot(ex, [fnum(r, "psnr") for r in zs], marker="o")
    axes[0, 1].set_title("Zero-shot PSNR vs checkpoint")
    axes[0, 1].set_xlabel("checkpoint_epoch")
    axes[0, 1].set_ylabel("PSNR")

    ex2 = [int(r["checkpoint_epoch"]) for r in ad200]
    axes[1, 0].plot(ex2, [fnum(r, "fa_mae") for r in ad200], marker="o")
    axes[1, 0].set_title("Adapt@200 FA MAE vs checkpoint")
    axes[1, 0].set_xlabel("checkpoint_epoch")
    axes[1, 0].set_ylabel("FA MAE")

    axes[1, 1].plot(ex2, [fnum(r, "psnr") for r in ad200], marker="o")
    axes[1, 1].set_title("Adapt@200 PSNR vs checkpoint")
    axes[1, 1].set_xlabel("checkpoint_epoch")
    axes[1, 1].set_ylabel("PSNR")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "checkpoint_vs_test_metrics.png", dpi=150)
    plt.close(fig)

    # Training history (Phase3 or current)
    hist_rows = list(csv.DictReader(open(history_csv, newline="", encoding="utf-8")))
    epochs = [int(float(r["epoch"])) for r in hist_rows]

    def hcol(r, *names):
        for n in names:
            if n in r and r[n] not in (None, ""):
                return float(r[n])
        return float("nan")

    train_l = [hcol(r, "train_signal_loss", "train_mse") for r in hist_rows]
    val_l = [hcol(r, "val_signal_loss", "val_mse") for r in hist_rows]

    # Write tidy history extract
    with open(out_dir / "training_val_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_zero_shot_loss"])
        w.writeheader()
        for e, t, v in zip(epochs, train_l, val_l):
            w.writerow({"epoch": e, "train_loss": t, "val_zero_shot_loss": v})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_l)
    axes[0].set_title("Train signal loss vs epoch")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train_loss")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, val_l, color="C1")
    axes[1].set_title("Val zero-shot loss vs epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val_zero_shot_loss")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_val_loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"[Phase4A] plots → {out_dir}")


def diagnose(metrics_csv: Path, history_csv: Path) -> dict[str, Any]:
    rows = list(csv.DictReader(open(metrics_csv, newline="", encoding="utf-8")))
    hist = list(csv.DictReader(open(history_csv, newline="", encoding="utf-8")))

    zs = sorted(
        [r for r in rows if r["mode"] == "zero_shot"],
        key=lambda r: int(r["checkpoint_epoch"]),
    )
    ad200 = sorted(
        [r for r in rows if r["mode"] == "latent_adaptation" and int(float(r["adapt_iteration"])) == 200],
        key=lambda r: int(r["checkpoint_epoch"]),
    )

    def series(rs, key):
        return [float(r[key]) for r in rs]

    fa_zs = series(zs, "fa_mae")
    psnr_zs = series(zs, "psnr")
    fa_ad = series(ad200, "fa_mae")
    psnr_ad = series(ad200, "psnr")
    epochs_ck = [int(r["checkpoint_epoch"]) for r in zs]

    # Later vs earlier (epoch 10 vs last)
    i10 = epochs_ck.index(10) if 10 in epochs_ck else 0
    ilast = -1
    fa_zs_improved = fa_zs[ilast] < fa_zs[i10] - 1e-4
    psnr_zs_improved = psnr_zs[ilast] > psnr_zs[i10] + 0.05
    fa_zs_worsened = fa_zs[ilast] > fa_zs[i10] + 1e-4
    psnr_zs_worsened = psnr_zs[ilast] < psnr_zs[i10] - 0.05

    # Adaptation at last ckpt: signal vs FA
    last_ep = epochs_ck[ilast]
    ad_last = sorted(
        [
            r
            for r in rows
            if r["mode"] == "latent_adaptation" and int(r["checkpoint_epoch"]) == last_ep
        ],
        key=lambda r: int(float(r["adapt_iteration"])),
    )
    if len(ad_last) >= 2:
        fa0, faT = float(ad_last[0]["fa_mae"]), float(ad_last[-1]["fa_mae"])
        p0, pT = float(ad_last[0]["psnr"]), float(ad_last[-1]["psnr"])
        adapt_fa_improved = faT < fa0 - 1e-4
        adapt_signal_improved = pT > p0 + 0.05
    else:
        fa0 = faT = p0 = pT = float("nan")
        adapt_fa_improved = adapt_signal_improved = False

    train_l = [float(r.get("train_signal_loss", r.get("train_mse", "nan"))) for r in hist]
    train_down = bool(len(train_l) > 10 and train_l[-1] < 0.5 * float(np.nanmean(train_l[: max(1, len(train_l)//5)])))

    labels: list[str] = []
    # A: checkpoint-selection — later θ improves on unseen
    if (fa_zs_improved or psnr_zs_improved) and not (fa_zs_worsened and psnr_zs_worsened):
        labels.append("A_checkpoint_selection")
    # B: overfit — train down, unseen worsens
    if train_down and (fa_zs_worsened or psnr_zs_worsened) and not fa_zs_improved:
        labels.append("B_population_overfit_train")
    # C: signal vs DTI mismatch under adaptation
    if adapt_signal_improved and not adapt_fa_improved:
        labels.append("C_signal_DTI_mismatch")
    # D: latent adaptation helps both
    if adapt_signal_improved and adapt_fa_improved:
        labels.append("D_latent_adaptation_effective")

    if not labels:
        labels.append("INCONCLUSIVE")

    return {
        "checkpoint_epochs": epochs_ck,
        "zero_shot_fa_mae": fa_zs,
        "zero_shot_psnr": psnr_zs,
        "adapt200_fa_mae": fa_ad,
        "adapt200_psnr": psnr_ad,
        "fa_zero_shot_epoch10": fa_zs[i10],
        "fa_zero_shot_last": fa_zs[ilast],
        "psnr_zero_shot_epoch10": psnr_zs[i10],
        "psnr_zero_shot_last": psnr_zs[ilast],
        "adapt_last_ckpt_fa0": fa0,
        "adapt_last_ckpt_faT": faT,
        "adapt_last_ckpt_psnr0": p0,
        "adapt_last_ckpt_psnrT": pT,
        "train_loss_decreased": train_down,
        "diagnosis_labels": labels,
        "primary_diagnosis": labels[0],
        "notes": [
            "A: later checkpoints improve unseen metrics → epoch-10 best was val-protocol artifact.",
            "B: train improves while unseen worsens → overfitting to train subjects.",
            "C: adaptation improves signal but not FA/MD → signal/parameter mismatch.",
            "D: adaptation improves both signal and DTI → latent mechanism preliminarily effective.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4-A Checkpoint Selection Diagnostic")
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase4a_checkpoint_diag.yaml"))
    ap.add_argument("--exp-dir", default="")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing epoch_* checkpoints")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    split = split_from_config(cfg)
    save_epochs = [int(e) for e in cfg.get("save_checkpoint_epochs", [10, 20, 50, 100, 150])]
    adapt_iters = [int(x) for x in cfg.get("adapt_iterations", [0, 10, 50, 100, 200])]

    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        exp_dir = make_experiment_dir(tag=str(cfg.get("experiment_tag", "population_dti_phase4a")))
    save_yaml(exp_dir / "config" / "run_config.yaml", cfg)
    save_json(exp_dir / "config" / "split.json", split)

    print("=" * 72)
    print("Phase 4-A Checkpoint Selection Diagnostic")
    print(f"  exp={exp_dir}")
    print(f"  snapshots={save_epochs}")
    print("=" * 72)

    if not args.skip_train:
        train_population(cfg, exp_dir=exp_dir)
    else:
        print("[Phase4A] skip-train: using existing checkpoints", flush=True)

    device = resolve_device(str(cfg.get("device", "auto")))
    trad_root = Path(cfg["trad_root"])
    test_id = split["test"][0]
    subj = load_subject_bundle(subject_id=test_id, cfg=cfg, trad_dir=trad_root / test_id)
    print(f"[Phase4A] loaded unseen test {test_id} vols={subj.n_volumes}", flush=True)

    all_rows: list[dict[str, Any]] = []
    for ep in save_epochs:
        ckpt_dir = exp_dir / "checkpoints" / f"epoch_{ep:04d}"
        theta_path = ckpt_dir / "theta.pt"
        if not theta_path.is_file():
            raise FileNotFoundError(f"missing snapshot {theta_path}")
        model = _build_model(cfg, split["train"], device)
        load_theta(theta_path, model, map_location=device)
        model.freeze_theta()
        print(f"\n[Phase4A] === checkpoint epoch {ep} ===", flush=True)
        t0 = time.time()
        zs_row = eval_zero_shot(model, subj, cfg, device, ep)
        all_rows.append(zs_row)
        print(
            f"  zero-shot FA={zs_row['fa_mae']:.4f} PSNR={zs_row['psnr']:.3f} RelMSE={zs_row['signal_relmse']:.4f}",
            flush=True,
        )
        # reload fresh model for adaptation (clean z)
        model = _build_model(cfg, split["train"], device)
        load_theta(theta_path, model, map_location=device)
        model.freeze_theta()
        ad_rows = eval_latent_adaptation(model, subj, cfg, device, ep, adapt_iters)
        all_rows.extend(ad_rows)
        print(f"  checkpoint {ep} done in {time.time()-t0:.1f}s", flush=True)

    metrics_csv = exp_dir / "metrics" / "checkpoint_vs_test_metrics.csv"
    _write_csv(metrics_csv, all_rows)

    # Prefer current training history; fall back to Phase3 path in config
    hist = exp_dir / "metrics" / "training_history.csv"
    if not hist.is_file():
        hist = Path(cfg.get("phase3_history", ""))
    plot_phase4a(metrics_csv, hist, exp_dir / "plots")

    diagnosis = diagnose(metrics_csv, hist)
    save_json(exp_dir / "metrics" / "phase4a_diagnosis.json", diagnosis)

    print("\n===== PHASE 4-A DIAGNOSIS =====")
    print(f"  primary: {diagnosis['primary_diagnosis']}")
    print(f"  labels:  {diagnosis['diagnosis_labels']}")
    print(f"  FA zero-shot ep10 → last: {diagnosis['fa_zero_shot_epoch10']:.4f} → {diagnosis['fa_zero_shot_last']:.4f}")
    print(f"  PSNR zero-shot ep10 → last: {diagnosis['psnr_zero_shot_epoch10']:.3f} → {diagnosis['psnr_zero_shot_last']:.3f}")
    print(f"  Adapt@last FA 0→200: {diagnosis['adapt_last_ckpt_fa0']:.4f} → {diagnosis['adapt_last_ckpt_faT']:.4f}")
    print(f"  Adapt@last PSNR 0→200: {diagnosis['adapt_last_ckpt_psnr0']:.3f} → {diagnosis['adapt_last_ckpt_psnrT']:.3f}")
    print(f"\n[Phase4A] done → {exp_dir}")


if __name__ == "__main__":
    main()
