#!/usr/bin/env python
"""Supplement Phase 6-A with high-lambda sweep; merge into existing exp dir."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import load_subject_bundle
from data.split import split_from_config
from predict import predict_maps
from run_phase6 import (
    DEFAULT_PHASE4A,
    DEFAULT_PHASE5,
    KEY_ITERS,
    _build_model,
    _fro,
    _load_phase5_refs,
    diagnose_phase6,
    plot_pareto_c,
    plot_phase6a,
    run_mixed_adaptation,
)
from utils_io import _write_csv, load_latents, load_theta, load_yaml, resolve_device, save_json

EXP_DIR = ROOT / "experiments" / "population_dti_phase6" / "20260830_141502"
HIGH = (50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0)


def frow(r: dict) -> dict:
    out = dict(r)
    for k, v in r.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    if "tag" not in out or out["tag"] == "":
        out["tag"] = f"lambdaDTI={out.get('lambda_DTI')}"
    return out


def main() -> None:
    phase4a = DEFAULT_PHASE4A
    phase5 = DEFAULT_PHASE5
    exp_dir = EXP_DIR
    cfg = dict(load_yaml(phase4a / "config" / "run_config.yaml"))
    cfg["adapt_lr"] = float(cfg.get("adapt_lr", cfg.get("lr", 1e-3)))
    split = split_from_config(cfg)
    device = resolve_device(str(cfg.get("device", "auto")))
    ckpt = phase4a / "checkpoints" / "epoch_0150"

    model = _build_model(cfg, split["train"], device)
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()
    train_latents = {k: v.float() for k, v in load_latents(ckpt / "latents.pt").items()}
    mean_train_norm = float(np.mean([float(torch.linalg.vector_norm(v)) for v in train_latents.values()]))

    trad = Path(cfg["trad_root"])
    subj = load_subject_bundle(subject_id=split["test"][0], cfg=cfg, trad_dir=trad / split["test"][0])
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
    dti_ref = float(
        np.mean(_fro(D0[mask].astype(np.float64) - np.asarray(subj.ref["D"], dtype=np.float64)[mask]))
    )
    print(f"dti_ref={dti_ref:.6e}", flush=True)

    extra_traj: list[dict] = []
    extra_final: list[dict] = []
    for lam in HIGH:
        load_theta(ckpt / "theta.pt", model, map_location=device)
        model.freeze_theta()
        tag = f"lambdaDTI={lam}"
        print(f"\n--- HIGH {tag} ---", flush=True)
        rows, zf = run_mixed_adaptation(
            model=model,
            subj=subj,
            cfg=cfg,
            device=device,
            lambda_DTI=float(lam),
            lambda_z=0.0,
            max_iter=1000,
            key_iters=KEY_ITERS,
            D0_vol=D0,
            seed=42,
            tag=tag,
            dti_ref_scale=dti_ref,
        )
        extra_traj.extend(rows)
        extra_final.append(rows[-1])
        torch.save({"z": zf.cpu(), "lambda_DTI": lam, "lambda_z": 0.0}, exp_dir / "lambda_sweep" / f"z_lambda_{lam}.pt")
        _write_csv(exp_dir / "lambda_sweep" / f"trajectory_lambda_{lam}.csv", rows)
        print(
            "FINAL",
            {k: rows[-1][k] for k in ("lambda_DTI", "latent_norm", "PSNR", "FA_MAE", "mean_D_fro_error_vs_wls")},
            flush=True,
        )

    with open(exp_dir / "metrics" / "lambda_sweep_final.csv", encoding="utf-8") as f:
        old_final = [frow(r) for r in csv.DictReader(f)]
    # drop any previous high-λ rows if re-run
    old_final = [r for r in old_final if float(r["lambda_DTI"]) not in set(HIGH)]
    with open(exp_dir / "metrics" / "lambda_sweep.csv", encoding="utf-8") as f:
        old_traj = [frow(r) for r in csv.DictReader(f)]
    old_traj = [r for r in old_traj if float(r["lambda_DTI"]) not in set(HIGH)]

    sweep_final = old_final + extra_final
    sweep_traj = old_traj + extra_traj
    _write_csv(exp_dir / "metrics" / "lambda_sweep.csv", sweep_traj)
    _write_csv(exp_dir / "metrics" / "lambda_sweep_final.csv", sweep_final)
    _write_csv(exp_dir / "metrics" / "lambda_sweep_high.csv", extra_traj)

    norm_final: list[dict] = []
    nf = exp_dir / "metrics" / "norm_regularization_final.csv"
    if nf.exists():
        with open(nf, encoding="utf-8") as f:
            norm_final = [frow(r) for r in csv.DictReader(f)]

    plot_phase6a(sweep_final, sweep_traj, exp_dir / "plots")
    refs = _load_phase5_refs(phase5)
    z0_live = next((r for r in sweep_traj if int(r["iteration"]) == 0 and float(r["lambda_DTI"]) == 0.0), None)
    if z0_live is not None:
        refs["z0"] = {
            k: float(z0_live[k])
            for k in ("PSNR", "FA_MAE", "MD_MAE", "mean_D_fro_error_vs_wls", "latent_norm", "RelMSE")
        }

    diag = diagnose_phase6(sweep_final, norm_final, refs, mean_train_norm)
    save_json(exp_dir / "phase6_diagnosis.json", diag)

    pareto_rows = list(diag["pareto_points"])
    for r in sweep_final + norm_final:
        pareto_rows.append(
            {
                "name": r.get("tag", str(r.get("lambda_DTI"))),
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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(refs["z0"]["PSNR"], refs["z0"]["FA_MAE"], s=140, marker="*", label="z=0", zorder=5)
    ax.scatter(refs["phase5_signal"]["PSNR"], refs["phase5_signal"]["FA_MAE"], s=100, marker="s", label="P5 signal", zorder=5)
    ax.scatter(refs["phase5_dti"]["PSNR"], refs["phase5_dti"]["FA_MAE"], s=100, marker="D", label="P5 DTI oracle", zorder=5)
    for r in sweep_final:
        ax.scatter(r["PSNR"], r["FA_MAE"], s=55, label=f"λ={r['lambda_DTI']}")
    ax.set_xlabel("PSNR ↑")
    ax.set_ylabel("FA MAE ↓")
    ax.set_title("Phase 6 Pareto (+ high-λ)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(exp_dir / "plots" / "pareto_full_PSNR_FA.png", dpi=150)
    plt.close(fig)

    print(
        "UPDATED DIAG",
        json.dumps(
            {
                k: diag[k]
                for k in (
                    "Q1_mixed_objective_relieves_conflict",
                    "Q2_exists_lambda_both_better_than_z0",
                    "Q2_both_better_configs",
                    "best_compromise",
                    "Go_NoGo",
                )
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
