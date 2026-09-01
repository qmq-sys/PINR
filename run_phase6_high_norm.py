#!/usr/bin/env python
"""Phase 6-B supplement at high λ_DTI where FA actually improves."""
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
    plot_phase6b,
    run_mixed_adaptation,
)
from utils_io import _write_csv, load_latents, load_theta, load_yaml, resolve_device, save_json

EXP_DIR = ROOT / "experiments" / "population_dti_phase6" / "20260830_141502"
# High-λ where FA improves; test whether λ_z recovers signal
LAMBDA_DTI = (1000.0, 5000.0, 10000.0)
LAMBDA_Z = (0.0, 0.01, 0.1, 1.0)


def frow(r: dict) -> dict:
    out = dict(r)
    for k, v in r.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            pass
    if "tag" not in out or not out["tag"]:
        out["tag"] = f"lambdaDTI={out.get('lambda_DTI')}_lambdaz={out.get('lambda_z', 0)}"
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
        model, subj.train_coords, subj.train_flat_idx, subj.shape_xyz, model.zero_z(device=device), device, want_D=True
    )
    D0 = maps0["D"]
    mask = subj.common_mask
    dti_ref = float(np.mean(_fro(D0[mask].astype(np.float64) - np.asarray(subj.ref["D"], dtype=np.float64)[mask])))

    traj: list[dict] = []
    finals: list[dict] = []
    for ld in LAMBDA_DTI:
        for lz in LAMBDA_Z:
            load_theta(ckpt / "theta.pt", model, map_location=device)
            model.freeze_theta()
            tag = f"lambdaDTI={ld}_lambdaz={lz}"
            print(f"\n--- HIGH-NORM {tag} ---", flush=True)
            rows, zf = run_mixed_adaptation(
                model=model,
                subj=subj,
                cfg=cfg,
                device=device,
                lambda_DTI=float(ld),
                lambda_z=float(lz),
                max_iter=1000,
                key_iters=KEY_ITERS,
                D0_vol=D0,
                seed=42,
                tag=tag,
                dti_ref_scale=dti_ref,
            )
            traj.extend(rows)
            finals.append(rows[-1])
            torch.save(
                {"z": zf.cpu(), "lambda_DTI": ld, "lambda_z": lz},
                exp_dir / "norm_reg" / f"z_high_ld{ld}_lz{lz}.pt",
            )
            print(
                "FINAL",
                {
                    k: rows[-1][k]
                    for k in ("lambda_DTI", "lambda_z", "latent_norm", "PSNR", "FA_MAE", "mean_D_fro_error_vs_wls")
                },
                flush=True,
            )

    # merge with existing norm finals
    nf = exp_dir / "metrics" / "norm_regularization_final.csv"
    old = []
    if nf.exists():
        with open(nf, encoding="utf-8") as f:
            old = [frow(r) for r in csv.DictReader(f)]
    # keep old user-grid rows (λ_DTI <= 10)
    old = [r for r in old if float(r["lambda_DTI"]) <= 10.0]
    norm_final = old + finals

    nt = exp_dir / "metrics" / "norm_regularization.csv"
    old_t = []
    if nt.exists():
        with open(nt, encoding="utf-8") as f:
            old_t = [frow(r) for r in csv.DictReader(f)]
    old_t = [r for r in old_t if float(r["lambda_DTI"]) <= 10.0]
    norm_traj = old_t + traj

    _write_csv(exp_dir / "metrics" / "norm_regularization.csv", norm_traj)
    _write_csv(exp_dir / "metrics" / "norm_regularization_final.csv", norm_final)
    _write_csv(exp_dir / "metrics" / "norm_regularization_high.csv", traj)
    plot_phase6b(norm_final, exp_dir / "plots")

    with open(exp_dir / "metrics" / "lambda_sweep_final.csv", encoding="utf-8") as f:
        sweep_final = [frow(r) for r in csv.DictReader(f)]
    with open(exp_dir / "metrics" / "lambda_sweep.csv", encoding="utf-8") as f:
        sweep_traj = [frow(r) for r in csv.DictReader(f)]

    plot_phase6a(sweep_final, sweep_traj, exp_dir / "plots")
    refs = _load_phase5_refs(phase5)
    z0 = next(r for r in sweep_traj if int(r["iteration"]) == 0 and float(r["lambda_DTI"]) == 0.0)
    refs["z0"] = {k: float(z0[k]) for k in ("PSNR", "FA_MAE", "MD_MAE", "mean_D_fro_error_vs_wls", "latent_norm", "RelMSE")}

    diag = diagnose_phase6(sweep_final, norm_final, refs, mean_train_norm)
    # enrich with high-λ specific notes
    both = []
    for r in sweep_final + norm_final:
        if r["PSNR"] > refs["z0"]["PSNR"] + 0.3 and r["FA_MAE"] < refs["z0"]["FA_MAE"] - 0.001:
            both.append(r)
    fa_better = [r for r in sweep_final + norm_final if r["FA_MAE"] < refs["z0"]["FA_MAE"] - 0.001]
    diag["high_lambda_FA_improving_configs"] = [
        {
            "tag": r.get("tag"),
            "lambda_DTI": r["lambda_DTI"],
            "lambda_z": r.get("lambda_z", 0),
            "PSNR": r["PSNR"],
            "FA_MAE": r["FA_MAE"],
            "latent_norm": r["latent_norm"],
            "D_err": r["mean_D_fro_error_vs_wls"],
        }
        for r in fa_better
    ]
    diag["Q2_exists_lambda_both_better_than_z0"] = len(both) > 0
    diag["Q2_both_better_configs"] = both
    save_json(exp_dir / "phase6_diagnosis.json", diag)

    pareto_rows = list(diag["pareto_points"])
    for r in sweep_final + norm_final:
        pareto_rows.append(
            {
                "name": r.get("tag"),
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

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(refs["z0"]["PSNR"], refs["z0"]["FA_MAE"], s=160, marker="*", label="z=0", zorder=6, c="black")
    ax.scatter(refs["phase5_signal"]["PSNR"], refs["phase5_signal"]["FA_MAE"], s=110, marker="s", label="P5 signal", zorder=5)
    ax.scatter(refs["phase5_dti"]["PSNR"], refs["phase5_dti"]["FA_MAE"], s=110, marker="D", label="P5 DTI oracle", zorder=5)
    for r in sweep_final:
        ax.scatter(r["PSNR"], r["FA_MAE"], s=50, label=f"6A λ={r['lambda_DTI']}")
    for r in finals:
        ax.scatter(r["PSNR"], r["FA_MAE"], s=40, marker="^", alpha=0.85, label=f"6B λD={r['lambda_DTI']} λz={r['lambda_z']}")
    ax.axhline(refs["z0"]["FA_MAE"], color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(refs["z0"]["PSNR"], color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("PSNR ↑")
    ax.set_ylabel("FA MAE ↓")
    ax.set_title("Phase 6 Pareto (user λ + high λ + norm reg)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=5, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(exp_dir / "plots" / "pareto_full_PSNR_FA.png", dpi=150)
    plt.close(fig)

    print(json.dumps({k: diag[k] for k in ("Q1_mixed_objective_relieves_conflict", "Q2_exists_lambda_both_better_than_z0", "Q3_phase5_dti_z_is_runaway", "Q4_norm_regularization_helps", "best_compromise", "Go_NoGo", "high_lambda_FA_improving_configs")}, indent=2, default=str))


if __name__ == "__main__":
    main()
