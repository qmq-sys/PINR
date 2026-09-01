"""Post-hoc diagnosis for Phase 3 scientific validation (no model changes)."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from utils_io import save_json


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def diagnose_phase3(exp_dir: Path) -> dict[str, Any]:
    exp_dir = Path(exp_dir)
    hist = _read_csv(exp_dir / "metrics" / "training_history.csv")
    adapt = _read_csv(exp_dir / "metrics" / "adaptation_curve.csv")
    pe_path = exp_dir / "metrics" / "parameter_efficiency.json"
    pe = json.loads(pe_path.read_text(encoding="utf-8")) if pe_path.is_file() else {}

    train_losses = [float(r.get("train_signal_loss", r.get("train_mse", "nan"))) for r in hist]
    val_losses = [float(r.get("val_signal_loss", r.get("val_mse", "nan"))) for r in hist]

    # Convergence heuristics (last 20% of epochs)
    n = len(train_losses)
    k = max(1, n // 5)
    early = float(np.nanmean(train_losses[:k])) if n else float("nan")
    late = float(np.nanmean(train_losses[-k:])) if n else float("nan")
    late_std = float(np.nanstd(train_losses[-k:])) if n else float("nan")
    train_improved = bool(np.isfinite(early) and np.isfinite(late) and late < 0.5 * early)
    train_plateau = bool(np.isfinite(late_std) and late_std < max(1e-4, 0.05 * abs(late)))

    adapt_lat = [r for r in adapt if r.get("mode") == "latent_adaptation"]
    adapt_lat = sorted(adapt_lat, key=lambda r: int(float(r.get("adapt_iter", 0))))
    zs = next((r for r in adapt if r.get("mode") == "zero_shot"), None)
    if zs is None and adapt_lat:
        zs = adapt_lat[0]

    def _f(r, key):
        try:
            return float(r.get(key, "nan"))
        except (TypeError, ValueError):
            return float("nan")

    zero_shot = {
        "FA_MAE": _f(zs, "FA_MAE") if zs else float("nan"),
        "MD_MAE": _f(zs, "MD_MAE") if zs else float("nan"),
        "signal_PSNR": _f(zs, "signal_PSNR") if zs else float("nan"),
        "signal_RelMSE": _f(zs, "signal_RelMSE") if zs else float("nan"),
    }
    adapt_curve = [
        {
            "adapt_iter": int(float(r["adapt_iter"])),
            "FA_MAE": _f(r, "FA_MAE"),
            "MD_MAE": _f(r, "MD_MAE"),
            "signal_PSNR": _f(r, "signal_PSNR"),
            "signal_RelMSE": _f(r, "signal_RelMSE"),
        }
        for r in adapt_lat
    ]

    # Adaptation improvement
    if len(adapt_curve) >= 2:
        fa0, faT = adapt_curve[0]["FA_MAE"], adapt_curve[-1]["FA_MAE"]
        p0, pT = adapt_curve[0]["signal_PSNR"], adapt_curve[-1]["signal_PSNR"]
        r0, rT = adapt_curve[0]["signal_RelMSE"], adapt_curve[-1]["signal_RelMSE"]
        fa_improved = bool(np.isfinite(fa0) and np.isfinite(faT) and faT < fa0 - 1e-4)
        signal_improved = bool(
            (np.isfinite(p0) and np.isfinite(pT) and pT > p0 + 0.05)
            or (np.isfinite(r0) and np.isfinite(rT) and rT < r0 * 0.95)
        )
    else:
        fa_improved = signal_improved = False
        fa0 = faT = p0 = pT = r0 = rT = float("nan")

    # Independent baseline if present
    indep = {}
    for p in (exp_dir / "baselines").glob("full_retrain_*.json") if (exp_dir / "baselines").is_dir() else []:
        indep = json.loads(p.read_text(encoding="utf-8"))
        break

    indep_fa = float(indep.get("FA_MAE", float("nan"))) if indep else float("nan")
    pop_fa = float(faT) if np.isfinite(faT) else zero_shot["FA_MAE"]
    gap_vs_indep = (
        float(pop_fa - indep_fa) if np.isfinite(pop_fa) and np.isfinite(indep_fa) else float("nan")
    )

    # Underfitting flags
    theta_underfit = bool(
        (not train_improved)
        or (np.isfinite(late) and late > 0.2)  # still high S0-norm MSE
        or (np.isfinite(gap_vs_indep) and gap_vs_indep > 0.03 and not train_plateau)
    )
    # Latent underfit: signal barely moves under adaptation OR FA stuck while theta looks OK
    latent_underfit = bool(
        train_improved
        and train_plateau
        and (not signal_improved)
        and (not fa_improved)
    )

    diagnosis = {
        "epochs_run": n,
        "train_loss_early_mean": early,
        "train_loss_late_mean": late,
        "train_loss_late_std": late_std,
        "train_improved": train_improved,
        "train_plateau": train_plateau,
        "zero_shot": zero_shot,
        "adaptation_curve": adapt_curve,
        "adaptation_FA_improved": fa_improved,
        "adaptation_signal_improved": signal_improved,
        "FA_MAE_adapt0": fa0,
        "FA_MAE_adaptT": faT,
        "PSNR_adapt0": p0,
        "PSNR_adaptT": pT,
        "RelMSE_adapt0": r0,
        "RelMSE_adaptT": rT,
        "independent_FA_MAE": indep_fa,
        "population_best_FA_MAE": pop_fa,
        "FA_gap_population_minus_independent": gap_vs_indep,
        "parameter_efficiency": pe,
        "theta_underfitting_suspected": theta_underfit,
        "latent_underfitting_suspected": latent_underfit,
        "notes": [
            "Val loss is zero-shot on held-out val subject (z=0), not a trained latent.",
            "Do not treat 5-epoch smoke results as performance conclusions.",
            "If Population << Independent after 100–200 epochs: report diagnosis, do not change architecture.",
        ],
    }
    save_json(exp_dir / "metrics" / "phase3_diagnosis.json", diagnosis)
    return diagnosis
