#!/usr/bin/env python
"""Phase 3 Scientific Validation runner.

A Independent full retrain | B Shared seen | C zero-shot | D latent adaptation
No architecture / loss / INR baseline source changes.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapt_subject import adapt_from_experiment  # noqa: E402
from data.split import split_from_config  # noqa: E402
from diagnose import diagnose_phase3  # noqa: E402
from full_retrain import run_full_retraining  # noqa: E402
from plot_curves import plot_adaptation_curve, plot_training_curve  # noqa: E402
from shared_baseline import run_shared_seen_baseline  # noqa: E402
from train_population import train_population  # noqa: E402
from utils_io import load_yaml, make_experiment_dir, save_json, save_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 Scientific Validation")
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase3_pilot0.yaml"))
    ap.add_argument("--exp-dir", default="")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-adapt", action="store_true")
    ap.add_argument("--skip-independent", action="store_true")
    ap.add_argument("--skip-shared", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    split = split_from_config(cfg)
    exp_dir = Path(args.exp_dir) if args.exp_dir else make_experiment_dir(
        tag=str(cfg.get("experiment_tag", "population_dti_phase3"))
    )
    save_yaml(exp_dir / "config" / "run_config.yaml", cfg)
    save_json(exp_dir / "config" / "split.json", split)

    print("=" * 72)
    print("Phase 3 Scientific Validation — 100% DTI")
    print(f"  exp={exp_dir}")
    print(f"  train={split['train']}")
    print(f"  val  ={split['val']}")
    print(f"  test ={split['test']}")
    print(f"  population_epochs={cfg.get('epochs')}")
    print(f"  independent_epochs={cfg.get('full_retrain_epochs')}")
    print(f"  shared_epochs={cfg.get('shared_epochs')}")
    print("=" * 72)

    try:
        # C/D path: population train + zero-shot + adaptation
        if not args.skip_train:
            train_population(cfg, exp_dir=exp_dir)
        else:
            hist = exp_dir / "metrics" / "training_history.csv"
            if hist.is_file():
                plot_training_curve(hist, exp_dir / "plots" / "training_curve.png")

        if not args.skip_adapt:
            adapt_from_experiment(exp_dir, cfg)
            curve = exp_dir / "metrics" / "adaptation_curve.csv"
            if curve.is_file():
                for tid in split["test"]:
                    plot_adaptation_curve(
                        curve,
                        exp_dir / "plots" / f"{tid}_adaptation_curve.png",
                        subject_id=tid,
                    )

        # A: Independent full retrain on unseen test
        if not args.skip_independent:
            for tid in split["test"]:
                run_full_retraining(
                    subject_id=tid,
                    cfg=cfg,
                    out_dir=exp_dir / "baselines",
                )

        # B: Shared seen-subject baseline on train subjects only
        if not args.skip_shared:
            run_shared_seen_baseline(
                subject_ids=split["train"],
                cfg=cfg,
                out_dir=exp_dir / "baselines" / "shared_seen",
            )

        diagnosis = diagnose_phase3(exp_dir)
        print("\n===== PHASE 3 DIAGNOSIS =====")
        for k in (
            "epochs_run",
            "train_loss_early_mean",
            "train_loss_late_mean",
            "train_improved",
            "train_plateau",
            "zero_shot",
            "adaptation_FA_improved",
            "adaptation_signal_improved",
            "independent_FA_MAE",
            "population_best_FA_MAE",
            "FA_gap_population_minus_independent",
            "parameter_efficiency",
            "theta_underfitting_suspected",
            "latent_underfitting_suspected",
        ):
            print(f"  {k}: {diagnosis.get(k)}")
        print(f"\n[Phase3] done → {exp_dir}")
        print(f"[Phase3] diagnosis → {exp_dir / 'metrics' / 'phase3_diagnosis.json'}")
    except Exception as e:
        save_json(exp_dir / "logs" / "error.json", {"error": str(e), "traceback": traceback.format_exc()})
        print(f"[Phase3] FAILED: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
