#!/usr/bin/env python
"""Pilot-0 runner: population train → zero-shot → latent adaptation → optional full retrain.

Does not modify INR Independent/Shared baselines.
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
from full_retrain import run_full_retraining  # noqa: E402
from plot_adaptation import plot_adaptation_curve  # noqa: E402
from train_population import train_population  # noqa: E402
from utils_io import load_yaml, make_experiment_dir, save_json, save_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Population-DTI-INR Pilot-0")
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot0.yaml"))
    ap.add_argument("--skip-full-retrain", action="store_true", help="skip Independent full retrain (A)")
    ap.add_argument("--exp-dir", default="", help="reuse existing experiment dir (skip train)")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    split = split_from_config(cfg)

    print("=" * 72)
    print("Population-DTI-INR Pilot-0")
    print(f"  train={split['train']}")
    print(f"  val  ={split['val']}")
    print(f"  test ={split['test']}")
    print(f"  sampling_fraction={cfg.get('sampling_fraction')}")
    print("=" * 72)
    print(
        "CONFLICT note (Shared baseline B vs unseen):\n"
        "  SharedSpatialDTIINR requires subject_id in training embedding table.\n"
        "  It cannot run unseen zero-shot without changing the baseline.\n"
        "  Pilot-0 therefore runs: Population C/D + Independent full_retrain (A).\n"
        "  Shared (B) remains untouched as seen-subject baseline elsewhere.\n"
    )

    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        exp_dir = make_experiment_dir(tag=str(cfg.get("experiment_tag", "population_dti")))
    save_yaml(exp_dir / "config" / "run_config.yaml", cfg)
    save_json(exp_dir / "config" / "split.json", split)

    try:
        if not args.skip_train:
            train_population(cfg, exp_dir=exp_dir)
        adapt_from_experiment(exp_dir, cfg)

        curve = exp_dir / "metrics" / "adaptation_curve.csv"
        if curve.is_file():
            for tid in split["test"]:
                plot_adaptation_curve(
                    curve,
                    exp_dir / "plots" / f"{tid}_adaptation_curve.png",
                    subject_id=tid,
                )

        if not args.skip_full_retrain:
            for tid in split["test"]:
                run_full_retraining(
                    subject_id=tid,
                    cfg=cfg,
                    out_dir=exp_dir / "baselines",
                )

        print(f"\n[Pilot-0] SUCCESS → {exp_dir}")
    except Exception as e:
        err = {"error": str(e), "traceback": traceback.format_exc()}
        save_json(exp_dir / "logs" / "error.json", err)
        print(f"[Pilot-0] FAILED: {e}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
