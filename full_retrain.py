"""Full Independent INR retraining baseline on an unseen subject (comparison interface).

Does NOT modify INR baselines — imports train_one_independent_subject as a library call.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from utils_io import package_root, resolve_device, save_json

_INR = package_root().parent / "INR"
if str(_INR) not in sys.path:
    sys.path.insert(0, str(_INR))

from inr.model import SpatialDTIINR  # noqa: E402
from inr.train_independent import train_one_independent_subject  # noqa: E402


def count_trainable(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_full_retraining(
    *,
    subject_id: str,
    cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """
    Unseen subject → fresh SpatialDTIINR → full θ_s optimization.

    Returns summary metrics + wall time + trainable param count.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(cfg.get("device", "auto")))

    inr_cfg = {
        "hcp_root": cfg["hcp_root"],
        "b0_threshold": cfg["b0_threshold"],
        "shell_tol": cfg["shell_tol"],
        "dti_shells": list(cfg.get("dti_shells", [1000.0])),
    }
    trad_dir = (
        Path(cfg.get("trad_root", str(package_root().parent / "INR" / "outputs" / "step1_traditional_dti")))
        / subject_id
    )
    subj_out = out_dir / f"independent_{subject_id}"
    subj_out.mkdir(parents=True, exist_ok=True)

    probe = SpatialDTIINR(
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 4)),
        pe_freqs=int(cfg.get("pe_freqs", 8)),
    )
    n_params = count_trainable(probe)
    del probe

    epochs = int(cfg.get("full_retrain_epochs", cfg.get("epochs", 20)))
    t0 = time.time()
    result = train_one_independent_subject(
        subject_id=subject_id,
        cfg=inr_cfg,
        out_dir=subj_out,
        trad_dir=trad_dir,
        device=device,
        epochs=epochs,
        batch_voxels=int(cfg.get("batch_voxels", 4096)),
        lr=float(cfg.get("lr", 1e-3)),
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 4)),
        pe_freqs=int(cfg.get("pe_freqs", 8)),
        log_every=int(cfg.get("log_every", 5)),
        eval_every=int(cfg.get("eval_every", 50)),
        seed=int(cfg.get("seed", 42)),
        skip_traditional_if_exists=bool(cfg.get("skip_traditional_if_exists", True)),
    )
    wall = time.time() - t0

    fa_mae = md_mae = float("nan")
    dwi_rel = float("nan")
    metrics_path = subj_out / "metrics.json"
    if metrics_path.is_file():
        obj = json.loads(metrics_path.read_text(encoding="utf-8"))
        fa_mae = float(obj.get("parameter_metrics", {}).get("FA", {}).get("MAE", float("nan")))
        md_mae = float(obj.get("parameter_metrics", {}).get("MD", {}).get("MAE", float("nan")))
        dwi_rel = float(obj.get("dwi", {}).get("relative_mse", float("nan")))

    summary = {
        "mode": "full_retraining",
        "subject_id": subject_id,
        "FA_MAE": fa_mae,
        "MD_MAE": md_mae,
        "DWI_RelMSE": dwi_rel,
        "note_signal": (
            "Independent baseline reports RelMSE; Population uses S0-norm PSNR — "
            "compare FA/MD primarily in Pilot-0"
        ),
        "training_iterations_epochs": epochs,
        "wall_clock_time_sec": wall,
        "trainable_parameter_count": n_params,
        "out_dir": str(subj_out),
        "train_result_keys": list(result.keys()) if isinstance(result, dict) else [],
    }
    save_json(out_dir / f"full_retrain_{subject_id}.json", summary)
    print(f"[full_retrain] {subject_id} FA_MAE={fa_mae:.6f} wall={wall:.1f}s params={n_params}")
    return summary
