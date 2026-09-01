"""Shared INR seen-subject baseline (B) — calls INR train_shared without modifying INR."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from utils_io import package_root, resolve_device, save_json

_INR = package_root().parent / "INR"
if str(_INR) not in sys.path:
    sys.path.insert(0, str(_INR))

from inr.train_shared import train_shared_inr  # noqa: E402


def run_shared_seen_baseline(
    *,
    subject_ids: list[str],
    cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """
    Train SharedSpatialDTIINR on the given subjects (must be the train split).

    This is a seen-subject baseline only — cannot evaluate unseen test IDs
    without changing the Shared baseline (forbidden).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(cfg.get("device", "auto")))
    trad_root = Path(
        cfg.get("trad_root", str(package_root().parent / "INR" / "outputs" / "step1_traditional_dti"))
    )
    inr_cfg = {
        "hcp_root": cfg["hcp_root"],
        "b0_threshold": cfg["b0_threshold"],
        "shell_tol": cfg["shell_tol"],
        "dti_shells": list(cfg.get("dti_shells", [1000.0])),
    }
    # Shared baseline keeps its own default latent_dim=32 unless overridden for fair logging
    shared_latent = int(cfg.get("shared_latent_dim", 32))
    epochs = int(cfg.get("shared_epochs", cfg.get("epochs", 150)))

    print(
        f"[SharedBaseline-B] subjects={subject_ids} epochs={epochs} "
        f"latent_dim={shared_latent} (seen-subject only)",
        flush=True,
    )
    result = train_shared_inr(
        subject_ids=subject_ids,
        cfg=inr_cfg,
        out_root=out_dir,
        trad_root=trad_root,
        device=device,
        latent_dim=shared_latent,
        epochs=epochs,
        batch_voxels=int(cfg.get("batch_voxels", 4096)),
        lr=float(cfg.get("lr", 1e-3)),
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 4)),
        pe_freqs=int(cfg.get("pe_freqs", 8)),
        log_every=int(cfg.get("log_every", 10)),
        seed=int(cfg.get("seed", 42)),
        skip_traditional_if_exists=bool(cfg.get("skip_traditional_if_exists", True)),
        save_maps=True,
        tag="SharedINR-seen",
    )
    summary = {
        "mode": "shared_seen_baseline",
        "subjects": subject_ids,
        "epochs": epochs,
        "latent_dim": shared_latent,
        "note": "Seen-subject only; not applicable to unseen test without baseline change.",
        "result": {k: result[k] for k in result if k != "rows"},
        "rows": result.get("rows", []),
    }
    save_json(out_dir / "shared_seen_summary.json", summary)
    return summary
