"""Volume sampling — wraps INR volume_sampling v2 (per-subject, no global indices)."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_INR_ROOT = Path(__file__).resolve().parents[2] / "INR"
if str(_INR_ROOT) not in sys.path:
    sys.path.insert(0, str(_INR_ROOT))

from inr.hcp_io import shell_volume_mask  # noqa: E402
from inr.volume_sampling import (  # noqa: E402
    assemble_subject_indices,
    build_subject_shell_protocol,
)


def build_indices_for_subject(
    subject_id: str,
    bvals: np.ndarray,
    *,
    fraction: float,
    b0_threshold: float = 50.0,
    shell_tol: float = 200.0,
    shells: tuple[float, ...] = (1000.0,),
    base_seed: int = 20260822,
) -> np.ndarray:
    """
    Return local indices on the shell-masked volume axis for this subject.

    fraction=1.0 → all volumes on the DTI shell mask (b0 + b1000).
    """
    vol_m = shell_volume_mask(
        bvals,
        b0_threshold=float(b0_threshold),
        shell_tol=float(shell_tol),
        shells=tuple(shells),
        include_b0=True,
    )
    if float(fraction) >= 1.0 - 1e-12:
        return np.arange(int(np.count_nonzero(vol_m)), dtype=np.int64)

    # Map fraction to nested protocol level key
    levels = (("100%", 1.0), ("50%", 0.5), ("25%", 0.25), ("10%", 0.1))
    # Build protocol for this subject only
    proto = build_subject_shell_protocol(
        subject_id,
        bvals,
        vol_m,
        b0_threshold=float(b0_threshold),
        shell_tol=float(shell_tol),
        shells=tuple(shells),
        base_seed=int(base_seed),
        levels=levels,
    )
    # Pick closest supported fraction key
    frac_key = _fraction_key(float(fraction))
    return assemble_subject_indices(proto, frac_key)


def _fraction_key(fraction: float) -> str:
    mapping = {1.0: "1.0", 0.5: "0.5", 0.25: "0.25", 0.1: "0.1"}
    for k, v in mapping.items():
        if abs(float(fraction) - k) < 1e-9:
            return v
    raise ValueError(f"unsupported sampling fraction {fraction}; use 1.0/0.5/0.25/0.1")


def sampling_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "fraction": float(cfg.get("sampling_fraction", 1.0)),
        "base_seed": int(cfg.get("sampling_base_seed", 20260822)),
        "shells": list(cfg.get("dti_shells", [1000.0])),
        "b0_threshold": float(cfg.get("b0_threshold", 50.0)),
        "shell_tol": float(cfg.get("shell_tol", 200.0)),
        "strategy": "subject_specific_shell_stratified_nested_v2",
    }
