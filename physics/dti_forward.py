"""Thin wrapper around INR DTI forward — single scale from config."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_INR_ROOT = Path(__file__).resolve().parents[2] / "INR"
if str(_INR_ROOT) not in sys.path:
    sys.path.insert(0, str(_INR_ROOT))

from inr.physics import compute_fa_md_ad_rd, dti_forward_signal  # noqa: E402


def predict_signal(
    S0: torch.Tensor,
    D: torch.Tensor,
    bvals: torch.Tensor,
    bvecs: torch.Tensor,
    *,
    b_scale: float = 1.0,
) -> torch.Tensor:
    """S = S0 * exp(-(b/b_scale) * g^T D g). Returns [V, N]."""
    return dti_forward_signal(S0, D, bvals, bvecs, b_scale=float(b_scale))


__all__ = ["predict_signal", "dti_forward_signal", "compute_fa_md_ad_rd"]
