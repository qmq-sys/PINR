"""DTI parameter decode — identical math to INR SpatialDTIINR / SharedSpatialDTIINR.

Baseline classes are not imported/modified; formulas are mirrored here as pure functions.
"""
from __future__ import annotations

import torch


def decode_dti_parameters(params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode MLP outputs to (S0, D).

    params: [V, 7] = logS0(1) + L_raw(6)
    S0: [V]
    D:  [V, 3, 3] SPD via Cholesky LL^T

    Matches INR/inr/model.py SpatialDTIINR.forward and
    INR/inr/shared_model.py _params_to_s0_d.
    """
    if params.ndim != 2 or int(params.shape[-1]) != 7:
        raise ValueError(f"params must be [V,7], got {tuple(params.shape)}")

    logS0 = params[:, 0:1].clamp(-5.0, 12.0)
    L_raw = params[:, 1:]
    S0 = torch.exp(logS0).squeeze(-1)

    L11 = torch.exp(L_raw[:, 0].clamp(-8.0, -1.5))
    L22 = torch.exp(L_raw[:, 1].clamp(-8.0, -1.5))
    L33 = torch.exp(L_raw[:, 2].clamp(-8.0, -1.5))
    L21 = L_raw[:, 3].tanh() * 0.05
    L31 = L_raw[:, 4].tanh() * 0.05
    L32 = L_raw[:, 5].tanh() * 0.05
    zeros = torch.zeros_like(L11)
    row1 = torch.stack([L11, zeros, zeros], dim=-1)
    row2 = torch.stack([L21, L22, zeros], dim=-1)
    row3 = torch.stack([L31, L32, L33], dim=-1)
    Lmat = torch.stack([row1, row2, row3], dim=-2)
    D = Lmat @ Lmat.transpose(-1, -2)
    return S0, D


def decode_s0_from_logit(s0_raw: torch.Tensor) -> torch.Tensor:
    """S0 head: single logit → positive S0 (same clamp/exp as baseline logS0)."""
    logS0 = s0_raw.reshape(-1, 1).clamp(-5.0, 12.0)
    return torch.exp(logS0).squeeze(-1)


def decode_d_from_cholesky_raw(L_raw: torch.Tensor) -> torch.Tensor:
    """D head: 6 Cholesky params → SPD D [V,3,3] (same as baseline)."""
    if L_raw.ndim != 2 or int(L_raw.shape[-1]) != 6:
        raise ValueError(f"L_raw must be [V,6], got {tuple(L_raw.shape)}")
    L11 = torch.exp(L_raw[:, 0].clamp(-8.0, -1.5))
    L22 = torch.exp(L_raw[:, 1].clamp(-8.0, -1.5))
    L33 = torch.exp(L_raw[:, 2].clamp(-8.0, -1.5))
    L21 = L_raw[:, 3].tanh() * 0.05
    L31 = L_raw[:, 4].tanh() * 0.05
    L32 = L_raw[:, 5].tanh() * 0.05
    zeros = torch.zeros_like(L11)
    row1 = torch.stack([L11, zeros, zeros], dim=-1)
    row2 = torch.stack([L21, L22, zeros], dim=-1)
    row3 = torch.stack([L31, L32, L33], dim=-1)
    Lmat = torch.stack([row1, row2, row3], dim=-2)
    return Lmat @ Lmat.transpose(-1, -2)
