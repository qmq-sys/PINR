"""Spatial Fourier encoder — reuses INR FourierFeatures without modifying INR."""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

_INR_ROOT = Path(__file__).resolve().parents[2] / "INR"
if str(_INR_ROOT) not in sys.path:
    sys.path.insert(0, str(_INR_ROOT))

from inr.model import FourierFeatures  # noqa: E402


class SpatialEncoder(nn.Module):
    """PE(x) = FourierFeatures(x); coordinates expected in [-1, 1]^3."""

    def __init__(self, pe_freqs: int = 8, include_input: bool = True):
        super().__init__()
        self.pe = FourierFeatures(n_freqs=int(pe_freqs), include_input=bool(include_input))

    @property
    def out_dim(self) -> int:
        return int(self.pe.out_dim)

    def forward(self, xyz_m11):
        return self.pe(xyz_m11)
