"""Shared MLP trunk + S0 / D heads."""
from __future__ import annotations

import torch
import torch.nn as nn

from .decode import decode_d_from_cholesky_raw, decode_s0_from_logit


def _init_s0_bias(bias: torch.Tensor) -> None:
    # Match Independent/Shared: logS0 bias ~ 6.0 → S0 ~ 400
    nn.init.constant_(bias, 6.0)


def _init_d_bias(bias: torch.Tensor) -> None:
    # Match Independent/Shared isotropic D ~ 0.001
    with torch.no_grad():
        bias.copy_(
            torch.tensor([-3.45, -3.45, -3.45, 0.0, 0.0, 0.0], dtype=bias.dtype, device=bias.device)
        )


class ParameterField(nn.Module):
    """
    u = concat(PE(x), z)
    h = trunk(u)
    S0 = S0_head(h)
    D   = CholeskyDecode(D_head(h))
    """

    def __init__(self, in_dim: int, hidden: int = 128, layers: int = 4):
        super().__init__()
        trunk: list[nn.Module] = []
        last = int(in_dim)
        for _ in range(int(layers)):
            trunk.append(nn.Linear(last, int(hidden)))
            trunk.append(nn.ReLU(inplace=True))
            last = int(hidden)
        self.trunk = nn.Sequential(*trunk)
        self.s0_head = nn.Linear(last, 1)
        self.d_head = nn.Linear(last, 6)
        nn.init.zeros_(self.s0_head.weight)
        nn.init.zeros_(self.d_head.weight)
        _init_s0_bias(self.s0_head.bias)
        _init_d_bias(self.d_head.bias)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          h:  [V, H]
          S0: [V]
          D:  [V, 3, 3]
        """
        h = self.trunk(u)
        S0 = decode_s0_from_logit(self.s0_head(h))
        D = decode_d_from_cholesky_raw(self.d_head(h))
        return h, S0, D
