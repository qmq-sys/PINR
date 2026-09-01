"""Subject latent table (ID-keyed) + factory for unseen z_new."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class SubjectLatentTable(nn.Module):
    """
    Train-time latents keyed by subject ID string.

    Not an nn.Embedding(num_subjects) — each z is an independent Parameter
    in a ParameterDict so get_z(sid) works by ID.
    """

    def __init__(self, subject_ids: Iterable[str], latent_dim: int = 16, init_std: float = 0.01):
        super().__init__()
        self.latent_dim = int(latent_dim)
        ids = [str(s).strip() for s in subject_ids]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate subject ids: {ids}")
        self.subject_ids: list[str] = list(ids)
        table: dict[str, nn.Parameter] = {}
        for sid in self.subject_ids:
            table[sid] = nn.Parameter(torch.randn(self.latent_dim) * float(init_std))
        self.z = nn.ParameterDict(table)

    def has(self, sid: str) -> bool:
        return str(sid).strip() in self.z

    def get_z(self, sid: str) -> torch.Tensor:
        key = str(sid).strip()
        if key not in self.z:
            raise KeyError(f"subject '{key}' not in training latent table: {self.subject_ids}")
        return self.z[key]

    def forward_lookup(self, sid: str, n_voxels: int) -> torch.Tensor:
        z = self.get_z(sid)
        return z.unsqueeze(0).expand(int(n_voxels), -1)

    def state_dict_by_id(self) -> dict[str, torch.Tensor]:
        return {sid: self.get_z(sid).detach().cpu().clone() for sid in self.subject_ids}

    def load_state_dict_by_id(self, latents: dict[str, torch.Tensor]) -> None:
        for sid, z in latents.items():
            key = str(sid).strip()
            if key not in self.z:
                raise KeyError(f"cannot load latent for unknown sid {key}")
            t = torch.as_tensor(z, dtype=self.get_z(key).dtype)
            if t.numel() != self.latent_dim:
                raise ValueError(f"latent dim mismatch for {key}: {t.numel()} vs {self.latent_dim}")
            with torch.no_grad():
                self.get_z(key).copy_(t.reshape(self.latent_dim))


def zero_z(
    latent_dim: int = 16,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a zero latent vector (not a Parameter)."""
    t = torch.zeros(int(latent_dim), dtype=dtype)
    return t if device is None else t.to(device)


def new_z(
    latent_dim: int = 16,
    *,
    trainable: bool = True,
    init: str = "zeros",
    init_std: float = 0.01,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> nn.Parameter:
    """
    Create a new latent unrelated to the training subject mapping.

    init: 'zeros' | 'randn'
    """
    d = int(latent_dim)
    if init == "zeros":
        data = torch.zeros(d, dtype=dtype)
    elif init == "randn":
        data = torch.randn(d, dtype=dtype) * float(init_std)
    else:
        raise ValueError(f"unknown init={init!r}")
    if device is not None:
        data = data.to(device)
    return nn.Parameter(data, requires_grad=bool(trainable))
