"""PopulationDTIINR: PE(x) + z → trunk → (S0, D)."""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .parameter_field import ParameterField
from .spatial_encoder import SpatialEncoder
from .subject_latent import SubjectLatentTable, new_z, zero_z


class PopulationDTIINR(nn.Module):
    """
    Population-trained DTI INR.

    PE(x) = FourierFeatures(x)
    u = concat(PE(x), z_s)
    h = trunk_theta(u)
    S0, D = heads(h)
    """

    def __init__(
        self,
        train_subject_ids: Iterable[str] | None = None,
        *,
        latent_dim: int = 16,
        hidden: int = 128,
        layers: int = 4,
        pe_freqs: int = 8,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.pe_freqs = int(pe_freqs)
        self.encoder = SpatialEncoder(pe_freqs=self.pe_freqs, include_input=True)
        self.field = ParameterField(
            in_dim=self.encoder.out_dim + self.latent_dim,
            hidden=self.hidden,
            layers=self.layers,
        )
        ids = list(train_subject_ids or [])
        self.latents: SubjectLatentTable | None
        if ids:
            self.latents = SubjectLatentTable(ids, latent_dim=self.latent_dim)
        else:
            self.latents = None

    # ---- latent API ----
    def get_z(self, sid: str) -> torch.Tensor:
        if self.latents is None:
            raise RuntimeError("no training latent table attached")
        return self.latents.get_z(sid)

    def zero_z(self, *, device: torch.device | None = None) -> torch.Tensor:
        return zero_z(self.latent_dim, device=device)

    def new_z(self, trainable: bool = True, *, device: torch.device | None = None, init: str = "zeros") -> nn.Parameter:
        return new_z(self.latent_dim, trainable=trainable, init=init, device=device)

    def theta_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters()) + list(self.field.parameters())

    def freeze_theta(self) -> None:
        for p in self.theta_parameters():
            p.requires_grad_(False)

    def unfreeze_theta(self) -> None:
        for p in self.theta_parameters():
            p.requires_grad_(True)

    def theta_state_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for k, v in self.encoder.state_dict().items():
            out[f"encoder.{k}"] = v
        for k, v in self.field.state_dict().items():
            out[f"field.{k}"] = v
        return out

    def load_theta_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True) -> None:
        enc = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
        fld = {k[len("field.") :]: v for k, v in state.items() if k.startswith("field.")}
        self.encoder.load_state_dict(enc, strict=strict)
        self.field.load_state_dict(fld, strict=strict)

    def forward_with_z(self, xyz_m11: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        xyz_m11: [V, 3]
        z: [latent_dim] or [V, latent_dim] or [1, latent_dim]
        returns h, S0, D
        """
        pe = self.encoder(xyz_m11)
        if z.ndim == 1:
            z_b = z.unsqueeze(0).expand(pe.shape[0], -1)
        elif z.ndim == 2 and int(z.shape[0]) == 1:
            z_b = z.expand(pe.shape[0], -1)
        elif z.ndim == 2 and int(z.shape[0]) == int(pe.shape[0]):
            z_b = z
        else:
            raise ValueError(f"z shape {tuple(z.shape)} incompatible with xyz {tuple(xyz_m11.shape)}")
        if int(z_b.shape[-1]) != self.latent_dim:
            raise ValueError(f"z dim {z_b.shape[-1]} != latent_dim {self.latent_dim}")
        u = torch.cat([pe, z_b], dim=-1)
        return self.field(u)

    def forward(self, xyz_m11: torch.Tensor, subject_id: str | None = None, z: torch.Tensor | None = None):
        if z is None:
            if subject_id is None:
                raise ValueError("provide subject_id or z")
            if self.latents is None:
                raise RuntimeError("no latent table; pass z explicitly")
            z = self.latents.get_z(subject_id)
        h, S0, D = self.forward_with_z(xyz_m11, z)
        return S0, D
