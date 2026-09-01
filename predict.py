"""Predict DTI maps for a subject under a given latent z."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from metrics.evaluator import compute_dti_scalars_from_D
from models.population_dti_inr import PopulationDTIINR


@torch.no_grad()
def predict_maps(
    model: PopulationDTIINR,
    coords: np.ndarray | torch.Tensor,
    flat_idx: np.ndarray,
    shape_xyz: tuple[int, int, int],
    z: torch.Tensor,
    device: torch.device,
    *,
    chunk: int = 65536,
    want_D: bool = True,
) -> dict[str, np.ndarray]:
    model.eval()
    if isinstance(coords, np.ndarray):
        coords_t = torch.from_numpy(coords)
    else:
        coords_t = coords
    X, Y, Z = shape_xyz
    n = int(coords_t.shape[0])
    S0_vol = np.zeros((X * Y * Z,), dtype=np.float32)
    FA = np.zeros((X * Y * Z,), dtype=np.float32)
    MD = np.zeros((X * Y * Z,), dtype=np.float32)
    AD = np.zeros((X * Y * Z,), dtype=np.float32)
    RD = np.zeros((X * Y * Z,), dtype=np.float32)
    D_vol = np.zeros((X * Y * Z, 3, 3), dtype=np.float32) if want_D else None

    z = z.to(device)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        xyz = coords_t[sl].to(device)
        S0, D = model(xyz, z=z)
        scalars = compute_dti_scalars_from_D(D.detach().float().cpu().numpy())
        idx = flat_idx[sl]
        S0_vol[idx] = S0.detach().float().cpu().numpy()
        FA[idx] = scalars["FA"]
        MD[idx] = scalars["MD"]
        AD[idx] = scalars["AD"]
        RD[idx] = scalars["RD"]
        if D_vol is not None:
            D_vol[idx] = D.detach().float().cpu().numpy()

    out: dict[str, Any] = {
        "S0": S0_vol.reshape(X, Y, Z),
        "FA": FA.reshape(X, Y, Z),
        "MD": MD.reshape(X, Y, Z),
        "AD": AD.reshape(X, Y, Z),
        "RD": RD.reshape(X, Y, Z),
    }
    if D_vol is not None:
        out["D"] = D_vol.reshape(X, Y, Z, 3, 3)
    return out
