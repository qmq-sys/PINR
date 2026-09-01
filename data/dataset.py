"""Subject DTI dataset loading (brain voxels + shell-masked DWI)."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_INR_ROOT = Path(__file__).resolve().parents[2] / "INR"
if str(_INR_ROOT) not in sys.path:
    sys.path.insert(0, str(_INR_ROOT))

from inr.coords import masked_coords_and_indices  # noqa: E402
from inr.hcp_io import load_hcp_subject, normalize_bvecs, shell_volume_mask  # noqa: E402
from inr.train_independent import load_or_fit_wls_reference  # noqa: E402

from .sampling import build_indices_for_subject


@dataclass
class SubjectBundle:
    subject_id: str
    shape_xyz: tuple[int, int, int]
    affine: np.ndarray
    brain_mask: np.ndarray
    common_mask: np.ndarray  # brain & WLS valid
    train_coords: np.ndarray  # brain voxels [-1,1]
    train_flat_idx: np.ndarray
    eval_coords: np.ndarray  # common mask
    eval_flat_idx: np.ndarray
    dwi_flat: np.ndarray  # [X*Y*Z, N] float32
    bvals: np.ndarray
    bvecs: np.ndarray
    ref: dict[str, Any]
    n_volumes: int
    sampling_fraction: float


def load_subject_bundle(
    *,
    subject_id: str,
    cfg: dict[str, Any],
    trad_dir: Path,
    sampling_fraction: float | None = None,
) -> SubjectBundle:
    sid = str(subject_id).strip()
    frac = float(cfg.get("sampling_fraction", 1.0) if sampling_fraction is None else sampling_fraction)
    b0_thr = float(cfg["b0_threshold"])
    shell_tol = float(cfg["shell_tol"])
    shells = tuple(cfg.get("dti_shells", [1000.0]))

    bundle = load_hcp_subject(cfg["hcp_root"], sid, b0_threshold=b0_thr)
    data = bundle["data"]
    bvals_all = bundle["bvals"]
    bvecs_all = normalize_bvecs(bvals_all, bundle["bvecs"], b0_threshold=b0_thr)
    brain = bundle["brain_mask"]
    affine = bundle["affine"]

    vol_m = shell_volume_mask(
        bvals_all,
        b0_threshold=b0_thr,
        shell_tol=shell_tol,
        shells=shells,
        include_b0=True,
    )
    dwi = data[..., vol_m].astype(np.float32)
    bvals_u = bvals_all[vol_m].astype(np.float32)
    bvecs_u = bvecs_all[vol_m].astype(np.float32)

    local_idx = build_indices_for_subject(
        sid,
        bvals_all,
        fraction=frac,
        b0_threshold=b0_thr,
        shell_tol=shell_tol,
        shells=shells,
        base_seed=int(cfg.get("sampling_base_seed", 20260822)),
    )
    dwi = dwi[..., local_idx]
    bvals_u = bvals_u[local_idx]
    bvecs_u = bvecs_u[local_idx]

    # INR cfg subset for WLS
    inr_cfg = {
        "b0_threshold": b0_thr,
        "shell_tol": shell_tol,
        "dti_shells": list(shells),
    }
    ref = load_or_fit_wls_reference(
        bundle=bundle,
        trad_dir=Path(trad_dir),
        cfg=inr_cfg,
        skip_if_exists=bool(cfg.get("skip_traditional_if_exists", True)),
    )

    train_coords, train_flat = masked_coords_and_indices(brain)
    common = brain & ref["valid_mask"]
    if int(np.count_nonzero(common)) == 0:
        raise RuntimeError(f"{sid}: empty common_mask (brain & WLS_valid)")
    eval_coords, eval_flat = masked_coords_and_indices(common)

    X, Y, Z = [int(x) for x in brain.shape]
    dwi_flat = dwi.reshape(X * Y * Z, -1).astype(np.float32)

    return SubjectBundle(
        subject_id=sid,
        shape_xyz=(X, Y, Z),
        affine=np.asarray(affine, dtype=np.float32),
        brain_mask=brain.astype(bool),
        common_mask=common.astype(bool),
        train_coords=train_coords.astype(np.float32),
        train_flat_idx=train_flat.astype(np.int64),
        eval_coords=eval_coords.astype(np.float32),
        eval_flat_idx=eval_flat.astype(np.int64),
        dwi_flat=dwi_flat,
        bvals=bvals_u,
        bvecs=bvecs_u,
        ref=ref,
        n_volumes=int(bvals_u.shape[0]),
        sampling_fraction=frac,
    )


def s0_obs_from_batch(target: torch.Tensor, bvals: torch.Tensor, b0_threshold: float) -> torch.Tensor:
    """Per-voxel observed S0 from b0 channels. target [V,N], returns [V,1]."""
    b0 = bvals.reshape(-1) < float(b0_threshold)
    if bool(torch.any(b0).item()):
        s0 = target[:, b0].mean(dim=-1, keepdim=True)
    else:
        s0 = target.max(dim=-1, keepdim=True).values
    return s0.clamp_min(1.0)
