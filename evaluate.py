"""Evaluate Population-DTI-INR predictions vs WLS reference + signal metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from data.dataset import SubjectBundle, s0_obs_from_batch
from metrics.evaluator import dti_parameter_metrics, format_metrics_row, signal_metrics
from models.population_dti_inr import PopulationDTIINR
from physics.dti_forward import predict_signal
from predict import predict_maps


@torch.no_grad()
def evaluate_subject(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    z: torch.Tensor,
    *,
    device: torch.device,
    cfg: dict[str, Any],
    mode: str,
    max_signal_voxels: int = 65536,
    seed: int = 0,
    adapt_iter: int | None = None,
) -> dict[str, Any]:
    model.eval()
    b_scale = float(cfg.get("b_scale", 1.0))
    b0_thr = float(cfg["b0_threshold"])
    max_signal = float(cfg.get("max_signal", 1.0))
    sig_norm = str(cfg.get("signal_normalization", "s0"))

    # Parameter maps on common mask voxels (full brain coords for volume fill)
    maps = predict_maps(
        model,
        subj.train_coords,
        subj.train_flat_idx,
        subj.shape_xyz,
        z,
        device,
        want_D=True,
    )
    dti = dti_parameter_metrics(
        {"FA": maps["FA"], "MD": maps["MD"], "AD": maps["AD"], "RD": maps["RD"]},
        {
            "FA": subj.ref["FA"],
            "MD": subj.ref["MD"],
            "AD": subj.ref["AD"],
            "RD": subj.ref["RD"],
        },
        subj.common_mask,
    )

    # Signal metrics on subsample of common-mask voxels
    rng = np.random.default_rng(seed)
    n = int(subj.eval_coords.shape[0])
    sel = np.arange(n) if n <= max_signal_voxels else rng.choice(n, size=max_signal_voxels, replace=False)
    xyz = torch.from_numpy(subj.eval_coords[sel]).to(device)
    target = torch.from_numpy(subj.dwi_flat[subj.eval_flat_idx[sel]]).to(device)
    bvals_t = torch.from_numpy(subj.bvals).to(device)
    bvecs_t = torch.from_numpy(subj.bvecs).to(device)
    S0, D = model(xyz, z=z.to(device))
    pred = predict_signal(S0, D, bvals_t, bvecs_t, b_scale=b_scale)

    if sig_norm.lower() in {"s0", "s0_norm", "normalized"}:
        s0 = s0_obs_from_batch(target, bvals_t, b0_thr)
        pred_n = (pred / s0).detach().cpu().numpy()
        obs_n = (target / s0).detach().cpu().numpy()
    else:
        pred_n = pred.detach().cpu().numpy()
        obs_n = target.detach().cpu().numpy()
        # if raw, max_signal from config still used for PSNR (user responsibility)

    # Mid-slice plane for SSIM: use mean over directions at eval voxels → scatter not grid.
    # Use flattened S0-norm signals for MSE/PSNR; SSIM on reshaped block.
    sig = signal_metrics(pred_n, obs_n, max_signal=max_signal, compute_ssim=True)

    theta_frozen = not any(p.requires_grad for p in model.theta_parameters())
    z_trainable = bool(getattr(z, "requires_grad", False))
    print(
        f"  [eval] subject_id={subj.subject_id} mode={mode} "
        f"z={tuple(z.shape)} FA_MAE={dti['FA']['MAE']:.6f} "
        f"MD_MAE={dti['MD']['MAE']:.6f} PSNR={sig['PSNR']:.3f} RelMSE={sig['RelMSE']:.6e} "
        f"theta_frozen={theta_frozen} z_trainable={z_trainable}",
        flush=True,
    )

    row = format_metrics_row(
        subj.subject_id,
        mode,
        dti,
        sig,
        adapt_iter=adapt_iter,
        theta_frozen=theta_frozen,
        z_trainable=z_trainable,
        sampling_fraction=subj.sampling_fraction,
        n_volumes=subj.n_volumes,
    )
    return {
        "row": row,
        "dti": dti,
        "signal": sig,
        "maps": maps,
    }
