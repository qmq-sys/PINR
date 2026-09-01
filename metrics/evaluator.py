"""Unified DTI / signal metrics for Population-DTI-INR."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # pragma: no cover
    _ssim = None


def _masked(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)[np.asarray(mask, dtype=bool)]


def scalar_error_stats(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, median abs err, P95 abs err on mask."""
    m = np.asarray(mask, dtype=bool)
    p = _masked(pred, m)
    r = _masked(ref, m)
    if p.size == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "median_abs": float("nan"),
            "P95_abs": float("nan"),
            "n": 0,
        }
    err = np.abs(p - r)
    return {
        "MAE": float(np.mean(err)),
        "RMSE": float(np.sqrt(np.mean((p - r) ** 2))),
        "median_abs": float(np.median(err)),
        "P95_abs": float(np.percentile(err, 95.0)),
        "n": int(p.size),
    }


def compute_dti_scalars_from_D(D: np.ndarray, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """
    D[...,3,3] → FA, MD, AD, RD with descending eigenvalues.

    FA = sqrt(3/2) * sqrt(sum_i (λi-MD)^2 / sum_i λi^2)
    """
    Df = np.asarray(D, dtype=np.float64)
    Df = 0.5 * (Df + np.swapaxes(Df, -1, -2))
    Df = np.nan_to_num(Df, nan=0.0, posinf=0.0, neginf=0.0)
    eye = np.eye(3, dtype=np.float64)
    Df = Df + float(eps) * eye
    evals = np.linalg.eigvalsh(Df)  # ascending
    l3, l2, l1 = evals[..., 0], evals[..., 1], evals[..., 2]
    l1 = np.clip(l1, 0.0, None)
    l2 = np.clip(l2, 0.0, None)
    l3 = np.clip(l3, 0.0, None)
    md = (l1 + l2 + l3) / 3.0
    ad = l1
    rd = 0.5 * (l2 + l3)
    num = (l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2
    den = l1**2 + l2**2 + l3**2
    fa = np.sqrt(1.5 * num / np.maximum(den, eps))
    fa = np.clip(fa, 0.0, 1.0).astype(np.float32)
    return {
        "FA": fa,
        "MD": md.astype(np.float32),
        "AD": ad.astype(np.float32),
        "RD": rd.astype(np.float32),
    }


def dti_parameter_metrics(
    pred_maps: dict[str, np.ndarray],
    ref_maps: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    out: dict[str, Any] = {"n_voxels": int(np.count_nonzero(m)), "reference": "WLS-DTI"}
    for key in ("FA", "MD", "AD", "RD"):
        out[key] = scalar_error_stats(pred_maps[key], ref_maps[key], m)
    return out


def signal_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    *,
    max_signal: float = 1.0,
    compute_ssim: bool = True,
) -> dict[str, float]:
    """
    pred/obs: arrays of equal shape (already S0-normalized if configured).

    PSNR = 10 * log10(MAX_SIGNAL^2 / MSE)
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    o = np.asarray(obs, dtype=np.float64).ravel()
    if p.size == 0 or p.shape != o.shape:
        raise ValueError(f"pred/obs shape mismatch: {p.shape} vs {o.shape}")
    mse = float(np.mean((p - o) ** 2))
    rmse = float(math.sqrt(mse))
    # RelMSE = ||pred-obs||^2 / (||obs||^2 + eps)
    rel_mse = float(np.sum((p - o) ** 2) / (np.sum(o * o) + 1e-8))
    max_s = float(max_signal)
    if mse <= 0.0:
        psnr = float("inf")
    else:
        psnr = float(10.0 * math.log10((max_s * max_s) / mse))

    ssim_val = float("nan")
    if compute_ssim and _ssim is not None:
        # Prefer 2D mid-slice if volumes; else treat as 1D windowed SSIM on flattened
        pa = np.asarray(pred, dtype=np.float64)
        oa = np.asarray(obs, dtype=np.float64)
        try:
            if pa.ndim >= 2:
                # Reduce last dims by mean if needed to get a 2D plane
                while pa.ndim > 2:
                    pa = pa.mean(axis=-1)
                    oa = oa.mean(axis=-1)
                if pa.ndim == 2 and min(pa.shape) >= 7:
                    data_range = max(max_s, float(np.max(oa) - np.min(oa)), 1e-8)
                    ssim_val = float(_ssim(oa, pa, data_range=data_range))
            if not math.isfinite(ssim_val) and p.size >= 49:
                # reshape to approximate square for a coarse SSIM
                side = int(math.sqrt(p.size))
                side = max(7, side - (side % 1))
                n = side * side
                if n <= p.size:
                    data_range = max(max_s, float(np.max(o[:n]) - np.min(o[:n])), 1e-8)
                    ssim_val = float(
                        _ssim(o[:n].reshape(side, side), p[:n].reshape(side, side), data_range=data_range)
                    )
        except Exception:
            ssim_val = float("nan")

    return {
        "MSE": mse,
        "RMSE": rmse,
        "RelMSE": rel_mse,
        "PSNR": psnr,
        "SSIM": ssim_val,
        "n_values": int(p.size),
        "MAX_SIGNAL": max_s,
    }


def format_metrics_row(subject_id: str, mode: str, dti: dict[str, Any], sig: dict[str, float], **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "subject_id": subject_id,
        "mode": mode,
        "signal_MSE": sig.get("MSE"),
        "signal_RMSE": sig.get("RMSE"),
        "signal_RelMSE": sig.get("RelMSE"),
        "signal_PSNR": sig.get("PSNR"),
        "signal_SSIM": sig.get("SSIM"),
        "FA_MAE": dti.get("FA", {}).get("MAE"),
        "FA_RMSE": dti.get("FA", {}).get("RMSE"),
        "FA_median_abs": dti.get("FA", {}).get("median_abs"),
        "FA_P95_abs": dti.get("FA", {}).get("P95_abs"),
        "MD_MAE": dti.get("MD", {}).get("MAE"),
        "MD_RMSE": dti.get("MD", {}).get("RMSE"),
        "MD_median_abs": dti.get("MD", {}).get("median_abs"),
        "MD_P95_abs": dti.get("MD", {}).get("P95_abs"),
        "AD_MAE": dti.get("AD", {}).get("MAE"),
        "AD_RMSE": dti.get("AD", {}).get("RMSE"),
        "AD_median_abs": dti.get("AD", {}).get("median_abs"),
        "AD_P95_abs": dti.get("AD", {}).get("P95_abs"),
        "RD_MAE": dti.get("RD", {}).get("MAE"),
        "RD_RMSE": dti.get("RD", {}).get("RMSE"),
        "RD_median_abs": dti.get("RD", {}).get("median_abs"),
        "RD_P95_abs": dti.get("RD", {}).get("P95_abs"),
        "n_voxels": dti.get("n_voxels"),
    }
    row.update(extra)
    return row
