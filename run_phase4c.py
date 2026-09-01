#!/usr/bin/env python
"""Phase 4-C: Direct D-field sensitivity diagnostic (inference only).

Fixed theta = Phase4-A epoch_0150. Compare D(x,z) across latents on unseen 106319.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.dataset import load_subject_bundle
from data.split import split_from_config
from models.population_dti_inr import PopulationDTIINR
from predict import predict_maps
from utils_io import (
    _write_csv,
    load_latents,
    load_theta,
    load_yaml,
    make_experiment_dir,
    resolve_device,
    save_json,
    save_yaml,
)

DEFAULT_PHASE4A = ROOT / "experiments" / "population_dti_phase4a" / "20260829_192714"
DEFAULT_PHASE4B = ROOT / "experiments" / "population_dti_phase4b" / "20260829_201921"
EPS = 1e-12


def _build_model(cfg: dict[str, Any], train_ids: list[str], device: torch.device) -> PopulationDTIINR:
    return PopulationDTIINR(
        train_subject_ids=train_ids,
        latent_dim=int(cfg.get("latent_dim", 16)),
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 4)),
        pe_freqs=int(cfg.get("pe_freqs", 8)),
    ).to(device)


def _eigs_desc(D: np.ndarray) -> np.ndarray:
    """D [V,3,3] -> eigenvalues descending [V,3]."""
    Df = 0.5 * (D + np.swapaxes(D, -1, -2))
    Df = np.nan_to_num(Df, nan=0.0, posinf=0.0, neginf=0.0)
    eye = np.eye(3, dtype=np.float64)
    Df = Df + float(EPS) * eye
    evals = np.linalg.eigvalsh(Df.astype(np.float64))  # ascending
    evals = np.clip(evals, 0.0, None)
    return evals[..., ::-1].copy()  # lambda1 >= lambda2 >= lambda3


def _fa_md_ad_rd(l123: np.ndarray) -> dict[str, np.ndarray]:
    l1, l2, l3 = l123[..., 0], l123[..., 1], l123[..., 2]
    md = (l1 + l2 + l3) / 3.0
    ad = l1
    rd = 0.5 * (l2 + l3)
    num = (l1 - md) ** 2 + (l2 - md) ** 2 + (l3 - md) ** 2
    den = l1**2 + l2**2 + l3**2
    fa = np.sqrt(1.5 * num / np.maximum(den, EPS))
    fa = np.clip(fa, 0.0, 1.0)
    return {"FA": fa, "MD": md, "AD": ad, "RD": rd}


def _frobenius(D: np.ndarray) -> np.ndarray:
    """||D||_F per voxel. D [V,3,3]."""
    return np.sqrt(np.sum(D.astype(np.float64) ** 2, axis=(-2, -1)))


def _component_stats(D: np.ndarray) -> dict[str, float]:
    """Stats over unique symmetric components of D [V,3,3]."""
    # Store all 9 for completeness; also report lower-triangular unique
    names = ["D00", "D01", "D02", "D10", "D11", "D12", "D20", "D21", "D22"]
    out: dict[str, float] = {}
    flat = D.reshape(-1, 9)
    for i, name in enumerate(names):
        c = flat[:, i]
        out[f"{name}_mean"] = float(np.mean(c))
        out[f"{name}_std"] = float(np.std(c))
        out[f"{name}_min"] = float(np.min(c))
        out[f"{name}_max"] = float(np.max(c))
    return out


def analyze_field(
    latent_id: str,
    D: np.ndarray,
    mask: np.ndarray,
    D0: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    """D, D0 as full volumes [X,Y,Z,3,3]; stats on mask voxels."""
    m = np.asarray(mask, dtype=bool)
    Dv = D[m].astype(np.float64)  # [V,3,3]
    l123 = _eigs_desc(Dv)
    scalars = _fa_md_ad_rd(l123)
    fro = _frobenius(Dv)

    row: dict[str, Any] = {
        "latent_id": latent_id,
        "n_voxels": int(Dv.shape[0]),
        "mean_FA": float(np.mean(scalars["FA"])),
        "std_FA": float(np.std(scalars["FA"])),
        "mean_MD": float(np.mean(scalars["MD"])),
        "std_MD": float(np.std(scalars["MD"])),
        "mean_AD": float(np.mean(scalars["AD"])),
        "std_AD": float(np.std(scalars["AD"])),
        "mean_RD": float(np.mean(scalars["RD"])),
        "std_RD": float(np.std(scalars["RD"])),
        "mean_lambda1": float(np.mean(l123[:, 0])),
        "std_lambda1": float(np.std(l123[:, 0])),
        "mean_lambda2": float(np.mean(l123[:, 1])),
        "std_lambda2": float(np.std(l123[:, 1])),
        "mean_lambda3": float(np.mean(l123[:, 2])),
        "std_lambda3": float(np.std(l123[:, 2])),
        "mean_D_frobenius": float(np.mean(fro)),
        "std_D_frobenius": float(np.std(fro)),
    }
    row.update(_component_stats(Dv))

    maps_delta: dict[str, np.ndarray] | None = None
    if D0 is not None:
        D0v = D0[m].astype(np.float64)
        l0 = _eigs_desc(D0v)
        s0 = _fa_md_ad_rd(l0)
        dD = Dv - D0v
        fro0 = _frobenius(D0v)
        fro_d = _frobenius(dD)
        rel = fro_d / (fro0 + EPS)

        # Orientation vs magnitude: compare eigenvector alignment of principal direction
        # Use absolute cosine of v1; also eigenvalue-only relative change
        # For orientation: eigh of each tensor
        def _v1(Dbatch: np.ndarray) -> np.ndarray:
            Df = 0.5 * (Dbatch + np.swapaxes(Dbatch, -1, -2))
            Df = Df + EPS * np.eye(3)
            # ascending eigenvectors; last column is v1
            _, evecs = np.linalg.eigh(Df.astype(np.float64))
            return evecs[..., :, 2]

        v1 = _v1(Dv)
        v10 = _v1(D0v)
        cos = np.abs(np.sum(v1 * v10, axis=-1))
        mean_abs_cos_v1 = float(np.mean(cos))

        d_lam = l123 - l0
        # Relative eigenvalue magnitude change (vs ||lambda0||)
        eig_mag0 = np.linalg.norm(l0, axis=-1)
        eig_mag_d = np.linalg.norm(d_lam, axis=-1)
        rel_eig = eig_mag_d / (eig_mag0 + EPS)

        row.update(
            {
                "mean_D_change": float(np.mean(fro_d)),
                "relative_D_change": float(np.mean(rel)),
                "median_relative_D_change": float(np.median(rel)),
                "p95_relative_D_change": float(np.percentile(rel, 95)),
                "mean_delta_lambda1": float(np.mean(d_lam[:, 0])),
                "mean_delta_lambda2": float(np.mean(d_lam[:, 1])),
                "mean_delta_lambda3": float(np.mean(d_lam[:, 2])),
                "mean_abs_delta_lambda1": float(np.mean(np.abs(d_lam[:, 0]))),
                "mean_abs_delta_lambda2": float(np.mean(np.abs(d_lam[:, 1]))),
                "mean_abs_delta_lambda3": float(np.mean(np.abs(d_lam[:, 2]))),
                "mean_relative_eigenvalue_change": float(np.mean(rel_eig)),
                "mean_abs_cos_v1": mean_abs_cos_v1,  # ~1 => orientation unchanged
                "mean_delta_FA": float(np.mean(scalars["FA"] - s0["FA"])),
                "mean_delta_MD": float(np.mean(scalars["MD"] - s0["MD"])),
                "mean_delta_AD": float(np.mean(scalars["AD"] - s0["AD"])),
                "mean_delta_RD": float(np.mean(scalars["RD"] - s0["RD"])),
                "mean_abs_delta_FA": float(np.mean(np.abs(scalars["FA"] - s0["FA"]))),
                "mean_abs_delta_MD": float(np.mean(np.abs(scalars["MD"] - s0["MD"]))),
            }
        )

        # Full-volume delta maps
        X, Y, Z = mask.shape
        fro_map = np.zeros((X, Y, Z), dtype=np.float32)
        fa_map = np.zeros((X, Y, Z), dtype=np.float32)
        md_map = np.zeros((X, Y, Z), dtype=np.float32)
        fro_map[m] = fro_d.astype(np.float32)
        fa_map[m] = (scalars["FA"] - s0["FA"]).astype(np.float32)
        md_map[m] = (scalars["MD"] - s0["MD"]).astype(np.float32)
        maps_delta = {
            "delta_D_frobenius": fro_map,
            "delta_FA": fa_map,
            "delta_MD": md_map,
        }
    else:
        row.update(
            {
                "mean_D_change": 0.0,
                "relative_D_change": 0.0,
                "median_relative_D_change": 0.0,
                "p95_relative_D_change": 0.0,
                "mean_delta_lambda1": 0.0,
                "mean_delta_lambda2": 0.0,
                "mean_delta_lambda3": 0.0,
                "mean_abs_delta_lambda1": 0.0,
                "mean_abs_delta_lambda2": 0.0,
                "mean_abs_delta_lambda3": 0.0,
                "mean_relative_eigenvalue_change": 0.0,
                "mean_abs_cos_v1": 1.0,
                "mean_delta_FA": 0.0,
                "mean_delta_MD": 0.0,
                "mean_delta_AD": 0.0,
                "mean_delta_RD": 0.0,
                "mean_abs_delta_FA": 0.0,
                "mean_abs_delta_MD": 0.0,
            }
        )

    return row, maps_delta


def diagnose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {r["latent_id"]: r for r in rows}
    non_zero = [r for r in rows if r["latent_id"] != "z=0"]

    rels = [float(r["relative_D_change"]) for r in non_zero]
    max_rel = float(max(rels)) if rels else 0.0
    mean_rel = float(np.mean(rels)) if rels else 0.0

    # Significant D change? use relative Frobenius threshold (e.g. 1% mean)
    A_significant = bool(mean_rel > 0.01 or max_rel > 0.02)

    # Orientation preserved if mean |cos v1| high
    cos_vals = [float(r["mean_abs_cos_v1"]) for r in non_zero]
    mean_cos = float(np.mean(cos_vals)) if cos_vals else 1.0
    # Eigenvalue relative change
    eig_rels = [float(r["mean_relative_eigenvalue_change"]) for r in non_zero]
    mean_eig_rel = float(np.mean(eig_rels)) if eig_rels else 0.0

    # C: change mainly eigenvalue magnitude vs orientation
    # If orientation stable (cos>~0.95) and some eig/D change exists -> magnitude-dominated
    # If cos drops a lot -> orientation also changes
    C_mainly_eigenvalue_magnitude = bool(mean_cos > 0.95 and (mean_eig_rel > 1e-4 or mean_rel > 1e-4))
    C_orientation_also = bool(mean_cos < 0.90)

    # D: z_new vs train latents sensitivity
    zn = by.get("z_new")
    trains = [by[k] for k in ("z_101309", "z_102715", "z_103515") if k in by]
    if zn and trains:
        zn_rel = float(zn["relative_D_change"])
        tr_rel = float(np.mean([float(t["relative_D_change"]) for t in trains]))
        D_different = bool(abs(zn_rel - tr_rel) > 0.5 * max(tr_rel, 1e-6) or zn_rel < 0.5 * tr_rel)
    else:
        zn_rel = tr_rel = float("nan")
        D_different = False

    # E: supports Phase4-B claim if D changes are small
    E_supports_weak_geometry = bool(mean_rel < 0.05 and max_rel < 0.10)

    return {
        "A_latent_significantly_changes_D": A_significant,
        "A_detail": {
            "mean_relative_D_change_across_latents": mean_rel,
            "max_relative_D_change": max_rel,
            "threshold_note": "significant if mean_rel>0.01 or max_rel>0.02",
        },
        "B_change_magnitude": {
            "per_latent_relative_D_change": {
                r["latent_id"]: float(r["relative_D_change"]) for r in non_zero
            },
            "per_latent_mean_D_change_frobenius": {
                r["latent_id"]: float(r["mean_D_change"]) for r in non_zero
            },
            "per_latent_mean_abs_delta_FA": {
                r["latent_id"]: float(r["mean_abs_delta_FA"]) for r in non_zero
            },
        },
        "C_mainly_eigenvalue_magnitude_not_orientation": C_mainly_eigenvalue_magnitude,
        "C_orientation_also_changes": C_orientation_also,
        "C_detail": {
            "mean_abs_cos_v1": mean_cos,
            "mean_relative_eigenvalue_change": mean_eig_rel,
            "interpretation": (
                "high |cos v1| (~1) => principal direction stable; "
                "relative_D / relative_eig quantify magnitude change"
            ),
        },
        "D_z_new_sensitivity_differs_from_train": D_different,
        "D_detail": {
            "z_new_relative_D_change": zn_rel,
            "train_mean_relative_D_change": tr_rel,
        },
        "E_supports_phase4b_weak_DTI_control": E_supports_weak_geometry,
        "answers": {
            "A": (
                "YES — latent induces measurable D(x,z) change"
                if A_significant
                else "NO — D(x,z) barely changes across latents (direct matrix evidence)"
            ),
            "B": {
                "mean_relative_Frobenius_change": mean_rel,
                "max_relative_Frobenius_change": max_rel,
                "per_latent": {r["latent_id"]: float(r["relative_D_change"]) for r in non_zero},
            },
            "C": (
                "YES — orientation (v1) largely preserved; change is mostly eigenvalue/magnitude"
                if C_mainly_eigenvalue_magnitude and not C_orientation_also
                else (
                    "MIXED — both magnitude and orientation shift"
                    if C_orientation_also
                    else "unclear / negligible D change"
                )
            ),
            "D": (
                "YES — z_new D-sensitivity differs from train latents"
                if D_different
                else "NO — z_new D-sensitivity similar in scale to train latents"
            ),
            "E": (
                "YES — direct D-field evidence supports weak geometry control by z"
                if E_supports_weak_geometry
                else "NO — D-field changes are large enough to challenge that claim"
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4-C Direct D-field Sensitivity")
    ap.add_argument("--phase4a-dir", default=str(DEFAULT_PHASE4A))
    ap.add_argument("--phase4b-dir", default=str(DEFAULT_PHASE4B))
    args = ap.parse_args()

    phase4a = Path(args.phase4a_dir)
    phase4b = Path(args.phase4b_dir)
    cfg = load_yaml(phase4a / "config" / "run_config.yaml")
    split = split_from_config(cfg)
    device = resolve_device(str(cfg.get("device", "auto")))

    ckpt = phase4a / "checkpoints" / "epoch_0150"
    z_new_path = phase4b / "checkpoints" / "z_new.pt"
    if not (ckpt / "theta.pt").is_file():
        raise FileNotFoundError(ckpt / "theta.pt")
    if not z_new_path.is_file():
        raise FileNotFoundError(z_new_path)

    exp_dir = make_experiment_dir(tag="population_dti_phase4c")
    save_yaml(exp_dir / "config" / "run_config.yaml", cfg)
    save_json(
        exp_dir / "config" / "sources.json",
        {
            "theta": str(ckpt / "theta.pt"),
            "latents": str(ckpt / "latents.pt"),
            "z_new": str(z_new_path),
        },
    )

    print("=" * 72)
    print("Phase 4-C Direct D-field Sensitivity Diagnostic")
    print(f"  theta={ckpt / 'theta.pt'}")
    print(f"  out={exp_dir}")
    print("=" * 72)

    model = _build_model(cfg, split["train"], device)
    load_theta(ckpt / "theta.pt", model, map_location=device)
    model.freeze_theta()
    model.eval()

    train_latents = load_latents(ckpt / "latents.pt")
    z_new = torch.load(z_new_path, map_location="cpu", weights_only=False)["z_new"].float()

    trad = Path(cfg["trad_root"])
    test_id = split["test"][0]
    subj = load_subject_bundle(subject_id=test_id, cfg=cfg, trad_dir=trad / test_id)
    mask = subj.brain_mask  # all valid brain voxels as requested
    print(f"[Phase4C] subject={test_id} brain_voxels={int(mask.sum())}", flush=True)

    latent_bank = {
        "z=0": torch.zeros(model.latent_dim),
        "z_101309": train_latents["101309"].float(),
        "z_102715": train_latents["102715"].float(),
        "z_103515": train_latents["103515"].float(),
        "z_new": z_new,
    }

    fields: dict[str, np.ndarray] = {}
    for lid, z in latent_bank.items():
        print(f"[Phase4C] predicting D for {lid} ...", flush=True)
        maps = predict_maps(
            model,
            subj.train_coords,
            subj.train_flat_idx,
            subj.shape_xyz,
            z.to(device),
            device,
            want_D=True,
        )
        fields[lid] = maps["D"]
        # quick sanity
        print(f"  D shape={maps['D'].shape} FA_mean={float(maps['FA'][mask].mean()):.4f}", flush=True)

    D0 = fields["z=0"]
    rows: list[dict[str, Any]] = []
    for lid, D in fields.items():
        row, dmaps = analyze_field(lid, D, mask, D0=None if lid == "z=0" else D0)
        rows.append(row)
        print(
            f"  [{lid}] rel_D={row['relative_D_change']:.6e} "
            f"dFA={row['mean_delta_FA']:.6e} |cos v1|={row['mean_abs_cos_v1']:.4f}",
            flush=True,
        )
        if dmaps is not None:
            np.savez_compressed(
                exp_dir / "maps" / f"delta_{lid.replace('=', '')}.npz",
                **dmaps,
            )

    _write_csv(exp_dir / "metrics" / "direct_d_sensitivity.csv", rows)
    diag = diagnose(rows)
    save_json(exp_dir / "metrics" / "phase4c_diagnosis.json", diag)

    # Optional mid-slice viz of Frobenius delta for z_new
    try:
        import matplotlib.pyplot as plt

        zmid = mask.shape[2] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, lid in zip(axes, ["z_101309", "z_new", "z_102715"]):
            p = exp_dir / "maps" / f"delta_{lid}.npz"
            if p.is_file():
                arr = np.load(p)["delta_D_frobenius"][:, :, zmid]
                im = ax.imshow(arr.T, origin="lower", cmap="magma")
                ax.set_title(f"||ΔD||_F {lid}")
                fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(exp_dir / "plots" / "delta_D_frobenius_midslice.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[Phase4C] plot skipped: {e}", flush=True)

    print("\n===== PHASE 4-C DIAGNOSIS =====")
    for k in ("A", "C", "D", "E"):
        print(f"  {k}: {diag['answers'][k]}")
    print(f"  B: {diag['answers']['B']}")
    print(f"\n[Phase4C] done → {exp_dir}")


if __name__ == "__main__":
    main()
