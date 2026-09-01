"""Unseen-subject inference: zero-shot and latent-only adaptation."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from data.dataset import SubjectBundle, load_subject_bundle, s0_obs_from_batch
from data.split import split_from_config
from evaluate import evaluate_subject
from models.population_dti_inr import PopulationDTIINR
from physics.dti_forward import predict_signal
from utils_io import (
    append_csv_row,
    load_theta,
    load_yaml,
    package_root,
    resolve_device,
    save_json,
    save_yaml,
)


def _build_model_from_cfg(cfg: dict[str, Any], train_ids: list[str], device: torch.device) -> PopulationDTIINR:
    model = PopulationDTIINR(
        train_subject_ids=train_ids,
        latent_dim=int(cfg.get("latent_dim", 16)),
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 4)),
        pe_freqs=int(cfg.get("pe_freqs", 8)),
    ).to(device)
    return model


def _mse_loss(pred, target, bvals, cfg):
    mode = str(cfg.get("signal_normalization", "s0")).lower()
    b0 = float(cfg["b0_threshold"])
    if mode in {"s0", "s0_norm", "normalized"}:
        s0 = s0_obs_from_batch(target, bvals, b0)
        return F.mse_loss(pred / s0, target / s0)
    return F.mse_loss(pred, target)


def run_zero_shot(
    *,
    model: PopulationDTIINR,
    subj: SubjectBundle,
    cfg: dict[str, Any],
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    model.freeze_theta()
    model.eval()
    z = model.zero_z(device=device)
    assert not any(p.requires_grad for p in model.theta_parameters())
    print(
        f"[zero_shot] subject_id={subj.subject_id} theta_frozen=True "
        f"z_trainable={z.requires_grad if hasattr(z, 'requires_grad') else False} "
        f"z={tuple(z.shape)}",
        flush=True,
    )
    if model.latents is not None and model.latents.has(subj.subject_id):
        raise RuntimeError(
            f"CONFLICT: unseen test subject {subj.subject_id} is in training latent table"
        )

    result = evaluate_subject(
        model, subj, z, device=device, cfg=cfg, mode="zero_shot", adapt_iter=0, seed=int(cfg.get("seed", 42))
    )
    maps = result["maps"]
    np.savez_compressed(
        out_dir / "maps" / f"{subj.subject_id}_zero_shot.npz",
        S0=maps["S0"],
        FA=maps["FA"],
        MD=maps["MD"],
        AD=maps["AD"],
        RD=maps["RD"],
    )
    save_json(out_dir / "metrics" / f"{subj.subject_id}_zero_shot.json", result["row"])
    append_csv_row(out_dir / "metrics" / "adaptation_curve.csv", result["row"])
    return result


def run_latent_adaptation(
    *,
    model: PopulationDTIINR,
    subj: SubjectBundle,
    cfg: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    checkpoints: list[int] | None = None,
) -> list[dict[str, Any]]:
    """
    Freeze theta; optimize z_new only.
    Record metrics at iteration checkpoints (including 0 = zero init before steps).
    """
    ckpts = list(checkpoints or cfg.get("adapt_iterations", [0, 10, 50, 100, 200]))
    ckpts = sorted(set(int(x) for x in ckpts))
    max_iter = max(ckpts) if ckpts else 0

    model.freeze_theta()
    for p in model.theta_parameters():
        if p.requires_grad:
            raise RuntimeError("theta not frozen before latent adaptation")

    z_new = model.new_z(trainable=True, device=device, init="zeros")
    lr = float(cfg.get("adapt_lr", cfg.get("lr", 1e-3)))
    opt = torch.optim.Adam([z_new], lr=lr)

    batch = int(cfg.get("batch_voxels", 4096))
    b_scale = float(cfg.get("b_scale", 1.0))
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    coords_t = torch.from_numpy(subj.train_coords)
    bvals_t = torch.from_numpy(subj.bvals).to(device)
    bvecs_t = torch.from_numpy(subj.bvecs).to(device)
    n_vox = int(coords_t.shape[0])

    adapt_cfg = {
        "mode": "latent_adaptation",
        "subject_id": subj.subject_id,
        "adapt_iterations": ckpts,
        "adapt_lr": lr,
        "batch_voxels": batch,
        "latent_dim": model.latent_dim,
        "theta_frozen": True,
    }
    save_yaml(out_dir / "checkpoints" / f"{subj.subject_id}_adapt_config.yaml", adapt_cfg)

    history: list[dict[str, Any]] = []
    results_at_ckpt: dict[int, dict[str, Any]] = {}

    def _eval_at(it: int) -> dict[str, Any]:
        with torch.no_grad():
            z_eval = z_new.detach().clone()
        res = evaluate_subject(
            model,
            subj,
            z_eval,
            device=device,
            cfg=cfg,
            mode="latent_adaptation",
            adapt_iter=it,
            seed=seed,
        )
        row = dict(res["row"])
        row["adapt_iter"] = it
        append_csv_row(out_dir / "metrics" / "adaptation_curve.csv", row)
        append_csv_row(out_dir / "metrics" / f"{subj.subject_id}_adapt_history.csv", row)
        torch.save({"z_new": z_new.detach().cpu(), "adapt_iter": it}, out_dir / "checkpoints" / f"{subj.subject_id}_z_new_iter{it}.pt")
        np.savez_compressed(
            out_dir / "maps" / f"{subj.subject_id}_adapt_iter{it}.npz",
            S0=res["maps"]["S0"],
            FA=res["maps"]["FA"],
            MD=res["maps"]["MD"],
            AD=res["maps"]["AD"],
            RD=res["maps"]["RD"],
        )
        save_json(out_dir / "metrics" / f"{subj.subject_id}_adapt_iter{it}.json", row)
        return res

    t0 = time.time()
    # iter 0 before any update
    if 0 in ckpts:
        results_at_ckpt[0] = _eval_at(0)
        history.append(results_at_ckpt[0]["row"])

    print(
        f"[latent_adaptation] subject_id={subj.subject_id} max_iter={max_iter} "
        f"theta_frozen=True z_trainable={z_new.requires_grad} z={tuple(z_new.shape)}",
        flush=True,
    )

    for it in range(1, max_iter + 1):
        model.train()  # dropout N/A; allows grad on z
        # Keep theta frozen
        model.freeze_theta()
        sel = rng.integers(0, n_vox, size=batch, endpoint=False)
        xyz = coords_t[sel].to(device)
        idx = subj.train_flat_idx[sel]
        target = torch.from_numpy(subj.dwi_flat[idx]).to(device)

        S0, D = model(xyz, z=z_new)
        pred = predict_signal(S0, D, bvals_t, bvecs_t, b_scale=b_scale)
        loss = _mse_loss(pred, target, bvals_t, cfg)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Sanity: theta grads must be None
        for p in model.theta_parameters():
            if p.grad is not None and float(p.grad.abs().sum()) > 0:
                raise RuntimeError("theta received gradients during latent adaptation")
        opt.step()

        if it == 1:
            print(
                f"  [adapt] iter=1 xyz={tuple(xyz.shape)} S0={tuple(S0.shape)} "
                f"D={tuple(D.shape)} S_pred={tuple(pred.shape)} loss={float(loss.detach()):.6e} "
                f"theta_frozen={not any(p.requires_grad for p in model.theta_parameters())} "
                f"z_trainable={z_new.requires_grad}",
                flush=True,
            )

        if it in ckpts:
            results_at_ckpt[it] = _eval_at(it)
            history.append(results_at_ckpt[it]["row"])
            print(f"  [adapt] checkpoint iter={it} wall={time.time()-t0:.1f}s", flush=True)

    torch.save({"z_new": z_new.detach().cpu(), "adapt_iter": max_iter}, out_dir / "checkpoints" / f"{subj.subject_id}_z_new.pt")
    return history


def adapt_from_experiment(exp_dir: Path, cfg: dict[str, Any] | None = None) -> None:
    exp_dir = Path(exp_dir)
    cfg = cfg or load_yaml(exp_dir / "config" / "run_config.yaml")
    split = split_from_config(cfg)
    device = resolve_device(str(cfg.get("device", "auto")))
    ckpt_dir = exp_dir / "checkpoints" / "best"
    if not (ckpt_dir / "theta.pt").is_file():
        ckpt_dir = exp_dir / "checkpoints" / "last"

    model = _build_model_from_cfg(cfg, split["train"], device)
    load_theta(ckpt_dir / "theta.pt", model, map_location=device)
    model.freeze_theta()

    trad_root = Path(cfg.get("trad_root", str(package_root().parent / "INR" / "outputs" / "step1_traditional_dti")))
    for tid in split["test"]:
        subj = load_subject_bundle(subject_id=tid, cfg=cfg, trad_dir=trad_root / tid)
        print(f"[adapt] unseen test subject {tid} vols={subj.n_volumes}", flush=True)
        run_zero_shot(model=model, subj=subj, cfg=cfg, device=device, out_dir=exp_dir)
        run_latent_adaptation(model=model, subj=subj, cfg=cfg, device=device, out_dir=exp_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--config", default="")
    args = ap.parse_args()
    cfg = load_yaml(args.config) if args.config else None
    adapt_from_experiment(Path(args.exp_dir), cfg)


if __name__ == "__main__":
    main()
