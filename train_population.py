"""Population training: min_{θ,{z_s}} Σ_s MSE(S_pred, S_obs)."""
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
from models.population_dti_inr import PopulationDTIINR
from physics.dti_forward import predict_signal
from utils_io import (
    load_yaml,
    make_experiment_dir,
    package_root,
    resolve_device,
    save_json,
    save_population_checkpoint,
    save_yaml,
)


def _signal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bvals: torch.Tensor,
    *,
    signal_normalization: str,
    b0_threshold: float,
) -> torch.Tensor:
    """MSE on raw or S0-normalized signals."""
    mode = str(signal_normalization).lower()
    if mode in {"s0", "s0_norm", "normalized"}:
        s0 = s0_obs_from_batch(target, bvals, b0_threshold)
        pred_n = pred / s0
        tgt_n = target / s0
        return F.mse_loss(pred_n, tgt_n)
    if mode in {"none", "raw", "absolute"}:
        return F.mse_loss(pred, target)
    raise ValueError(f"unknown signal_normalization={signal_normalization!r}")


def _train_subject_epoch(
    model: PopulationDTIINR,
    opt: torch.optim.Optimizer,
    subj: SubjectBundle,
    *,
    device: torch.device,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    model.train()
    batch = int(cfg.get("batch_voxels", 4096))
    b_scale = float(cfg.get("b_scale", 1.0))
    b0_thr = float(cfg["b0_threshold"])
    sig_norm = str(cfg.get("signal_normalization", "s0"))

    coords_t = torch.from_numpy(subj.train_coords)
    bvals_t = torch.from_numpy(subj.bvals).to(device)
    bvecs_t = torch.from_numpy(subj.bvecs).to(device)
    n_vox = int(coords_t.shape[0])
    n_steps = max(1, int(np.ceil(n_vox / batch)))
    losses: list[float] = []

    z = model.get_z(subj.subject_id)
    for _ in range(n_steps):
        sel = rng.integers(0, n_vox, size=batch, endpoint=False)
        xyz = coords_t[sel].to(device)
        idx = subj.train_flat_idx[sel]
        target = torch.from_numpy(subj.dwi_flat[idx]).to(device)

        S0, D = model(xyz, subject_id=subj.subject_id)
        pred = predict_signal(S0, D, bvals_t, bvecs_t, b_scale=b_scale)
        loss = _signal_loss(pred, target, bvals_t, signal_normalization=sig_norm, b0_threshold=b0_thr)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))

        if len(losses) == 1:
            print(
                f"  [train] subject_id={subj.subject_id} "
                f"latent={tuple(z.shape)} xyz={tuple(xyz.shape)} "
                f"S0={tuple(S0.shape)} D={tuple(D.shape)} "
                f"S_pred={tuple(pred.shape)} loss={losses[-1]:.6e} "
                f"theta_grad={any(p.requires_grad for p in model.theta_parameters())} "
                f"z_grad={z.requires_grad}",
                flush=True,
            )
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _eval_subject_loss(
    model: PopulationDTIINR,
    subj: SubjectBundle,
    *,
    device: torch.device,
    cfg: dict[str, Any],
    max_voxels: int = 32768,
    seed: int = 0,
    mode: str = "train_latent",
) -> float:
    """
    mode:
      train_latent — use z from training table (train subjects only)
      zero_shot    — z=0 (for val subjects not in the latent table)
    """
    model.eval()
    rng = np.random.default_rng(seed)
    b_scale = float(cfg.get("b_scale", 1.0))
    b0_thr = float(cfg["b0_threshold"])
    sig_norm = str(cfg.get("signal_normalization", "s0"))
    n = int(subj.eval_coords.shape[0])
    sel = np.arange(n) if n <= max_voxels else rng.choice(n, size=max_voxels, replace=False)
    xyz = torch.from_numpy(subj.eval_coords[sel]).to(device)
    target = torch.from_numpy(subj.dwi_flat[subj.eval_flat_idx[sel]]).to(device)
    bvals_t = torch.from_numpy(subj.bvals).to(device)
    bvecs_t = torch.from_numpy(subj.bvecs).to(device)
    if mode == "zero_shot":
        z = model.zero_z(device=device)
        S0, D = model(xyz, z=z)
    else:
        S0, D = model(xyz, subject_id=subj.subject_id)
    pred = predict_signal(S0, D, bvals_t, bvecs_t, b_scale=b_scale)
    loss = _signal_loss(pred, target, bvals_t, signal_normalization=sig_norm, b0_threshold=b0_thr)
    return float(loss.cpu())

def train_population(cfg: dict[str, Any], exp_dir: Path | None = None) -> Path:
    split = split_from_config(cfg)
    train_ids = split["train"]
    val_ids = split["val"]
    device = resolve_device(str(cfg.get("device", "auto")))
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if exp_dir is None:
        exp_dir = make_experiment_dir(tag=str(cfg.get("experiment_tag", "population_dti")))
    save_yaml(exp_dir / "config" / "run_config.yaml", cfg)

    trad_root = Path(cfg.get("trad_root", str(INR_ROOT_DEFAULT())))
    print(f"[PopulationTrain] exp={exp_dir}")
    print(f"[PopulationTrain] train={train_ids} val={val_ids} test={split['test']}")
    print(f"[PopulationTrain] device={device} sampling_fraction={cfg.get('sampling_fraction', 1.0)}")

    subjects: list[SubjectBundle] = []
    for sid in train_ids:
        subj = load_subject_bundle(subject_id=sid, cfg=cfg, trad_dir=trad_root / sid)
        subjects.append(subj)
        print(
            f"  loaded train {sid}: brain={subj.train_coords.shape[0]} "
            f"common={subj.eval_coords.shape[0]} vols={subj.n_volumes}",
            flush=True,
        )

    val_subjects: list[SubjectBundle] = []
    for sid in val_ids:
        subj = load_subject_bundle(subject_id=sid, cfg=cfg, trad_dir=trad_root / sid)
        val_subjects.append(subj)
        print(
            f"  loaded val {sid}: brain={subj.train_coords.shape[0]} "
            f"common={subj.eval_coords.shape[0]} vols={subj.n_volumes}",
            flush=True,
        )

    model = PopulationDTIINR(
        train_subject_ids=train_ids,
        latent_dim=int(cfg.get("latent_dim", 16)),
        hidden=int(cfg.get("hidden", 128)),
        layers=int(cfg.get("layers", 4)),
        pe_freqs=int(cfg.get("pe_freqs", 8)),
    ).to(device)

    # Ensure unseen test IDs are NOT in latent table
    for tid in split["test"]:
        if model.latents is not None and model.latents.has(tid):
            raise RuntimeError(f"CONFLICT: test subject {tid} present in training latent table")

    mapping = {sid: i for i, sid in enumerate(train_ids)}
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))

    epochs = int(cfg.get("epochs", 30))
    log_every = int(cfg.get("log_every", 1))
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        order = list(subjects)
        rng.shuffle(order)
        train_losses = []
        for subj in order:
            ls = _train_subject_epoch(model, opt, subj, device=device, cfg=cfg, rng=rng)
            train_losses.append(ls)
        train_mean = float(np.mean(train_losses)) if train_losses else float("nan")

        val_mean = float("nan")
        if val_subjects:
            # CONFLICT resolution: val IDs are NOT in z_train table.
            # Monitor with zero-shot (z=0) so early-stopping reflects unseen-style generalization.
            # See docs note in run_pilot0 / architecture report.
            val_losses = [
                _eval_subject_loss(
                    model, vs, device=device, cfg=cfg, seed=seed + epoch, mode="zero_shot"
                )
                for vs in val_subjects
            ]
            val_mean = float(np.mean(val_losses)) if val_losses else float("nan")
        row = {
            "epoch": epoch,
            "train_signal_loss": train_mean,
            "val_signal_loss": val_mean,
            "train_mse": train_mean,  # alias
            "val_mse": val_mean,  # alias
            "wall_time_sec": time.time() - t0,
        }
        history.append(row)
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(
                f"[PopulationTrain] epoch {epoch}/{epochs} "
                f"train_signal_loss={train_mean:.6e} val_signal_loss={val_mean:.6e} "
                f"wall={row['wall_time_sec']:.1f}s",
                flush=True,
            )

        score = val_mean if np.isfinite(val_mean) else train_mean
        if score < best_val:
            best_val = score
            best_epoch = epoch
            save_population_checkpoint(
                exp_dir / "checkpoints" / "best",
                model=model,
                optimizer=opt,
                subject_mapping=mapping,
                config=cfg,
                history_rows=history,
            )

        # Optional epoch snapshots for diagnostics (does not change early-stopping).
        save_epochs = {int(e) for e in (cfg.get("save_checkpoint_epochs") or [])}
        if epoch in save_epochs:
            save_population_checkpoint(
                exp_dir / "checkpoints" / f"epoch_{epoch:04d}",
                model=model,
                optimizer=opt,
                subject_mapping=mapping,
                config={**cfg, "checkpoint_epoch": epoch},
                history_rows=history,
            )
            print(f"[PopulationTrain] saved snapshot epoch_{epoch:04d}", flush=True)

    save_population_checkpoint(
        exp_dir / "checkpoints" / "last",
        model=model,
        optimizer=opt,
        subject_mapping=mapping,
        config=cfg,
        history_rows=history,
    )

    # Canonical training history + curve under metrics/plots
    from utils_io import _write_csv  # noqa: WPS433

    hist_path = exp_dir / "metrics" / "training_history.csv"
    _write_csv(hist_path, history)
    # also copy under checkpoints root alias
    _write_csv(exp_dir / "checkpoints" / "training_history.csv", history)

    try:
        from plot_curves import plot_training_curve

        plot_training_curve(hist_path, exp_dir / "plots" / "training_curve.png")
    except Exception as e:
        print(f"[PopulationTrain] warning: training curve plot failed: {e}", flush=True)

    from param_efficiency import parameter_efficiency_report

    pe = parameter_efficiency_report(model)
    save_json(exp_dir / "metrics" / "parameter_efficiency.json", pe)
    print(f"[PopulationTrain] param_efficiency={pe}", flush=True)

    save_json(
        exp_dir / "metrics" / "train_summary.json",
        {
            "best_epoch": best_epoch,
            "best_val_signal_loss": best_val,
            "epochs": epochs,
            "train_subjects": train_ids,
            "val_subjects": val_ids,
            "test_subjects": split["test"],
            "training_time_sec": time.time() - t0,
            "parameter_efficiency": pe,
        },
    )
    print(f"[PopulationTrain] done best_epoch={best_epoch} best_score={best_val:.6e} → {exp_dir}")
    return exp_dir


def INR_ROOT_DEFAULT() -> Path:
    return package_root().parent / "INR" / "outputs" / "step1_traditional_dti"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Population-DTI-INR")
    ap.add_argument("--config", default=str(package_root() / "configs" / "pilot0.yaml"))
    ap.add_argument("--exp-dir", default="")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    exp = Path(args.exp_dir) if args.exp_dir else None
    train_population(cfg, exp_dir=exp)


if __name__ == "__main__":
    main()
