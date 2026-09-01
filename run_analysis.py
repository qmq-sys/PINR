#!/usr/bin/env python
"""One-click PINR experiment analysis: eval → metrics → figures → report.

Does not modify models/, physics/, train/, or checkpoint format.
"""
from __future__ import annotations

import argparse
import json
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

from adapt_subject import _build_model_from_cfg, run_latent_adaptation, run_zero_shot
from data.dataset import load_subject_bundle
from data.split import split_from_config
from evaluate import evaluate_subject
from generate_report import generate_report
from metrics.evaluator import dti_parameter_metrics, format_metrics_row
from utils_io import _write_csv, load_theta, load_yaml, package_root, resolve_device, save_json
from visualization.scatter_plots import plot_scalar_scatter
from visualization.training_curves import plot_training_curves
from visualization.visualize_maps import save_subject_map_figures


def ensure_results_tree(results_dir: Path) -> Path:
    results_dir = Path(results_dir)
    for sub in (
        "metrics",
        "figures/maps",
        "figures/scatter",
        "figures/curves",
        "report",
        "maps",
        "checkpoints",
        "logs",
        "plots",
        "config",
    ):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)
    return results_dir


def resolve_checkpoint(exp_dir: Path, checkpoint: str | None) -> Path:
    exp_dir = Path(exp_dir)
    if checkpoint:
        c = Path(checkpoint)
        if c.is_file():
            return c
        cand = exp_dir / "checkpoints" / checkpoint
        if (cand / "theta.pt").is_file():
            return cand / "theta.pt"
        if cand.with_suffix(".pt").is_file():
            return cand.with_suffix(".pt")
        if cand.is_file():
            return cand
    for rel in (
        "checkpoints/best/theta.pt",
        "checkpoints/last/theta.pt",
        "checkpoints/epoch_0150/theta.pt",
    ):
        p = exp_dir / rel
        if p.is_file():
            return p
    # any epoch_* / theta.pt
    matches = sorted(exp_dir.glob("checkpoints/epoch_*/theta.pt"))
    if matches:
        return matches[-1]
    raise FileNotFoundError(f"No theta.pt under {exp_dir}/checkpoints")


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject_id": row.get("subject_id"),
        "mode": row.get("mode"),
        "sampling": row.get("sampling_fraction", row.get("sampling")),
        "PSNR": row.get("signal_PSNR", row.get("PSNR")),
        "SSIM": row.get("signal_SSIM", row.get("SSIM")),
        "MSE": row.get("signal_MSE", row.get("MSE")),
        "RMSE": row.get("signal_RMSE", row.get("RMSE")),
        "Relative_Error": row.get("signal_RelMSE", row.get("Relative_Error")),
        "FA_MAE": row.get("FA_MAE"),
        "FA_RMSE": row.get("FA_RMSE"),
        "FA_CC": row.get("FA_Pearson", row.get("FA_CC")),
        "MD_MAE": row.get("MD_MAE"),
        "MD_RMSE": row.get("MD_RMSE"),
        "MD_CC": row.get("MD_Pearson", row.get("MD_CC")),
        "AD_MAE": row.get("AD_MAE"),
        "RD_MAE": row.get("RD_MAE"),
        "adapt_iter": row.get("adapt_iter"),
        "n_voxels": row.get("n_voxels"),
    }


def _signal_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject_id": row.get("subject_id"),
        "mode": row.get("mode"),
        "sampling_fraction": row.get("sampling_fraction"),
        "adapt_iter": row.get("adapt_iter"),
        "MSE": row.get("signal_MSE"),
        "RMSE": row.get("signal_RMSE"),
        "PSNR": row.get("signal_PSNR"),
        "SSIM": row.get("signal_SSIM"),
        "Relative_Error": row.get("signal_RelMSE"),
    }


def _dti_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject_id": row.get("subject_id"),
        "mode": row.get("mode"),
        "sampling_fraction": row.get("sampling_fraction"),
        "adapt_iter": row.get("adapt_iter"),
        "FA_MAE": row.get("FA_MAE"),
        "FA_RMSE": row.get("FA_RMSE"),
        "FA_Pearson": row.get("FA_Pearson"),
        "MD_MAE": row.get("MD_MAE"),
        "MD_RMSE": row.get("MD_RMSE"),
        "MD_Pearson": row.get("MD_Pearson"),
        "AD_MAE": row.get("AD_MAE"),
        "RD_MAE": row.get("RD_MAE"),
        "n_voxels": row.get("n_voxels"),
    }


def metrics_from_existing_maps(
    *,
    subj,
    maps_npz: Path,
    mode: str,
    adapt_iter: int | None,
    signal_fallback: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    data = np.load(maps_npz)
    pred = {k: data[k] for k in ("FA", "MD", "AD", "RD") if k in data.files}
    dti = dti_parameter_metrics(
        pred,
        {k: subj.ref[k] for k in ("FA", "MD", "AD", "RD")},
        subj.common_mask,
    )
    if signal_fallback is None:
        sig = {"MSE": float("nan"), "RMSE": float("nan"), "RelMSE": float("nan"), "PSNR": float("nan"), "SSIM": float("nan")}
    else:
        sig = {
            "MSE": float(signal_fallback.get("signal_MSE", signal_fallback.get("MSE", float("nan")))),
            "RMSE": float(signal_fallback.get("signal_RMSE", signal_fallback.get("RMSE", float("nan")))),
            "RelMSE": float(signal_fallback.get("signal_RelMSE", signal_fallback.get("RelMSE", float("nan")))),
            "PSNR": float(signal_fallback.get("signal_PSNR", signal_fallback.get("PSNR", float("nan")))),
            "SSIM": float(signal_fallback.get("signal_SSIM", signal_fallback.get("SSIM", float("nan")))),
        }
    row = format_metrics_row(
        subj.subject_id,
        mode,
        dti,
        sig,
        adapt_iter=adapt_iter,
        sampling_fraction=subj.sampling_fraction,
        n_volumes=subj.n_volumes,
        source_maps=str(maps_npz),
    )
    maps = {**pred}
    if "S0" in data.files:
        maps["S0"] = data["S0"]
    return row, maps


def write_metric_tables(results_dir: Path, rows: list[dict[str, Any]]) -> None:
    summaries = [_summary_row(r) for r in rows]
    signals = [_signal_row(r) for r in rows]
    dtis = [_dti_row(r) for r in rows]
    _write_csv(results_dir / "metrics" / "summary.csv", summaries)
    _write_csv(results_dir / "metrics" / "signal_metrics.csv", signals)
    _write_csv(results_dir / "metrics" / "dti_metrics.csv", dtis)


def make_figures_for_row(
    *,
    results_dir: Path,
    subj,
    row: dict[str, Any],
    maps: dict[str, np.ndarray],
) -> None:
    mode = str(row.get("mode", "unknown"))
    sid = str(row.get("subject_id"))
    save_subject_map_figures(
        subject_id=sid,
        mode=mode,
        pred_maps=maps,
        ref_maps=subj.ref,
        mask=subj.common_mask,
        maps_dir=results_dir / "figures" / "maps",
    )
    # scatter (FA / MD)
    scatter_dir = results_dir / "figures" / "scatter" / f"{sid}_{mode}"
    plot_scalar_scatter(
        subj.ref["FA"],
        maps["FA"],
        subj.common_mask,
        xlabel="reference FA (WLS)",
        ylabel="PINR FA",
        title=f"{sid} {mode} FA",
        out_path=scatter_dir / "FA_scatter.png",
    )
    plot_scalar_scatter(
        subj.ref["MD"],
        maps["MD"],
        subj.common_mask,
        xlabel="reference MD (WLS)",
        ylabel="PINR MD",
        title=f"{sid} {mode} MD",
        out_path=scatter_dir / "MD_scatter.png",
    )
    # also flat copies under figures/scatter/
    plot_scalar_scatter(
        subj.ref["FA"],
        maps["FA"],
        subj.common_mask,
        xlabel="reference FA (WLS)",
        ylabel="PINR FA",
        title=f"{sid} {mode} FA",
        out_path=results_dir / "figures" / "scatter" / f"{sid}_{mode}_FA_scatter.png",
    )
    plot_scalar_scatter(
        subj.ref["MD"],
        maps["MD"],
        subj.common_mask,
        xlabel="reference MD (WLS)",
        ylabel="PINR MD",
        title=f"{sid} {mode} MD",
        out_path=results_dir / "figures" / "scatter" / f"{sid}_{mode}_MD_scatter.png",
    )


def _mean_key(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for r in rows:
        try:
            v = float(r.get(key))
            if v == v:
                vals.append(v)
        except Exception:
            pass
    return float(sum(vals) / len(vals)) if vals else float("nan")


def print_terminal_summary(summary_rows: list[dict[str, Any]]) -> None:
    zero = [r for r in summary_rows if "zero" in str(r.get("mode", "")).lower()]
    adapt = [r for r in summary_rows if "adapt" in str(r.get("mode", "")).lower() or "latent" in str(r.get("mode", "")).lower()]
    # final adapt per subject
    best: dict[str, dict] = {}
    for r in adapt:
        sid = str(r.get("subject_id"))
        it = int(float(r.get("adapt_iter") or 0))
        if sid not in best or it >= int(float(best[sid].get("adapt_iter") or 0)):
            best[sid] = r
    adapt_f = list(best.values()) if best else adapt

    use = adapt_f if adapt_f else zero
    print("\n================================")
    print("PINR Experiment Summary")
    print("")
    print("Signal:")
    print(f"PSNR: {_mean_key(use, 'PSNR'):.4f}")
    print(f"SSIM: {_mean_key(use, 'SSIM'):.4f}")
    print("")
    print("DTI:")
    print(f"FA correlation: {_mean_key(use, 'FA_CC'):.4f}")
    print(f"MD correlation: {_mean_key(use, 'MD_CC'):.4f}")
    print("")
    print("Generalization:")
    print(f"Zero-shot FA_CC: {_mean_key(zero, 'FA_CC'):.4f}  PSNR: {_mean_key(zero, 'PSNR'):.4f}")
    print(f"Adaptation FA_CC: {_mean_key(adapt_f, 'FA_CC'):.4f}  PSNR: {_mean_key(adapt_f, 'PSNR'):.4f}")
    print("================================\n")


def run_analysis(
    *,
    exp_dir: Path,
    results_dir: Path,
    checkpoint: str | None = None,
    subjects: list[str] | None = None,
    adapt_iters: list[int] | None = None,
    reuse_maps: bool = False,
    skip_adapt: bool = False,
) -> Path:
    exp_dir = Path(exp_dir)
    results_dir = ensure_results_tree(results_dir)
    cfg_path = exp_dir / "config" / "run_config.yaml"
    if not cfg_path.is_file():
        # phase folders sometimes keep config elsewhere
        alt = list(exp_dir.glob("**/run_config.yaml"))
        if alt:
            cfg_path = alt[0]
        else:
            raise FileNotFoundError(f"run_config.yaml not found under {exp_dir}")
    cfg = dict(load_yaml(cfg_path))
    split = split_from_config(cfg)
    device = resolve_device(str(cfg.get("device", "auto")))
    theta_path = resolve_checkpoint(exp_dir, checkpoint)
    trad = Path(cfg["trad_root"])

    test_ids = list(subjects or split.get("test") or [])
    if not test_ids:
        raise RuntimeError("No test subjects in config/split")

    meta = {
        "exp_dir": str(exp_dir.resolve()),
        "checkpoint": str(theta_path.resolve()),
        "train_subjects": list(split.get("train") or []),
        "val_subjects": list(split.get("val") or []),
        "test_subjects": test_ids,
        "sampling_fraction": cfg.get("sampling_fraction"),
        "reuse_maps": reuse_maps,
    }
    save_json(results_dir / "report" / "meta.json", meta)

    all_rows: list[dict[str, Any]] = []
    figure_subjects: list[tuple[Any, dict, dict]] = []

    if reuse_maps:
        for tid in test_ids:
            subj = load_subject_bundle(subject_id=tid, cfg=cfg, trad_dir=trad / tid)
            # zero-shot maps
            zs = exp_dir / "maps" / f"{tid}_zero_shot.npz"
            zs_json = exp_dir / "metrics" / f"{tid}_zero_shot.json"
            if zs.is_file():
                fb = json.loads(zs_json.read_text(encoding="utf-8")) if zs_json.is_file() else None
                row, maps = metrics_from_existing_maps(
                    subj=subj, maps_npz=zs, mode="zero_shot", adapt_iter=0, signal_fallback=fb
                )
                all_rows.append(row)
                figure_subjects.append((subj, row, maps))
            # prefer largest adapt iter available
            adapt_maps = list(exp_dir.glob(f"maps/{tid}_adapt_iter*.npz"))
            if adapt_maps and not skip_adapt:
                def _iter_of(p: Path) -> int:
                    try:
                        return int(p.stem.split("iter")[-1])
                    except Exception:
                        return -1

                last = max(adapt_maps, key=_iter_of)
                it = _iter_of(last)
                js = exp_dir / "metrics" / f"{tid}_adapt_iter{it}.json"
                fb = json.loads(js.read_text(encoding="utf-8")) if js.is_file() else None
                row, maps = metrics_from_existing_maps(
                    subj=subj,
                    maps_npz=last,
                    mode="latent_adaptation",
                    adapt_iter=it,
                    signal_fallback=fb,
                )
                all_rows.append(row)
                figure_subjects.append((subj, row, maps))
    else:
        model = _build_model_from_cfg(cfg, split["train"], device)
        load_theta(theta_path, model, map_location=device)
        model.freeze_theta()
        iters = adapt_iters if adapt_iters is not None else list(cfg.get("adapt_iterations", [0, 10, 50, 100, 200]))
        # evaluate into results_dir (and also keep maps there)
        for tid in test_ids:
            subj = load_subject_bundle(subject_id=tid, cfg=cfg, trad_dir=trad / tid)
            print(f"[analysis] subject={tid}", flush=True)
            if skip_adapt or (iters == [0]):
                z = model.zero_z(device=device)
                res = evaluate_subject(
                    model, subj, z, device=device, cfg=cfg, mode="zero_shot", adapt_iter=0, seed=int(cfg.get("seed", 42))
                )
                row = res["row"]
                maps = res["maps"]
                np.savez_compressed(
                    results_dir / "maps" / f"{tid}_zero_shot.npz",
                    S0=maps["S0"], FA=maps["FA"], MD=maps["MD"], AD=maps["AD"], RD=maps["RD"],
                )
                save_json(results_dir / "metrics" / f"{tid}_zero_shot.json", row)
                all_rows.append(row)
                figure_subjects.append((subj, row, maps))
            else:
                # reuse adapt_subject helpers writing into results_dir
                run_zero_shot(model=model, subj=subj, cfg=cfg, device=device, out_dir=results_dir)
                hist = run_latent_adaptation(
                    model=model, subj=subj, cfg=cfg, device=device, out_dir=results_dir, checkpoints=iters
                )
                # reload final maps for figures
                zs = results_dir / "maps" / f"{tid}_zero_shot.npz"
                if zs.is_file():
                    row, maps = metrics_from_existing_maps(
                        subj=subj,
                        maps_npz=zs,
                        mode="zero_shot",
                        adapt_iter=0,
                        signal_fallback=json.loads((results_dir / "metrics" / f"{tid}_zero_shot.json").read_text(encoding="utf-8")),
                    )
                    # prefer live Pearson from just-written eval: use hist[0] if present
                    if hist:
                        all_rows.append(dict(hist[0]))
                    else:
                        all_rows.append(row)
                    figure_subjects.append((subj, all_rows[-1], maps))
                if hist:
                    last = hist[-1]
                    it = int(last.get("adapt_iter", max(iters)))
                    mp = results_dir / "maps" / f"{tid}_adapt_iter{it}.npz"
                    if mp.is_file():
                        _, maps = metrics_from_existing_maps(
                            subj=subj, maps_npz=mp, mode="latent_adaptation", adapt_iter=it, signal_fallback=last
                        )
                        all_rows.append(dict(last))
                        figure_subjects.append((subj, last, maps))

    # de-duplicate rows by (subject, mode, adapt_iter)
    dedup: dict[tuple, dict] = {}
    for r in all_rows:
        key = (str(r.get("subject_id")), str(r.get("mode")), str(r.get("adapt_iter")))
        dedup[key] = r
    all_rows = list(dedup.values())

    write_metric_tables(results_dir, all_rows)

    # figures
    for subj, row, maps in figure_subjects:
        make_figures_for_row(results_dir=results_dir, subj=subj, row=row, maps=maps)

    plot_training_curves(exp_dir, results_dir / "figures" / "curves")
    save_json(results_dir / "report" / "meta.json", meta)
    generate_report(results_dir, meta=meta)

    summary_rows = [_summary_row(r) for r in all_rows]
    print_terminal_summary(summary_rows)
    print(f"[analysis] results → {results_dir.resolve()}")
    return results_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="PINR interpretability analysis one-click runner")
    ap.add_argument("--exp-dir", required=True, help="experiment directory with config/ + checkpoints/")
    ap.add_argument(
        "--results-dir",
        default="",
        help="output results/ (default: <package>/results)",
    )
    ap.add_argument("--checkpoint", default="", help="theta path or checkpoint folder name (best|last|epoch_0150)")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--adapt-iters", default="", help="comma list e.g. 0,200 (default: config adapt_iterations)")
    ap.add_argument("--reuse-maps", action="store_true", help="recompute DTI metrics/figures from existing maps/*.npz")
    ap.add_argument("--skip-adapt", action="store_true", help="only zero-shot")
    args = ap.parse_args()

    iters = None
    if args.adapt_iters.strip():
        iters = [int(x) for x in args.adapt_iters.split(",") if x.strip() != ""]

    results = Path(args.results_dir) if args.results_dir else package_root() / "results"
    run_analysis(
        exp_dir=Path(args.exp_dir),
        results_dir=results,
        checkpoint=args.checkpoint or None,
        subjects=args.subjects,
        adapt_iters=iters,
        reuse_maps=bool(args.reuse_maps),
        skip_adapt=bool(args.skip_adapt),
    )


if __name__ == "__main__":
    main()
