"""Generate results/report/experiment_report.md from summary metrics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from utils_io import load_json, package_root, save_json


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [_f(r.get(key)) for r in rows]
    vals = [v for v in vals if v == v]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _filter_mode(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    m = mode.lower()
    out = []
    for r in rows:
        rm = str(r.get("mode", "")).lower()
        if m == "zero_shot" and ("zero" in rm):
            out.append(r)
        elif m == "latent_adaptation" and ("adapt" in rm or "latent" in rm):
            # prefer final adapt rows if adapt_iter present: take max per subject later
            out.append(r)
    return out


def _final_adapt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep latest adapt_iter per subject for latent_adaptation."""
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        sid = str(r.get("subject_id", ""))
        it = int(float(r.get("adapt_iter", r.get("adapt_iteration", 0)) or 0))
        prev = best.get(sid)
        if prev is None or it >= int(float(prev.get("adapt_iter", prev.get("adapt_iteration", 0)) or 0)):
            best[sid] = r
    return list(best.values())


def build_report_text(
    *,
    summary_rows: list[dict[str, Any]],
    meta: dict[str, Any],
) -> str:
    zero = _filter_mode(summary_rows, "zero_shot")
    adapt = _final_adapt(_filter_mode(summary_rows, "latent_adaptation"))

    def block(rows: list[dict[str, Any]], label: str) -> str:
        if not rows:
            return f"### {label}\n\n_No rows._\n"
        return (
            f"### {label}\n\n"
            f"- n subjects: {len(rows)}\n"
            f"- mean PSNR: {_mean(rows, 'PSNR'):.4f}\n"
            f"- mean SSIM: {_mean(rows, 'SSIM'):.4f}\n"
            f"- mean Relative Error (RelMSE): {_mean(rows, 'Relative_Error'):.6e}\n"
            f"- mean FA_MAE: {_mean(rows, 'FA_MAE'):.6f}\n"
            f"- mean FA_CC: {_mean(rows, 'FA_CC'):.4f}\n"
            f"- mean MD_MAE: {_mean(rows, 'MD_MAE'):.6e}\n"
            f"- mean MD_CC: {_mean(rows, 'MD_CC'):.4f}\n"
        )

    fa_cc_adapt = _mean(adapt, "FA_CC") if adapt else _mean(zero, "FA_CC")
    fa_cc_zero = _mean(zero, "FA_CC")
    md_cc_adapt = _mean(adapt, "MD_CC") if adapt else _mean(zero, "MD_CC")
    md_cc_zero = _mean(zero, "MD_CC")
    psnr = _mean(adapt, "PSNR") if adapt else _mean(zero, "PSNR")
    ssim = _mean(adapt, "SSIM") if adapt else _mean(zero, "SSIM")
    rel = _mean(adapt, "Relative_Error") if adapt else _mean(zero, "Relative_Error")

    if fa_cc_adapt == fa_cc_adapt and fa_cc_adapt > 0.85:
        conclusion = "PINR successfully reconstructs DTI parameters."
    else:
        conclusion = "Further optimization required."

    train_ids = meta.get("train_subjects", [])
    test_ids = meta.get("test_subjects", [])
    sampling = meta.get("sampling_fraction", "n/a")

    lines = [
        "# PINR Experiment Report",
        "",
        "## Dataset",
        "",
        f"- subject number (train+val+test listed): train={len(train_ids)}, test={len(test_ids)}",
        f"- sampling ratio: {sampling}",
        f"- training subjects: {', '.join(map(str, train_ids)) if train_ids else 'n/a'}",
        f"- testing subjects: {', '.join(map(str, test_ids)) if test_ids else 'n/a'}",
        f"- experiment: `{meta.get('exp_dir', '')}`",
        f"- checkpoint: `{meta.get('checkpoint', '')}`",
        "",
        "## Signal Reconstruction",
        "",
        f"- average PSNR: {psnr:.4f}",
        f"- average SSIM: {ssim:.4f}",
        f"- average Relative MSE: {rel:.6e}",
        "",
        "**Interpretation:** Signal reconstruction quality measures how well predicted "
        "DWI intensities match observations (S0-normalized when configured). "
        "Higher PSNR/SSIM and lower RelMSE indicate better signal fit.",
        "",
        "## Microstructure Reconstruction",
        "",
        f"- FA MAE: {_mean(adapt or zero, 'FA_MAE'):.6f}",
        f"- FA Correlation: {fa_cc_adapt:.4f}",
        f"- MD MAE: {_mean(adapt or zero, 'MD_MAE'):.6e}",
        f"- MD Correlation: {md_cc_adapt:.4f}",
        "",
        "**Interpretation:** Whether PINR recovers diffusion tensor information "
        "is judged primarily by FA/MD agreement with WLS reference on brain ∩ WLS_valid. "
        "High Pearson correlation with low MAE suggests microstructure recovery.",
        "",
        "## Population Generalization",
        "",
        block(zero, "Zero-shot (z = 0)"),
        block(adapt, "Latent adaptation"),
        "",
        "### Comparison",
        "",
        f"| Mode | FA_CC | MD_CC | PSNR |",
        f"|------|------:|------:|-----:|",
        f"| Zero-shot | {fa_cc_zero:.4f} | {md_cc_zero:.4f} | {_mean(zero, 'PSNR'):.4f} |",
        f"| Adaptation | {fa_cc_adapt:.4f} | {md_cc_adapt:.4f} | {_mean(adapt, 'PSNR'):.4f} |",
        "",
        "**Interpretation:** Whether population latent improves unseen subject prediction "
        "is shown by adaptation vs zero-shot FA/MD correlation and signal metrics. "
        "Improved signal with degraded FA still indicates an objective mismatch.",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
        f"- Decision threshold: FA correlation > 0.85 → success (observed FA_CC={fa_cc_adapt:.4f}).",
        "",
    ]
    return "\n".join(lines)


def generate_report(results_dir: Path, meta: dict[str, Any] | None = None) -> Path:
    results_dir = Path(results_dir)
    summary = _read_csv(results_dir / "metrics" / "summary.csv")
    meta_path = results_dir / "report" / "meta.json"
    meta = meta or (load_json(meta_path) if meta_path.is_file() else {})
    text = build_report_text(summary_rows=summary, meta=meta)
    out = results_dir / "report" / "experiment_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate PINR experiment_report.md")
    ap.add_argument(
        "--results-dir",
        default=str(package_root() / "results"),
        help="results/ directory containing metrics/summary.csv",
    )
    args = ap.parse_args()
    out = generate_report(Path(args.results_dir))
    print(f"[report] wrote {out}")


if __name__ == "__main__":
    main()
