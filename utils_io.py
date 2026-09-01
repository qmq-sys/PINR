"""Experiment paths, config load, checkpoint I/O."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parent
INR_ROOT = ROOT.parent / "INR"
if str(INR_ROOT) not in sys.path:
    sys.path.insert(0, str(INR_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def package_root() -> Path:
    return ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def save_yaml(path: str | Path, obj: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_experiment_dir(base: str | Path | None = None, tag: str = "population_dti") -> Path:
    root = Path(base) if base else package_root() / "experiments" / tag
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = root / stamp
    for sub in ("config", "checkpoints", "metrics", "maps", "logs", "plots"):
        (exp / sub).mkdir(parents=True, exist_ok=True)
    return exp


def resolve_device(name: str = "auto") -> torch.device:
    if name in ("", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_population_checkpoint(
    out_dir: Path,
    *,
    model,
    optimizer: torch.optim.Optimizer | None,
    subject_mapping: dict[str, int],
    config: dict[str, Any],
    history_rows: list[dict[str, Any]] | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.theta_state_dict(), out_dir / "theta.pt")
    latents = model.latents.state_dict_by_id() if model.latents is not None else {}
    torch.save(latents, out_dir / "latents.pt")
    save_json(out_dir / "subject_mapping.json", subject_mapping)
    save_yaml(out_dir / "config.yaml", config)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), out_dir / "optimizer.pt")
    if history_rows is not None:
        _write_csv(out_dir / "training_history.csv", history_rows)


def load_theta(path: Path, model, map_location=None) -> None:
    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_theta_state_dict(state)


def load_latents(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return {str(k): v for k, v in obj.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    keys = list(row.keys())
    if not write_header:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing = next(reader, None)
            if existing:
                keys = existing
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
