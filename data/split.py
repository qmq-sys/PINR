"""Subject-level split helpers."""
from __future__ import annotations

from typing import Any, Iterable


def validate_subject_level_split(
    train: Iterable[str],
    val: Iterable[str],
    test: Iterable[str],
) -> dict[str, list[str]]:
    tr = [str(s).strip() for s in train]
    va = [str(s).strip() for s in val]
    te = [str(s).strip() for s in test]
    for name, xs in (("train", tr), ("val", va), ("test", te)):
        if len(xs) != len(set(xs)):
            raise ValueError(f"duplicate ids in {name}: {xs}")
    sets = {"train": set(tr), "val": set(va), "test": set(te)}
    overlap_tv = sets["train"] & sets["val"]
    overlap_tt = sets["train"] & sets["test"]
    overlap_vt = sets["val"] & sets["test"]
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"subject-level split leak: train∩val={overlap_tv}, "
            f"train∩test={overlap_tt}, val∩test={overlap_vt}"
        )
    if not tr:
        raise ValueError("train subjects must be non-empty")
    if not te:
        raise ValueError("test subjects must be non-empty")
    return {"train": tr, "val": va, "test": te}


def split_from_config(cfg: dict[str, Any]) -> dict[str, list[str]]:
    split = cfg.get("split") or {}
    return validate_subject_level_split(
        split.get("train", []),
        split.get("val", []),
        split.get("test", []),
    )
