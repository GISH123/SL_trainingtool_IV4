from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import configuration as cfg

_NUMERIC_PREFIX = re.compile(r"^\s*(\d+)")


def canonical_class_name(name: str) -> str:
    """Return the canonical class folder name used by every split/inference path."""
    raw = str(name).strip()
    lowered = raw.lower().replace("-", "_").replace(" ", "_")
    aliases = {x.lower() for x in cfg.NON_CARD_ALIASES}
    if lowered in aliases:
        return cfg.NON_CARD_OUTPUT_NAME

    match = _NUMERIC_PREFIX.match(lowered)
    if match:
        value = int(match.group(1))
        if value in (0, 54) and (value == 54 or "card" in lowered or lowered == "0"):
            return cfg.NON_CARD_OUTPUT_NAME
        if 1 <= value <= 52:
            return str(value)
    return raw


def class_sort_key(name: str):
    canonical = canonical_class_name(name)
    if canonical == cfg.NON_CARD_OUTPUT_NAME:
        return (0, 0, canonical)
    if canonical.isdigit():
        return (1, int(canonical), canonical)
    return (2, 0, canonical.lower())


def sorted_class_names(names: Iterable[str]) -> list[str]:
    return sorted({canonical_class_name(n) for n in names}, key=class_sort_key)


def save_class_names(names: Iterable[str], path: Path = cfg.CLASS_NAMES_PATH) -> list[str]:
    ordered = sorted_class_names(names)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    return ordered


def load_class_names(path: Path = cfg.CLASS_NAMES_PATH) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {path}. Run Step1_split_dataset.py first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid class mapping: {path}")
    return [str(x) for x in data]


def class_index_map(class_names: Iterable[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}
