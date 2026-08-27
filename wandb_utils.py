"""
Helpers for structuring Weights & Biases metadata for easy run filtering.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _normalize_key_part(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def flatten_wandb_config(
    data: Mapping[str, Any],
    *,
    prefix: str | None = None,
) -> dict[str, Any]:
    """Flatten nested config dicts into scalar top-level keys for W&B filters."""
    flattened: dict[str, Any] = {}
    normalized_prefix = _normalize_key_part(prefix) if prefix else ""

    for raw_key, value in data.items():
        key = _normalize_key_part(raw_key)
        full_key = f"{normalized_prefix}_{key}" if normalized_prefix else key

        if isinstance(value, Mapping):
            flattened.update(flatten_wandb_config(value, prefix=full_key))
            continue

        if isinstance(value, Path):
            flattened[full_key] = value.as_posix()
            continue

        if _is_scalar(value):
            flattened[full_key] = value
            continue

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            flattened[f"{full_key}_count"] = len(value)
            continue

    return flattened


def short_model_name(model_name: str) -> str:
    stripped = (model_name or "").strip()
    if not stripped:
        return "unknown"
    return stripped.rsplit("/", 1)[-1]


def resolve_run_mode(results_root: Path | None) -> str:
    if results_root is None:
        return "nonblocking"
    root_parts = {part.lower() for part in results_root.parts}
    return "blocking" if "blocking" in root_parts else "nonblocking"


def _make_wandb_tag(label: str, value: Any) -> str | None:
    if value is None:
        return None

    normalized_label = _normalize_key_part(label).replace("_", "-")
    normalized_value = re.sub(r"\s+", "-", str(value).strip())
    normalized_value = re.sub(r"[^A-Za-z0-9._:/=-]+", "-", normalized_value).strip("-")
    if not normalized_value:
        return None
    return f"{normalized_label}:{normalized_value[:120]}"


def build_wandb_tags(pairs: Iterable[tuple[str, Any]]) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()

    for label, value in pairs:
        tag = _make_wandb_tag(label, value)
        if tag is None or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)

    return tags
