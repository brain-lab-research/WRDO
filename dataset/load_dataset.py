"""
Load project JSON datasets from the local repository.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATASET_DIR = Path(__file__).resolve().parent

SPLITS = ("train", "val", "test")
HARMTYPES = ("harmless", "harmful")

SPLIT_DATASET_RELATIVE_TEMPLATE = "splits/{harmtype}_{split}.json"

PROCESSED_DATASET_ALIASES = {
    "maliciousinstruct": "malicious_instruct",
}
PROCESSED_DATASET_NAMES = (
    "advbench",
    "tdc2023",
    "malicious_instruct",
    "harmbench_val",
    "harmbench_test",
    "jailbreakbench",
    "strongreject",
    "alpaca",
)


def _read_json_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _local_dataset_path(relative_path: str | Path) -> Path:
    return DATASET_DIR / Path(relative_path)


def _load_dataset_json(relative_path: str | Path) -> Any:
    path = _local_dataset_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Local dataset file not found: {path}")
    return _read_json_file(path)


def _normalize_processed_dataset_name(dataset_name: str) -> str:
    normalized = (dataset_name or "").strip()
    normalized = PROCESSED_DATASET_ALIASES.get(normalized, normalized)
    if normalized not in PROCESSED_DATASET_NAMES:
        valid_datasets = ", ".join(PROCESSED_DATASET_NAMES)
        raise AssertionError(f"Valid datasets: {valid_datasets}")
    return normalized


def load_dataset_split(
    harmtype: str,
    split: str,
    instructions_only: bool = False,
):
    assert harmtype in HARMTYPES
    assert split in SPLITS

    dataset = _load_dataset_json(
        SPLIT_DATASET_RELATIVE_TEMPLATE.format(harmtype=harmtype, split=split)
    )

    if instructions_only:
        dataset = [d["instruction"] for d in dataset]

    return dataset


def load_dataset(dataset_name, instructions_only: bool = False):
    dataset_name = _normalize_processed_dataset_name(dataset_name)
    dataset = _load_dataset_json(f"processed/{dataset_name}.json")

    if instructions_only:
        dataset = [d["instruction"] for d in dataset]

    return dataset
