"""
Utilities for loading and processing data.
"""

from typing import Dict, List

from dataset.load_dataset import PROCESSED_DATASET_NAMES, load_dataset


def load_combined_dataset() -> List[Dict]:
    """Load every processed local dataset into one instruction pool."""
    all_data: List[Dict] = []

    for dataset_name in sorted(PROCESSED_DATASET_NAMES):
        for item in load_dataset(dataset_name):
            if not isinstance(item, dict):
                continue
            instruction = item.get("instruction")
            if not isinstance(instruction, str) or not instruction.strip():
                continue

            combined_item = dict(item)
            combined_item["source"] = dataset_name
            all_data.append(combined_item)

    return all_data


def extract_response_after_think(response: str) -> str:
    """
    Extract the part of the response after the </think> tag.

    Args:
        response: Full model response

    Returns:
        Part of response after </think> with leading whitespace stripped
    """
    parts = response.split('</think>')
    if len(parts) > 1:
        return parts[1].strip()
    return response.strip()
