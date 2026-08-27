"""
Functions for computing, saving, and loading refusal directions.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from heretic.model import Model
from heretic.utils import empty_cache


def compute_refusal_direction(model: Model, harmful_questions: List[str], good_prompts: List[str]) -> torch.Tensor:
    """
    Compute refusal direction for a list of harmful questions.
    
    Args:
        model: Model to use
        harmful_questions: List of prompts used to induce refusals
        good_prompts: List of good prompts
    
    Returns:
        refusal_directions: Tensor of refusal directions per layer
    """
    print(f"  Getting residuals for {len(good_prompts)} good prompts...")
    good_residuals = model.get_residuals_batched(good_prompts)
    
    print(f"  Getting residuals for {len(harmful_questions)} harmful questions...")
    bad_residuals = model.get_residuals_batched(harmful_questions)
    
    good_mean = good_residuals.mean(dim=0)
    bad_mean = bad_residuals.mean(dim=0)
    
    refusal_directions = F.normalize(
        bad_mean - good_mean,
        p=2,
        dim=1
    )
    
    return refusal_directions


def save_refusal_directions(
    extracted_directions: List[torch.Tensor],
    bad_tags: List[str],
    model_name: str,
    save_path: Path
) -> None:
    """
    Save refusal directions to file.
    
    Args:
        extracted_directions: List of refusal direction tensors
        bad_tags: Tags for which directions were computed
        model_name: Model name
        save_path: Save path
    """
    save_data = {
        "extracted_directions": extracted_directions,
        "bad_tags": bad_tags,
        "model_name": model_name,
        "n_directions": len(extracted_directions),
        "timestamp": datetime.now().isoformat()
    }
    torch.save(save_data, save_path)
    print(f"  Directions saved to: {save_path}")


def load_refusal_directions(
    load_path: Path,
    expected_tags: List[str] = None
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Load refusal directions from file.
    
    Args:
        load_path: Path to saved directions
        expected_tags: Expected tag list (for validation)
    
    Returns:
        extracted_directions: List of tensors
        bad_tags: Tags from file
    """
    save_data = torch.load(load_path, map_location='cpu')
    extracted_directions = save_data["extracted_directions"]
    bad_tags = save_data["bad_tags"]
    
    if expected_tags is not None:
        if bad_tags != expected_tags:
            print(f"  Warning: file tags do not match expected!")
            print(f"    Expected: {len(expected_tags)} tags")
            print(f"    In file: {len(bad_tags)} tags")
    
    print(f"  Loaded {len(extracted_directions)} directions from file")
    print(f"  Model: {save_data.get('model_name', 'unknown')}")
    print(f"  Saved at: {save_data.get('timestamp', 'unknown')}")
    
    return extracted_directions, bad_tags
