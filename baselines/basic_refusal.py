#!/usr/bin/env python3
"""
Single-direction refusal baseline adapted from refusal_direction for Heretic.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ABLITERATION_PARAMS,
    BASIC_REFUSAL_CONFIG,
    MODEL_BATCH_SIZE,
    MODEL_NAME,
    get_method_results_dir,
)
from data_utils import extract_response_after_think
from dataset.load_dataset import load_dataset_split
from evaluate.metrics import (
    evaluate_responses,
    is_unsafe_rate_evaluation_metric,
    resolve_evaluation_metric,
)
from heretic.config import Settings
from heretic.model import Model
from model_utils import apply_abliteration_with_hyperparams
from wandb_utils import build_wandb_tags, flatten_wandb_config, resolve_run_mode, short_model_name

torch.set_grad_enabled(False)

BASELINE_RESULTS_DIR = get_method_results_dir("basic_refusal")
ANSWERS_DIR = BASELINE_RESULTS_DIR / "answers"
SELECTION_DIR = BASELINE_RESULTS_DIR / "selection"
ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
SELECTION_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTION_SAMPLE_OFFSETS = {
    ("harmful", "train"): 0,
    ("harmless", "train"): 1,
    ("harmful", "val"): 2,
    ("harmless", "val"): 3,
    ("harmful", "test"): 4,
    ("harmless", "test"): 5,
}


def _batched(items: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _format_run_name_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _sanitize_run_name_part(value: Any) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def _shorten_run_name(name: str, max_length: int = 200) -> str:
    if len(name) <= max_length:
        return name

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    prefix_length = max_length - len(digest) - 1
    shortened_prefix = name[:prefix_length].rstrip("_-")
    if not shortened_prefix:
        return digest
    return f"{shortened_prefix}_{digest}"


def _build_basic_refusal_run_name(
    *,
    model_name: str,
    evaluation_backend: str,
    abliteration_params: dict[str, Any],
    selection_config: dict[str, Any],
    results_root: Path | None = None,
    timestamp_str: str | None = None,
) -> str:
    run_mode = resolve_run_mode(results_root)

    parts = [
        run_mode,
        f"model_{_sanitize_run_name_part(model_name)}",
        "basic_refusal",
        f"eval_{_sanitize_run_name_part(evaluation_backend)}",
        f"abl_max{_format_run_name_value(abliteration_params['max_weight'])}",
        f"min{_format_run_name_value(abliteration_params['min_weight'])}",
        f"pos{_format_run_name_value(abliteration_params['max_weight_position'])}",
        f"dist{_format_run_name_value(abliteration_params['min_weight_distance'])}",
        f"seed{_format_run_name_value(selection_config['sample_seed'])}",
        f"kl{_format_run_name_value(selection_config['kl_threshold'])}",
        f"prune{_format_run_name_value(selection_config['prune_layer_percentage'])}",
        timestamp_str or datetime.now().strftime("%Y%m%d_%H%M%S"),
    ]
    return "_".join(parts)


def _parse_env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _define_model_state_eval_metrics() -> None:
    """Register shared wandb series for clean/edited harmfulness comparisons."""
    if not WANDB_AVAILABLE or not hasattr(wandb, "define_metric"):
        return

    wandb.define_metric("model_state_eval/point_index")
    wandb.define_metric(
        "model_state_eval/harmfulness_on_full_dataset",
        step_metric="model_state_eval/point_index",
    )


def _wandb_log(payload: dict[str, Any]) -> None:
    """Log wandb metrics while skipping unset values."""
    if not WANDB_AVAILABLE:
        return

    filtered_payload = {key: value for key, value in payload.items() if value is not None}
    if filtered_payload:
        wandb.log(filtered_payload)


def _log_model_state_eval(
    prefixes: Sequence[str],
    point_index: int,
    point_name: str,
    harmfulness_on_full_dataset: Optional[float],
) -> None:
    """Log comparable clean/edited harmfulness series to wandb."""
    if not WANDB_AVAILABLE:
        return

    payload: dict[str, Any] = {
        "model_state_eval/point_index": point_index,
        "model_state_eval/point_name": point_name,
    }

    if harmfulness_on_full_dataset is not None:
        for prefix in prefixes:
            payload[f"{prefix}/harmfulness_on_full_dataset"] = harmfulness_on_full_dataset
            if prefix in {"clean_model", "edited_model", "best_value_model"}:
                payload[f"{prefix}/harmfulness"] = harmfulness_on_full_dataset
        payload["model_state_eval/harmfulness_on_full_dataset"] = harmfulness_on_full_dataset

    _wandb_log(payload)


def _build_settings() -> Settings:
    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]] if sys.argv else ["script"]
        return Settings(
            model=MODEL_NAME,
            batch_size=MODEL_BATCH_SIZE,
            max_response_length=int(os.getenv("MAX_RESPONSE_LENGTH", "2048")),
            system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful assistant."),
        )
    finally:
        sys.argv = original_argv


def load_classifier_categories() -> list[dict[str, Any]]:
    return []


def sample_dataset_split(
    harmtype: str,
    split: str,
    sample_size: int | None,
    sample_seed: int,
) -> list[str]:
    dataset = list(load_dataset_split(harmtype=harmtype, split=split, instructions_only=True))
    if not dataset:
        return []

    if sample_size is None:
        return dataset

    if sample_size <= 0:
        return []

    sample_count = min(int(sample_size), len(dataset))
    offset = SELECTION_SAMPLE_OFFSETS[(harmtype, split)]
    rng = random.Random(int(sample_seed) + offset)
    return rng.sample(dataset, sample_count)


def load_selection_datasets(selection_config: dict[str, Any]) -> dict[str, list[str]]:
    sample_seed = int(selection_config["sample_seed"])
    n_train = selection_config["n_train"]
    n_val = selection_config["n_val"]
    n_test = selection_config["n_test"]
    return {
        "harmful_train": sample_dataset_split("harmful", "train", n_train, sample_seed),
        "harmless_train": sample_dataset_split("harmless", "train", n_train, sample_seed),
        "harmful_val": sample_dataset_split("harmful", "val", n_val, sample_seed),
        "harmless_val": sample_dataset_split("harmless", "val", n_val, sample_seed),
        "harmful_test": sample_dataset_split("harmful", "test", n_test, sample_seed),
        "harmless_test": sample_dataset_split("harmless", "test", n_test, sample_seed),
    }


def build_chat_prompts(model: Model, prompts: Sequence[str]) -> list[str]:
    chats = [model.get_chat(prompt) for prompt in prompts]
    return model.format_chats(
        chats,
        add_generation_prompt=True,
    )


def compute_eoi_suffix_token_ids(model: Model) -> list[int]:
    probe_prompts = ["__EoiProbeA__", "__DistinctEoiProbeB__"]
    tokenized_prompts = []

    for chat_prompt in build_chat_prompts(model, probe_prompts):
        encoded = model.tokenizer(
            chat_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenized_prompts.append(encoded["input_ids"][0].tolist())

    common_suffix: list[int] = []
    for lhs, rhs in zip(reversed(tokenized_prompts[0]), reversed(tokenized_prompts[1])):
        if lhs != rhs:
            break
        common_suffix.append(int(lhs))

    if not common_suffix:
        return [int(tokenized_prompts[0][-1])]

    common_suffix.reverse()
    return common_suffix


def build_refusal_token_ids(model: Model) -> list[int]:
    token_ids: set[int] = set()
    markers = getattr(model.settings, "refusal_markers", [])
    for marker in markers:
        normalized = str(marker).strip()
        if not normalized:
            continue
        for variant in (normalized, f" {normalized}"):
            encoded = model.tokenizer(
                variant,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0].tolist()
            if encoded:
                token_ids.add(int(encoded[0]))

    if not token_ids:
        raise ValueError("Could not derive refusal token ids from model refusal markers.")

    return sorted(token_ids)


def compute_refusal_scores_from_logprobs(
    logprobs: torch.Tensor,
    refusal_token_ids: Sequence[int],
    epsilon: float = 1e-8,
) -> torch.Tensor:
    scores = logprobs.to(torch.float64)
    refusal_probs = scores[:, list(refusal_token_ids)].exp().sum(dim=-1)
    non_refusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(non_refusal_probs + epsilon)


def compute_mean_kl(clean_logprobs: torch.Tensor, modified_logprobs: torch.Tensor) -> torch.Tensor:
    clean = clean_logprobs.to(torch.float64)
    modified = modified_logprobs.to(torch.float64)
    clean_probs = clean.exp()
    return torch.sum(clean_probs * (clean - modified), dim=-1).mean()


def filter_instructions_by_scores(
    instructions: Sequence[str],
    scores: Sequence[float],
    *,
    keep_positive: bool,
) -> list[str]:
    if keep_positive:
        return [instruction for instruction, score in zip(instructions, scores) if float(score) > 0.0]
    return [instruction for instruction, score in zip(instructions, scores) if float(score) < 0.0]


def filter_instructions_by_refusal_sign(
    model: Model,
    instructions: Sequence[str],
    refusal_token_ids: Sequence[int],
    *,
    keep_positive: bool,
    label: str,
) -> list[str]:
    if not instructions:
        return []

    logprobs = model.get_logprobs_batched(list(instructions))
    scores = compute_refusal_scores_from_logprobs(logprobs, refusal_token_ids).tolist()
    filtered = filter_instructions_by_scores(
        instructions,
        scores,
        keep_positive=keep_positive,
    )
    if filtered:
        return filtered

    print(f"Warning: filtering removed all prompts for {label}; using the unfiltered sample.")
    return list(instructions)


def get_mean_activations(
    model: Model,
    instructions: Sequence[str],
    positions: Sequence[int],
) -> torch.Tensor:
    if not instructions:
        raise ValueError("Cannot compute mean activations for an empty instruction set.")

    layers = list(model.get_layers())
    n_layers = len(layers)
    hidden_size = int(model.model.config.hidden_size)
    cache = torch.zeros((len(positions), n_layers, hidden_size), dtype=torch.float64)
    handles = []

    def build_hook(layer_index: int):
        def hook_fn(_module, args):
            activation = args[0].detach()
            selected = activation[:, list(positions), :].to(dtype=torch.float64, device="cpu")
            cache[:, layer_index, :] += selected.sum(dim=0)

        return hook_fn

    try:
        for layer_index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(build_hook(layer_index)))

        with torch.no_grad():
            for batch in _batched(list(instructions), model.settings.batch_size):
                chat_prompts = build_chat_prompts(model, batch)
                inputs = model.tokenizer(
                    chat_prompts,
                    return_tensors="pt",
                    padding=True,
                    return_token_type_ids=False,
                ).to(model.model.device)
                model.model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return cache / float(len(instructions))


def get_mean_diff(
    model: Model,
    harmful_instructions: Sequence[str],
    harmless_instructions: Sequence[str],
    positions: Sequence[int],
) -> torch.Tensor:
    harmful_mean = get_mean_activations(model, harmful_instructions, positions)
    harmless_mean = get_mean_activations(model, harmless_instructions, positions)
    return (harmful_mean - harmless_mean).to(torch.float32)


def expand_direction_to_refusal_tensor(direction: torch.Tensor, n_layers: int) -> torch.Tensor:
    vector = F.normalize(direction.to(torch.float32), p=2, dim=0)
    expanded = vector.repeat(n_layers + 1, 1)
    expanded[0] = 0.0
    return expanded


def normalize_candidate_direction(candidate_direction: torch.Tensor) -> torch.Tensor:
    return F.normalize(candidate_direction.to(torch.float32), p=2, dim=0)


def get_selection_abliteration_params(abliteration_params: dict[str, float]) -> dict[str, float]:
    return {
        "max_weight": abs(float(abliteration_params["max_weight"])),
        "max_weight_position": float(abliteration_params["max_weight_position"]),
        "min_weight": abs(float(abliteration_params["min_weight"])),
        "min_weight_distance": float(abliteration_params["min_weight_distance"]),
    }


def invert_abliteration_params(abliteration_params: dict[str, float]) -> dict[str, float]:
    return {
        "max_weight": -float(abliteration_params["max_weight"]),
        "max_weight_position": float(abliteration_params["max_weight_position"]),
        "min_weight": -float(abliteration_params["min_weight"]),
        "min_weight_distance": float(abliteration_params["min_weight_distance"]),
    }


def evaluate_candidate_direction(
    model: Model,
    candidate_direction: torch.Tensor,
    harmful_instructions: Sequence[str],
    harmless_instructions: Sequence[str],
    refusal_token_ids: Sequence[int],
    clean_harmless_logprobs: torch.Tensor,
    abliteration_params: dict[str, float],
) -> dict[str, float]:
    n_layers = len(model.get_layers())
    refusal_directions = expand_direction_to_refusal_tensor(candidate_direction, n_layers)

    model.reload_model()
    apply_abliteration_with_hyperparams(
        model,
        refusal_directions,
        abliteration_params["max_weight"],
        abliteration_params["max_weight_position"],
        abliteration_params["min_weight"],
        abliteration_params["min_weight_distance"],
        n_layers,
    )
    harmful_logprobs = model.get_logprobs_batched(list(harmful_instructions))
    harmful_refusal_mean = float(
        compute_refusal_scores_from_logprobs(harmful_logprobs, refusal_token_ids).mean().item()
    )
    modified_harmless_logprobs = model.get_logprobs_batched(list(harmless_instructions))
    kl_div_score = float(compute_mean_kl(clean_harmless_logprobs, modified_harmless_logprobs).item())

    model.reload_model()
    steering_params = invert_abliteration_params(abliteration_params)
    apply_abliteration_with_hyperparams(
        model,
        refusal_directions,
        steering_params["max_weight"],
        steering_params["max_weight_position"],
        steering_params["min_weight"],
        steering_params["min_weight_distance"],
        n_layers,
    )
    steering_logprobs = model.get_logprobs_batched(list(harmless_instructions))
    steering_refusal_mean = float(
        compute_refusal_scores_from_logprobs(steering_logprobs, refusal_token_ids).mean().item()
    )

    return {
        "refusal_score": harmful_refusal_mean,
        "steering_score": steering_refusal_mean,
        "kl_div_score": kl_div_score,
    }


def should_filter_direction(
    *,
    refusal_score: float,
    steering_score: float,
    kl_div_score: float,
    layer: int,
    n_layers: int,
    kl_threshold: float,
    induce_refusal_threshold: float,
    prune_layer_percentage: float,
) -> bool:
    if np.isnan(refusal_score) or np.isnan(steering_score) or np.isnan(kl_div_score):
        return True
    if prune_layer_percentage is not None and layer >= int(n_layers * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
        return True
    return False


def save_selection_artifacts(
    selection_run_dir: Path,
    *,
    mean_diffs: torch.Tensor,
    selected_direction: torch.Tensor,
    metadata: dict[str, Any],
    all_scores: list[dict[str, Any]],
    filtered_scores: list[dict[str, Any]],
) -> dict[str, str]:
    selection_run_dir.mkdir(parents=True, exist_ok=True)

    mean_diffs_path = selection_run_dir / "mean_diffs.pt"
    selected_direction_path = selection_run_dir / "selected_direction.pt"
    direction_metadata_path = selection_run_dir / "direction_metadata.json"
    direction_evaluations_path = selection_run_dir / "direction_evaluations.json"
    filtered_evaluations_path = selection_run_dir / "direction_evaluations_filtered.json"

    torch.save(mean_diffs, mean_diffs_path)
    torch.save(selected_direction, selected_direction_path)
    direction_metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    direction_evaluations_path.write_text(
        json.dumps(all_scores, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    filtered_evaluations_path.write_text(
        json.dumps(filtered_scores, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "mean_diffs_path": str(mean_diffs_path),
        "selected_direction_path": str(selected_direction_path),
        "direction_metadata_path": str(direction_metadata_path),
        "direction_evaluations_path": str(direction_evaluations_path),
        "direction_evaluations_filtered_path": str(filtered_evaluations_path),
    }


def select_direction(
    model: Model,
    harmful_selection_prompts: Sequence[str],
    harmless_selection_prompts: Sequence[str],
    mean_diffs: torch.Tensor,
    *,
    token_positions: Sequence[int],
    token_labels: Sequence[str],
    refusal_token_ids: Sequence[int],
    selection_config: dict[str, Any],
    abliteration_params: dict[str, float],
    selection_run_dir: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    if not harmful_selection_prompts or not harmless_selection_prompts:
        raise ValueError("Direction selection requires both harmful and harmless validation prompts.")

    n_positions, n_layers, _hidden_size = mean_diffs.shape
    if n_positions != len(token_positions):
        raise ValueError("token_positions length does not match mean_diffs positions dimension.")

    selection_abliteration = get_selection_abliteration_params(abliteration_params)
    clean_harmful_logprobs = model.get_logprobs_batched(list(harmful_selection_prompts))
    clean_harmless_logprobs = model.get_logprobs_batched(list(harmless_selection_prompts))
    baseline_harmful_refusal = float(
        compute_refusal_scores_from_logprobs(clean_harmful_logprobs, refusal_token_ids).mean().item()
    )
    baseline_harmless_refusal = float(
        compute_refusal_scores_from_logprobs(clean_harmless_logprobs, refusal_token_ids).mean().item()
    )

    all_scores: list[dict[str, Any]] = []
    filtered_scores: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    for pos_index, source_pos in enumerate(token_positions):
        for source_layer in range(n_layers):
            candidate_direction = normalize_candidate_direction(mean_diffs[pos_index, source_layer])
            candidate_metrics = evaluate_candidate_direction(
                model,
                candidate_direction,
                harmful_selection_prompts,
                harmless_selection_prompts,
                refusal_token_ids,
                clean_harmless_logprobs,
                selection_abliteration,
            )
            record = {
                "position_index": pos_index,
                "position": int(source_pos),
                "position_label": token_labels[pos_index],
                "layer": int(source_layer),
                "refusal_score": candidate_metrics["refusal_score"],
                "steering_score": candidate_metrics["steering_score"],
                "kl_div_score": candidate_metrics["kl_div_score"],
            }
            all_scores.append(record)

            filtered_out = should_filter_direction(
                refusal_score=record["refusal_score"],
                steering_score=record["steering_score"],
                kl_div_score=record["kl_div_score"],
                layer=source_layer,
                n_layers=n_layers,
                kl_threshold=float(selection_config["kl_threshold"]),
                induce_refusal_threshold=float(selection_config["induce_refusal_threshold"]),
                prune_layer_percentage=float(selection_config["prune_layer_percentage"]),
            )
            if filtered_out:
                continue

            filtered_scores.append(record)
            if best_record is None or record["refusal_score"] < best_record["refusal_score"]:
                best_record = record

    filtered_scores = sorted(filtered_scores, key=lambda item: item["refusal_score"])
    if best_record is None:
        print("Warning: all candidate directions were filtered out; falling back to the best unfiltered candidate.")
        best_record = min(all_scores, key=lambda item: item["refusal_score"])
        filtered_scores = [best_record]

    selected_direction = normalize_candidate_direction(
        mean_diffs[best_record["position_index"], best_record["layer"]]
    )
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "baseline_harmful_refusal_score": baseline_harmful_refusal,
        "baseline_harmless_refusal_score": baseline_harmless_refusal,
        "selection_config": selection_config,
        "selection_abliteration_params": selection_abliteration,
        "selection_split": "val",
        "selected_position_index": int(best_record["position_index"]),
        "selected_position": int(best_record["position"]),
        "selected_position_label": str(best_record["position_label"]),
        "selected_layer": int(best_record["layer"]),
        "selected_refusal_score": float(best_record["refusal_score"]),
        "selected_steering_score": float(best_record["steering_score"]),
        "selected_kl_div_score": float(best_record["kl_div_score"]),
        "token_positions": [int(position) for position in token_positions],
        "token_labels": [str(label) for label in token_labels],
        "refusal_token_ids": [int(token_id) for token_id in refusal_token_ids],
        "n_candidates": len(all_scores),
        "n_filtered_candidates": len(filtered_scores),
    }
    artifact_paths = save_selection_artifacts(
        selection_run_dir,
        mean_diffs=mean_diffs,
        selected_direction=selected_direction,
        metadata=metadata,
        all_scores=all_scores,
        filtered_scores=filtered_scores,
    )
    return metadata, artifact_paths


def apply_selected_direction(
    model: Model,
    selected_direction: torch.Tensor,
    abliteration_params: dict[str, float],
) -> None:
    n_layers = len(model.get_layers())
    refusal_directions = expand_direction_to_refusal_tensor(selected_direction, n_layers)
    apply_abliteration_with_hyperparams(
        model,
        refusal_directions,
        float(abliteration_params["max_weight"]),
        float(abliteration_params["max_weight_position"]),
        float(abliteration_params["min_weight"]),
        float(abliteration_params["min_weight_distance"]),
        n_layers,
    )


def save_answers(
    answers_path: Path,
    answers_data: dict[str, Any],
) -> None:
    answers_path.write_text(
        json.dumps(answers_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_best_model_checkpoint(
    *,
    model: Model,
    output_dir: Path,
    metadata: dict[str, Any],
    max_shard_size: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving final edited model to: {output_dir}")
    model.model.save_pretrained(output_dir, max_shard_size=max_shard_size)
    model.tokenizer.save_pretrained(output_dir)
    metadata_file = output_dir / "basic_refusal_best_model_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Best edited model metadata saved to: {metadata_file}")


def main() -> None:
    print("=" * 80)
    print("STARTING EXPERIMENT: BASIC REFUSAL (SINGLE-DIRECTION SELECTION)")
    print("=" * 80)
    started_at = datetime.now()
    run_stamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_name = _build_basic_refusal_run_name(
        model_name=MODEL_NAME,
        evaluation_backend=os.getenv("EVALUATION_BACKEND", "wildguard"),
        abliteration_params=ABLITERATION_PARAMS,
        selection_config=BASIC_REFUSAL_CONFIG,
        results_root=BASELINE_RESULTS_DIR,
        timestamp_str=run_stamp,
    )
    run_artifact_name = _shorten_run_name(run_name)
    print(f"Start time: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME}")
    print(f"Run artifact name: {run_artifact_name}")
    print(f"Selection config: {BASIC_REFUSAL_CONFIG}")
    print(f"Abliteration params: {ABLITERATION_PARAMS}")
    print()

    print("Loading selection datasets...")
    selection_datasets = load_selection_datasets(BASIC_REFUSAL_CONFIG)
    for name, dataset in selection_datasets.items():
        print(f"  {name}: {len(dataset)}")

    settings = _build_settings()
    print("\nLoading model...")
    model = Model(settings)
    print("Model loaded successfully!")

    classifier_categories = load_classifier_categories()
    evaluation_metric = resolve_evaluation_metric()
    uses_unsafe_rate_metric = is_unsafe_rate_evaluation_metric(evaluation_metric)

    refusal_token_ids = build_refusal_token_ids(model)
    suffix_token_ids = compute_eoi_suffix_token_ids(model)
    suffix_positions = list(range(-len(suffix_token_ids), 0))
    suffix_token_labels = model.tokenizer.batch_decode(
        [[token_id] for token_id in suffix_token_ids]
    )
    print(f"Using {len(suffix_token_ids)} end-of-instruction suffix tokens for direction extraction")

    if BASIC_REFUSAL_CONFIG["filter_train"]:
        selection_datasets["harmful_train"] = filter_instructions_by_refusal_sign(
            model,
            selection_datasets["harmful_train"],
            refusal_token_ids,
            keep_positive=True,
            label="harmful_train",
        )
        selection_datasets["harmless_train"] = filter_instructions_by_refusal_sign(
            model,
            selection_datasets["harmless_train"],
            refusal_token_ids,
            keep_positive=False,
            label="harmless_train",
        )
    if BASIC_REFUSAL_CONFIG["filter_val"]:
        selection_datasets["harmful_val"] = filter_instructions_by_refusal_sign(
            model,
            selection_datasets["harmful_val"],
            refusal_token_ids,
            keep_positive=True,
            label="harmful_val",
        )
        selection_datasets["harmless_val"] = filter_instructions_by_refusal_sign(
            model,
            selection_datasets["harmless_val"],
            refusal_token_ids,
            keep_positive=False,
            label="harmless_val",
        )

    if not selection_datasets["harmful_train"] or not selection_datasets["harmless_train"]:
        raise ValueError("basic_refusal requires non-empty harmful/harmless training samples.")
    if not selection_datasets["harmful_val"] or not selection_datasets["harmless_val"]:
        raise ValueError("basic_refusal requires non-empty harmful/harmless validation samples.")
    if not selection_datasets["harmful_test"]:
        raise ValueError("basic_refusal requires non-empty harmful test samples.")

    dataset_sizes = {name: len(dataset) for name, dataset in selection_datasets.items()}
    harmful_test_questions = list(selection_datasets["harmful_test"])

    run_mode = resolve_run_mode(BASELINE_RESULTS_DIR)
    if WANDB_AVAILABLE:
        wandb_config = {
            "method": "basic_refusal",
            "model": MODEL_NAME,
            "model_name": MODEL_NAME,
            "model_short_name": short_model_name(MODEL_NAME),
            "run_mode": run_mode,
            "evaluation_backend": os.getenv("EVALUATION_BACKEND", "wildguard"),
            "evaluation_metric": evaluation_metric,
            "selection_config": dict(BASIC_REFUSAL_CONFIG),
            "abliteration_params": dict(ABLITERATION_PARAMS),
            "dataset_sizes": dataset_sizes,
            "harmful_test_question_count": len(harmful_test_questions),
            "suffix_token_count": len(suffix_token_ids),
            "run_artifact_name": run_artifact_name,
        }
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "anonymized"),
            name=run_name,
            job_type="basic_refusal",
            group=f"basic_refusal/{short_model_name(MODEL_NAME)}",
            tags=build_wandb_tags(
                [
                    ("method", "basic_refusal"),
                    ("model", short_model_name(MODEL_NAME)),
                    ("eval_backend", os.getenv("EVALUATION_BACKEND", "wildguard")),
                    ("eval_metric", evaluation_metric),
                    ("run_mode", run_mode),
                ]
            ),
            config={
                **wandb_config,
                **flatten_wandb_config(BASIC_REFUSAL_CONFIG, prefix="selection"),
                **flatten_wandb_config(ABLITERATION_PARAMS, prefix="abl"),
                **flatten_wandb_config(dataset_sizes, prefix="dataset_size"),
            },
        )
        _define_model_state_eval_metrics()

    print("\nGenerating clean responses for harmful test prompts...")
    original_responses = [
        extract_response_after_think(response)
        for response in model.get_responses_batched(harmful_test_questions)
    ]

    print("\nComputing candidate refusal directions...")
    mean_diffs = get_mean_diff(
        model,
        selection_datasets["harmful_train"],
        selection_datasets["harmless_train"],
        suffix_positions,
    )
    selection_run_dir = SELECTION_DIR / run_artifact_name
    selection_metadata, selection_artifacts = select_direction(
        model,
        selection_datasets["harmful_val"],
        selection_datasets["harmless_val"],
        mean_diffs,
        token_positions=suffix_positions,
        token_labels=suffix_token_labels,
        refusal_token_ids=refusal_token_ids,
        selection_config=BASIC_REFUSAL_CONFIG,
        abliteration_params=ABLITERATION_PARAMS,
        selection_run_dir=selection_run_dir,
    )
    selected_direction = torch.load(selection_artifacts["selected_direction_path"], map_location="cpu")
    print(
        f"Selected direction: position={selection_metadata['selected_position']} "
        f"layer={selection_metadata['selected_layer']} "
        f"refusal={selection_metadata['selected_refusal_score']:.4f} "
        f"steering={selection_metadata['selected_steering_score']:.4f} "
        f"kl={selection_metadata['selected_kl_div_score']:.4f}"
    )
    _wandb_log(
        {
            "selection/baseline_harmful_refusal_score": selection_metadata["baseline_harmful_refusal_score"],
            "selection/baseline_harmless_refusal_score": selection_metadata["baseline_harmless_refusal_score"],
            "selection/selected_position_index": selection_metadata["selected_position_index"],
            "selection/selected_position": selection_metadata["selected_position"],
            "selection/selected_layer": selection_metadata["selected_layer"],
            "selection/selected_refusal_score": selection_metadata["selected_refusal_score"],
            "selection/selected_steering_score": selection_metadata["selected_steering_score"],
            "selection/selected_kl_div_score": selection_metadata["selected_kl_div_score"],
            "selection/n_candidates": selection_metadata["n_candidates"],
            "selection/n_filtered_candidates": selection_metadata["n_filtered_candidates"],
        }
    )

    print("\nApplying final selected direction...")
    model.reload_model()
    apply_selected_direction(model, selected_direction, ABLITERATION_PARAMS)

    print("Generating edited responses for harmful test prompts...")
    modified_responses = [
        extract_response_after_think(response)
        for response in model.get_responses_batched(harmful_test_questions)
    ]

    original_scores, original_score_data = evaluate_responses(
        harmful_test_questions,
        original_responses,
        classifier_categories,
        description="original harmful test responses",
    )
    modified_scores, modified_score_data = evaluate_responses(
        harmful_test_questions,
        modified_responses,
        classifier_categories,
        description="edited harmful test responses",
    )
    original_mean = float(np.mean(original_scores)) if original_scores else None
    original_median = float(np.median(original_scores)) if original_scores else None
    modified_mean = float(np.mean(modified_scores)) if modified_scores else None
    modified_median = float(np.median(modified_scores)) if modified_scores else None
    score_delta = (
        modified_mean - original_mean
        if original_mean is not None and modified_mean is not None
        else None
    )
    score_reduction = (
        original_mean - modified_mean
        if original_mean is not None and modified_mean is not None
        else None
    )

    _log_model_state_eval(
        prefixes=["clean_model", "original_model"],
        point_index=0,
        point_name="clean",
        harmfulness_on_full_dataset=original_mean,
    )
    _log_model_state_eval(
        prefixes=["edited_model", "best_value_model"],
        point_index=1,
        point_name="edited",
        harmfulness_on_full_dataset=modified_mean,
    )
    _wandb_log(
        {
            "results/original_mean_harmfulness": original_mean,
            "results/original_median_harmfulness": original_median,
            "results/original_count": len(original_scores),
            "results/modified_mean_harmfulness": modified_mean,
            "results/modified_median_harmfulness": modified_median,
            "results/modified_count": len(modified_scores),
            "results/mean_harmfulness_delta": score_delta,
            "results/mean_harmfulness_reduction": score_reduction,
            "results/evaluation_metric": evaluation_metric,
            "results/original_mean_unsafe_rate": original_mean if uses_unsafe_rate_metric else None,
            "results/original_median_unsafe_rate": original_median if uses_unsafe_rate_metric else None,
            "results/modified_mean_unsafe_rate": modified_mean if uses_unsafe_rate_metric else None,
            "results/modified_median_unsafe_rate": modified_median if uses_unsafe_rate_metric else None,
            "results/unsafe_rate_delta": score_delta if uses_unsafe_rate_metric else None,
            "results/unsafe_rate_reduction": score_reduction if uses_unsafe_rate_metric else None,
        }
    )

    experiment_config = {
        "method": "basic_refusal",
        "category": "basic_refusal",
        "evaluation_metric": evaluation_metric,
        "param_key": f"selected_pos_{selection_metadata['selected_position']}_layer_{selection_metadata['selected_layer']}",
        "run_name": run_name,
        "run_artifact_name": run_artifact_name,
        "hyperparameters": dict(ABLITERATION_PARAMS),
        "abliteration_params": dict(ABLITERATION_PARAMS),
        "selection_config": dict(BASIC_REFUSAL_CONFIG),
        "selected_direction_path": selection_artifacts["selected_direction_path"],
        "selection_artifacts": selection_artifacts,
        "selection_metadata": selection_metadata,
        "dataset_sizes": {name: len(dataset) for name, dataset in selection_datasets.items()},
        "timestamp": started_at.isoformat(),
    }
    experiment_info = {
        "category": experiment_config["category"],
        "hyperparameters": experiment_config["hyperparameters"],
        "param_key": experiment_config["param_key"],
        "timestamp": experiment_config["timestamp"],
        "run_name": run_name,
        "run_artifact_name": run_artifact_name,
        "selected_direction_path": experiment_config["selected_direction_path"],
    }

    answers_data: dict[str, Any] = {
        "timestamp": started_at.isoformat(),
        "questions": harmful_test_questions,
        "experiment_config": experiment_config,
        "experiment_info": experiment_info,
        "harmful_questions": {
            "questions": harmful_test_questions,
            "original_responses": original_responses,
            "modified_responses": modified_responses,
            "original_evaluations": original_score_data,
            "modified_evaluations": modified_score_data,
        },
        "score_statistics": {
            "metric": evaluation_metric,
            "original_mean": original_mean,
            "original_median": original_median,
            "modified_mean": modified_mean,
            "modified_median": modified_median,
            "original_count": len(original_scores),
            "modified_count": len(modified_scores),
        },
    }
    if uses_unsafe_rate_metric:
        answers_data["score_statistics"]["original_mean_unsafe_rate"] = original_mean
        answers_data["score_statistics"]["original_median_unsafe_rate"] = original_median
        answers_data["score_statistics"]["modified_mean_unsafe_rate"] = modified_mean
        answers_data["score_statistics"]["modified_median_unsafe_rate"] = modified_median
        answers_data["score_statistics"]["unsafe_rate_delta"] = score_delta
        answers_data["score_statistics"]["unsafe_rate_reduction"] = score_reduction
    answers_file = ANSWERS_DIR / f"answers_{run_artifact_name}.json"

    save_best_model = _parse_env_bool("RESULTS_SAVE_BEST_MODEL", False)
    best_model_max_shard_size = os.getenv("RESULTS_BEST_MODEL_MAX_SHARD_SIZE", "5GB")
    best_model_dir = BASELINE_RESULTS_DIR / "models" / f"best_model_{run_artifact_name}"
    answers_data["experiment_config"]["best_model_saved"] = save_best_model
    answers_data["experiment_config"]["best_model_dir"] = (
        _project_relative(best_model_dir) if save_best_model else None
    )
    answers_data["experiment_config"]["answers_file"] = _project_relative(answers_file)
    save_answers(answers_file, answers_data)

    if save_best_model:
        _save_best_model_checkpoint(
            model=model,
            output_dir=best_model_dir,
            max_shard_size=best_model_max_shard_size,
            metadata={
                "kind": "basic_refusal_best_model",
                "model": MODEL_NAME,
                "base_model": MODEL_NAME,
                "method": "basic_refusal",
                "evaluation_backend": os.getenv("EVALUATION_BACKEND", "wildguard"),
                "run_name": run_name,
                "run_artifact_name": run_artifact_name,
                "answers_file": _project_relative(answers_file),
                "results_root": _project_relative(BASELINE_RESULTS_DIR.parent),
                "method_results_dir": _project_relative(BASELINE_RESULTS_DIR),
                "selection_dir": _project_relative(selection_run_dir),
                "selected_direction_path": selection_artifacts["selected_direction_path"],
                "selected_position": selection_metadata["selected_position"],
                "selected_layer": selection_metadata["selected_layer"],
                "abliteration_params": dict(ABLITERATION_PARAMS),
                "selection_config": dict(BASIC_REFUSAL_CONFIG),
                "max_shard_size": best_model_max_shard_size,
                "timestamp": run_stamp,
            },
        )
    else:
        print("Skipping best edited model checkpoint save.")

    _wandb_log(
        {
            "artifacts/answers_file": _project_relative(answers_file),
            "artifacts/selection_dir": _project_relative(selection_run_dir),
            "artifacts/best_model_saved": save_best_model,
            "artifacts/best_model_dir": _project_relative(best_model_dir) if save_best_model else None,
        }
    )

    if WANDB_AVAILABLE:
        wandb.finish()

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)
    print(f"Results directory: {BASELINE_RESULTS_DIR}")
    print(f"Answers file: {answers_file}")
    print(f"Selection artifacts: {selection_run_dir}")


if __name__ == "__main__":
    main()
