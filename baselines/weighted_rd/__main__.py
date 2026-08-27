#!/usr/bin/env python3
"""
GRPO-IS baseline: GRPO with Importance Sampling for training direction weights.

Entry point: python -m baselines.weighted_rd
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import os
import random
import re
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from heretic.config import Settings
from heretic.utils import empty_cache, load_prompts

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

from config import (
    MODEL_NAME, MODEL_BATCH_SIZE, GOOD_PROMPTS_DATASET, RESULTS_DIR,
    GRPO_CONFIG, ABLITERATION_PARAMS,
    EVALUATION_BACKEND, DEBUG, get_method_results_dir,
)
from dataset.load_dataset import load_dataset_split
from data_utils import load_combined_dataset, extract_response_after_think
from refusal_directions import (
    compute_refusal_direction, save_refusal_directions, load_refusal_directions,
)
from model_utils import LearnableDirectionWeights, apply_abliteration_with_hyperparams
from wandb_utils import build_wandb_tags, flatten_wandb_config, resolve_run_mode, short_model_name

from baselines.weighted_rd.optuna_optimizer import optimize_weights_with_optuna
from baselines.weighted_rd.trainer import train_grpo_is_step
from baselines.weighted_rd.runtime_config import (
    resolve_weighted_rd_direction_prompt_count,
    resolve_weighted_rd_direction_prompt_seed,
    resolve_weighted_rd_debug_noise_scale,
    resolve_weighted_rd_debug_question_count,
    resolve_weighted_rd_optimizer_method,
    resolve_weighted_rd_optuna_n_trials,
    resolve_weighted_rd_optuna_sampler,
    resolve_weighted_rd_optuna_sampler_seed,
    resolve_weighted_rd_optuna_weight_max,
    resolve_weighted_rd_optuna_weight_min,
    resolve_weighted_rd_reward_metric,
    resolve_weighted_rd_reward_sign,
    resolve_weighted_rd_weights_mode,
    resolve_weighted_rd_weights_init_type,
    is_weighted_rd_unsafe_rate_metric,
    validate_weighted_rd_optimizer_compatibility,
    validate_weighted_rd_optimizer_method,
    validate_weighted_rd_optuna_sampler,
    validate_weighted_rd_optuna_weight_range,
    validate_weighted_rd_reward_metric,
    validate_weighted_rd_reward_sign,
    validate_weighted_rd_weights_mode,
    validate_weighted_rd_weights_init_type,
)


TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _parse_env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, or on/off; got {raw_value!r}."
    )


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_project_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _save_best_model_checkpoint(
    *,
    model,
    output_dir: Path,
    metadata: Dict[str, Any],
    max_shard_size: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving final best edited model to: {output_dir}")
    model.model.save_pretrained(output_dir, max_shard_size=max_shard_size)
    model.tokenizer.save_pretrained(output_dir)
    metadata_file = output_dir / "weighted_rd_best_model_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    empty_cache()
    print(f"Best edited model metadata saved to: {metadata_file}")


def _load_classifier_categories() -> List[Dict[str, str]]:
    return []


def _evaluate_model_harmfulness(
    model,
    questions: List[str],
    classifier_categories: List[Dict[str, str]],
    reward_metric: str = "harmfulness",
) -> Dict[str, Any]:
    """Evaluate the current model state on the selected harmfulness batch."""
    from baselines.weighted_rd.reward import compute_reward

    responses_raw = model.get_responses_batched(questions)
    responses = []
    for raw in responses_raw:
        extracted = extract_response_after_think(raw)
        responses.append(extracted if extracted else raw.strip())
    scores = [int(score) for score in compute_reward(
        questions,
        responses,
        classifier_categories,
        EVALUATION_BACKEND,
        reward_metric=reward_metric,
    )]
    mean_harmfulness = float(np.mean(scores)) if scores else None
    result = {
        "responses": responses,
        "scores": scores,
        "mean_harmfulness": mean_harmfulness,
    }
    if is_weighted_rd_unsafe_rate_metric(reward_metric):
        result["mean_unsafe_rate"] = mean_harmfulness
        result["best_unsafe_rate"] = float(np.max(scores)) if scores else None
    return result


def _save_scalar_weights_distribution_plot(
    weights: torch.Tensor,
    output_path: Path,
    state_label: str,
    dataset_label: str,
    optimizer_method: str,
    weights_init_type: str,
) -> None:
    """Save a polished scalar-weight distribution histogram as a PDF."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = weights.detach().cpu().float().view(-1).numpy()
    count = int(values.size)
    mean_value = float(np.mean(values)) if count else 0.0
    std_value = float(np.std(values)) if count else 0.0
    min_value = float(np.min(values)) if count else 0.0
    max_value = float(np.max(values)) if count else 0.0
    bins = min(40, max(10, count // 4)) if count else 10

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        values,
        bins=bins,
        color="#4C78A8",
        edgecolor="#1F2933",
        linewidth=0.8,
        alpha=0.85,
    )
    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.5, label="Zero")
    ax.axvline(mean_value, color="#D55E00", linestyle="-", linewidth=1.8, label="Mean")
    ax.set_xlabel("Scalar coefficient value", fontsize=12)
    ax.set_ylabel("Number of scalar coefficients", fontsize=12)
    ax.set_title(
        f"{state_label} Scalar Coefficient Value Distribution",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    subtitle = (
        f"Dataset: {dataset_label} | Optimizer: {optimizer_method} | "
        f"Weight init: {weights_init_type}"
    )
    ax.text(
        0.5,
        1.01,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4A5568",
    )
    stats_text = (
        f"Count: {count}\n"
        f"Mean: {mean_value:.6f}\n"
        f"Std: {std_value:.6f}\n"
        f"Min: {min_value:.6f}\n"
        f"Max: {max_value:.6f}"
    )
    ax.text(
        0.98,
        0.96,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F7FAFC",
            "edgecolor": "#CBD5E0",
            "alpha": 0.95,
        },
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _load_combined_dataset_questions() -> List[str]:
    """Load non-empty instructions from the full combined local dataset."""
    return [
        item["instruction"].strip()
        for item in load_combined_dataset()
        if isinstance(item.get("instruction"), str) and item["instruction"].strip()
    ]


def _load_harmless_split_questions(split: str) -> List[str]:
    """Load non-empty harmless instructions from the shared train/val/test split files."""
    questions = load_dataset_split(
        harmtype="harmless",
        split=split,
        instructions_only=True,
    )
    return [question.strip() for question in questions if isinstance(question, str) and question.strip()]


def _sample_fixed_questions(
    questions: List[str],
    sample_count: int,
    sample_seed: int,
) -> List[str]:
    """Deterministically sample a fixed prompt subset without disturbing global RNG state."""
    if sample_count >= len(questions):
        return list(questions)

    rng = random.Random(sample_seed)
    return rng.sample(questions, k=sample_count)


def _sample_training_questions(
    question_pool: List[str],
    debug_question_count: int,
) -> List[str]:
    """Sample the current training batch from the combined dataset."""
    effective_question_count = min(MODEL_BATCH_SIZE, len(question_pool))
    if DEBUG:
        effective_question_count = min(effective_question_count, debug_question_count)

    if effective_question_count < len(question_pool):
        return random.sample(question_pool, k=effective_question_count)
    return list(question_pool)


def _build_directions_cache_path(
    results_dir: Path,
    model_name: str,
    direction_count: int | None = None,
    prompt_seed: int | None = None,
) -> Path:
    """Build the cache path for combined-dataset refusal directions."""
    model_safe = model_name.replace("/", "_")
    return results_dir / (
        f"refusal_directions_{model_safe}_combined_dataset_"
        f"n{direction_count}_seed{prompt_seed}.pt"
    )


def _load_or_compute_refusal_directions(
    model,
    direction_identifiers: List[str],
    good_prompts: List[str],
    directions_file: Path,
) -> List[torch.Tensor]:
    """Reuse cached refusal directions when identifiers match exactly, otherwise rebuild them."""
    if directions_file.exists():
        extracted_directions, loaded_tags = load_refusal_directions(
            directions_file,
            expected_tags=direction_identifiers,
        )
        if loaded_tags == direction_identifiers:
            device = next(model.get_layers()[0].parameters()).device
            return [direction.to(device) for direction in extracted_directions]

    extracted_directions = []
    for direction_identifier in direction_identifiers:
        refusal_dir = compute_refusal_direction(model, [direction_identifier], good_prompts)
        extracted_directions.append(refusal_dir)
    save_refusal_directions(extracted_directions, direction_identifiers, MODEL_NAME, directions_file)
    return extracted_directions


def _format_run_name_value(value: Any) -> str:
    """Format run-name scalars compactly and consistently."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _sanitize_run_name_part(value: Any) -> str:
    """Normalize free-form values into wandb-friendly run-name fragments."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "unknown"


def _shorten_run_name(name: str, max_length: int = 200) -> str:
    """Keep artifact folder names informative without risking path-length issues."""
    if len(name) <= max_length:
        return name

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    prefix_length = max_length - len(digest) - 1
    shortened_prefix = name[:prefix_length].rstrip("_-")
    if not shortened_prefix:
        return digest
    return f"{shortened_prefix}_{digest}"


def _build_weighted_rd_run_name(
    optimizer_method: str,
    weights_mode: str,
    weights_init_type: str,
    grpo_config: Dict[str, Any],
    abliteration_params: Dict[str, Any],
    optuna_config: Dict[str, Any],
    model_name: str,
    reward_metric: str = "harmfulness",
    evaluation_backend: str = "wildguard",
    results_root: Optional[Path] = None,
    timestamp_str: Optional[str] = None,
) -> str:
    """Build a readable run name with common and optimizer-specific knobs."""
    run_mode = resolve_run_mode(results_root)

    parts = [
        run_mode,
        f"model_{_sanitize_run_name_part(model_name)}",
        _sanitize_run_name_part(optimizer_method),
        "combined_dataset",
        f"weights_{_sanitize_run_name_part(weights_mode)}",
        f"init_{_sanitize_run_name_part(weights_init_type)}",
        f"reward_{_sanitize_run_name_part(reward_metric)}",
        f"eval_{_sanitize_run_name_part(evaluation_backend)}",
        f"abl_max_weight{_format_run_name_value(abliteration_params['max_weight'])}",
    ]

    if optimizer_method == "grpo":
        parts.extend([
            f"n_groups{_format_run_name_value(grpo_config['n_groups'])}",
            f"n_epochs{_format_run_name_value(grpo_config['n_epochs'])}",
            f"lr{_format_run_name_value(grpo_config['learning_rate'])}",
            f"noise{_format_run_name_value(grpo_config['noise_scale'])}",
            f"ref_alpha{_format_run_name_value(grpo_config['ref_alpha'])}",
            f"is_clip{_format_run_name_value(grpo_config['is_clip_ratio'])}",
            f"clip{_format_run_name_value(grpo_config['clip_ratio'])}",
            f"loss_{_sanitize_run_name_part(grpo_config['loss_agg_mode'])}",
            f"beta1{_format_run_name_value(grpo_config['beta_1'])}",
            f"beta2{_format_run_name_value(grpo_config['beta_2'])}",
        ])
    else:
        parts.extend([
            f"sampler_{_sanitize_run_name_part(optuna_config['sampler'])}",
            f"n_trials{_format_run_name_value(optuna_config['n_trials'])}",
            f"sampler_seed{_format_run_name_value(optuna_config['sampler_seed'])}",
            f"weight_min{_format_run_name_value(optuna_config['weight_min'])}",
            f"weight_max{_format_run_name_value(optuna_config['weight_max'])}",
            f"beta1{_format_run_name_value(grpo_config['beta_1'])}",
            f"beta2{_format_run_name_value(grpo_config['beta_2'])}",
        ])

    parts.append(timestamp_str or datetime.now().strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts)


def _define_model_state_eval_metrics() -> None:
    """Register shared wandb series for harmfulness-only comparisons."""
    if not WANDB_AVAILABLE or not hasattr(wandb, "define_metric"):
        return

    wandb.define_metric("model_state_eval/point_index")
    wandb.define_metric(
        "model_state_eval/harmfulness_on_full_dataset",
        step_metric="model_state_eval/point_index",
    )


def _log_model_state_eval(
    prefixes: List[str],
    point_index: int,
    point_name: str,
    harmfulness_on_full_dataset: Optional[float],
) -> None:
    """Log harmfulness-only wandb metrics for clean/best model comparisons."""
    if not WANDB_AVAILABLE:
        return

    payload: Dict[str, Any] = {
        "model_state_eval/point_index": point_index,
        "model_state_eval/point_name": point_name,
    }

    if harmfulness_on_full_dataset is not None:
        for prefix in prefixes:
            payload[f"{prefix}/harmfulness_on_full_dataset"] = harmfulness_on_full_dataset
            if prefix in {"clean_model", "best_value_model"}:
                payload[f"{prefix}/harmfulness"] = harmfulness_on_full_dataset
        payload["model_state_eval/harmfulness_on_full_dataset"] = harmfulness_on_full_dataset

    wandb.log(payload)


def _run_grpo_training(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    model,
    question_pool: List[str],
    effective_noise_scale: float,
    classifier_categories: List[Dict[str, str]],
    n_layers: int,
    debug_question_count: int,
    reward_sign: float,
    reward_metric: str,
    harmless_kl_questions: List[str],
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(direction_weights.parameters(), lr=GRPO_CONFIG["learning_rate"])
    training_history = []
    best_epoch = None
    best_train_batch_questions: List[str] = []
    best_train_batch_mean_reward: Optional[float] = None
    best_train_batch_mean_objective = float("-inf")
    best_train_batch_mean_harmfulness: Optional[float] = None
    best_train_batch_mean_unsafe_rate: Optional[float] = None
    best_train_batch_best_unsafe_rate: Optional[float] = None
    best_train_batch_mean_kl: Optional[float] = None
    best_train_batch_harmful_mean_kl: Optional[float] = None
    best_train_batch_harmless_mean_kl: Optional[float] = None
    best_train_batch_kl_penalty: Optional[float] = None
    best_train_batch_harmless_questions: List[str] = []
    best_weights = direction_weights.weights.detach().cpu().clone()
    print("\n" + "=" * 80)
    print("GRPO-IS TRAINING")
    print("=" * 80)

    for epoch in range(GRPO_CONFIG["n_epochs"]):
        epoch_questions = _sample_training_questions(
            question_pool=question_pool,
            debug_question_count=debug_question_count,
        )
        print(f"\nEpoch {epoch + 1}/{GRPO_CONFIG['n_epochs']}")
        print(
            f"  Training on random batch of {len(epoch_questions)} question(s) "
            f"sampled from {len(question_pool)} available"
        )
        epoch_harmless_questions: List[str] = []
        if float(GRPO_CONFIG["beta_2"]) != 0.0:
            epoch_harmless_questions = _sample_training_questions(
                question_pool=harmless_kl_questions,
                debug_question_count=debug_question_count,
            )
            print(
                f"  Harmless KL on random batch of {len(epoch_harmless_questions)} question(s) "
                f"sampled from {len(harmless_kl_questions)} available"
            )
        metrics = train_grpo_is_step(
            direction_weights=direction_weights,
            extracted_directions=extracted_directions,
            model=model,
            questions=epoch_questions,
            n_groups=GRPO_CONFIG["n_groups"],
            noise_scale=effective_noise_scale,
            abliteration_params=ABLITERATION_PARAMS,
            optimizer=optimizer,
            classifier_categories=classifier_categories,
            n_layers=n_layers,
            ref_alpha=GRPO_CONFIG["ref_alpha"],
            is_clip_ratio=GRPO_CONFIG["is_clip_ratio"],
            clip_ratio=GRPO_CONFIG["clip_ratio"],
            loss_agg_mode=GRPO_CONFIG["loss_agg_mode"],
            reward_sign=reward_sign,
            reward_metric=reward_metric,
            beta_1=GRPO_CONFIG["beta_1"],
            beta_2=GRPO_CONFIG["beta_2"],
            harmless_questions=epoch_harmless_questions,
            backend=EVALUATION_BACKEND,
        )
        training_history.append({
            "epoch": epoch + 1,
            "n_questions": len(epoch_questions),
            **metrics,
        })
        mean_harmfulness = metrics.get("mean_harmfulness")
        mean_kl = metrics.get("mean_kl")
        kl_penalty = metrics.get("kl_penalty")
        mean_objective = metrics.get("mean_objective")
        if mean_harmfulness is not None and mean_kl is not None and mean_objective is not None:
            print(
                f"  Mean objective: {mean_objective:.3f}, Mean reward: {metrics['mean_reward']:.3f}, "
                f"Best reward: {metrics['best_reward']:.3f} "
                f"(mean harmfulness={mean_harmfulness:.3f}, mean kl={mean_kl:.3f}, "
                f"kl penalty={kl_penalty:.3f})"
            )
        else:
            print(f"  Mean reward: {metrics['mean_reward']:.3f}, Best: {metrics['best_reward']:.3f}")
        if metrics["mean_objective"] > best_train_batch_mean_objective:
            best_epoch = epoch + 1
            best_train_batch_mean_reward = float(metrics["mean_reward"])
            best_train_batch_mean_objective = float(metrics["mean_objective"])
            best_train_batch_mean_harmfulness = (
                float(mean_harmfulness) if mean_harmfulness is not None else None
            )
            best_train_batch_mean_unsafe_rate = (
                float(metrics["mean_unsafe_rate"])
                if metrics.get("mean_unsafe_rate") is not None
                else None
            )
            best_train_batch_best_unsafe_rate = (
                float(metrics["best_unsafe_rate"])
                if metrics.get("best_unsafe_rate") is not None
                else None
            )
            best_train_batch_mean_kl = float(mean_kl) if mean_kl is not None else None
            best_train_batch_harmful_mean_kl = (
                float(metrics["harmful_mean_kl"])
                if metrics.get("harmful_mean_kl") is not None
                else None
            )
            best_train_batch_harmless_mean_kl = (
                float(metrics["harmless_mean_kl"])
                if metrics.get("harmless_mean_kl") is not None
                else None
            )
            best_train_batch_kl_penalty = (
                float(metrics["kl_penalty"])
                if metrics.get("kl_penalty") is not None
                else None
            )
            best_train_batch_harmless_questions = list(epoch_harmless_questions)
            best_train_batch_questions = list(epoch_questions)
            best_weights = direction_weights.weights.detach().cpu().clone()

        if WANDB_AVAILABLE:
            wandb.log({f"train/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}, step=epoch + 1)

    if best_epoch is None:
        raise ValueError("GRPO training finished without producing a best checkpoint.")

    return {
        "training_history": training_history,
        "optimization_history": [],
        "optuna_config": None,
        "best_epoch": best_epoch,
        "best_weights": best_weights,
        "best_train_batch_questions": best_train_batch_questions,
        "best_train_batch_mean_reward": best_train_batch_mean_reward,
        "best_train_batch_mean_objective": best_train_batch_mean_objective,
        "best_train_batch_mean_harmfulness": best_train_batch_mean_harmfulness,
        "best_train_batch_mean_unsafe_rate": best_train_batch_mean_unsafe_rate,
        "best_train_batch_best_unsafe_rate": best_train_batch_best_unsafe_rate,
        "best_train_batch_mean_kl": best_train_batch_mean_kl,
        "best_train_batch_harmful_mean_kl": best_train_batch_harmful_mean_kl,
        "best_train_batch_harmless_mean_kl": best_train_batch_harmless_mean_kl,
        "best_train_batch_kl_penalty": best_train_batch_kl_penalty,
        "best_train_batch_harmless_questions": best_train_batch_harmless_questions,
    }


def _final_evaluate(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    model,
    evaluation_questions: List[str],
    classifier_categories: List[Dict[str, str]],
    n_layers: int,
    reward_metric: str = "harmfulness",
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    with torch.no_grad():
        final_direction = direction_weights([d.detach() for d in extracted_directions])

    model.reload_model()
    apply_abliteration_with_hyperparams(
        model,
        final_direction,
        ABLITERATION_PARAMS["max_weight"] * GRPO_CONFIG["ref_alpha"],
        ABLITERATION_PARAMS["max_weight_position"],
        ABLITERATION_PARAMS["min_weight"] * GRPO_CONFIG["ref_alpha"],
        ABLITERATION_PARAMS["min_weight_distance"],
        n_layers,
    )

    harmfulness_result = _evaluate_model_harmfulness(
        model=model,
        questions=evaluation_questions,
        classifier_categories=classifier_categories,
        reward_metric=reward_metric,
    )

    result = {
        "responses": harmfulness_result["responses"],
        "scores": harmfulness_result["scores"],
        "final_mean_harmfulness": harmfulness_result["mean_harmfulness"],
    }
    if is_weighted_rd_unsafe_rate_metric(reward_metric):
        result["mean_unsafe_rate"] = harmfulness_result.get("mean_unsafe_rate")
        result["best_unsafe_rate"] = harmfulness_result.get("best_unsafe_rate")
    return result


def main():
    direction_prompt_limit = resolve_weighted_rd_direction_prompt_count(
        os.getenv("WEIGHTED_RD_DIRECTION_PROMPT_COUNT")
    )
    direction_prompt_seed = resolve_weighted_rd_direction_prompt_seed(
        os.getenv("WEIGHTED_RD_DIRECTION_PROMPT_SEED")
    )
    optimizer_method = resolve_weighted_rd_optimizer_method(os.getenv("OPTIMIZER_METHOD"))
    weights_mode = resolve_weighted_rd_weights_mode(os.getenv("WEIGHTS_MODE"))
    weights_init_type = resolve_weighted_rd_weights_init_type(os.getenv("WEIGHTS_INIT_TYPE"))
    debug_question_count = resolve_weighted_rd_debug_question_count(os.getenv("DEBUG_N_QUESTIONS"))
    debug_noise_scale = resolve_weighted_rd_debug_noise_scale(
        base_noise_scale=GRPO_CONFIG["noise_scale"],
        env_value=os.getenv("DEBUG_NOISE_SCALE"),
    )
    optuna_sampler = resolve_weighted_rd_optuna_sampler(os.getenv("OPTUNA_SAMPLER"))
    optuna_n_trials = resolve_weighted_rd_optuna_n_trials(os.getenv("OPTUNA_N_TRIALS"))
    optuna_sampler_seed = resolve_weighted_rd_optuna_sampler_seed(os.getenv("OPTUNA_SAMPLER_SEED"))
    optuna_weight_min = resolve_weighted_rd_optuna_weight_min(os.getenv("OPTUNA_WEIGHT_MIN"))
    optuna_weight_max = resolve_weighted_rd_optuna_weight_max(os.getenv("OPTUNA_WEIGHT_MAX"))
    reward_sign = resolve_weighted_rd_reward_sign(os.getenv("REWARD_SIGN"))
    reward_metric = resolve_weighted_rd_reward_metric(
        os.getenv("REWARD_METRIC"),
        backend=EVALUATION_BACKEND,
    )
    effective_noise_scale = debug_noise_scale if DEBUG else GRPO_CONFIG["noise_scale"]

    validate_weighted_rd_optimizer_method(optimizer_method)
    validate_weighted_rd_weights_mode(weights_mode)
    validate_weighted_rd_weights_init_type(weights_init_type)
    validate_weighted_rd_optimizer_compatibility(optimizer_method, weights_mode)
    validate_weighted_rd_optuna_sampler(optuna_sampler)
    validate_weighted_rd_optuna_weight_range(optuna_weight_min, optuna_weight_max)
    validate_weighted_rd_reward_sign(reward_sign)
    validate_weighted_rd_reward_metric(reward_metric, EVALUATION_BACKEND)
    score_metric_label = "unsafe rate" if is_weighted_rd_unsafe_rate_metric(reward_metric) else "harmfulness"
    dataset_label = "combined_dataset"

    optuna_config = {
        "sampler": optuna_sampler,
        "n_trials": optuna_n_trials,
        "sampler_seed": optuna_sampler_seed,
        "weight_min": optuna_weight_min,
        "weight_max": optuna_weight_max,
    }
    run_timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = _build_weighted_rd_run_name(
        optimizer_method=optimizer_method,
        weights_mode=weights_mode,
        weights_init_type=weights_init_type,
        grpo_config=GRPO_CONFIG,
        abliteration_params=ABLITERATION_PARAMS,
        optuna_config=optuna_config,
        model_name=MODEL_NAME,
        reward_metric=reward_metric,
        evaluation_backend=EVALUATION_BACKEND,
        results_root=RESULTS_DIR,
        timestamp_str=run_timestamp_str,
    )
    run_artifact_name = _shorten_run_name(run_name)

    print("=" * 80)
    print("GRPO-IS: GRPO with Importance Sampling")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GRPO Config: {GRPO_CONFIG}")
    print(f"Abliteration Params: {ABLITERATION_PARAMS}")
    print(f"Optimizer Method: {optimizer_method}")
    print(f"Dataset: {dataset_label}")
    print(f"Run artifact name: {run_artifact_name}")
    print(f"Weights Mode: {weights_mode}")
    print(f"Weights Init Type: {weights_init_type}")
    print(f"Batch Size: {MODEL_BATCH_SIZE}")
    print(f"Evaluation backend: {EVALUATION_BACKEND}")
    print(f"Reward metric: {reward_metric}")
    print(f"Reward sign: {reward_sign:+g}")
    print(
        "Direction prompt sampling: "
        f"count={direction_prompt_limit}, "
        f"seed={direction_prompt_seed}"
    )
    if optimizer_method == "optuna":
        print(f"Optuna Config: {optuna_config}")
    print()

    GRPO_RESULTS_DIR = get_method_results_dir("weighted_rd")
    GRPO_ANSWERS_DIR = GRPO_RESULTS_DIR / "answers"
    GRPO_ANSWERS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    optimization_questions = _load_combined_dataset_questions()
    if not optimization_questions:
        raise ValueError("No instructions found in the combined local dataset.")
    full_evaluation_questions = list(optimization_questions)
    direction_source = "dataset/processed/*.json"
    optimization_source = "combined local processed datasets"
    final_evaluation_source = optimization_source
    print(
        f"Loaded {len(optimization_questions)} combined-dataset question(s) "
        "for optimization and evaluation"
    )

    print("\nLoading model...")
    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]] if sys.argv else ["script"]
        settings = Settings(
            model=MODEL_NAME, batch_size=MODEL_BATCH_SIZE,
            max_response_length=2048, system_prompt="You are a helpful assistant.",
        )
    finally:
        sys.argv = original_argv

    from heretic.model import Model
    model = Model(settings)
    n_layers = len(model.get_layers())
    print(f"Number of layers: {n_layers}")

    print("Loading good prompts...")
    good_prompts = load_prompts(GOOD_PROMPTS_DATASET)

    direction_sample_count = (
        len(optimization_questions)
        if direction_prompt_limit is None
        else min(direction_prompt_limit, len(optimization_questions))
    )
    direction_prompts = _sample_fixed_questions(
        optimization_questions,
        direction_sample_count,
        direction_prompt_seed,
    )
    direction_prompt_count = len(direction_prompts)
    print(
        f"Sampled {direction_prompt_count} combined-dataset prompt(s) "
        "for refusal directions"
    )
    directions_file = _build_directions_cache_path(
        results_dir=RESULTS_DIR,
        model_name=MODEL_NAME,
        direction_count=direction_prompt_count,
        prompt_seed=direction_prompt_seed,
    )

    extracted_directions = _load_or_compute_refusal_directions(
        model=model,
        direction_identifiers=direction_prompts,
        good_prompts=good_prompts,
        directions_file=directions_file,
    )

    n_directions = len(extracted_directions)
    hidden_size = extracted_directions[0].shape[1]

    print(f"Using weights mode '{weights_mode}'")
    print(f"Using weights init type '{weights_init_type}'")
    print(f"Using optimizer method '{optimizer_method}'")

    direction_weights = LearnableDirectionWeights(
        n_directions=n_directions, n_layers=n_layers, hidden_size=hidden_size,
        init_type=weights_init_type, topic_idx=0, mode=weights_mode,
    )
    device = extracted_directions[0].device
    direction_weights = direction_weights.to(device)
    weights_shape = list(direction_weights.weights.shape)
    initial_weights = direction_weights.weights.detach().cpu().clone()
    print(f"Trainable weights shape: {weights_shape}")

    train_question_count = min(MODEL_BATCH_SIZE, len(optimization_questions))
    if DEBUG:
        train_question_count = min(train_question_count, debug_question_count)

    print(
        f"Training batch size will be {train_question_count} question(s) sampled from "
        f"{len(optimization_questions)} combined-dataset question(s) "
        f"using batch_size={MODEL_BATCH_SIZE}"
    )

    if DEBUG and train_question_count > 0:
        print(
            f"DEBUG mode enabled: using {train_question_count} question(s) from the optimization pool "
            f"and rollout noise_scale={effective_noise_scale:g} "
            f"(base noise_scale={GRPO_CONFIG['noise_scale']:g})"
        )
    print(
        f"Loaded {len(optimization_questions)} total optimization question(s) "
        f"and {len(full_evaluation_questions)} full-evaluation question(s)"
    )

    classifier_categories = _load_classifier_categories()

    run_mode = resolve_run_mode(RESULTS_DIR)
    if WANDB_AVAILABLE:
        wandb_config = {
            "method": "weighted_rd",
            "model": MODEL_NAME,
            "model_name": MODEL_NAME,
            "model_short_name": short_model_name(MODEL_NAME),
            "run_mode": run_mode,
            "run_artifact_name": run_artifact_name,
            "dataset": dataset_label,
            "n_directions": n_directions,
            "n_layers": n_layers,
            "optimizer_method": optimizer_method,
            "weights_mode": weights_mode,
            "weights_init_type": weights_init_type,
            "weights_shape": weights_shape,
            "direction_source": direction_source,
            "optimization_source": optimization_source,
            "final_evaluation_source": final_evaluation_source,
            "direction_prompt_count": direction_prompt_count,
            "direction_prompt_seed": direction_prompt_seed,
            "optimization_question_pool_size": len(optimization_questions),
            "full_evaluation_question_count": len(full_evaluation_questions),
            "train_question_count": train_question_count,
            "reward_metric": reward_metric,
            "reward_sign": reward_sign,
            "evaluation_backend": EVALUATION_BACKEND,
            "debug": DEBUG,
            "debug_question_count": debug_question_count,
            "effective_noise_scale": effective_noise_scale,
            "grpo_config": GRPO_CONFIG,
            "abliteration_params": ABLITERATION_PARAMS,
            "optuna_config": optuna_config if optimizer_method == "optuna" else None,
        }
        wandb_filter_config = {
            **flatten_wandb_config(GRPO_CONFIG, prefix="grpo"),
            **flatten_wandb_config(ABLITERATION_PARAMS, prefix="abl"),
        }
        if optimizer_method == "optuna":
            wandb_filter_config.update(flatten_wandb_config(optuna_config, prefix="optuna"))
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "anonymized"),
            name=run_name,
            job_type="weighted_rd",
            group=f"weighted_rd/{short_model_name(MODEL_NAME)}/{optimizer_method}",
            tags=build_wandb_tags(
                [
                    ("method", "weighted_rd"),
                    ("model", short_model_name(MODEL_NAME)),
                    ("optimizer", optimizer_method),
                    ("dataset", dataset_label),
                    ("weights_mode", weights_mode),
                    ("weights_init", weights_init_type),
                    ("reward_metric", reward_metric),
                    ("eval_backend", EVALUATION_BACKEND),
                    ("run_mode", run_mode),
                ]
            ),
            config={**wandb_config, **wandb_filter_config},
        )
        _define_model_state_eval_metrics()

    print("\n" + "=" * 80)
    print("CLEAN MODEL EVALUATION")
    print("=" * 80)
    print(
        f"Evaluating clean model on the combined dataset: "
        f"{len(full_evaluation_questions)} question(s); training will use batches "
        f"of {train_question_count} question(s)"
    )
    model.reload_model()
    clean_harmfulness_result = _evaluate_model_harmfulness(
        model=model,
        questions=full_evaluation_questions,
        classifier_categories=classifier_categories,
        reward_metric=reward_metric,
    )
    clean_mean_harmfulness = clean_harmfulness_result["mean_harmfulness"]
    print(
        f"Clean model {score_metric_label}: {clean_mean_harmfulness:.3f}"
        if clean_mean_harmfulness is not None
        else f"Clean model {score_metric_label}: n/a"
    )

    _log_model_state_eval(
        prefixes=["clean_model"],
        point_index=0,
        point_name="clean",
        harmfulness_on_full_dataset=clean_mean_harmfulness,
    )

    print("\n" + "=" * 80)
    print("INITIALIZED WEIGHTS EVALUATION")
    print("=" * 80)
    print(
        f"Evaluating initialized weights on full evaluation split: "
        f"{len(full_evaluation_questions)} question(s)"
    )
    with torch.no_grad():
        initialized_direction = direction_weights(
            [direction.detach() for direction in extracted_directions]
        )
    model.reload_model()
    apply_abliteration_with_hyperparams(
        model,
        initialized_direction,
        ABLITERATION_PARAMS["max_weight"] * GRPO_CONFIG["ref_alpha"],
        ABLITERATION_PARAMS["max_weight_position"],
        ABLITERATION_PARAMS["min_weight"] * GRPO_CONFIG["ref_alpha"],
        ABLITERATION_PARAMS["min_weight_distance"],
        n_layers,
    )
    initialized_harmfulness_result = _evaluate_model_harmfulness(
        model=model,
        questions=full_evaluation_questions,
        classifier_categories=classifier_categories,
        reward_metric=reward_metric,
    )
    initialized_mean_harmfulness = initialized_harmfulness_result["mean_harmfulness"]
    print(
        f"Initialized weights {score_metric_label}: {initialized_mean_harmfulness:.3f}"
        if initialized_mean_harmfulness is not None
        else f"Initialized weights {score_metric_label}: n/a"
    )
    model.reload_model()

    harmless_kl_questions: List[str] = []
    if float(GRPO_CONFIG["beta_2"]) != 0.0:
        harmless_kl_questions = _load_harmless_split_questions("val")
        if not harmless_kl_questions:
            raise ValueError(
                "beta_2 is non-zero, but no harmless val questions found for KL regularization."
            )

    if optimizer_method == "grpo":
        if harmless_kl_questions:
            print(
                f"  Each GRPO epoch will also compute KL on a random harmless batch "
                f"of {min(MODEL_BATCH_SIZE, len(harmless_kl_questions))} question(s) "
                f"sampled from {len(harmless_kl_questions)} available"
            )
        optimization_result = _run_grpo_training(
            direction_weights=direction_weights,
            extracted_directions=extracted_directions,
            model=model,
            classifier_categories=classifier_categories,
            question_pool=optimization_questions,
            effective_noise_scale=effective_noise_scale,
            n_layers=n_layers,
            debug_question_count=debug_question_count,
            reward_sign=reward_sign,
            reward_metric=reward_metric,
            harmless_kl_questions=harmless_kl_questions,
        )
    else:
        print("\n" + "=" * 80)
        print("OPTUNA OPTIMIZATION")
        print("=" * 80)
        print(
            f"  Each Optuna trial will train on a random batch of {train_question_count} "
            f"question(s) sampled from {len(optimization_questions)} available"
        )
        if harmless_kl_questions:
            print(
                f"  Each Optuna trial will also compute KL on a random harmless batch "
                f"of {min(MODEL_BATCH_SIZE, len(harmless_kl_questions))} question(s) "
                f"sampled from {len(harmless_kl_questions)} available"
            )
        else:
            print("  Harmless KL regularization disabled because beta_2=0")
        optimization_result = optimize_weights_with_optuna(
            direction_weights=direction_weights,
            extracted_directions=extracted_directions,
            model=model,
            questions=optimization_questions,
            abliteration_params=ABLITERATION_PARAMS,
            classifier_categories=classifier_categories,
            n_layers=n_layers,
            ref_alpha=GRPO_CONFIG["ref_alpha"],
            reward_sign=reward_sign,
            beta_1=GRPO_CONFIG["beta_1"],
            beta_2=GRPO_CONFIG["beta_2"],
            n_trials=optuna_n_trials,
            sampler_name=optuna_sampler,
            sampler_seed=optuna_sampler_seed,
            weight_min=optuna_weight_min,
            weight_max=optuna_weight_max,
            question_sampler=lambda: _sample_training_questions(
                question_pool=optimization_questions,
                debug_question_count=debug_question_count,
            ),
            harmless_question_sampler=(
                lambda: _sample_training_questions(
                    question_pool=harmless_kl_questions,
                    debug_question_count=debug_question_count,
                )
                if harmless_kl_questions
                else None
            ),
            loss_agg_mode=GRPO_CONFIG["loss_agg_mode"],
            backend=EVALUATION_BACKEND,
            reward_metric=reward_metric,
        )
        optimization_result["training_history"] = []
        optimization_result["optuna_config"] = optuna_config
        print(
            f"Best Optuna trial: #{optimization_result['best_trial_number']} "
            f"with mean objective={optimization_result['best_value']:.3f} "
            f"(mean harmfulness={optimization_result.get('best_mean_harmfulness')}, "
            f"mean unsafe rate={optimization_result.get('best_mean_unsafe_rate')}, "
            f"mean kl={optimization_result.get('best_mean_kl')}, "
            f"harmful kl={optimization_result.get('best_harmful_mean_kl')}, "
            f"harmless kl={optimization_result.get('best_harmless_mean_kl')}, "
            f"kl penalty={optimization_result.get('best_kl_penalty')})"
        )
        if WANDB_AVAILABLE:
            wandb.log(
                {
                    "optuna/best_trial_number": optimization_result["best_trial_number"],
                    "optuna/best_value": optimization_result["best_value"],
                    "optuna/best_objective": optimization_result.get("best_mean_objective", optimization_result["best_value"]),
                    "optuna/best_harmfulness": optimization_result.get("best_mean_harmfulness"),
                    "optuna/best_unsafe_rate": optimization_result.get("best_mean_unsafe_rate"),
                    "optuna/best_kl": optimization_result.get("best_mean_kl"),
                    "optuna/best_harmful_kl": optimization_result.get("best_harmful_mean_kl"),
                    "optuna/best_harmless_kl": optimization_result.get("best_harmless_mean_kl"),
                    "optuna/best_kl_penalty": optimization_result.get("best_kl_penalty"),
                }
            )

    evaluation_questions = full_evaluation_questions
    if optimizer_method == "grpo":
        with torch.no_grad():
            direction_weights.weights.data.copy_(
                optimization_result["best_weights"].to(
                    device=direction_weights.weights.device,
                    dtype=direction_weights.weights.dtype,
                )
            )

    final_result = _final_evaluate(
        direction_weights=direction_weights,
        extracted_directions=extracted_directions,
        model=model,
        evaluation_questions=evaluation_questions,
        classifier_categories=classifier_categories,
        n_layers=n_layers,
        reward_metric=reward_metric,
    )

    final_mean_harmfulness = final_result["final_mean_harmfulness"]
    final_mean_unsafe_rate = final_result.get("mean_unsafe_rate")
    final_best_unsafe_rate = final_result.get("best_unsafe_rate")
    if optimizer_method == "grpo":
        optimal_objective = optimization_result["best_train_batch_mean_objective"]
        optimal_harmfulness = optimization_result["best_train_batch_mean_harmfulness"]
        optimal_harmfulness_source = "best_train_batch_mean_objective"
        optimal_objective_source = "best_train_batch_mean_objective"
    else:
        best_mean_objective = optimization_result.get("best_mean_objective")
        best_mean_harmfulness = optimization_result.get("best_mean_harmfulness")
        optimal_objective = (
            best_mean_objective
            if best_mean_objective is not None
            else optimization_result["best_value"]
        )
        optimal_harmfulness = (
            best_mean_harmfulness
            if best_mean_harmfulness is not None
            else optimal_objective
        )
        optimal_harmfulness_source = "best_trial_mean_objective"
        optimal_objective_source = "best_trial_mean_objective"

    best_batch_model_harmfulness_result = _evaluate_model_harmfulness(
        model=model,
        questions=full_evaluation_questions,
        classifier_categories=classifier_categories,
        reward_metric=reward_metric,
    )
    best_batch_model_mean_harmfulness = best_batch_model_harmfulness_result["mean_harmfulness"]

    best_model_prefixes = ["best_value_model"]
    if optimizer_method == "grpo":
        best_model_prefixes.append("best_batch_model")

    _log_model_state_eval(
        prefixes=best_model_prefixes,
        point_index=1,
        point_name="best_value",
        harmfulness_on_full_dataset=best_batch_model_mean_harmfulness,
    )

    timestamp_str = run_timestamp_str
    save_best_model = _parse_env_bool("RESULTS_SAVE_BEST_MODEL", False)
    best_model_max_shard_size = os.getenv("RESULTS_BEST_MODEL_MAX_SHARD_SIZE", "5GB")
    best_model_dir = (
        GRPO_RESULTS_DIR
        / "models"
        / f"best_model_{run_artifact_name}"
    )

    if weights_mode == "scalar":
        initial_weights_plot_file = (
            GRPO_RESULTS_DIR
            / f"scalar_weights_distribution_initial_{run_artifact_name}.pdf"
        )
        final_weights_plot_file = (
            GRPO_RESULTS_DIR
            / f"scalar_weights_distribution_final_{run_artifact_name}.pdf"
        )
        _save_scalar_weights_distribution_plot(
            weights=initial_weights,
            output_path=initial_weights_plot_file,
            state_label="Initial",
            dataset_label=dataset_label,
            optimizer_method=optimizer_method,
            weights_init_type=weights_init_type,
        )
        print(f"Initial scalar weights distribution saved to: {initial_weights_plot_file}")
        _save_scalar_weights_distribution_plot(
            weights=direction_weights.weights.data.cpu(),
            output_path=final_weights_plot_file,
            state_label="Final",
            dataset_label=dataset_label,
            optimizer_method=optimizer_method,
            weights_init_type=weights_init_type,
        )
        print(f"Final scalar weights distribution saved to: {final_weights_plot_file}")
    else:
        print(f"Skipping scalar weight distribution plots for weights_mode='{weights_mode}'")

    answers_data = {
        "experiment_config": {
            "model": MODEL_NAME,
            "dataset": dataset_label,
            "direction_source": direction_source,
            "optimization_source": optimization_source,
            "final_evaluation_source": final_evaluation_source,
            "direction_prompt_count": direction_prompt_count,
            "direction_prompt_seed": direction_prompt_seed,
            "optimization_question_pool_size": len(optimization_questions),
            "full_evaluation_question_count": len(full_evaluation_questions),
            "n_questions": len(evaluation_questions), "n_directions": n_directions,
            "optimizer_method": optimizer_method,
            "reward_metric": reward_metric,
            "reward_sign": reward_sign,
            "weights_mode": weights_mode, "weights_init_type": weights_init_type,
            "weights_shape": weights_shape,
            "run_name": run_name,
            "run_artifact_name": run_artifact_name,
            "best_model_saved": save_best_model,
            "best_model_dir": _project_relative(best_model_dir) if save_best_model else None,
            "grpo_config": GRPO_CONFIG, "abliteration_params": ABLITERATION_PARAMS,
            "training_history": optimization_result["training_history"],
            "optuna_config": optimization_result["optuna_config"],
        },
        "final_weights": direction_weights.weights.data.cpu().tolist(),
        "questions": evaluation_questions,
        "responses": final_result["responses"],
        "final_scores": final_result["scores"],
        "final_mean_harmfulness": final_mean_harmfulness,
        "optimal_objective": optimal_objective,
        "optimal_objective_source": optimal_objective_source,
        "optimal_harmfulness": optimal_harmfulness,
        "optimal_harmfulness_source": optimal_harmfulness_source,
        "score_statistics": {
            "mean": final_mean_harmfulness,
            "median": float(np.median(final_result["scores"])) if final_result["scores"] else None,
            "count": len(final_result["scores"]),
        },
        "timestamp": datetime.now().isoformat(),
    }
    if is_weighted_rd_unsafe_rate_metric(reward_metric):
        answers_data["final_mean_unsafe_rate"] = final_mean_unsafe_rate
        answers_data["final_best_unsafe_rate"] = final_best_unsafe_rate
        answers_data["score_statistics"]["mean_unsafe_rate"] = final_mean_unsafe_rate
        answers_data["score_statistics"]["best_unsafe_rate"] = final_best_unsafe_rate
    if optimizer_method == "grpo":
        answers_data["experiment_config"]["best_grpo_epoch"] = optimization_result["best_epoch"]
        answers_data["experiment_config"]["best_train_batch_mean_reward"] = optimization_result["best_train_batch_mean_reward"]
        answers_data["experiment_config"]["best_train_batch_mean_objective"] = optimization_result["best_train_batch_mean_objective"]
        answers_data["experiment_config"]["best_train_batch_mean_harmfulness"] = optimization_result["best_train_batch_mean_harmfulness"]
        if is_weighted_rd_unsafe_rate_metric(reward_metric):
            answers_data["experiment_config"]["best_train_batch_mean_unsafe_rate"] = optimization_result["best_train_batch_mean_unsafe_rate"]
        answers_data["experiment_config"]["best_train_batch_best_unsafe_rate"] = optimization_result["best_train_batch_best_unsafe_rate"]
        answers_data["experiment_config"]["best_train_batch_mean_kl"] = optimization_result["best_train_batch_mean_kl"]
        answers_data["experiment_config"]["best_train_batch_harmful_mean_kl"] = optimization_result["best_train_batch_harmful_mean_kl"]
        answers_data["experiment_config"]["best_train_batch_harmless_mean_kl"] = optimization_result["best_train_batch_harmless_mean_kl"]
        answers_data["experiment_config"]["best_train_batch_kl_penalty"] = optimization_result["best_train_batch_kl_penalty"]
        answers_data["experiment_config"]["best_train_batch_size"] = len(optimization_result["best_train_batch_questions"])
        answers_data["experiment_config"]["best_train_batch_harmless_batch_size"] = len(
            optimization_result["best_train_batch_harmless_questions"]
        )
    else:
        answers_data["experiment_config"]["best_trial_mean_objective"] = optimization_result.get("best_mean_objective")
        answers_data["experiment_config"]["best_trial_mean_harmfulness"] = optimization_result.get("best_mean_harmfulness")
        if is_weighted_rd_unsafe_rate_metric(reward_metric):
            answers_data["experiment_config"]["best_trial_mean_unsafe_rate"] = optimization_result.get("best_mean_unsafe_rate")
            answers_data["experiment_config"]["best_trial_best_unsafe_rate"] = optimization_result.get("best_unsafe_rate")
        answers_data["experiment_config"]["best_trial_mean_kl"] = optimization_result.get("best_mean_kl")
        answers_data["experiment_config"]["best_trial_harmful_mean_kl"] = optimization_result.get("best_harmful_mean_kl")
        answers_data["experiment_config"]["best_trial_harmless_mean_kl"] = optimization_result.get("best_harmless_mean_kl")
        answers_data["experiment_config"]["best_trial_kl_penalty"] = optimization_result.get("best_kl_penalty")
        answers_data["experiment_config"]["best_trial_harmless_batch_size"] = len(
            optimization_result.get("best_trial_harmless_questions", [])
        )
    if optimization_result["optimization_history"]:
        answers_data["optimization_history"] = optimization_result["optimization_history"]
    answers_file = GRPO_ANSWERS_DIR / f"answers_{run_artifact_name}.json"
    with open(answers_file, "w", encoding="utf-8") as f:
        json.dump(answers_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {answers_file}")

    weights_pt_file = GRPO_RESULTS_DIR / f"coefficients_{run_artifact_name}.pt"
    torch.save({
        "weights": direction_weights.weights.data.cpu(),
        "metadata": {
            "model": MODEL_NAME,
            "dataset": dataset_label,
            "direction_source": direction_source,
            "optimization_source": optimization_source,
            "final_evaluation_source": final_evaluation_source,
            "direction_prompt_count": direction_prompt_count,
            "direction_prompt_seed": direction_prompt_seed,
            "n_directions": n_directions,
            "n_layers": n_layers,
            "optimizer_method": optimizer_method,
            "reward_metric": reward_metric,
            "weights_mode": weights_mode,
            "weights_init_type": weights_init_type,
            "weights_shape": weights_shape,
            "run_name": run_name,
            "run_artifact_name": run_artifact_name,
        },
    }, weights_pt_file)
    print(f"Coefficients saved to: {weights_pt_file}")

    if save_best_model:
        _save_best_model_checkpoint(
            model=model,
            output_dir=best_model_dir,
            max_shard_size=best_model_max_shard_size,
            metadata={
                "kind": "weighted_rd_best_model",
                "model": MODEL_NAME,
                "base_model": MODEL_NAME,
                "dataset": dataset_label,
                "optimizer_method": optimizer_method,
                "reward_metric": reward_metric,
                "weights_mode": weights_mode,
                "weights_init_type": weights_init_type,
                "run_name": run_name,
                "run_artifact_name": run_artifact_name,
                "answers_file": _project_relative(answers_file),
                "coefficients_file": _project_relative(weights_pt_file),
                "results_root": _project_relative(RESULTS_DIR),
                "method_results_dir": _project_relative(GRPO_RESULTS_DIR),
                "max_shard_size": best_model_max_shard_size,
                "timestamp": timestamp_str,
            },
        )
    else:
        print("Skipping best edited model checkpoint save.")

    if WANDB_AVAILABLE:
        wandb.finish()

    print("=" * 80)


if __name__ == "__main__":
    main()
