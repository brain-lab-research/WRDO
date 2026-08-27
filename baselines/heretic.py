#!/usr/bin/env python3
"""
Heretic baseline orchestrator.

Mirrors `baselines.basic_refusal` structurally but replaces the single-
direction selection step with an Optuna TPE study (see
`baselines.heretic_optim`). All weight ablation goes through the local
`heretic.model.Model.abliterate`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    HERETIC_CONFIG,
    MODEL_BATCH_SIZE,
    MODEL_NAME,
    get_method_results_dir,
)
from data_utils import extract_response_after_think
from evaluate.metrics import (
    evaluate_responses,
    is_unsafe_rate_evaluation_metric,
    resolve_evaluation_metric,
)
from heretic.config import Settings
from heretic.model import Model
from wandb_utils import (
    build_wandb_tags,
    flatten_wandb_config,
    resolve_run_mode,
    short_model_name,
)

from baselines.basic_refusal import (
    _parse_env_bool,
    _project_relative,
    _sanitize_run_name_part,
    _shorten_run_name,
    _wandb_log,
    build_refusal_token_ids,
    compute_eoi_suffix_token_ids,
    filter_instructions_by_refusal_sign,
    get_mean_diff,
    load_classifier_categories,
    load_selection_datasets,
    save_answers,
)
from baselines.heretic_optim import (
    apply_trial,
    auto_select_trial,
    build_refusal_directions_tensor,
    extract_pareto_front,
    make_objective,
    reconstruct_params_from_trial,
    run_heretic_study,
    trial_to_record,
)

torch.set_grad_enabled(False)
logger = logging.getLogger(__name__)

BASELINE_RESULTS_DIR = get_method_results_dir("heretic")
ANSWERS_DIR = BASELINE_RESULTS_DIR / "answers"
SELECTION_DIR = BASELINE_RESULTS_DIR / "selection"
ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
SELECTION_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _build_settings() -> Settings:
    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]] if sys.argv else ["script"]
        return Settings(
            model=MODEL_NAME,
            batch_size=MODEL_BATCH_SIZE,
            max_response_length=int(os.getenv("MAX_RESPONSE_LENGTH", "256")),
            system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful assistant."),
            n_trials=int(HERETIC_CONFIG["n_trials"]),
            n_startup_trials=int(HERETIC_CONFIG["n_startup_trials"]),
            kl_divergence_scale=float(HERETIC_CONFIG["kl_divergence_scale"]),
            refusal_markers=list(HERETIC_CONFIG["refusal_markers"]),
        )
    finally:
        sys.argv = original_argv


def _build_run_name(
    *,
    model_name: str,
    evaluation_backend: str,
    cfg: dict[str, Any],
    results_root: Path | None,
    timestamp_str: str,
) -> str:
    run_mode = resolve_run_mode(results_root)
    parts = [
        run_mode,
        f"model_{_sanitize_run_name_part(model_name)}",
        "heretic",
        f"mode_{cfg['mode']}",
        f"eval_{_sanitize_run_name_part(evaluation_backend)}",
        f"trials{int(cfg['n_trials'])}",
        f"startup{int(cfg['n_startup_trials'])}",
        f"klscale{cfg['kl_divergence_scale']:g}",
        f"kltarget{cfg['kl_divergence_target']:g}",
        f"refthr{int(cfg['refusal_threshold_count'])}",
        f"seed{int(cfg['seed'])}",
        timestamp_str,
    ]
    return "_".join(parts)


def _trial_logger(trial: optuna.Trial, payload: dict[str, Any]) -> None:
    scope = payload["scope"]
    _wandb_log(
        {
            f"heretic/trial/{trial.number}/refusals": payload["refusals"],
            f"heretic/trial/{trial.number}/kl_divergence": payload["kl_divergence"],
            f"heretic/trial/{trial.number}/kld_score": payload["kld_score"],
            f"heretic/trial/{trial.number}/refusal_objective": payload["refusal_objective"],
            f"heretic/trial/{trial.number}/scope_is_global": 1 if scope == "global" else 0,
        }
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
    metadata_file = output_dir / "heretic_best_model_metadata.json"
    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Best edited model metadata saved to: {metadata_file}")


def _maybe_filter(
    model: Model,
    selection_datasets: dict[str, list[str]],
    cfg: dict[str, Any],
    refusal_token_ids: list[int],
) -> None:
    if cfg["filter_train"]:
        selection_datasets["harmful_train"] = filter_instructions_by_refusal_sign(
            model, selection_datasets["harmful_train"], refusal_token_ids,
            keep_positive=True, label="harmful_train",
        )
        selection_datasets["harmless_train"] = filter_instructions_by_refusal_sign(
            model, selection_datasets["harmless_train"], refusal_token_ids,
            keep_positive=False, label="harmless_train",
        )
    if cfg["filter_val"]:
        selection_datasets["harmful_val"] = filter_instructions_by_refusal_sign(
            model, selection_datasets["harmful_val"], refusal_token_ids,
            keep_positive=True, label="harmful_val",
        )
        selection_datasets["harmless_val"] = filter_instructions_by_refusal_sign(
            model, selection_datasets["harmless_val"], refusal_token_ids,
            keep_positive=False, label="harmless_val",
        )


def _init_wandb(
    *,
    run_name: str,
    run_artifact_name: str,
    run_mode: str,
    cfg: dict[str, Any],
    dataset_sizes: dict[str, int],
    harmful_test_count: int,
    suffix_token_count: int,
    evaluation_metric: str,
) -> None:
    if not WANDB_AVAILABLE:
        return
    wandb_config = {
        "method": "heretic",
        "model": MODEL_NAME,
        "model_name": MODEL_NAME,
        "model_short_name": short_model_name(MODEL_NAME),
        "run_mode": run_mode,
        "evaluation_backend": os.getenv("EVALUATION_BACKEND", "wildguard"),
        "evaluation_metric": evaluation_metric,
        "heretic_config": cfg,
        "dataset_sizes": dataset_sizes,
        "harmful_test_question_count": harmful_test_count,
        "suffix_token_count": suffix_token_count,
        "run_artifact_name": run_artifact_name,
    }
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "anonymized"),
        name=run_name,
        job_type="heretic",
        group=f"heretic/{short_model_name(MODEL_NAME)}",
        tags=build_wandb_tags(
            [
                ("method", "heretic"),
                ("mode", cfg["mode"]),
                ("model", short_model_name(MODEL_NAME)),
                ("eval_backend", os.getenv("EVALUATION_BACKEND", "wildguard")),
                ("eval_metric", evaluation_metric),
                ("run_mode", run_mode),
            ]
        ),
        config={
            **wandb_config,
            **flatten_wandb_config(cfg, prefix="heretic"),
            **flatten_wandb_config(dataset_sizes, prefix="dataset_size"),
        },
    )


def main() -> None:
    print("=" * 80)
    print("STARTING EXPERIMENT: HERETIC (OPTUNA-TPE REFUSAL-DIRECTION SEARCH)")
    print("=" * 80)
    started_at = datetime.now()
    run_stamp = started_at.strftime("%Y%m%d_%H%M%S")
    cfg = dict(HERETIC_CONFIG)
    run_name = _build_run_name(
        model_name=MODEL_NAME,
        evaluation_backend=os.getenv("EVALUATION_BACKEND", "wildguard"),
        cfg=cfg,
        results_root=BASELINE_RESULTS_DIR,
        timestamp_str=run_stamp,
    )
    run_artifact_name = _shorten_run_name(run_name)
    print(f"Start time: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME}")
    print(f"Run artifact name: {run_artifact_name}")
    print(f"Heretic config: {cfg}")
    print()

    print("Loading selection datasets...")
    selection_datasets = load_selection_datasets(cfg)
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
    print(
        f"Using {len(suffix_token_ids)} end-of-instruction suffix tokens "
        f"for direction extraction"
    )

    _maybe_filter(model, selection_datasets, cfg, refusal_token_ids)

    if not selection_datasets["harmful_train"] or not selection_datasets["harmless_train"]:
        raise ValueError("heretic baseline requires non-empty harmful/harmless training samples.")
    if not selection_datasets["harmful_val"] or not selection_datasets["harmless_val"]:
        raise ValueError("heretic baseline requires non-empty harmful/harmless validation samples.")
    if not selection_datasets["harmful_test"]:
        raise ValueError("heretic baseline requires non-empty harmful test samples.")

    dataset_sizes = {name: len(d) for name, d in selection_datasets.items()}
    harmful_test_questions = list(selection_datasets["harmful_test"])
    harmful_val_prompts = list(selection_datasets["harmful_val"])
    harmless_val_prompts = list(selection_datasets["harmless_val"])

    run_mode = resolve_run_mode(BASELINE_RESULTS_DIR)
    _init_wandb(
        run_name=run_name,
        run_artifact_name=run_artifact_name,
        run_mode=run_mode,
        cfg=cfg,
        dataset_sizes=dataset_sizes,
        harmful_test_count=len(harmful_test_questions),
        suffix_token_count=len(suffix_token_ids),
        evaluation_metric=evaluation_metric,
    )

    print("\nGenerating clean responses for harmful test prompts...")
    original_responses = [
        extract_response_after_think(r)
        for r in model.get_responses_batched(harmful_test_questions)
    ]

    print("\nGenerating clean responses for harmful val prompts (base refusal count)...")
    # Use raw responses (no think-tag stripping) for the refusal classifier,
    # matching upstream Heretic and avoiding false-positive refusals on
    # truncated reasoning outputs (see code-review notes).
    base_val_responses = list(model.get_responses_batched(harmful_val_prompts))
    print("\nCaching base logprobs on harmless val prompts (KL reference)...")
    base_logprobs = model.get_logprobs_batched(harmless_val_prompts).detach().cpu()

    print("\nComputing candidate refusal directions...")
    mean_diffs = get_mean_diff(
        model,
        selection_datasets["harmful_train"],
        selection_datasets["harmless_train"],
        suffix_positions,
    )
    refusal_directions = build_refusal_directions_tensor(
        mean_diffs,
        orthogonalize=bool(cfg["orthogonalize_direction"]),
    )

    selection_run_dir = SELECTION_DIR / run_artifact_name
    selection_run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(refusal_directions, selection_run_dir / "refusal_directions.pt")
    torch.save(mean_diffs, selection_run_dir / "mean_diffs.pt")

    print(
        f"\nRunning Optuna TPE study with n_trials={cfg['n_trials']} "
        f"(startup={cfg['n_startup_trials']}, seed={cfg['seed']}) ..."
    )
    n_layers = len(model.get_layers())
    objective, base_refusals = make_objective(
        model,
        refusal_directions=refusal_directions,
        harmful_val=harmful_val_prompts,
        harmless_val=harmless_val_prompts,
        base_responses=base_val_responses,
        base_logprobs=base_logprobs,
        cfg=cfg,
        n_layers=n_layers,
        # Identity postprocess: pass raw responses to the refusal classifier
        # (upstream Heretic does not strip think tags inside the loop).
        response_postprocess=lambda r: r,
        on_trial=_trial_logger,
    )
    study = run_heretic_study(
        objective,
        n_trials=int(cfg["n_trials"]),
        n_startup_trials=int(cfg["n_startup_trials"]),
        seed=int(cfg["seed"]),
        study_dir=selection_run_dir,
        study_name=f"heretic_{run_artifact_name}",
    )
    print(f"Base refusals on harmful val: {base_refusals}/{len(harmful_val_prompts)}")

    pareto = extract_pareto_front(study)
    if not pareto:
        raise RuntimeError("Heretic study produced no completed trials.")
    print(f"\nPareto front size: {len(pareto)}")
    for trial in pareto:
        print(
            f"  trial #{trial.number}: "
            f"refusals={trial.user_attrs['refusals']}, "
            f"kl={trial.user_attrs['kl_divergence']:.4f}, "
            f"scope={trial.user_attrs['scope']}"
        )

    chosen_trial, selection_reason = auto_select_trial(
        pareto,
        refusal_threshold_count=int(cfg["refusal_threshold_count"]),
    )
    print(
        f"\nSelected trial #{chosen_trial.number} via {selection_reason}: "
        f"refusals={chosen_trial.user_attrs['refusals']}, "
        f"kl={chosen_trial.user_attrs['kl_divergence']:.4f}"
    )

    pareto_records = [trial_to_record(t) for t in pareto]
    (selection_run_dir / "pareto.json").write_text(
        json.dumps(pareto_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    selected_record = trial_to_record(
        chosen_trial,
        extra={
            "selection_reason": selection_reason,
            "base_refusals": int(base_refusals),
            "n_val_harmful": int(len(harmful_val_prompts)),
            "pareto_size": int(len(pareto)),
        },
    )
    (selection_run_dir / "selected.json").write_text(
        json.dumps(selected_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _wandb_log(
        {
            "heretic/selected/trial_number": chosen_trial.number,
            "heretic/selected/refusals": int(chosen_trial.user_attrs["refusals"]),
            "heretic/selected/kl_divergence": float(chosen_trial.user_attrs["kl_divergence"]),
            "heretic/selected/scope_is_global": 1 if chosen_trial.user_attrs["scope"] == "global" else 0,
            "heretic/selected/pareto_size": len(pareto),
            "heretic/selected/base_refusals": int(base_refusals),
        }
    )

    print("\nApplying chosen trial to model...")
    chosen_scope = str(chosen_trial.user_attrs["scope"])
    # For per-layer scope, direction_index_used is None and apply_trial ignores
    # the direction_index argument; for global scope it carries the float value
    # sampled during the trial.
    chosen_direction_index = chosen_trial.user_attrs.get("direction_index_used")
    chosen_params = reconstruct_params_from_trial(
        chosen_trial, model.get_abliterable_components()
    )
    apply_trial(
        model,
        refusal_directions,
        chosen_scope,
        float(chosen_direction_index) if chosen_direction_index is not None else 0.0,
        chosen_params,
    )

    print("Generating edited responses for harmful test prompts...")
    modified_responses = [
        extract_response_after_think(r)
        for r in model.get_responses_batched(harmful_test_questions)
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
        "method": "heretic",
        "category": "heretic",
        "evaluation_metric": evaluation_metric,
        "param_key": f"trial_{chosen_trial.number}_{chosen_scope.replace(' ', '_')}",
        "run_name": run_name,
        "run_artifact_name": run_artifact_name,
        "heretic_config": cfg,
        "selected_trial": selected_record,
        "pareto_summary": [
            {
                "trial_number": r["trial_number"],
                "refusals": r["refusals"],
                "kl_divergence": r["kl_divergence"],
            }
            for r in pareto_records
        ],
        "selected_direction_path": str(selection_run_dir / "refusal_directions.pt"),
        "selection_dir": _project_relative(selection_run_dir),
        "dataset_sizes": dataset_sizes,
        "timestamp": started_at.isoformat(),
    }
    experiment_info = {
        "category": experiment_config["category"],
        "hyperparameters": selected_record,
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
                "kind": "heretic_best_model",
                "model": MODEL_NAME,
                "base_model": MODEL_NAME,
                "method": "heretic",
                "mode": cfg["mode"],
                "evaluation_backend": os.getenv("EVALUATION_BACKEND", "wildguard"),
                "run_name": run_name,
                "run_artifact_name": run_artifact_name,
                "answers_file": _project_relative(answers_file),
                "results_root": _project_relative(BASELINE_RESULTS_DIR.parent),
                "method_results_dir": _project_relative(BASELINE_RESULTS_DIR),
                "selection_dir": _project_relative(selection_run_dir),
                "selected_trial": selected_record,
                "heretic_config": cfg,
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
