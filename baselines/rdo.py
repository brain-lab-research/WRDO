#!/usr/bin/env python3
"""
RDO baseline orchestrator.

Mirrors `baselines.basic_refusal` structurally but replaces the candidate-
direction grid search with gradient-based optimization (see
`baselines.rdo_optim`). All weight ablation goes through the local
`heretic.model.Model.abliterate` via `model_utils.apply_abliteration_with_hyperparams`.
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
import torch

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ABLITERATION_PARAMS,
    MODEL_BATCH_SIZE,
    MODEL_NAME,
    RDO_CONFIG,
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
from model_utils import apply_abliteration_with_hyperparams
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
    build_chat_prompts,
    build_refusal_token_ids,
    compute_eoi_suffix_token_ids,
    expand_direction_to_refusal_tensor,
    filter_instructions_by_refusal_sign,
    get_mean_diff,
    load_classifier_categories,
    load_selection_datasets,
    save_answers,
)
from baselines.rdo_optim import (
    build_rdo_dataset,
    generate_intervention_targets,
    installed_ablation_hooks,
    orthogonalize_embedding,
    pick_seed_direction,
    train_refusal_direction,
)

# `baselines.basic_refusal` runs `torch.set_grad_enabled(False)` at import
# time. RDO training needs grad enabled inside its `torch.enable_grad()`
# blocks; we also re-enable the global flag here so the model's internal
# `torch.is_grad_enabled()` checks during forward report True. Target
# generation and final evaluation use explicit `torch.no_grad()` contexts.
torch.set_grad_enabled(True)

logger = logging.getLogger(__name__)

BASELINE_RESULTS_DIR = get_method_results_dir("rdo")
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
            max_response_length=int(os.getenv("MAX_RESPONSE_LENGTH", "2048")),
            system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful assistant."),
        )
    finally:
        sys.argv = original_argv


def _build_run_name(
    *,
    model_name: str,
    evaluation_backend: str,
    abliteration_params: dict[str, Any],
    cfg: dict[str, Any],
    results_root: Path | None,
    timestamp_str: str,
) -> str:
    run_mode = resolve_run_mode(results_root)
    parts = [
        run_mode,
        f"model_{_sanitize_run_name_part(model_name)}",
        "rdo",
        f"mode_{cfg['mode']}",
        f"apply_{cfg['apply_method']}",
        f"emb_{'on' if cfg['ablate_embedding'] else 'off'}",
        f"eval_{_sanitize_run_name_part(evaluation_backend)}",
        f"abl_max{abliteration_params['max_weight']:g}",
        f"min{abliteration_params['min_weight']:g}",
        f"pos{abliteration_params['max_weight_position']:g}",
        f"dist{abliteration_params['min_weight_distance']:g}",
        f"lr{cfg['lr']:g}",
        f"effbs{int(cfg['effective_batch_size'])}",
        f"ep{int(cfg['epochs'])}",
        f"abl{cfg['ablation_lambda']:g}_add{cfg['addition_lambda']:g}_ret{cfg['retain_lambda']:g}",
        f"seed{int(cfg['seed'])}",
        timestamp_str,
    ]
    return "_".join(parts)


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
    metadata_file = output_dir / "rdo_best_model_metadata.json"
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
    abliteration_params: dict[str, Any],
    dataset_sizes: dict[str, int],
    harmful_test_count: int,
    suffix_token_count: int,
    evaluation_metric: str,
) -> None:
    if not WANDB_AVAILABLE:
        return
    wandb_config = {
        "method": "rdo",
        "model": MODEL_NAME,
        "model_name": MODEL_NAME,
        "model_short_name": short_model_name(MODEL_NAME),
        "run_mode": run_mode,
        "evaluation_backend": os.getenv("EVALUATION_BACKEND", "wildguard"),
        "evaluation_metric": evaluation_metric,
        "rdo_config": cfg,
        "abliteration_params": abliteration_params,
        "dataset_sizes": dataset_sizes,
        "harmful_test_question_count": harmful_test_count,
        "suffix_token_count": suffix_token_count,
        "run_artifact_name": run_artifact_name,
    }
    apply_method = str(cfg["apply_method"])
    ablate_embedding = bool(cfg["ablate_embedding"])
    # Prefix the run name with the apply method so it's the FIRST thing visible
    # in the wandb runs list, even if the rest of the name gets truncated.
    emb_suffix = "_emb" if (apply_method == "weights" and ablate_embedding) else ""
    wandb_run_name = f"[rdo-{apply_method}{emb_suffix}] {run_name}"
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "anonymized"),
        name=wandb_run_name,
        # job_type and group also encode apply_method so the wandb UI can
        # filter / sub-group "activation" vs "weights" runs separately.
        job_type=f"rdo-{apply_method}",
        group=f"rdo/{apply_method}/{short_model_name(MODEL_NAME)}",
        tags=build_wandb_tags(
            [
                ("method", "rdo"),
                ("apply", apply_method),
                ("emb", "on" if ablate_embedding else "off"),
                ("mode", cfg["mode"]),
                ("model", short_model_name(MODEL_NAME)),
                ("eval_backend", os.getenv("EVALUATION_BACKEND", "wildguard")),
                ("eval_metric", evaluation_metric),
                ("run_mode", run_mode),
            ]
        ),
        config={
            **wandb_config,
            **flatten_wandb_config(cfg, prefix="rdo"),
            **flatten_wandb_config(abliteration_params, prefix="abl"),
            **flatten_wandb_config(dataset_sizes, prefix="dataset_size"),
        },
    )


def _step_logger(step: int, payload: dict[str, float]) -> None:
    if not WANDB_AVAILABLE:
        return
    _wandb_log({f"rdo/step/{step}/{k}": v for k, v in payload.items() if k != "step"})


def main() -> None:
    print("=" * 80)
    print("STARTING EXPERIMENT: RDO (GRADIENT-BASED REFUSAL-DIRECTION OPTIMIZATION)")
    print("=" * 80)
    started_at = datetime.now()
    run_stamp = started_at.strftime("%Y%m%d_%H%M%S")
    cfg = dict(RDO_CONFIG)
    abliteration_params = dict(ABLITERATION_PARAMS)
    run_name = _build_run_name(
        model_name=MODEL_NAME,
        evaluation_backend=os.getenv("EVALUATION_BACKEND", "wildguard"),
        abliteration_params=abliteration_params,
        cfg=cfg,
        results_root=BASELINE_RESULTS_DIR,
        timestamp_str=run_stamp,
    )
    run_artifact_name = _shorten_run_name(run_name)
    print(f"Start time: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME}")
    print(f"Run artifact name: {run_artifact_name}")
    print(f"RDO config: {cfg}")
    print(f"Abliteration params: {abliteration_params}")
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
    suffix_token_labels = model.tokenizer.batch_decode(
        [[token_id] for token_id in suffix_token_ids]
    )
    print(f"Using {len(suffix_token_ids)} end-of-instruction suffix tokens")

    _maybe_filter(model, selection_datasets, cfg, refusal_token_ids)

    if not selection_datasets["harmful_train"] or not selection_datasets["harmless_train"]:
        raise ValueError("rdo baseline requires non-empty harmful/harmless training samples.")
    if not selection_datasets["harmful_val"] or not selection_datasets["harmless_val"]:
        raise ValueError("rdo baseline requires non-empty harmful/harmless validation samples.")
    if not selection_datasets["harmful_test"]:
        raise ValueError("rdo baseline requires non-empty harmful test samples.")

    dataset_sizes = {name: len(d) for name, d in selection_datasets.items()}
    harmful_test_questions = list(selection_datasets["harmful_test"])

    run_mode = resolve_run_mode(BASELINE_RESULTS_DIR)
    _init_wandb(
        run_name=run_name,
        run_artifact_name=run_artifact_name,
        run_mode=run_mode,
        cfg=cfg,
        abliteration_params=abliteration_params,
        dataset_sizes=dataset_sizes,
        harmful_test_count=len(harmful_test_questions),
        suffix_token_count=len(suffix_token_ids),
        evaluation_metric=evaluation_metric,
    )

    selection_run_dir = SELECTION_DIR / run_artifact_name
    selection_run_dir.mkdir(parents=True, exist_ok=True)

    # ----- Clean responses on harmful test (for the delta vs edited model) -----
    print("\nGenerating clean responses for harmful test prompts...")
    with torch.no_grad():
        original_responses = [
            extract_response_after_think(r)
            for r in model.get_responses_batched(harmful_test_questions)
        ]

    # ----- Mean diffs + seed direction selection (basic_refusal helper) -----
    print("\nComputing candidate mean diffs...")
    with torch.no_grad():
        mean_diffs = get_mean_diff(
            model,
            selection_datasets["harmful_train"],
            selection_datasets["harmless_train"],
            suffix_positions,
        )
    print("\nPicking seed direction via basic_refusal.select_direction...")
    v_seed, best_layer, alpha, selection_metadata, selection_artifacts = pick_seed_direction(
        model,
        mean_diffs=mean_diffs,
        harmful_val=selection_datasets["harmful_val"],
        harmless_val=selection_datasets["harmless_val"],
        token_positions=suffix_positions,
        token_labels=suffix_token_labels,
        refusal_token_ids=refusal_token_ids,
        selection_config=cfg,
        abliteration_params=abliteration_params,
        selection_run_dir=selection_run_dir,
    )
    print(
        f"Seed: layer={best_layer}, "
        f"pos={selection_metadata['selected_position']}, alpha={alpha:.4f}, "
        f"refusal={selection_metadata['selected_refusal_score']:.4f}, "
        f"kl={selection_metadata['selected_kl_div_score']:.4f}"
    )
    torch.save(v_seed.cpu(), selection_run_dir / "v_seed.pt")
    _wandb_log({
        "rdo/seed/best_layer": best_layer,
        "rdo/seed/best_pos": int(selection_metadata["selected_position"]),
        "rdo/seed/alpha": alpha,
        "rdo/seed/refusal_score": selection_metadata["selected_refusal_score"],
        "rdo/seed/steering_score": selection_metadata["selected_steering_score"],
        "rdo/seed/kl_div_score": selection_metadata["selected_kl_div_score"],
    })

    # ----- Reload base model after select_direction's abliteration test -----
    # `select_direction` mutated the weights while scoring candidates; restore
    # the original model before generating targets and training.
    print("\nReloading base model after seed-selection scoring...")
    model.reload_model()

    # ----- Chat-format prompts for target generation + training -----
    harmful_train_chat = build_chat_prompts(model, selection_datasets["harmful_train"])
    harmless_train_chat = build_chat_prompts(model, selection_datasets["harmless_train"])
    # Match lengths (upstream truncates harmless to len(harmful)).
    pair_len = min(len(harmful_train_chat), len(harmless_train_chat))
    harmful_train_chat = harmful_train_chat[:pair_len]
    harmless_train_chat = harmless_train_chat[:pair_len]
    print(f"Paired harmful/harmless chat prompts: {pair_len}")

    # ----- Generate three target sets (one-shot) -----
    print("\nGenerating intervention targets (ablation / addition / retain)...")
    tgt_bs = int(cfg["target_generation_batch_size"])
    n_tgt = int(cfg["num_target_tokens"])
    ablation_targets = generate_intervention_targets(
        model, harmful_train_chat, kind="ablation",
        v_seed=v_seed, alpha=alpha, best_layer=best_layer,
        num_target_tokens=n_tgt, batch_size=tgt_bs,
    )
    addition_targets = generate_intervention_targets(
        model, harmless_train_chat, kind="addition",
        v_seed=v_seed, alpha=alpha, best_layer=best_layer,
        num_target_tokens=n_tgt, batch_size=tgt_bs,
    )
    retain_targets = generate_intervention_targets(
        model, harmless_train_chat, kind="retain",
        v_seed=v_seed, alpha=alpha, best_layer=best_layer,
        num_target_tokens=n_tgt, batch_size=tgt_bs,
    )

    # ----- Build training dataset + train v -----
    print("\nBuilding RDO training dataset...")
    train_dataset = build_rdo_dataset(
        model.tokenizer,
        harmful_chat_prompts=harmful_train_chat,
        harmless_chat_prompts=harmless_train_chat,
        ablation_targets=ablation_targets,
        addition_targets=addition_targets,
        retain_targets=retain_targets,
    )
    print(f"Train dataset size: {len(train_dataset)}")

    print("\nTraining refusal direction (RDO)...")
    training_result = train_refusal_direction(
        model,
        v_seed=v_seed,
        alpha=alpha,
        best_layer=best_layer,
        train_dataset=train_dataset,
        cfg=cfg,
        on_step=_step_logger,
    )
    v_final = training_result.best_v
    torch.save(v_final, selection_run_dir / "v_final.pt")
    (selection_run_dir / "train_losses.json").write_text(
        json.dumps({
            "train_losses": training_result.train_losses,
            "step_logs": training_result.step_logs,
            "lowest_loss": training_result.lowest_loss,
        }, indent=2),
        encoding="utf-8",
    )
    print(
        f"RDO done. Lowest loss: {training_result.lowest_loss:.4f}, "
        f"steps: {len(training_result.train_losses)}"
    )
    _wandb_log({
        "rdo/training/lowest_loss": training_result.lowest_loss,
        "rdo/training/num_steps": len(training_result.train_losses),
    })

    # ----- Apply v_final to the model -----
    # Two paths, controlled by RDO_APPLY_METHOD:
    #   "weights":    triangular-kernel weight surgery on attn.o_proj +
    #                 mlp.down_proj (heretic.Model.abliterate). Optional
    #                 embedding orthogonalization closes the Arditi gap.
    #   "activation": runtime hooks at 3 sites/layer with uniform coefficient
    #                 = ABLITERATION_MAX_WEIGHT. Matches upstream RDO's
    #                 intervene_with_fn_vector_ablation exactly.
    apply_method = str(cfg["apply_method"])
    ablate_emb = bool(cfg["ablate_embedding"])

    print(f"\nReloading base model (apply_method='{apply_method}')...")
    model.reload_model()
    n_layers = len(model.get_layers())

    if ablate_emb:
        if apply_method == "weights":
            print("Orthogonalizing input token embedding against v_final...")
            vocab_size = orthogonalize_embedding(model, v_final.to(model.model.device))
            print(f"Embedding orthogonalized ({vocab_size} rows).")
        else:
            print(
                "RDO_ABLATE_EMBEDDING=true is redundant in activation mode "
                "(layer-0 input hook already strips embedding-derived v); "
                "skipping."
            )

    if apply_method == "weights":
        print("Applying trained direction via weight surgery...")
        refusal_directions = expand_direction_to_refusal_tensor(v_final, n_layers)
        apply_abliteration_with_hyperparams(
            model,
            refusal_directions,
            float(abliteration_params["max_weight"]),
            float(abliteration_params["max_weight_position"]),
            float(abliteration_params["min_weight"]),
            float(abliteration_params["min_weight_distance"]),
            n_layers,
        )

        print("\nGenerating edited responses for harmful test prompts...")
        with torch.no_grad():
            modified_responses = [
                extract_response_after_think(r)
                for r in model.get_responses_batched(harmful_test_questions)
            ]
    else:
        # Activation mode: hooks live for the duration of test generation.
        # Uniform coefficient = ABLITERATION_MAX_WEIGHT (no triangular kernel,
        # no per-layer windowing). This is the "honest" Arditi/RDO-upstream
        # comparison.
        activation_coefficient = float(abliteration_params["max_weight"])
        print(
            f"Applying trained direction via runtime activation hooks "
            f"(coefficient={activation_coefficient:g}, all {n_layers} layers, "
            f"3 sites/layer)..."
        )
        v_buf = v_final.detach().to(model.model.device, dtype=torch.float32)
        layers = list(model.get_layers())
        with torch.no_grad():
            with installed_ablation_hooks(
                layers, v_buf, coefficient=activation_coefficient,
            ):
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

    _wandb_log({
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
    })

    experiment_config = {
        "method": "rdo",
        "category": "rdo",
        "evaluation_metric": evaluation_metric,
        "apply_method": apply_method,
        "ablate_embedding": ablate_emb,
        "param_key": f"rdo_layer_{best_layer}_alpha_{alpha:.3f}_apply_{apply_method}",
        "run_name": run_name,
        "run_artifact_name": run_artifact_name,
        "hyperparameters": dict(abliteration_params),
        "abliteration_params": dict(abliteration_params),
        "rdo_config": cfg,
        "selection_metadata": selection_metadata,
        "selection_artifacts": selection_artifacts,
        "rdo_seed": {
            "best_layer": best_layer,
            "best_pos": int(selection_metadata["selected_position"]),
            "alpha": alpha,
        },
        "rdo_training": {
            "lowest_loss": training_result.lowest_loss,
            "num_steps": len(training_result.train_losses),
            "v_final_path": str(selection_run_dir / "v_final.pt"),
            "v_seed_path": str(selection_run_dir / "v_seed.pt"),
        },
        "dataset_sizes": dataset_sizes,
        "selection_dir": _project_relative(selection_run_dir),
        "timestamp": started_at.isoformat(),
    }
    experiment_info = {
        "category": experiment_config["category"],
        "hyperparameters": experiment_config["hyperparameters"],
        "param_key": experiment_config["param_key"],
        "timestamp": experiment_config["timestamp"],
        "run_name": run_name,
        "run_artifact_name": run_artifact_name,
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
    # If we were in activation mode, the in-memory model is currently base
    # (hooks were torn down). To save a useful HF artifact we have to BAKE the
    # trained direction into weights via the same heretic.Model.abliterate
    # used by basic_refusal / heretic / weighted_rd, so the saved checkpoint
    # is a regular HF model that can be loaded and used without hooks.
    #
    # For activation eval, we ALSO orthogonalize the embedding regardless of
    # `ablate_emb` — the eval's layer-0 hook implicitly scrubs the
    # embedding's v-component, and we want the saved weight image to
    # approximate that behavior (Arditi (2024) §5.2 equivalence).
    saved_via_weight_bake = False
    saved_with_embedding_ortho = bool(ablate_emb)
    if save_best_model and apply_method == "activation":
        print(
            "\nBaking trained direction into weights for HF checkpoint "
            "(activation mode used hooks during eval; we apply via "
            "Model.abliterate + embedding orthogonalization for the saved "
            "artifact)..."
        )
        model.reload_model()
        vocab_size = orthogonalize_embedding(model, v_final.to(model.model.device))
        print(f"Embedding orthogonalized ({vocab_size} rows).")
        refusal_directions = expand_direction_to_refusal_tensor(v_final, n_layers)
        apply_abliteration_with_hyperparams(
            model,
            refusal_directions,
            float(abliteration_params["max_weight"]),
            float(abliteration_params["max_weight_position"]),
            float(abliteration_params["min_weight"]),
            float(abliteration_params["min_weight_distance"]),
            n_layers,
        )
        saved_via_weight_bake = True
        saved_with_embedding_ortho = True

    answers_data["experiment_config"]["best_model_saved"] = save_best_model
    answers_data["experiment_config"]["best_model_dir"] = (
        _project_relative(best_model_dir) if save_best_model else None
    )
    answers_data["experiment_config"]["best_model_saved_via_weight_bake"] = (
        saved_via_weight_bake
    )
    answers_data["experiment_config"]["best_model_saved_with_embedding_ortho"] = (
        saved_with_embedding_ortho
    )
    answers_data["experiment_config"]["answers_file"] = _project_relative(answers_file)
    save_answers(answers_file, answers_data)

    if save_best_model:
        _save_best_model_checkpoint(
            model=model,
            output_dir=best_model_dir,
            max_shard_size=best_model_max_shard_size,
            metadata={
                "kind": "rdo_best_model",
                "model": MODEL_NAME,
                "base_model": MODEL_NAME,
                "method": "rdo",
                "mode": cfg["mode"],
                "apply_method": apply_method,
                "ablate_embedding": ablate_emb,
                "saved_via_weight_bake": saved_via_weight_bake,
                "saved_with_embedding_ortho": saved_with_embedding_ortho,
                "evaluation_backend": os.getenv("EVALUATION_BACKEND", "wildguard"),
                "run_name": run_name,
                "run_artifact_name": run_artifact_name,
                "answers_file": _project_relative(answers_file),
                "results_root": _project_relative(BASELINE_RESULTS_DIR.parent),
                "method_results_dir": _project_relative(BASELINE_RESULTS_DIR),
                "selection_dir": _project_relative(selection_run_dir),
                "abliteration_params": dict(abliteration_params),
                "rdo_config": cfg,
                "max_shard_size": best_model_max_shard_size,
                "timestamp": run_stamp,
            },
        )
    else:
        print("Skipping best edited model checkpoint save.")

    _wandb_log({
        "artifacts/answers_file": _project_relative(answers_file),
        "artifacts/selection_dir": _project_relative(selection_run_dir),
        "artifacts/best_model_saved": save_best_model,
        "artifacts/best_model_dir": _project_relative(best_model_dir) if save_best_model else None,
        "artifacts/best_model_saved_via_weight_bake": saved_via_weight_bake,
        "artifacts/best_model_saved_with_embedding_ortho": saved_with_embedding_ortho,
    })

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
