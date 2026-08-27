"""
Optuna-based weight search for weighted_rd.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import optuna
import torch
from optuna.samplers import (
    CmaEsSampler,
    GPSampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)

from data_utils import extract_response_after_think
from model_utils import LearnableDirectionWeights, apply_abliteration_with_hyperparams

from baselines.weighted_rd.kl import compute_masked_mean_kl
from baselines.weighted_rd.log_probs import compute_sequence_log_probs
from baselines.weighted_rd.reward import compute_reward
from baselines.weighted_rd.runtime_config import is_weighted_rd_unsafe_rate_metric


def create_optuna_sampler(sampler_name: str, sampler_seed: int) -> optuna.samplers.BaseSampler:
    """Build a supported Optuna sampler from a short config name."""
    if sampler_name == "tpe":
        return TPESampler(seed=sampler_seed)
    if sampler_name == "random":
        return RandomSampler(seed=sampler_seed)
    if sampler_name == "gp":
        return GPSampler(seed=sampler_seed)
    if sampler_name == "cmaes":
        try:
            return CmaEsSampler(seed=sampler_seed)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Optuna sampler 'cmaes' requires the optional Python package 'cmaes'. "
                "Install it with `pip install cmaes` or switch OPTUNA_SAMPLER to "
                "'tpe', 'random', 'gp', or 'qmc'."
            ) from exc
    if sampler_name == "qmc":
        return QMCSampler(seed=sampler_seed)
    raise ValueError(
        "Unsupported Optuna sampler "
        f"'{sampler_name}'. Expected one of: tpe, random, gp, cmaes, qmc."
    )


def _suggest_scalar_weights(
    trial: optuna.Trial,
    n_directions: int,
    weight_min: float,
    weight_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = [
        trial.suggest_float(f"weight_{direction_idx}", weight_min, weight_max)
        for direction_idx in range(n_directions)
    ]
    return torch.tensor(values, device=device, dtype=dtype)


def _suggest_dense_weights(
    trial: optuna.Trial,
    n_directions: int,
    n_layers: int,
    hidden_size: int,
    weight_min: float,
    weight_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = []
    for direction_idx in range(n_directions):
        direction_values = []
        for layer_idx in range(n_layers + 1):
            layer_values = [
                trial.suggest_float(
                    f"weight_{direction_idx}_{layer_idx}_{hidden_idx}",
                    weight_min,
                    weight_max,
                )
                for hidden_idx in range(hidden_size)
            ]
            direction_values.append(layer_values)
        values.append(direction_values)
    return torch.tensor(values, device=device, dtype=dtype)


def suggest_trial_weights(
    trial: optuna.Trial,
    direction_weights: LearnableDirectionWeights,
    weight_min: float,
    weight_max: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Suggest a trial weight tensor matching the active parameterization mode."""
    if direction_weights.mode == "scalar":
        return _suggest_scalar_weights(
            trial=trial,
            n_directions=direction_weights.n_directions,
            weight_min=weight_min,
            weight_max=weight_max,
            device=device,
            dtype=dtype,
        )

    if direction_weights.mode == "dense":
        return _suggest_dense_weights(
            trial=trial,
            n_directions=direction_weights.n_directions,
            n_layers=direction_weights.n_layers,
            hidden_size=direction_weights.hidden_size,
            weight_min=weight_min,
            weight_max=weight_max,
            device=device,
            dtype=dtype,
        )

    raise ValueError(
        f"Unsupported LearnableDirectionWeights mode '{direction_weights.mode}' for Optuna search."
    )


def reconstruct_weights_from_params(
    params: Dict[str, float],
    direction_weights: LearnableDirectionWeights,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Rebuild a scalar vector or dense tensor from Optuna best-trial params."""
    if direction_weights.mode == "scalar":
        return torch.tensor(
            [
                params[f"weight_{direction_idx}"]
                for direction_idx in range(direction_weights.n_directions)
            ],
            device=device,
            dtype=dtype,
        )

    if direction_weights.mode == "dense":
        values = []
        for direction_idx in range(direction_weights.n_directions):
            direction_values = []
            for layer_idx in range(direction_weights.n_layers + 1):
                layer_values = [
                    params[f"weight_{direction_idx}_{layer_idx}_{hidden_idx}"]
                    for hidden_idx in range(direction_weights.hidden_size)
                ]
                direction_values.append(layer_values)
            values.append(direction_values)
        return torch.tensor(values, device=device, dtype=dtype)

    raise ValueError(
        f"Unsupported LearnableDirectionWeights mode '{direction_weights.mode}' for Optuna search."
    )


def summarize_weights(weights: torch.Tensor) -> Dict[str, Any]:
    """Compact stats for dense Optuna history and lightweight diagnostics."""
    detached = weights.detach()
    return {
        "weights_shape": list(detached.shape),
        "weights_mean": float(detached.mean().item()),
        "weights_std": float(detached.std(unbiased=False).item()),
        "weights_min": float(detached.min().item()),
        "weights_max": float(detached.max().item()),
        "weights_norm": float(detached.norm().item()),
    }


def _extract_generated_responses(responses_raw: List[str]) -> List[str]:
    responses = []
    for raw in responses_raw:
        extracted = extract_response_after_think(raw)
        responses.append(extracted if extracted else raw.strip())
    return responses


def _compute_mean_kl_from_log_probs(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_probs: torch.Tensor,
    ref_response_mask: torch.Tensor,
    loss_agg_mode: str,
    batch_name: str,
) -> float:
    if tuple(ref_response_mask.shape) != tuple(response_mask.shape):
        raise ValueError(
            f"{batch_name} reference-policy response mask shape does not match "
            f"modified-policy mask shape: {tuple(ref_response_mask.shape)} vs "
            f"{tuple(response_mask.shape)}."
        )

    _, mean_kl_tensor = compute_masked_mean_kl(
        log_prob=policy_log_probs,
        ref_log_prob=ref_log_probs,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )
    return float(mean_kl_tensor.item())


def evaluate_weights(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    weights: torch.Tensor,
    model,
    questions: List[str],
    abliteration_params: Dict[str, float],
    classifier_categories: List[Dict[str, Any]],
    n_layers: int,
    ref_alpha: float,
    reward_sign: float = 1.0,
    beta_1: float = 0.0,
    beta_2: float = 1000.0,
    loss_agg_mode: str = "token-mean",
    backend: Optional[str] = None,
    harmless_questions: Optional[List[str]] = None,
    reward_metric: str = "harmfulness",
) -> Dict[str, Any]:
    """Evaluate a scalar vector or dense tensor of coefficients on the current question batch."""
    use_harmful_kl = float(beta_1) != 0.0
    use_harmless_kl = float(beta_2) != 0.0

    with torch.no_grad():
        combined_direction = direction_weights.combine_with_weights(
            [direction.detach() for direction in extracted_directions],
            weights.detach(),
        )

    model.reload_model()
    apply_abliteration_with_hyperparams(
        model,
        combined_direction,
        abliteration_params["max_weight"] * ref_alpha,
        abliteration_params["max_weight_position"],
        abliteration_params["min_weight"] * ref_alpha,
        abliteration_params["min_weight_distance"],
        n_layers,
    )

    responses_raw = model.get_responses_batched(questions)
    responses = _extract_generated_responses(responses_raw)

    harmless_questions = [
        question.strip()
        for question in (harmless_questions or [])
        if isinstance(question, str) and question.strip()
    ]
    if use_harmless_kl and not harmless_questions:
        raise ValueError("beta_2 is non-zero, but no harmless questions were provided for KL regularization.")

    harmless_responses: List[str] = []
    if use_harmless_kl:
        harmless_responses_raw = model.get_responses_batched(harmless_questions)
        harmless_responses = _extract_generated_responses(harmless_responses_raw)

    with torch.no_grad():
        policy_log_probs = None
        response_mask = None
        harmless_policy_log_probs = None
        harmless_response_mask = None
        if use_harmful_kl:
            policy_log_probs, response_mask = compute_sequence_log_probs(model, questions, responses)
        if use_harmless_kl:
            harmless_policy_log_probs, harmless_response_mask = compute_sequence_log_probs(
                model,
                harmless_questions,
                harmless_responses,
            )

    model.reload_model()
    with torch.no_grad():
        ref_log_probs = None
        ref_response_mask = None
        harmless_ref_log_probs = None
        harmless_ref_response_mask = None
        if use_harmful_kl:
            ref_log_probs, ref_response_mask = compute_sequence_log_probs(model, questions, responses)
        if use_harmless_kl:
            harmless_ref_log_probs, harmless_ref_response_mask = compute_sequence_log_probs(
                model,
                harmless_questions,
                harmless_responses,
            )

    harmful_mean_kl = 0.0
    if use_harmful_kl:
        harmful_mean_kl = _compute_mean_kl_from_log_probs(
            policy_log_probs=policy_log_probs,
            response_mask=response_mask,
            ref_log_probs=ref_log_probs,
            ref_response_mask=ref_response_mask,
            loss_agg_mode=loss_agg_mode,
            batch_name="Harmful",
        )
    harmless_mean_kl = 0.0
    if use_harmless_kl:
        harmless_mean_kl = _compute_mean_kl_from_log_probs(
            policy_log_probs=harmless_policy_log_probs,
            response_mask=harmless_response_mask,
            ref_log_probs=harmless_ref_log_probs,
            ref_response_mask=harmless_ref_response_mask,
            loss_agg_mode=loss_agg_mode,
            batch_name="Harmless",
        )

    harmfulness_scores = compute_reward(
        questions,
        responses,
        classifier_categories,
        backend,
        reward_metric=reward_metric,
    )
    rewards = [float(score) * float(reward_sign) for score in harmfulness_scores]

    mean_harmfulness = float(np.mean(harmfulness_scores)) if harmfulness_scores else 0.0
    best_harmfulness = float(np.max(harmfulness_scores)) if harmfulness_scores else 0.0
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    best_reward = float(np.max(rewards)) if rewards else 0.0
    mean_kl = harmful_mean_kl + harmless_mean_kl
    harmful_kl_penalty = float(beta_1) * harmful_mean_kl
    harmless_kl_penalty = float(beta_2) * harmless_mean_kl
    kl_penalty = harmful_kl_penalty + harmless_kl_penalty
    mean_objective = float(mean_reward - kl_penalty)
    result = {
        "weights": weights.detach().cpu().tolist(),
        "responses": responses,
        "scores": [float(score) for score in harmfulness_scores],
        "mean_reward": mean_reward,
        "best_reward": best_reward,
        "mean_harmfulness": mean_harmfulness,
        "best_harmfulness": best_harmfulness,
        "mean_kl": mean_kl,
        "harmful_mean_kl": harmful_mean_kl,
        "harmless_mean_kl": harmless_mean_kl,
        "kl_loss": mean_kl,
        "harmful_kl_penalty": harmful_kl_penalty,
        "harmless_kl_penalty": harmless_kl_penalty,
        "kl_penalty": kl_penalty,
        "mean_objective": mean_objective,
        "beta_1": float(beta_1),
        "beta_2": float(beta_2),
        "n_questions": len(questions),
        "harmless_n_questions": len(harmless_questions) if use_harmless_kl else 0,
        "weights_mode": direction_weights.mode,
    }
    if is_weighted_rd_unsafe_rate_metric(reward_metric):
        result["mean_unsafe_rate"] = mean_harmfulness
        result["best_unsafe_rate"] = best_harmfulness
    return result


def evaluate_scalar_weights(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    scalar_weights: torch.Tensor,
    model,
    questions: List[str],
    abliteration_params: Dict[str, float],
    classifier_categories: List[Dict[str, Any]],
    n_layers: int,
    ref_alpha: float,
    reward_sign: float = 1.0,
    beta_1: float = 0.0,
    beta_2: float = 1000.0,
    loss_agg_mode: str = "token-mean",
    backend: Optional[str] = None,
    harmless_questions: Optional[List[str]] = None,
    reward_metric: str = "harmfulness",
) -> Dict[str, Any]:
    """Backward-compatible wrapper for scalar Optuna evaluation."""
    return evaluate_weights(
        direction_weights=direction_weights,
        extracted_directions=extracted_directions,
        weights=scalar_weights,
        model=model,
        questions=questions,
        abliteration_params=abliteration_params,
        classifier_categories=classifier_categories,
        n_layers=n_layers,
        ref_alpha=ref_alpha,
        reward_sign=reward_sign,
        beta_1=beta_1,
        beta_2=beta_2,
        loss_agg_mode=loss_agg_mode,
        backend=backend,
        harmless_questions=harmless_questions,
        reward_metric=reward_metric,
    )


def evaluate_dense_weights(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    dense_weights: torch.Tensor,
    model,
    questions: List[str],
    abliteration_params: Dict[str, float],
    classifier_categories: List[Dict[str, Any]],
    n_layers: int,
    ref_alpha: float,
    reward_sign: float = 1.0,
    beta_1: float = 0.0,
    beta_2: float = 1000.0,
    loss_agg_mode: str = "token-mean",
    backend: Optional[str] = None,
    harmless_questions: Optional[List[str]] = None,
    reward_metric: str = "harmfulness",
) -> Dict[str, Any]:
    """Convenience wrapper for dense Optuna evaluation."""
    return evaluate_weights(
        direction_weights=direction_weights,
        extracted_directions=extracted_directions,
        weights=dense_weights,
        model=model,
        questions=questions,
        abliteration_params=abliteration_params,
        classifier_categories=classifier_categories,
        n_layers=n_layers,
        ref_alpha=ref_alpha,
        reward_sign=reward_sign,
        beta_1=beta_1,
        beta_2=beta_2,
        loss_agg_mode=loss_agg_mode,
        backend=backend,
        harmless_questions=harmless_questions,
        reward_metric=reward_metric,
    )


def optimize_weights_with_optuna(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    model,
    questions: Optional[List[str]],
    abliteration_params: Dict[str, float],
    classifier_categories: List[Dict[str, Any]],
    n_layers: int,
    ref_alpha: float,
    reward_sign: float,
    beta_1: float,
    beta_2: float,
    n_trials: int,
    sampler_name: str,
    sampler_seed: int,
    weight_min: float,
    weight_max: float,
    question_sampler: Optional[Callable[[], List[str]]] = None,
    harmless_questions: Optional[List[str]] = None,
    harmless_question_sampler: Optional[Callable[[], List[str]]] = None,
    loss_agg_mode: str = "token-mean",
    backend: Optional[str] = None,
    reward_metric: str = "harmfulness",
) -> Dict[str, Any]:
    """Search scalar or dense direction weights with Optuna and update the module in-place."""
    device = direction_weights.weights.device
    dtype = direction_weights.weights.dtype
    weights_mode = direction_weights.mode
    fallback_questions = list(questions) if questions is not None else None
    fallback_harmless_questions = (
        list(harmless_questions)
        if harmless_questions is not None
        else []
    )
    use_harmless_kl = float(beta_2) != 0.0

    if question_sampler is None and fallback_questions is None:
        raise ValueError("Optuna optimization requires either `questions` or `question_sampler`.")
    if use_harmless_kl and harmless_question_sampler is None and not fallback_harmless_questions:
        raise ValueError("beta_2 is non-zero, but no harmless questions were provided for Optuna KL regularization.")

    def objective(trial: optuna.Trial) -> float:
        trial_questions = (
            list(question_sampler())
            if question_sampler is not None
            else list(fallback_questions)
        )
        trial_harmless_questions = []
        if use_harmless_kl:
            trial_harmless_questions = (
                list(harmless_question_sampler())
                if harmless_question_sampler is not None
                else list(fallback_harmless_questions)
            )
        trial_weights = suggest_trial_weights(
            trial=trial,
            direction_weights=direction_weights,
            weight_min=weight_min,
            weight_max=weight_max,
            device=device,
            dtype=dtype,
        )
        trial_result = evaluate_weights(
            direction_weights=direction_weights,
            extracted_directions=extracted_directions,
            weights=trial_weights,
            model=model,
            questions=trial_questions,
            abliteration_params=abliteration_params,
            classifier_categories=classifier_categories,
            n_layers=n_layers,
            ref_alpha=ref_alpha,
            reward_sign=reward_sign,
            beta_1=beta_1,
            beta_2=beta_2,
            loss_agg_mode=loss_agg_mode,
            backend=backend,
            harmless_questions=trial_harmless_questions,
            reward_metric=reward_metric,
        )

        trial.set_user_attr("weights_mode", weights_mode)
        trial.set_user_attr("mean_reward", trial_result["mean_reward"])
        trial.set_user_attr("best_reward", trial_result["best_reward"])
        trial.set_user_attr("mean_harmfulness", trial_result["mean_harmfulness"])
        trial.set_user_attr("best_harmfulness", trial_result["best_harmfulness"])
        if is_weighted_rd_unsafe_rate_metric(reward_metric):
            trial.set_user_attr("mean_unsafe_rate", trial_result.get("mean_unsafe_rate"))
            trial.set_user_attr("best_unsafe_rate", trial_result.get("best_unsafe_rate"))
        trial.set_user_attr("mean_kl", trial_result["mean_kl"])
        trial.set_user_attr("harmful_mean_kl", trial_result.get("harmful_mean_kl"))
        trial.set_user_attr("harmless_mean_kl", trial_result.get("harmless_mean_kl"))
        trial.set_user_attr("harmful_kl_penalty", trial_result.get("harmful_kl_penalty"))
        trial.set_user_attr("harmless_kl_penalty", trial_result.get("harmless_kl_penalty"))
        trial.set_user_attr("kl_penalty", trial_result.get("kl_penalty"))
        trial.set_user_attr("mean_objective", trial_result["mean_objective"])
        trial.set_user_attr("n_questions", trial_result["n_questions"])
        trial.set_user_attr("harmless_n_questions", trial_result.get("harmless_n_questions"))
        trial.set_user_attr("questions", trial_questions)
        trial.set_user_attr("harmless_questions", trial_harmless_questions)
        for key, value in summarize_weights(trial_weights).items():
            trial.set_user_attr(key, value)
        if weights_mode == "scalar":
            trial.set_user_attr("weights", trial_result["weights"])

        return trial_result["mean_objective"]

    study = optuna.create_study(
        direction="maximize",
        sampler=create_optuna_sampler(sampler_name, sampler_seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_mean_harmfulness = best_trial.user_attrs.get("mean_harmfulness")
    best_mean_unsafe_rate = (
        best_trial.user_attrs.get("mean_unsafe_rate")
        if is_weighted_rd_unsafe_rate_metric(reward_metric)
        else None
    )
    best_unsafe_rate = (
        best_trial.user_attrs.get("best_unsafe_rate")
        if is_weighted_rd_unsafe_rate_metric(reward_metric)
        else None
    )
    best_mean_kl = best_trial.user_attrs.get("mean_kl")
    best_harmful_mean_kl = best_trial.user_attrs.get("harmful_mean_kl")
    best_harmless_mean_kl = best_trial.user_attrs.get("harmless_mean_kl")
    best_harmful_kl_penalty = best_trial.user_attrs.get("harmful_kl_penalty")
    best_harmless_kl_penalty = best_trial.user_attrs.get("harmless_kl_penalty")
    best_kl_penalty = best_trial.user_attrs.get("kl_penalty")
    best_mean_objective = best_trial.user_attrs.get("mean_objective")
    best_weights = reconstruct_weights_from_params(
        params=best_trial.params,
        direction_weights=direction_weights,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        direction_weights.weights.copy_(best_weights)

    optimization_history = []
    for trial in study.trials:
        history_item = {
            "trial_number": trial.number,
            "value": float(trial.value) if trial.value is not None else None,
            "mean_reward": trial.user_attrs.get("mean_reward"),
            "best_reward": trial.user_attrs.get("best_reward"),
            "mean_harmfulness": trial.user_attrs.get("mean_harmfulness"),
            "best_harmfulness": trial.user_attrs.get("best_harmfulness"),
            "mean_kl": trial.user_attrs.get("mean_kl"),
            "harmful_mean_kl": trial.user_attrs.get("harmful_mean_kl"),
            "harmless_mean_kl": trial.user_attrs.get("harmless_mean_kl"),
            "harmful_kl_penalty": trial.user_attrs.get("harmful_kl_penalty"),
            "harmless_kl_penalty": trial.user_attrs.get("harmless_kl_penalty"),
            "kl_penalty": trial.user_attrs.get("kl_penalty"),
            "mean_objective": trial.user_attrs.get("mean_objective"),
            "n_questions": trial.user_attrs.get("n_questions"),
            "harmless_n_questions": trial.user_attrs.get("harmless_n_questions"),
            "weights_mode": trial.user_attrs.get("weights_mode"),
            "state": trial.state.name,
        }
        if trial.user_attrs.get("weights_mode") == "scalar":
            history_item["weights"] = trial.user_attrs.get("weights")
        else:
            for key in (
                "weights_shape",
                "weights_mean",
                "weights_std",
                "weights_min",
                "weights_max",
                "weights_norm",
            ):
                history_item[key] = trial.user_attrs.get(key)
        if is_weighted_rd_unsafe_rate_metric(reward_metric):
            history_item["mean_unsafe_rate"] = trial.user_attrs.get("mean_unsafe_rate")
            history_item["best_unsafe_rate"] = trial.user_attrs.get("best_unsafe_rate")
        optimization_history.append(history_item)

    result = {
        "best_weights": best_weights.detach().cpu().tolist(),
        "best_value": float(best_trial.value),
        "best_trial_number": best_trial.number,
        "best_trial_questions": list(best_trial.user_attrs.get("questions", fallback_questions or [])),
        "best_trial_harmless_questions": list(
            best_trial.user_attrs.get("harmless_questions", fallback_harmless_questions)
        ),
        "best_mean_harmfulness": best_mean_harmfulness,
        "best_mean_kl": best_mean_kl,
        "best_harmful_mean_kl": best_harmful_mean_kl,
        "best_harmless_mean_kl": best_harmless_mean_kl,
        "best_harmful_kl_penalty": best_harmful_kl_penalty,
        "best_harmless_kl_penalty": best_harmless_kl_penalty,
        "best_kl_penalty": best_kl_penalty,
        "best_mean_objective": best_mean_objective,
        "sampler_name": sampler_name,
        "optimization_history": optimization_history,
    }
    if is_weighted_rd_unsafe_rate_metric(reward_metric):
        result["best_mean_unsafe_rate"] = best_mean_unsafe_rate
        result["best_unsafe_rate"] = best_unsafe_rate
    return result


def optimize_scalar_weights_with_optuna(
    direction_weights: LearnableDirectionWeights,
    extracted_directions: List[torch.Tensor],
    model,
    questions: Optional[List[str]],
    abliteration_params: Dict[str, float],
    classifier_categories: List[Dict[str, Any]],
    n_layers: int,
    ref_alpha: float,
    reward_sign: float,
    beta_1: float,
    beta_2: float,
    n_trials: int,
    sampler_name: str,
    sampler_seed: int,
    weight_min: float,
    weight_max: float,
    question_sampler: Optional[Callable[[], List[str]]] = None,
    harmless_questions: Optional[List[str]] = None,
    harmless_question_sampler: Optional[Callable[[], List[str]]] = None,
    loss_agg_mode: str = "token-mean",
    backend: Optional[str] = None,
    reward_metric: str = "harmfulness",
) -> Dict[str, Any]:
    """Backward-compatible scalar wrapper around the generic Optuna optimizer."""
    return optimize_weights_with_optuna(
        direction_weights=direction_weights,
        extracted_directions=extracted_directions,
        model=model,
        questions=questions,
        abliteration_params=abliteration_params,
        classifier_categories=classifier_categories,
        n_layers=n_layers,
        ref_alpha=ref_alpha,
        reward_sign=reward_sign,
        beta_1=beta_1,
        beta_2=beta_2,
        n_trials=n_trials,
        sampler_name=sampler_name,
        sampler_seed=sampler_seed,
        weight_min=weight_min,
        weight_max=weight_max,
        question_sampler=question_sampler,
        harmless_questions=harmless_questions,
        harmless_question_sampler=harmless_question_sampler,
        loss_agg_mode=loss_agg_mode,
        backend=backend,
        reward_metric=reward_metric,
    )
