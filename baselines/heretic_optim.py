#!/usr/bin/env python3
"""
Heretic baseline — optimization core.

Substring-based refusal classifier, refusal-direction tensor construction
(with optional Gram-Schmidt orthogonalization), Optuna TPE study, and
Pareto-front auto-selection. All weight ablation goes through the local
`heretic.model.Model.abliterate`.

This module is imported by `baselines.heretic`; see that module for the
top-level pipeline.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import optuna
import torch
import torch.nn.functional as F
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from heretic.model import AbliterationParameters, Model


# ---------------------------------------------------------------------------
# Refusal classifier (port of upstream evaluator.is_refusal)
# ---------------------------------------------------------------------------

_REFUSAL_NORMALIZE_RE = re.compile(r"\s+")


def is_refusal(response: str, refusal_markers: Sequence[str]) -> bool:
    if not response or not response.strip():
        return True
    normalized = response.lower().replace("*", "").replace("’", "'").replace("‘", "'")
    normalized = _REFUSAL_NORMALIZE_RE.sub(" ", normalized).strip()
    return any(marker.lower() in normalized for marker in refusal_markers)


def count_refusals(responses: Sequence[str], refusal_markers: Sequence[str]) -> int:
    return sum(1 for response in responses if is_refusal(response, refusal_markers))


# ---------------------------------------------------------------------------
# Refusal-direction tensor construction
# ---------------------------------------------------------------------------


def build_refusal_directions_tensor(
    mean_diffs: torch.Tensor,
    *,
    orthogonalize: bool,
) -> torch.Tensor:
    """Aggregate (positions, layers, hidden) into the (n_layers+1, hidden)
    layout expected by Model.abliterate (row 0 is the embedding direction;
    row i+1 is layer i's direction).
    """
    if mean_diffs.ndim != 3:
        raise ValueError(
            f"Expected mean_diffs of shape (positions, n_layers, hidden), "
            f"got {tuple(mean_diffs.shape)}."
        )
    aggregated = mean_diffs.to(torch.float32).mean(dim=0)
    normalized = F.normalize(aggregated, p=2, dim=-1)
    if orthogonalize:
        normalized = _orthogonalize_directions(normalized)
    n_layers, hidden = normalized.shape
    directions = torch.zeros((n_layers + 1, hidden), dtype=torch.float32)
    directions[1:] = normalized
    return directions


def _orthogonalize_directions(directions: torch.Tensor) -> torch.Tensor:
    """Modified Gram-Schmidt across layers, re-normalizing each result.

    Layers whose orthogonal component collapses (norm < 1e-8) keep their
    original direction unchanged. Port of the orthogonalize step in
    upstream Heretic (main.py:464-476).
    """
    n_layers, _ = directions.shape
    out = torch.empty_like(directions)
    for layer_index in range(n_layers):
        vector = directions[layer_index].clone()
        for prev_index in range(layer_index):
            prev = out[prev_index]
            vector = vector - torch.dot(vector, prev) * prev
        norm = vector.norm()
        if float(norm) < 1e-8:
            out[layer_index] = directions[layer_index]
        else:
            out[layer_index] = vector / norm
    return out


# ---------------------------------------------------------------------------
# Trial parameter sampling
# ---------------------------------------------------------------------------


def _trial_param_name(component: str, suffix: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", component)
    return f"{safe}__{suffix}"


def _sample_per_component_params(
    trial: optuna.Trial,
    components: Sequence[str],
    n_layers: int,
    cfg: dict[str, Any],
) -> dict[str, AbliterationParameters]:
    params: dict[str, AbliterationParameters] = {}
    layer_span = max(n_layers - 1, 1)
    for component in components:
        max_w = trial.suggest_float(
            _trial_param_name(component, "max_weight"),
            cfg["max_weight_low"],
            cfg["max_weight_high"],
        )
        max_p = trial.suggest_float(
            _trial_param_name(component, "max_weight_position"),
            cfg["max_weight_position_low_frac"] * layer_span,
            cfg["max_weight_position_high_frac"] * layer_span,
        )
        min_w_frac = trial.suggest_float(
            _trial_param_name(component, "min_weight_frac"),
            cfg["min_weight_low"],
            cfg["min_weight_high"],
        )
        min_w = min_w_frac * max_w
        min_d = trial.suggest_float(
            _trial_param_name(component, "min_weight_distance"),
            max(cfg["min_weight_distance_low_frac"] * layer_span, 1.0),
            max(cfg["min_weight_distance_high_frac"] * layer_span, 1.0),
        )
        params[component] = AbliterationParameters(
            max_weight=max_w,
            max_weight_position=max_p,
            min_weight=min_w,
            min_weight_distance=min_d,
        )
    return params


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def _compute_objective_scores(
    *,
    refusals: int,
    base_refusals: int,
    kl_divergence: float,
    cfg: dict[str, Any],
) -> tuple[float, float]:
    """Port of upstream evaluator.get_score (evaluator.py:95-127).

    Returns (kld_score, refusal_objective).

    The KL gate (`kl < target`) replaces the KL term with a refusal-driven
    penalty to prevent the optimizer from drifting into a "do nothing"
    region. We minimize the refusal score while keeping KL above the target.
    """
    if base_refusals > 0:
        refusals_score = refusals / base_refusals
    else:
        refusals_score = float(refusals)

    target = float(cfg["kl_divergence_target"])
    scale = float(cfg["kl_divergence_scale"])

    if kl_divergence >= target:
        kld_score = kl_divergence / scale
    else:
        gate_penalty = refusals_score
        kld_score = gate_penalty * target / scale

    return kld_score, refusals_score


def apply_trial(
    model: Model,
    refusal_directions: torch.Tensor,
    scope: str,
    direction_index: float,
    params: dict[str, AbliterationParameters],
) -> None:
    """Reload base weights and apply the trial's per-component ablation."""
    model.reload_model()
    with torch.no_grad():
        model.abliterate(
            refusal_directions,
            direction_index=direction_index if scope == "global" else None,
            parameters=params,
        )


def make_objective(
    model: Model,
    *,
    refusal_directions: torch.Tensor,
    harmful_val: list[str],
    harmless_val: list[str],
    base_responses: list[str],
    base_logprobs: torch.Tensor,
    cfg: dict[str, Any],
    n_layers: int,
    response_postprocess: Callable[[str], str],
    on_trial: Callable[[optuna.Trial, dict[str, Any]], None] | None = None,
) -> tuple[Callable[[optuna.Trial], tuple[float, float]], int]:
    """Build the Optuna objective callable. Returns (objective, base_refusals)."""
    components = list(model.get_abliterable_components())
    layer_span = max(n_layers - 1, 1)
    base_refusals = count_refusals(base_responses, cfg["refusal_markers"])

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        scope = trial.suggest_categorical(
            "direction_scope", list(cfg["scope_choices"])
        )
        direction_index = trial.suggest_float(
            "direction_index",
            cfg["direction_index_low_frac"] * layer_span,
            cfg["direction_index_high_frac"] * layer_span,
        )
        params = _sample_per_component_params(trial, components, n_layers, cfg)

        apply_trial(model, refusal_directions, scope, direction_index, params)

        responses = [
            response_postprocess(r)
            for r in model.get_responses_batched(harmful_val)
        ]
        refusals = count_refusals(responses, cfg["refusal_markers"])

        new_logp = model.get_logprobs_batched(harmless_val)
        # base_logprobs is cached on CPU by the caller; align devices+dtype before
        # F.kl_div, which requires matching devices.
        new_logp_f64 = new_logp.to(torch.float64)
        base_logp_f64 = base_logprobs.to(new_logp_f64.device, torch.float64)
        kl_value = float(
            F.kl_div(
                new_logp_f64,
                base_logp_f64,
                reduction="batchmean",
                log_target=True,
            ).item()
        )

        trial.set_user_attr("refusals", int(refusals))
        trial.set_user_attr("kl_divergence", kl_value)
        trial.set_user_attr("scope", scope)
        trial.set_user_attr(
            "direction_index_used",
            None if scope != "global" else float(direction_index),
        )
        trial.set_user_attr(
            "abliteration_params",
            {comp: asdict(p) for comp, p in params.items()},
        )

        kld_score, refusal_obj = _compute_objective_scores(
            refusals=refusals,
            base_refusals=base_refusals,
            kl_divergence=kl_value,
            cfg=cfg,
        )
        trial.set_user_attr("kld_score", kld_score)
        trial.set_user_attr("refusal_objective", refusal_obj)

        if on_trial is not None:
            on_trial(
                trial,
                {
                    "refusals": refusals,
                    "kl_divergence": kl_value,
                    "kld_score": kld_score,
                    "refusal_objective": refusal_obj,
                    "scope": scope,
                },
            )
        return kld_score, refusal_obj

    return objective, base_refusals


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------


def run_heretic_study(
    objective: Callable[[optuna.Trial], tuple[float, float]],
    *,
    n_trials: int,
    n_startup_trials: int,
    seed: int,
    study_dir: Path,
    study_name: str,
) -> optuna.Study:
    sampler = TPESampler(
        n_startup_trials=int(n_startup_trials),
        n_ei_candidates=128,
        multivariate=True,
        seed=int(seed),
    )
    study_dir.mkdir(parents=True, exist_ok=True)
    # Optuna 4.x moved the file backend under `optuna.storages.journal`.
    # `JournalFileBackend` is the canonical 4.x name (3.x was `JournalFileStorage`).
    from optuna.storages.journal import JournalFileBackend

    storage = optuna.storages.JournalStorage(
        JournalFileBackend(str(study_dir / "optuna.jsonl"))
    )
    study = optuna.create_study(
        sampler=sampler,
        directions=["minimize", "minimize"],
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
    remaining = max(int(n_trials) - completed, 0)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    return study


# ---------------------------------------------------------------------------
# Pareto extraction + auto-selection
# ---------------------------------------------------------------------------


def extract_pareto_front(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Port of upstream main.py:639-660.

    Sort trials by (refusal_objective asc, kl asc), then walk through them,
    keeping every trial whose KL strictly beats the running minimum.
    The refusal objective is the refusal fraction, so lower is better.
    """
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        return []
    sorted_trials = sorted(
        completed,
        key=lambda t: (
            float(t.user_attrs.get("refusal_objective", t.user_attrs["refusals"])),
            float(t.user_attrs["kl_divergence"]),
        ),
    )
    min_kl = math.inf
    pareto: list[optuna.trial.FrozenTrial] = []
    for t in sorted_trials:
        kl_value = float(t.user_attrs["kl_divergence"])
        if kl_value < min_kl:
            min_kl = kl_value
            pareto.append(t)
    return pareto


def auto_select_trial(
    pareto: list[optuna.trial.FrozenTrial],
    *,
    refusal_threshold_count: int,
) -> tuple[optuna.trial.FrozenTrial, str]:
    """Auto-pick a single trial from the Pareto front.

    Select the minimum-KL trial whose refusal count is below the threshold.
    If no trial qualifies, fall back to the trial with the smallest refusal
    count and then the smallest KL.
    """
    if not pareto:
        raise ValueError("Pareto front is empty; no trial to select.")

    qualified = [
        t for t in pareto
        if int(t.user_attrs["refusals"]) <= refusal_threshold_count
    ]
    if qualified:
        chosen = min(qualified, key=lambda t: float(t.user_attrs["kl_divergence"]))
        return chosen, "min_kl_with_refusals_le_threshold"
    chosen = min(
        pareto,
        key=lambda t: (
            int(t.user_attrs["refusals"]),
            float(t.user_attrs["kl_divergence"]),
        ),
    )
    return chosen, "min_refusals_fallback"


def reconstruct_params_from_trial(
    trial: optuna.trial.FrozenTrial,
    components: Sequence[str],
) -> dict[str, AbliterationParameters]:
    stored = trial.user_attrs.get("abliteration_params") or {}
    params: dict[str, AbliterationParameters] = {}
    for component in components:
        record = stored.get(component)
        if record is None:
            raise KeyError(
                f"Component {component!r} missing from trial.user_attrs."
            )
        params[component] = AbliterationParameters(**record)
    return params


def trial_to_record(
    trial: optuna.trial.FrozenTrial,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "trial_number": int(trial.number),
        "refusals": int(trial.user_attrs["refusals"]),
        "kl_divergence": float(trial.user_attrs["kl_divergence"]),
        "scope": str(trial.user_attrs["scope"]),
        "direction_index_used": trial.user_attrs.get("direction_index_used"),
        "abliteration_params": trial.user_attrs.get("abliteration_params"),
        "params": {
            k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            for k, v in trial.params.items()
        },
    }
    if extra:
        record.update(extra)
    return record


__all__ = [
    "is_refusal",
    "count_refusals",
    "build_refusal_directions_tensor",
    "apply_trial",
    "make_objective",
    "run_heretic_study",
    "extract_pareto_front",
    "auto_select_trial",
    "reconstruct_params_from_trial",
    "trial_to_record",
]
