"""Batch harmfulness evaluation helpers used by attack-mode baselines."""

import os

import numpy as np
from typing import Dict, List, Tuple

from evaluate.judges import (
    evaluate_harmfulness,
)
from config import EVALUATION_BACKEND

UNSAFE_RATE_EVALUATION_METRICS = frozenset({"unsafe_rate"})
UNSAFE_RATE_EVALUATION_BACKENDS = frozenset({"llamaguard", "wildguard", "qwen3guard"})
SUPPORTED_EVALUATION_METRICS = frozenset({"harmfulness", *UNSAFE_RATE_EVALUATION_METRICS})


def resolve_evaluation_metric(
    env_value: str | None = None,
    backend: str | None = None,
) -> str:
    """Resolve the scalar metric used by attack-mode evaluation summaries."""
    selected_backend = (backend or EVALUATION_BACKEND).strip().lower()
    raw_value = env_value if env_value is not None else os.getenv("EVALUATION_METRIC")

    if raw_value is None or not raw_value.strip():
        if selected_backend in UNSAFE_RATE_EVALUATION_BACKENDS:
            return "unsafe_rate"
        return "harmfulness"

    normalized = raw_value.strip().lower()
    if normalized not in SUPPORTED_EVALUATION_METRICS:
        raise ValueError(
            f"Unsupported EVALUATION_METRIC '{raw_value}'. "
            "Expected one of: harmfulness, unsafe_rate."
        )
    if normalized == "unsafe_rate" and selected_backend not in UNSAFE_RATE_EVALUATION_BACKENDS:
        raise ValueError(
            "EVALUATION_METRIC='unsafe_rate' requires "
            "EVALUATION_BACKEND in {'llamaguard', 'wildguard', 'qwen3guard'}."
        )
    return normalized


def is_unsafe_rate_evaluation_metric(metric: str) -> bool:
    """Return whether the selected evaluation metric is binary unsafe rate."""
    return metric.strip().lower() in UNSAFE_RATE_EVALUATION_METRICS


def evaluate_responses(
    questions: List[str],
    responses: List[str],
    classifier_categories: List[Dict],
    description: str = "",
) -> Tuple[List[float], List[Dict]]:
    """
    Evaluate harmfulness for a list of question-response pairs.

    Returns:
        Tuple of (list of numeric scores on the active metric scale, list of full score dicts)
    """
    selected_backend = EVALUATION_BACKEND.strip().lower()
    selected_metric = resolve_evaluation_metric(backend=selected_backend)
    scores: List[float] = []
    score_data_list: List[Dict] = []

    print(
        f"\nEvaluating {description} with backend: {EVALUATION_BACKEND} "
        f"(metric: {selected_metric})"
    )
    print(f"Total responses to evaluate: {len(responses)}")

    for i, (question, response) in enumerate(zip(questions, responses)):
        if not response:
            continue

        try:
            score_data = evaluate_harmfulness(
                question,
                response,
                classifier_categories=classifier_categories,
            )
            if is_unsafe_rate_evaluation_metric(selected_metric):
                label = str(score_data.get("label", "")).strip().lower()
                score_value = 1.0 if label == "unsafe" else 0.0
                scores.append(score_value)
                score_data_list.append({
                    "question": question,
                    "response": response,
                    "score": score_value,
                    "evaluation_metric": selected_metric,
                    "category_id": score_data.get("category_id", "unknown"),
                    "explanation": score_data.get("explanation", ""),
                    "full_evaluation": score_data,
                })
                continue

            score = score_data.get("score")
            if score is not None:
                score_value = float(score)
                scores.append(score_value)
                score_data_list.append({
                    "question": question,
                    "response": response,
                    "score": score_value,
                    "evaluation_metric": selected_metric,
                    "category_id": score_data.get("category_id", "unknown"),
                    "explanation": score_data.get("explanation", ""),
                    "full_evaluation": score_data,
                })
        except Exception as e:
            if i < 5:
                print(f"      Evaluation error: {e}")

        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(responses)} responses...")

    print(f"Got {len(scores)} scores for {description}")
    return scores, score_data_list
