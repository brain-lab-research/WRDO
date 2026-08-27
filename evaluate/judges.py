"""
Functions for evaluating model responses with guard backends.
"""

from typing import Dict, List, Optional

from config import EVALUATION_BACKEND


def evaluate_harmfulness(
    question: str,
    response: str,
    classifier_categories: Optional[List[Dict]] = None,
    backend: Optional[str] = None,
) -> dict:
    """Evaluate harmfulness with a configurable guard backend."""
    del classifier_categories  # Preserved for compatibility with existing callers.
    selected_backend = (backend or EVALUATION_BACKEND).strip().lower()

    if selected_backend == "llamaguard":
        from evaluate.evaluation_llamaguard import get_llamaguard_evaluator

        evaluator = get_llamaguard_evaluator()
        result = evaluator.evaluate(question, response)
        result.setdefault("category_id", "unknown")
        return result

    if selected_backend == "wildguard":
        from evaluate.evaluation_wildguard import get_wildguard_evaluator

        evaluator = get_wildguard_evaluator()
        result = evaluator.evaluate(question, response)
        result.setdefault("category_id", "unknown")
        return result

    if selected_backend == "qwen3guard":
        from evaluate.evaluation_qwen3guard import get_qwen3guard_evaluator

        evaluator = get_qwen3guard_evaluator()
        result = evaluator.evaluate(question, response)
        result.setdefault("category_id", "unknown")
        return result

    return {
        "score": None,
        "category_id": "unknown",
        "explanation": f"Unknown evaluation backend: {selected_backend!r}.",
    }
