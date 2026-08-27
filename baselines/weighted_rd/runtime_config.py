"""Runtime-only configuration helpers for weighted_rd."""

UNSAFE_RATE_GUARD_BACKENDS = frozenset({"llamaguard", "wildguard", "qwen3guard"})
UNSAFE_RATE_REWARD_METRICS = frozenset(
    {"unsafe_rate", "llamaguard_unsafe", "wildguard_unsafe", "qwen3guard_unsafe"}
)
SUPPORTED_REWARD_METRICS = frozenset({"harmfulness", *UNSAFE_RATE_REWARD_METRICS})


def resolve_weighted_rd_optimizer_method(env_value: str | None) -> str:
    """Default weighted_rd to the existing GRPO trainer unless overridden."""
    if env_value is None:
        return "grpo"

    normalized = env_value.strip()
    if not normalized:
        return "grpo"
    return normalized


def validate_weighted_rd_optimizer_method(method: str) -> None:
    """Validate weighted_rd optimizer selection."""
    if method not in {"grpo", "optuna"}:
        raise ValueError(
            f"weighted_rd only supports OPTIMIZER_METHOD in {{'grpo', 'optuna'}}, got '{method}'."
        )


def validate_weighted_rd_optimizer_compatibility(method: str, weights_mode: str) -> None:
    """Ensure the selected optimizer supports the requested weight parameterization."""
    return None


def resolve_weighted_rd_weights_mode(env_value: str | None) -> str:
    """Default weighted_rd to scalar-per-direction weights when env is unset."""
    if env_value is None:
        return "scalar"

    normalized = env_value.strip()
    if not normalized:
        return "scalar"
    return normalized


def validate_weighted_rd_weights_mode(mode: str) -> None:
    """Validate weighted_rd weight parameterization."""
    if mode not in {"scalar", "dense"}:
        raise ValueError(
            f"weighted_rd only supports WEIGHTS_MODE in {{'scalar', 'dense'}}, got '{mode}'."
        )


def resolve_weighted_rd_debug_question_count(
    env_value: str | None,
    default: int = 4,
) -> int:
    """Use a small but non-degenerate question count for debug smoke runs."""
    if env_value is None:
        return default

    normalized = env_value.strip()
    if not normalized:
        return default

    value = int(normalized)
    if value < 1:
        raise ValueError(f"DEBUG_N_QUESTIONS must be >= 1, got {value}")
    return value


def resolve_weighted_rd_direction_prompt_count(
    env_value: str | None,
    default: int = 128,
) -> int | None:
    """Resolve the number of combined-dataset prompts used to build directions."""
    if env_value is None or not env_value.strip():
        return default

    normalized = env_value.strip().lower()
    if normalized in ("all", "none", "null"):
        return None

    value = int(normalized)
    if value < 1:
        raise ValueError(f"WEIGHTED_RD_DIRECTION_PROMPT_COUNT must be >= 1, got {value}")
    return value


def resolve_weighted_rd_direction_prompt_seed(
    env_value: str | None,
    default: int = 42,
) -> int:
    """Resolve the sampling seed for combined-dataset direction prompts."""
    if env_value is None or not env_value.strip():
        return default
    return int(env_value.strip())


def resolve_weighted_rd_debug_noise_scale(
    base_noise_scale: float,
    env_value: str | None,
    default: float = 0.02,
) -> float:
    """Keep debug rollout noise small enough for a useful local training signal."""
    if env_value is None or not env_value.strip():
        return min(base_noise_scale, default)

    value = float(env_value.strip())
    if value < 0:
        raise ValueError(f"DEBUG_NOISE_SCALE must be >= 0, got {value}")
    return value


def resolve_weighted_rd_weights_init_type(env_value: str | None) -> str:
    """Default weighted_rd to a nonzero initialization when env is unset."""
    if env_value is None:
        return "average"

    normalized = env_value.strip()
    if not normalized:
        return "average"
    return normalized


def validate_weighted_rd_weights_init_type(
    init_type: str,
) -> None:
    """Reject zero init because the current differentiable path cannot leave it."""
    if init_type == "zero":
        raise ValueError(
            "weighted_rd does not support WEIGHTS_INIT_TYPE='zero': "
            "the current differentiable intervention path has a dead start at zero, "
            "so training cannot move off that point. Use 'average' (default)."
        )

    if init_type == "topic":
        raise ValueError(
            "weighted_rd does not support WEIGHTS_INIT_TYPE='topic' in combined-dataset mode. "
            "Use 'average'."
        )


def resolve_weighted_rd_optuna_n_trials(
    env_value: str | None,
    default: int = 50,
) -> int:
    """Resolve the Optuna trial budget."""
    if env_value is None or not env_value.strip():
        return default

    value = int(env_value.strip())
    if value < 1:
        raise ValueError(f"OPTUNA_N_TRIALS must be >= 1, got {value}")
    return value


def resolve_weighted_rd_optuna_sampler_seed(
    env_value: str | None,
    default: int = 42,
) -> int:
    """Resolve the Optuna sampler seed."""
    if env_value is None or not env_value.strip():
        return default
    return int(env_value.strip())


def resolve_weighted_rd_optuna_sampler(
    env_value: str | None,
    default: str = "tpe",
) -> str:
    """Resolve the Optuna sampler name."""
    if env_value is None:
        return default

    normalized = env_value.strip()
    if not normalized:
        return default
    return normalized


def validate_weighted_rd_optuna_sampler(sampler: str) -> None:
    """Validate the supported Optuna sampler choices."""
    if sampler not in {"tpe", "random", "gp", "cmaes", "qmc"}:
        raise ValueError(
            "weighted_rd only supports OPTUNA_SAMPLER in "
            "{'tpe', 'random', 'gp', 'cmaes', 'qmc'}, "
            f"got '{sampler}'."
        )


def resolve_weighted_rd_optuna_weight_min(
    env_value: str | None,
    default: float = -2.0,
) -> float:
    """Resolve the lower search bound for scalar Optuna coefficients."""
    if env_value is None or not env_value.strip():
        return default
    return float(env_value.strip())


def resolve_weighted_rd_optuna_weight_max(
    env_value: str | None,
    default: float = 2.0,
) -> float:
    """Resolve the upper search bound for scalar Optuna coefficients."""
    if env_value is None or not env_value.strip():
        return default
    return float(env_value.strip())


def validate_weighted_rd_optuna_weight_range(weight_min: float, weight_max: float) -> None:
    """Reject inverted scalar Optuna search ranges."""
    if weight_min > weight_max:
        raise ValueError(
            f"OPTUNA weight range must satisfy OPTUNA_WEIGHT_MIN <= OPTUNA_WEIGHT_MAX, "
            f"got {weight_min} > {weight_max}."
        )


def resolve_weighted_rd_reward_sign(
    env_value: str | None,
    default: float = 1.0,
) -> float:
    """Resolve the reward sign used to convert harmfulness into a training objective."""
    if env_value is None or not env_value.strip():
        return default
    return float(env_value.strip())


def validate_weighted_rd_reward_sign(reward_sign: float) -> None:
    """Reject zero reward scaling to avoid a degenerate objective."""
    if reward_sign == 0:
        raise ValueError("REWARD_SIGN must be non-zero.")


def resolve_weighted_rd_reward_metric(
    env_value: str | None,
    backend: str,
    default: str = "harmfulness",
) -> str:
    """Resolve the response-level reward metric used by weighted_rd."""
    normalized_backend = backend.strip().lower()
    if env_value is None or not env_value.strip():
        if normalized_backend in UNSAFE_RATE_GUARD_BACKENDS:
            return "unsafe_rate"
        return default
    return env_value.strip().lower()


def is_weighted_rd_unsafe_rate_metric(reward_metric: str) -> bool:
    """Return whether a reward metric is a binary unsafe-rate metric."""
    return reward_metric.strip().lower() in UNSAFE_RATE_REWARD_METRICS


def validate_weighted_rd_reward_metric(reward_metric: str, backend: str) -> None:
    """Validate reward metric compatibility with the selected evaluator backend."""
    normalized_backend = backend.strip().lower()
    if reward_metric not in SUPPORTED_REWARD_METRICS:
        raise ValueError(
            "weighted_rd only supports REWARD_METRIC in "
            "{'harmfulness', 'unsafe_rate', 'llamaguard_unsafe', "
            "'wildguard_unsafe', 'qwen3guard_unsafe'}, "
            f"got '{reward_metric}'."
        )

    if reward_metric == "unsafe_rate" and normalized_backend not in UNSAFE_RATE_GUARD_BACKENDS:
        raise ValueError(
            "REWARD_METRIC='unsafe_rate' requires "
            "EVALUATION_BACKEND in {'llamaguard', 'wildguard', 'qwen3guard'}."
        )

    if reward_metric == "llamaguard_unsafe" and normalized_backend != "llamaguard":
        raise ValueError(
            "REWARD_METRIC='llamaguard_unsafe' requires "
            "EVALUATION_BACKEND='llamaguard'."
        )

    if reward_metric == "wildguard_unsafe" and normalized_backend != "wildguard":
        raise ValueError(
            "REWARD_METRIC='wildguard_unsafe' requires "
            "EVALUATION_BACKEND='wildguard'."
        )

    if reward_metric == "qwen3guard_unsafe" and normalized_backend != "qwen3guard":
        raise ValueError(
            "REWARD_METRIC='qwen3guard_unsafe' requires "
            "EVALUATION_BACKEND='qwen3guard'."
        )
