"""
Minimal PPO and off-policy helpers used by weighted_rd.

These local helpers replace the small subset of vendored verl functionality
that this job needs at runtime.
"""

from enum import Enum
from typing import Any, Optional

import torch


SAFETY_BOUND = 20.0
ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | Enum):
    """Register a custom advantage estimator by name."""

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def masked_sum(
    values: torch.Tensor,
    mask: torch.Tensor,
    axis: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Compute the sum of values selected by mask."""
    mask = mask.to(dtype=values.dtype)
    valid_values = torch.where(mask.bool(), values, torch.zeros_like(values))
    return (valid_values * mask).sum(dim=axis)


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    axis: int | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Compute the mean of values selected by mask."""
    mask = mask.to(dtype=values.dtype)
    return masked_sum(values, mask, axis=axis) / (mask.sum(dim=axis) + 1e-8)


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
) -> torch.Tensor:
    """Aggregate token losses using the same modes weighted_rd already expects."""
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode in {"seq-mean-token-sum", "seq-mean-token-sum-norm"}:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).to(dtype=loss_mat.dtype)
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                loss_scale_factor = loss_mask.shape[-1]
            loss = loss / loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_token_counts = torch.sum(loss_mask, dim=-1)
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_token_counts + 1e-8)
        seq_mask = (seq_token_counts > 0).to(dtype=loss_mat.dtype)
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def kl_penalty(
    logprob: torch.FloatTensor,
    ref_logprob: torch.FloatTensor,
    kl_penalty: str,
) -> torch.FloatTensor:
    """Compute token-level KL estimates."""
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    backward_score = 0.5 * (logprob - ref_logprob).square()
    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(
    logprob: torch.FloatTensor,
    ref_logprob: torch.FloatTensor,
    kl_penalty: str,
) -> torch.FloatTensor:
    """Forward-pass KL estimator."""
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        raise NotImplementedError("full KL is not implemented in weighted_rd")

    raise NotImplementedError(f"Unsupported kl_penalty mode: {kl_penalty}")


def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[Any] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute the PPO clipped objective used by weighted_rd."""
    if config is None:
        raise ValueError("config must be provided for compute_policy_loss_vanilla")

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0) if hasattr(config, "get") else getattr(config, "clip_ratio_c", 3.0)

    if clip_ratio_c <= 1.0:
        raise ValueError(
            "clip_ratio_c for dual-clip PPO must be greater than 1.0, "
            f"got {clip_ratio_c}."
        )

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(
        (torch.gt(clip_pg_losses1, pg_losses3) & (advantages < 0)).float(),
        response_mask,
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    global_batch_info = getattr(config, "global_batch_info", {}) or {}
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **global_batch_info,
    )

    return pg_loss, {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }


def compute_rollout_correction_weights(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str = "token",
    rollout_is_threshold: float = 2.0,
    rollout_is_batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute truncated IS weights used by weighted_rd."""
    valid_is_levels = {"token", "sequence"}
    if rollout_is not in valid_is_levels:
        raise ValueError(f"Invalid rollout_is: {rollout_is}. Must be one of {valid_is_levels}.")
    if rollout_is_threshold <= 0:
        raise ValueError(f"rollout_is_threshold must be positive, got {rollout_is_threshold}.")

    if rollout_is == "token":
        log_ratio_for_metrics = log_ratio
        rollout_is_weights = torch.exp(torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND))
    else:
        log_ratio_sum = masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        log_ratio_for_metrics = log_ratio_sum
        rollout_is_weights = torch.exp(
            torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        ).expand_as(log_ratio)

    rollout_is_weights = rollout_is_weights * response_mask.to(dtype=rollout_is_weights.dtype)
    metrics = compute_is_metrics(
        rollout_is_weights=rollout_is_weights,
        log_ratio_for_metrics=log_ratio_for_metrics,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
    )

    rollout_is_weights = rollout_is_weights.clamp(max=rollout_is_threshold).detach()

    if rollout_is_batch_normalize:
        if rollout_is == "token":
            weights_mean = masked_mean(rollout_is_weights, response_mask)
        else:
            seq_weights = masked_mean(rollout_is_weights, response_mask, axis=-1)
            seq_mask = (response_mask.sum(dim=-1) > 0).to(dtype=rollout_is_weights.dtype)
            weights_mean = (seq_weights * seq_mask).sum() / seq_mask.sum().clamp_min(1e-8)

        if weights_mean > 1e-8:
            rollout_is_weights = rollout_is_weights / weights_mean
            metrics["rollout_is_batch_norm_factor"] = weights_mean.item()
        else:
            metrics["rollout_is_batch_norm_factor"] = 1.0

    return rollout_is_weights, metrics


def compute_is_metrics(
    rollout_is_weights: torch.Tensor,
    log_ratio_for_metrics: torch.Tensor,
    response_mask: torch.Tensor,
    rollout_is: str,
    rollout_is_threshold: float,
) -> dict[str, float]:
    """Compute summary metrics for IS weights."""
    if not response_mask.any():
        raise ValueError("response_mask must contain at least one valid token (1).")

    metrics: dict[str, float] = {}
    device = rollout_is_weights.device
    rollout_is_threshold_lower = 1.0 / rollout_is_threshold
    log_threshold_upper = torch.log(torch.tensor(rollout_is_threshold, device=device))
    log_threshold_lower = torch.log(torch.tensor(rollout_is_threshold_lower, device=device))

    if rollout_is == "sequence":
        log_max = log_ratio_for_metrics.max()
        log_min = log_ratio_for_metrics.min()
        metrics["rollout_is_max"] = torch.exp(torch.clamp(log_max, max=SAFETY_BOUND)).item()
        metrics["rollout_is_min"] = torch.exp(log_min).item()
        metrics["rollout_is_mean"] = masked_mean(rollout_is_weights, response_mask).item()
        metrics["rollout_is_ratio_fraction_high"] = (log_ratio_for_metrics > log_threshold_upper).float().mean().item()
        metrics["rollout_is_ratio_fraction_low"] = (log_ratio_for_metrics < log_threshold_lower).float().mean().item()
    else:
        metrics["rollout_is_mean"] = masked_mean(rollout_is_weights, response_mask).item()
        metrics["rollout_is_ratio_fraction_high"] = masked_mean(
            (rollout_is_weights > rollout_is_threshold).float(),
            response_mask,
        ).item()
        metrics["rollout_is_ratio_fraction_low"] = masked_mean(
            (rollout_is_weights < rollout_is_threshold_lower).float(),
            response_mask,
        ).item()
        mask_bool = response_mask.bool()
        metrics["rollout_is_max"] = rollout_is_weights.masked_fill(~mask_bool, float("-inf")).max().item()
        metrics["rollout_is_min"] = rollout_is_weights.masked_fill(~mask_bool, float("inf")).min().item()

    mask_count = response_mask.sum()
    if mask_count > 1:
        weights_for_std = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold)
        mean_clamped = masked_mean(weights_for_std, response_mask)
        rollout_is_var = masked_mean(weights_for_std.square(), response_mask) - mean_clamped.square()
        metrics["rollout_is_std"] = torch.sqrt(torch.clamp(rollout_is_var, min=0.0)).item()
    else:
        metrics["rollout_is_std"] = 0.0

    weights_for_ess = rollout_is_weights.clamp(min=rollout_is_threshold_lower, max=rollout_is_threshold)
    mean_for_ess = masked_mean(weights_for_ess, response_mask)
    is_weights_normalized = weights_for_ess / (mean_for_ess + 1e-8)
    metrics["rollout_is_eff_sample_size"] = (
        1.0 / masked_mean(is_weights_normalized.square(), response_mask).item()
    )

    if rollout_is_weights.dim() > 1:
        seq_mean_weights = masked_mean(rollout_is_weights, response_mask, axis=-1)
        metrics["rollout_is_seq_mean"] = seq_mean_weights.mean().item()
        metrics["rollout_is_seq_std"] = seq_mean_weights.std().item() if seq_mean_weights.numel() > 1 else 0.0
        metrics["rollout_is_seq_max"] = seq_mean_weights.max().item()
        metrics["rollout_is_seq_min"] = seq_mean_weights.min().item()
        seq_deviation = (seq_mean_weights - 1.0).abs()
        metrics["rollout_is_seq_max_deviation"] = seq_deviation.max().item()
        metrics["rollout_is_seq_fraction_high"] = (seq_mean_weights > rollout_is_threshold).float().mean().item()
        metrics["rollout_is_seq_fraction_low"] = (seq_mean_weights < rollout_is_threshold_lower).float().mean().item()

    return metrics


def compute_offpolicy_metrics(
    old_log_prob: torch.Tensor,
    rollout_log_prob: Optional[torch.Tensor],
    response_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute off-policy diagnostics for training vs rollout log-probs."""
    if not response_mask.any():
        raise ValueError("Expected at least one valid token in response_mask")

    metrics: dict[str, Any] = {}
    mean_log_prob_training = masked_mean(old_log_prob, response_mask, axis=-1)
    training_ppl = torch.exp(-mean_log_prob_training).mean()
    metrics["training_ppl"] = training_ppl.detach().item()
    metrics["training_log_ppl"] = (-mean_log_prob_training).mean().detach().item()

    if rollout_log_prob is not None:
        metrics["kl"] = masked_mean(rollout_log_prob - old_log_prob, response_mask).detach().item()

        log_ratio = old_log_prob - rollout_log_prob
        k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
        metrics["k3_kl"] = masked_mean(k3_kl_matrix, response_mask).detach().item()

        mean_log_prob_rollout = masked_mean(rollout_log_prob, response_mask, axis=-1)
        rollout_ppl = torch.exp(-mean_log_prob_rollout).mean()
        metrics["rollout_ppl"] = rollout_ppl.detach().item()
        metrics["rollout_log_ppl"] = (-mean_log_prob_rollout).mean().detach().item()

        log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
        metrics["log_ppl_diff"] = log_ppl_diff.mean().detach().item()
        metrics["log_ppl_abs_diff"] = log_ppl_diff.abs().mean().detach().item()
        metrics["log_ppl_diff_max"] = log_ppl_diff.max().detach().item()
        metrics["log_ppl_diff_min"] = log_ppl_diff.min().detach().item()
        metrics["ppl_ratio"] = torch.exp(log_ppl_diff).mean().detach().item()

        log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_token = torch.exp(log_ratio_safe)
        rho_squared_token = rho_token.square()
        metrics["chi2_token"] = (masked_mean(rho_squared_token, response_mask) - 1.0).detach().item()

        log_ratio_sum = masked_sum(log_ratio, response_mask, axis=-1)
        log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
        rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)
        metrics["chi2_seq"] = (rho_squared_seq.mean() - 1.0).detach().item()

    return metrics
