"""
KL helpers for weighted_rd regularization.
"""

from typing import Tuple

import torch

from baselines.weighted_rd.ppo_utils import agg_loss, kl_penalty


KL_LOSS_TYPE = "low_var_kl"


def compute_masked_mean_kl(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-token KL estimates and their masked aggregate."""
    kl_tensor = kl_penalty(
        logprob=log_prob,
        ref_logprob=ref_log_prob,
        kl_penalty=KL_LOSS_TYPE,
    )
    mean_kl = agg_loss(
        loss_mat=kl_tensor,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
    )
    return kl_tensor, mean_kl
