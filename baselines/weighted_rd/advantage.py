"""
GRPO-IS advantage estimator: IS-weighted baseline (setup.txt).
"""

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch

from baselines.weighted_rd.ppo_utils import register_adv_est


@register_adv_est("grpo_is")
def compute_grpo_is_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    is_weights: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[Any] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO-IS with importance-weighted baseline.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: grouping array (e.g. question id for each response)
        is_weights: (bs, response_length) or (bs,) - IS weights. If None, falls back to simple mean.
        epsilon: small value for numerical stability
        norm_adv_by_std_in_grpo: whether to divide by std
        config: optional algorithm config

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)
    print(f"      [advantage] {scores.shape[0]} responses, "
          f"{len(set(index.tolist()))} unique groups, "
          f"scores: mean={scores.mean():.3f}, std={scores.std():.3f}")

    id2score = defaultdict(list)
    id2is = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            if is_weights is not None:
                # For sequence-level IS, all valid tokens share the same weight; use max to avoid padding zeros
                w = is_weights[i].max() if is_weights[i].numel() > 0 else torch.tensor(1.0, device=scores.device)
                id2is[index[i]].append(w)

        for idx in id2score:
            group_scores = id2score[idx]
            if len(group_scores) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
                id2std[idx] = torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
            else:
                scores_tensor = torch.stack(group_scores)
                if is_weights is not None and idx in id2is and len(id2is[idx]) == len(group_scores):
                    weights_tensor = torch.stack(id2is[idx])
                    weights_sum = weights_tensor.sum()
                    if weights_sum > 1e-8:
                        id2mean[idx] = (weights_tensor * scores_tensor).sum() / weights_sum
                    else:
                        id2mean[idx] = scores_tensor.mean()
                else:
                    id2mean[idx] = scores_tensor.mean()
                id2std[idx] = scores_tensor.std()

        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
