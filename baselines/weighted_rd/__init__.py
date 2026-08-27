"""
GRPO with Importance Sampling for training direction weight coefficients.

Uses lightweight local PPO helpers with custom IS-corrected advantage and
differentiable abliteration hooks.
"""

__all__ = [
    "main",
    "compute_grpo_is_advantage",
    "compute_reward",
    "compute_sequence_log_probs",
    "register_abliteration_hooks",
    "remove_hooks",
]


def main():
    from .__main__ import main as _main

    return _main()
