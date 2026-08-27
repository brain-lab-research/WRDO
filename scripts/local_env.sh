#!/bin/bash
# Optional local defaults for release scripts.
# Do not put credentials or personal service endpoints in this file.

# ---- Weights & Biases ----
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-anonymized}"

# ---- Local results defaults ----
export RESULTS_SAVE_BEST_MODEL="${RESULTS_SAVE_BEST_MODEL:-false}"
export RESULTS_BEST_MODEL_MAX_SHARD_SIZE="${RESULTS_BEST_MODEL_MAX_SHARD_SIZE:-5GB}"
