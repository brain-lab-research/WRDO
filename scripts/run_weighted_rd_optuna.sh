#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[[ -f "$SCRIPT_DIR/local_env.sh" ]] && source "$SCRIPT_DIR/local_env.sh"
# Respect a preconfigured multi-GPU mask; fall back to a single visible GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set -euo pipefail

export WEIGHTED_RD_DIRECTION_PROMPT_COUNT="${WEIGHTED_RD_DIRECTION_PROMPT_COUNT:-128}"
export WEIGHTED_RD_DIRECTION_PROMPT_SEED="${WEIGHTED_RD_DIRECTION_PROMPT_SEED:-42}"
export OPTIMIZER_METHOD="optuna"
export OPTUNA_SAMPLER="${OPTUNA_SAMPLER:-tpe}"
export EVALUATION_BACKEND="${EVALUATION_BACKEND:-wildguard}"
export REWARD_METRIC="${REWARD_METRIC:-}"
export EVALUATE_LOCALITY="${EVALUATE_LOCALITY:-false}"
export MMLU_ENABLED="${MMLU_ENABLED:-false}"
export ACADEMIC_BENCHMARKS_ENABLED="${ACADEMIC_BENCHMARKS_ENABLED:-}"
export BENCHMARKS_ENABLED="${BENCHMARKS_ENABLED:-}"

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B-Base}"
export WANDB_PROJECT="${WANDB_PROJECT:-anonymized}"
export GRPO_BETA_1="${GRPO_BETA_1:-0.0}"
export GRPO_BETA_2="${GRPO_BETA_2:-1000.0}"

cd "$(dirname "$0")/.."
python -m baselines.weighted_rd
