#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[[ -f "$SCRIPT_DIR/local_env.sh" ]] && source "$SCRIPT_DIR/local_env.sh"
# Respect a preconfigured multi-GPU mask; fall back to a single visible GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set -euo pipefail

# ---- Fast smoke debug ----
# When true, weighted_rd uses only a small subset of questions and a smaller rollout noise scale.
export DEBUG="${DEBUG:-false}"
export DEBUG_N_QUESTIONS="${DEBUG_N_QUESTIONS:-4}"
export DEBUG_NOISE_SCALE="${DEBUG_NOISE_SCALE:-0.02}"

# ---- Model ----
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B-Base}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export WANDB_PROJECT="${WANDB_PROJECT:-anonymized}"

# ---- Refusal-direction prompt sampling ----
export WEIGHTED_RD_DIRECTION_PROMPT_COUNT="${WEIGHTED_RD_DIRECTION_PROMPT_COUNT:-128}"
export WEIGHTED_RD_DIRECTION_PROMPT_SEED="${WEIGHTED_RD_DIRECTION_PROMPT_SEED:-42}"

# ---- GRPO-IS training hyperparameters ----
export GRPO_N_GROUPS="${GRPO_N_GROUPS:-8}"
export GRPO_N_EPOCHS="${GRPO_N_EPOCHS:-100}"
export GRPO_LEARNING_RATE="${GRPO_LEARNING_RATE:-5e-3}"
export GRPO_NOISE_SCALE="${GRPO_NOISE_SCALE:-100}"

# ---- GRPO-IS specific: shared intervention scale for all sampled W rollout policies ----
export GRPO_REF_ALPHA="${GRPO_REF_ALPHA:-1.0}"
export IS_CLIP_RATIO="${IS_CLIP_RATIO:-5.0}"
export GRPO_CLIP_RATIO="${GRPO_CLIP_RATIO:-0.2}"
export GRPO_LOSS_AGG_MODE="${GRPO_LOSS_AGG_MODE:-token-mean}"
export GRPO_BETA_1="${GRPO_BETA_1:-0.0}"
export GRPO_BETA_2="${GRPO_BETA_2:-1000.0}"

# ---- Abliteration parameters ----
export ABLITERATION_MAX_WEIGHT="${ABLITERATION_MAX_WEIGHT:-2.0}"
export ABLITERATION_MAX_WEIGHT_POSITION="${ABLITERATION_MAX_WEIGHT_POSITION:-0.7}"
export ABLITERATION_MIN_WEIGHT="${ABLITERATION_MIN_WEIGHT:-0.1}"
export ABLITERATION_MIN_WEIGHT_DISTANCE="${ABLITERATION_MIN_WEIGHT_DISTANCE:-0.3}"

# ---- Weight initialization ----
export WEIGHTS_MODE="${WEIGHTS_MODE:-scalar}"
export WEIGHTS_INIT_TYPE="${WEIGHTS_INIT_TYPE:-average}"

# ---- Optimizer selection ----
export OPTIMIZER_METHOD="${OPTIMIZER_METHOD:-grpo}"

# ---- Optuna weight search ----
export OPTUNA_SAMPLER="${OPTUNA_SAMPLER:-cmaes}"
export OPTUNA_N_TRIALS="${OPTUNA_N_TRIALS:-50}"
export OPTUNA_SAMPLER_SEED="${OPTUNA_SAMPLER_SEED:-42}"
export OPTUNA_WEIGHT_MIN="${OPTUNA_WEIGHT_MIN:--2.0}"
export OPTUNA_WEIGHT_MAX="${OPTUNA_WEIGHT_MAX:-2.0}"

# ---- Evaluation backend ----
# Options: wildguard | llamaguard | qwen3guard
export EVALUATION_BACKEND="${EVALUATION_BACKEND:-wildguard}"
export REWARD_METRIC="${REWARD_METRIC:-}"

# ---- LlamaGuard settings (used by llamaguard) ----
export LLAMAGUARD_MODEL_NAME="${LLAMAGUARD_MODEL_NAME:-meta-llama/Meta-Llama-Guard-2-8B}"
export LLAMAGUARD_DEVICE="${LLAMAGUARD_DEVICE:-cuda}"
export LLAMAGUARD_DTYPE="${LLAMAGUARD_DTYPE:-bfloat16}"

# ---- WildGuard settings (used by wildguard) ----
export WILDGUARD_MODEL_NAME="${WILDGUARD_MODEL_NAME:-allenai/wildguard}"
export WILDGUARD_DEVICE="${WILDGUARD_DEVICE:-cuda}"
export WILDGUARD_DTYPE="${WILDGUARD_DTYPE:-bfloat16}"
export WILDGUARD_MAX_NEW_TOKENS="${WILDGUARD_MAX_NEW_TOKENS:-32}"

# ---- Qwen3Guard settings (used by qwen3guard) ----
export QWEN3GUARD_MODEL_NAME="${QWEN3GUARD_MODEL_NAME:-Qwen/Qwen3Guard-Gen-8B}"
export QWEN3GUARD_DEVICE="${QWEN3GUARD_DEVICE:-auto}"
export QWEN3GUARD_DTYPE="${QWEN3GUARD_DTYPE:-auto}"
export QWEN3GUARD_MAX_NEW_TOKENS="${QWEN3GUARD_MAX_NEW_TOKENS:-128}"

# ---- Harmfulness-only evaluation ----
export EVALUATE_LOCALITY="${EVALUATE_LOCALITY:-false}"
export MMLU_ENABLED="${MMLU_ENABLED:-false}"
export ACADEMIC_BENCHMARKS_ENABLED="${ACADEMIC_BENCHMARKS_ENABLED:-}"
export BENCHMARKS_ENABLED="${BENCHMARKS_ENABLED:-}"

cd "$(dirname "$0")/.."
python -m baselines.weighted_rd
