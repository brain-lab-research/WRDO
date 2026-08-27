#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[[ -f "$SCRIPT_DIR/local_env.sh" ]] && source "$SCRIPT_DIR/local_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set -euo pipefail
export EVALUATION_BACKEND="${EVALUATION_BACKEND:-wildguard}"
export EVALUATE_LOCALITY="${EVALUATE_LOCALITY:-false}"
export MMLU_ENABLED="${MMLU_ENABLED:-false}"
export BENCHMARKS_ENABLED="${BENCHMARKS_ENABLED:-}"

cd "$(dirname "$0")/.."
python -m baselines.heretic
