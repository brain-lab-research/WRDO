#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
[[ -f "$SCRIPT_DIR/local_env.sh" ]] && source "$SCRIPT_DIR/local_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set -euo pipefail
# Default to "activation" — honest RDO comparison (runtime hooks at 3 sites/layer,
# matches upstream geometry-of-refusal). Switch to "weights" for our weight-surgery
# variant via Model.abliterate.
export RDO_APPLY_METHOD="${RDO_APPLY_METHOD:-activation}"
# No effect in activation mode; set true only with RDO_APPLY_METHOD=weights for
# Arditi-equivalent weight surgery.
export RDO_ABLATE_EMBEDDING="${RDO_ABLATE_EMBEDDING:-false}"
export RESULTS_SAVE_BEST_MODEL="${RESULTS_SAVE_BEST_MODEL:-false}"
export EVALUATION_BACKEND="${EVALUATION_BACKEND:-wildguard}"
export EVALUATE_LOCALITY="${EVALUATE_LOCALITY:-false}"
export MMLU_ENABLED="${MMLU_ENABLED:-false}"
export BENCHMARKS_ENABLED="${BENCHMARKS_ENABLED:-}"

cd "$(dirname "$0")/.."
python -m baselines.rdo
