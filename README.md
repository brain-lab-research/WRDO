# WRDO: Weighted Refusal Direction Optimization in Aligned Language Models

## Overview

- `baselines/basic_refusal.py` - single-direction refusal baseline.
- `baselines/weighted_rd/` - WRDO with GRPO-IS or Optuna weight optimization.
- `baselines/heretic.py` - Heretic refusal-direction search baseline.
- `baselines/rdo.py` - RDO baseline with activation hooks or weight surgery.
- `dataset/` - local JSON datasets and train/validation/test splits.
- `evaluate/` - evaluation with WildGuard, LlamaGuard, or Qwen3Guard.
- `scripts/` - runnable experiment entrypoints.
- `configs/` - TOML presets documenting experiment parameters.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The main experiments require a CUDA-capable GPU, access to the selected
Hugging Face models, and enough disk space for model weights and outputs.

## Quick Start

Experiment settings are controlled through environment variables. Local
overrides can be placed in `scripts/local_env.sh`; the launch scripts
load this file automatically when it exists.

```bash
# Basic refusal-direction baseline
bash scripts/run_basic_refusal.sh

# WRDO with GRPO-IS
bash scripts/run_weighted_rd.sh

# WRDO with Optuna
bash scripts/run_weighted_rd_optuna.sh

# Heretic and RDO baselines
bash scripts/run_heretic.sh
bash scripts/run_rdo.sh
```

Common environment variables:

```bash
export MODEL_NAME="Qwen/Qwen3-8B-Base"
export CUDA_VISIBLE_DEVICES="0"
export EVALUATION_BACKEND="wildguard"   # wildguard | llamaguard | qwen3guard
export RESULTS_ROOT="results"
export WANDB_PROJECT="anonymized"
export DEBUG="true"                     # fast smoke run for weighted_rd
```

## Data and Outputs

Processed datasets are stored in `dataset/processed/`; prepared splits are
stored in `dataset/splits/`. Experiment outputs are written to
`results/<method>/`, including model responses, selected directions, metrics,
plots, checkpoints, and metadata. W&B logging is used when `wandb` is installed
and configured.

## Notes

By default, configuration is read from `config.py` and environment variables.
The TOML files in `configs/` serve as compact presets and parameter
documentation for specific experiment settings.
