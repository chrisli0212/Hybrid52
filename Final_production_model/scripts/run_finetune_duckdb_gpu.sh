#!/usr/bin/env bash
set -euo pipefail

# Finetune Stage1 checkpoints using current DuckDB-derived Tier3 data.
# Default paths can be overridden via env vars when needed.

DATA_ROOT="${DATA_ROOT:-/workspace/data/tier3_binary_2026_duckdb}"
HORIZON_DIR="${HORIZON_DIR:-horizon_30min}"
MODEL_DIR="${MODEL_DIR:-/workspace/Final_production_model/models/stage1}"

EPOCHS="${EPOCHS:-12}"
LR_BACKBONE="${LR_BACKBONE:-1e-5}"
LR_HEAD="${LR_HEAD:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MIN_SAVE_AUC="${MIN_SAVE_AUC:-0.51}"
MIN_IMPROVE_AUC="${MIN_IMPROVE_AUC:-0.002}"

python "/workspace/Final_production_model/scripts/finetune_backbone.py" \
  --data-root "${DATA_ROOT}" \
  --horizon-dir "${HORIZON_DIR}" \
  --model-dir "${MODEL_DIR}" \
  --epochs "${EPOCHS}" \
  --lr-backbone "${LR_BACKBONE}" \
  --lr-head "${LR_HEAD}" \
  --batch-size "${BATCH_SIZE}" \
  --min-save-auc "${MIN_SAVE_AUC}" \
  --min-improve-auc "${MIN_IMPROVE_AUC}"

