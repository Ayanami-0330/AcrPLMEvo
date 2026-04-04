#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_ROOT="$(cd "$ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$EXP_ROOT/results/runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EXP_ROOT/results/runs_frozen}"
RESULTS_CSV="${RESULTS_CSV:-$EXP_ROOT/results/experiments_frozen.csv}"
MODELS="${MODELS:-ankh,esm2,protbert,prott5}"
SEEDS="${SEEDS:-11,22,33,44,55}"
ADAPTER_TYPES="${ADAPTER_TYPES:-lora,dora}"
VARIANTS="${VARIANTS:-lm_only,lm_pssm}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/_state" "$EXP_ROOT/results/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$EXP_ROOT/results/logs/supplemental_frozen_eval_${STAMP}.log"

CMD="$PYTHON_BIN frozen_baseline/run_supplemental_frozen_eval.py --resume --adapter-root $ADAPTER_ROOT --output-root $OUTPUT_ROOT --results-csv $RESULTS_CSV --models $MODELS --seeds $SEEDS --adapter-types $ADAPTER_TYPES --variants $VARIANTS"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] $CMD --dry-run" | tee -a "$LOG_FILE"
  $PYTHON_BIN frozen_baseline/run_supplemental_frozen_eval.py --resume --adapter-root "$ADAPTER_ROOT" --output-root "$OUTPUT_ROOT" --results-csv "$RESULTS_CSV" --models "$MODELS" --seeds "$SEEDS" --adapter-types "$ADAPTER_TYPES" --variants "$VARIANTS" --dry-run | tee -a "$LOG_FILE"
else
  echo "supplemental frozen eval started at $(date)" | tee -a "$LOG_FILE"
  eval "$CMD" | tee -a "$LOG_FILE"
  echo "supplemental frozen eval finished at $(date)" | tee -a "$LOG_FILE"
fi
