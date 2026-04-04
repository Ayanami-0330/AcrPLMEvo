#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_ROOT="$(cd "$ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS="${MODELS:-ankh,esm2,protbert,prott5}"
SEEDS="${SEEDS:-11,22,33,44,55}"
EPOCHS="${EPOCHS:-8}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] $*"
  else
    eval "$*"
  fi
}

cd "$ROOT"
mkdir -p "$EXP_ROOT/results/logs" "$EXP_ROOT/results/runs"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$EXP_ROOT/results/logs/phase_a_${STAMP}.log"

echo "phase A started at $(date)" | tee -a "$LOG_FILE"

a_protocol="$PYTHON_BIN main.py run-10 --models $MODELS --seeds $SEEDS --epochs $EPOCHS --resume"

run_cmd "$a_protocol" | tee -a "$LOG_FILE"
run_cmd "find $EXP_ROOT/results/runs -name full_model_state.pt -type f -delete" | tee -a "$LOG_FILE"
run_cmd "$PYTHON_BIN main.py summary --output-root $EXP_ROOT/results" | tee -a "$LOG_FILE"

echo "phase A finished at $(date)" | tee -a "$LOG_FILE"
