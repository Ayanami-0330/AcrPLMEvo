#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
ATTEMPT_TIMEOUT="${ATTEMPT_TIMEOUT:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-15}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
STALL_LIMIT="${STALL_LIMIT:-20}"
PREFETCH_RETRIES="${PREFETCH_RETRIES:-200}"

if [[ -z "$MODEL" ]]; then
  echo "Usage: $0 <model_key>"
  echo "model_key in: protbert prott5 esm2_3b ankh_large"
  exit 1
fi

case "$MODEL" in
  protbert)
    CACHE_SUBDIR="models--Rostlab--prot_bert_bfd"
    PREFETCH_MODEL="protbert"
    ;;
  prott5)
    CACHE_SUBDIR="models--Rostlab--prot_t5_xl_uniref50"
    PREFETCH_MODEL="prott5"
    ;;
  esm2_3b)
    CACHE_SUBDIR="models--facebook--esm2_t36_3B_UR50D"
    PREFETCH_MODEL="esm2"
    ;;
  ankh_large)
    CACHE_SUBDIR="models--ElnaggarLab--ankh-large"
    PREFETCH_MODEL="ankh"
    ;;
  *)
    echo "Unsupported model key: $MODEL"
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$SCRIPT_DIR}"
LOG_FILE="$ROOT_DIR/${MODEL}_prefetch_supervisor.log"
HF_CACHE_DIR="${ACRPLMEVO_HF_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}"
CACHE_ROOT="$HF_CACHE_DIR/$CACHE_SUBDIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOCK_FILE="/tmp/prefetch_supervisor_${MODEL}.lock"

# Single-instance guard to prevent concurrent downloaders fighting over the same lock/incomplete blob.
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] supervisor already running for model=$MODEL (lock=$LOCK_FILE)"
  exit 0
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  :
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME:-lm-hf}" || true
fi

export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export HF_HUB_DISABLE_XET=1
: "${HF_HUB_ETAG_TIMEOUT:=120}"
: "${HF_HUB_DOWNLOAD_TIMEOUT:=1800}"
export HF_HUB_ETAG_TIMEOUT
export HF_HUB_DOWNLOAD_TIMEOUT

touch "$LOG_FILE"

echo "[$(date '+%F %T')] supervisor start model=$MODEL pid=$$ timeout=${ATTEMPT_TIMEOUT}s monitor=${MONITOR_INTERVAL}s stall_limit=${STALL_LIMIT}" | tee -a "$LOG_FILE"
attempt=0
while true; do
  attempt=$((attempt+1))
  echo "[$(date '+%F %T')] attempt=$attempt start" | tee -a "$LOG_FILE"
  set +e
  if [[ "$ATTEMPT_TIMEOUT" -gt 0 ]]; then
    timeout "$ATTEMPT_TIMEOUT" "$PYTHON_BIN" "$ROOT_DIR/prefetch_backbones.py" --models "$PREFETCH_MODEL" --cache-dir "$HF_CACHE_DIR" --disable-xet --retries "$PREFETCH_RETRIES" --sleep-seconds "$SLEEP_SECONDS" >> "$LOG_FILE" 2>&1 &
  else
    "$PYTHON_BIN" "$ROOT_DIR/prefetch_backbones.py" --models "$PREFETCH_MODEL" --cache-dir "$HF_CACHE_DIR" --disable-xet --retries "$PREFETCH_RETRIES" --sleep-seconds "$SLEEP_SECONDS" >> "$LOG_FILE" 2>&1 &
  fi
  worker_pid=$!
  stall_count=0
  prev_total=-1
  while kill -0 "$worker_pid" 2>/dev/null; do
    total_now=$(find "$CACHE_ROOT" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
    if [[ "$total_now" -gt "$prev_total" ]]; then
      if [[ "$prev_total" -ge 0 ]]; then
        echo "[$(date '+%F %T')] progress bytes=$total_now delta=$((total_now-prev_total))" | tee -a "$LOG_FILE"
      else
        echo "[$(date '+%F %T')] progress bytes=$total_now" | tee -a "$LOG_FILE"
      fi
      prev_total=$total_now
      stall_count=0
    else
      stall_count=$((stall_count+1))
      echo "[$(date '+%F %T')] no_growth bytes=$total_now stall_count=$stall_count/$STALL_LIMIT" | tee -a "$LOG_FILE"
      if [[ "$stall_count" -ge "$STALL_LIMIT" ]]; then
        echo "[$(date '+%F %T')] stall_detected kill worker pid=$worker_pid" | tee -a "$LOG_FILE"
        kill "$worker_pid" 2>/dev/null || true
        sleep 1
        kill -9 "$worker_pid" 2>/dev/null || true
        break
      fi
    fi
    sleep "$MONITOR_INTERVAL"
  done
  wait "$worker_pid"
  rc=$?
  set -e
  if [[ "$rc" -eq 124 ]]; then
    echo "[$(date '+%F %T')] attempt=$attempt timed_out after ${ATTEMPT_TIMEOUT}s" | tee -a "$LOG_FILE"
  elif [[ "$rc" -ne 0 ]]; then
    echo "[$(date '+%F %T')] attempt=$attempt exited rc=$rc" | tee -a "$LOG_FILE"
  else
    echo "[$(date '+%F %T')] attempt=$attempt exited rc=0" | tee -a "$LOG_FILE"
  fi

  "$PYTHON_BIN" - <<PY | tee -a "$LOG_FILE"
from pathlib import Path
root = Path("$CACHE_ROOT")
if not root.exists():
    print("status root_missing")
else:
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    inc = sorted(root.rglob("*.incomplete"))
    print("status total_bytes", total, "incomplete_count", len(inc))
    if inc:
        p = inc[-1]
        st = p.stat()
        print("latest_incomplete", st.st_size, int(st.st_mtime), p)
PY

  if grep -q "\[prefetch\] done ${PREFETCH_MODEL}" "$LOG_FILE"; then
    echo "[$(date '+%F %T')] ${MODEL} prefetch finished" | tee -a "$LOG_FILE"
    break
  fi

  echo "[$(date '+%F %T')] retry after ${SLEEP_SECONDS}s" | tee -a "$LOG_FILE"
  sleep "$SLEEP_SECONDS"
done
