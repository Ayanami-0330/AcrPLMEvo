#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
HF_CACHE_DIR="${ACRPLMEVO_HF_CACHE_DIR:-${HF_HOME:-$HOME/.cache/huggingface}}"
export HF_HOME="$HF_CACHE_DIR"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE_DIR"
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export ACRPLMEVO_OFFLINE=${ACRPLMEVO_OFFLINE:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS=${PREFETCH_MODELS:-protbert,prott5,esm2,ankh}
SKIP_PREFETCH=${SKIP_PREFETCH:-0}

if [[ "$SKIP_PREFETCH" != "1" ]]; then
	"$PYTHON_BIN" "$SCRIPT_DIR/prefetch_backbones.py" \
		--models "$MODELS" \
		--cache-dir "$HF_CACHE_DIR" \
		--disable-xet
fi

"$PYTHON_BIN" "$SCRIPT_DIR/main.py" run-10 "$@"