#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/workspace/qwen-qlora-lab"
if [ ! -d "$PROJECT_DIR" ]; then PROJECT_DIR="/root/zhl/qwen-qlora-lab"; fi
MODEL_DIR="$PROJECT_DIR/models"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$MODEL_DIR" "$LOG_DIR" /root/.cache/modelscope /root/.cache/huggingface
LOG="$LOG_DIR/model_downloads.log"
: > "$LOG"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
verify_model_dir() {
  local dir="$1"
  test -f "$dir/config.json" || return 1
  find "$dir" -maxdepth 2 -type f \( -name '*.safetensors' -o -name 'model*.bin' \) | grep -q . || return 1
  find "$dir" -maxdepth 2 -type f \( -name '*tokenizer*' -o -name 'vocab*' -o -name 'merges.txt' \) | grep -q . || return 1
}

download_one() {
  local model_id="$1"
  local target="$2"
  mkdir -p "$target"
  if verify_model_dir "$target"; then
    log "SKIP existing model: $model_id -> $target"
    du -sh "$target" | tee -a "$LOG"
    return 0
  fi
  log "Downloading with ModelScope: $model_id -> $target"
  if python - "$model_id" "$target" <<'PY'
import sys
from modelscope import snapshot_download
model_id, target = sys.argv[1], sys.argv[2]
snapshot_download(model_id, local_dir=target)
PY
  then
    if verify_model_dir "$target"; then
      log "ModelScope download OK: $model_id"
      du -sh "$target" | tee -a "$LOG"
      return 0
    fi
    log "ModelScope returned but verification failed: $model_id"
  else
    log "ModelScope download failed: $model_id"
  fi

  log "Trying Hugging Face fallback: $model_id"
  if command -v huggingface-cli >/dev/null 2>&1; then
    if huggingface-cli download "$model_id" --local-dir "$target" --local-dir-use-symlinks False; then
      if verify_model_dir "$target"; then
        log "Hugging Face download OK: $model_id"
        du -sh "$target" | tee -a "$LOG"
        return 0
      fi
    fi
  fi
  log "FAILED to download/verify: $model_id"
  return 1
}

download_one 'Qwen/Qwen3-1.7B' "$MODEL_DIR/Qwen3-1.7B"
download_one 'Qwen/Qwen3.5-0.8B' "$MODEL_DIR/Qwen3.5-0.8B"
if ! download_one 'Qwen/Qwen3.5-2B' "$MODEL_DIR/Qwen3.5-2B"; then
  log "Qwen3.5-2B unavailable; multimodal round will fallback to Qwen3.5-0.8B"
  touch "$MODEL_DIR/Qwen3.5-2B.DOWNLOAD_FAILED"
fi
log "Final model directory sizes:"
du -sh "$MODEL_DIR"/* 2>/dev/null | tee -a "$LOG" || true