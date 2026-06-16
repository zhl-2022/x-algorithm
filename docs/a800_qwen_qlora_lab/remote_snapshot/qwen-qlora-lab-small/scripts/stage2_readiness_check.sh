#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/zhl/qwen-qlora-lab}"
IMAGE="${IMAGE:-qwen-qlora-swift:latest}"
MODEL_DIR="$PROJECT_DIR/models/Qwen3-1.7B"
OUT_DIR="$PROJECT_DIR/outputs/qwen3_17b_text_effective"

echo "host: $(hostname)"
echo "date: $(date -Is)"
echo "project: $PROJECT_DIR"
echo

echo "== disk =="
df -h /root
echo

echo "== gpu =="
nvidia-smi
echo

echo "== required files =="
test -d "$PROJECT_DIR" && echo "ok project dir"
test -f "$MODEL_DIR/config.json" && echo "ok model config: $MODEL_DIR/config.json"

ckpt="$(find "$OUT_DIR" -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true)"
if [ -z "$ckpt" ]; then
  echo "missing checkpoint under $OUT_DIR" >&2
  exit 1
fi
echo "ok latest checkpoint: $ckpt"

docker image inspect "$IMAGE" >/dev/null
echo "ok docker image: $IMAGE"

echo
echo "== deploy containers =="
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'qwen3-17b|NAMES' || true

echo
echo "Stage 2 readiness check passed."
