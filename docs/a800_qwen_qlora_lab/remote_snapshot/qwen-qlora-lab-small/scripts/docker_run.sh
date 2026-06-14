#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/root/zhl/qwen-qlora-lab"
mkdir -p "$PROJECT_DIR/logs" /root/.cache/huggingface /root/.cache/modelscope
docker run --rm \
  --gpus device=0 \
  --ipc=host \
  --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HOME=/root/.cache/huggingface \
  -e MODELSCOPE_CACHE=/root/.cache/modelscope \
  -v "$PROJECT_DIR:/workspace/qwen-qlora-lab" \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/modelscope:/root/.cache/modelscope \
  -w /workspace/qwen-qlora-lab \
  qwen-qlora-swift:latest "$@"