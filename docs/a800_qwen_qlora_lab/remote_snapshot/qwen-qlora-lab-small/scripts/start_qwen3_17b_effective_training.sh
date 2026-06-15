#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/zhl/qwen-qlora-lab}"
mkdir -p "$PROJECT_DIR/logs"

bash "$PROJECT_DIR/scripts/stop_qwen3_17b_deploy.sh" || true

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
echo "free_mib=$free_mib"
if [ "$free_mib" -lt 20000 ]; then
  echo "Free GPU memory below 20000 MiB; aborting to avoid affecting existing services." >&2
  exit 1
fi

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
  qwen-qlora-swift:latest \
  bash scripts/train_qwen3_17b_effective_inside.sh
