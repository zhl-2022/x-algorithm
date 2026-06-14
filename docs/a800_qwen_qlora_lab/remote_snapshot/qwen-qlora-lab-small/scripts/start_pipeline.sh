#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/root/zhl/qwen-qlora-lab"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
if [ "$FREE_MIB" -lt 18000 ] && [ "${ALLOW_LOW_FREE:-0}" != "1" ]; then
  echo "A800 free memory is ${FREE_MIB} MiB, below 18000 MiB. Not starting. Set ALLOW_LOW_FREE=1 to override." | tee -a "$LOG_DIR/start_pipeline.log"
  exit 1
fi
if ! docker image inspect qwen-qlora-swift:latest >/dev/null 2>&1; then
  echo "Image qwen-qlora-swift:latest not found; build it first with scripts/build_image.sh" | tee -a "$LOG_DIR/start_pipeline.log"
  exit 1
fi
RUN_LOG="$LOG_DIR/pipeline_$(date '+%Y%m%d_%H%M%S').log"
nohup docker run --rm \
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
  qwen-qlora-swift:latest bash scripts/run_pipeline_inside.sh > "$RUN_LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOG_DIR/pipeline.pid"
echo "$RUN_LOG" > "$LOG_DIR/latest_pipeline_log.txt"
echo "Started Qwen QLoRA pipeline PID=$PID log=$RUN_LOG" | tee -a "$LOG_DIR/start_pipeline.log"