#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/zhl/qwen-qlora-lab}"
IMAGE="${IMAGE:-qwen-qlora-swift:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-17b-base-swift-deploy}"
PORT="${1:-${QWEN3_BASE_PORT:-18080}}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen3-1.7B}"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "Missing model config: $MODEL_DIR/config.json" >&2
  exit 1
fi

if ss -ltn | grep -q ":${PORT} "; then
  echo "Port ${PORT} is already in use. Choose another port." >&2
  exit 1
fi

existing_id="$(docker ps -aq --filter "name=^/${CONTAINER_NAME}$" | head -n 1 || true)"
if [ -n "$existing_id" ]; then
  running_id="$(docker ps -q --filter "name=^/${CONTAINER_NAME}$" | head -n 1 || true)"
  if [ -n "$running_id" ]; then
    echo "Container ${CONTAINER_NAME} is already running."
    docker ps --filter "name=^/${CONTAINER_NAME}$"
    exit 0
  fi
  docker rm "$existing_id" >/dev/null
fi

nvidia-smi > "$LOG_DIR/qwen3_17b_base_deploy.before.nvidia-smi.txt"

docker run -d \
  --name "$CONTAINER_NAME" \
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
  -p "${PORT}:${PORT}" \
  "$IMAGE" \
  bash -lc "swift deploy \
    --model models/Qwen3-1.7B \
    --model_type qwen3 \
    --template qwen3 \
    --torch_dtype bfloat16 \
    --infer_backend pt \
    --host 0.0.0.0 \
    --port ${PORT} \
    --served_model_name qwen3-1.7b-base \
    --max_length 2048 \
    --max_new_tokens 512 \
    --temperature 0.2 \
    --stream false" \
  > "$LOG_DIR/qwen3_17b_base_deploy.container_id.txt"

echo "Waiting for http://127.0.0.1:${PORT}/v1/models ..."
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$LOG_DIR/qwen3_17b_base_deploy.models.json"; then
    nvidia-smi > "$LOG_DIR/qwen3_17b_base_deploy.after.nvidia-smi.txt"
    echo "Qwen3-1.7B base deploy is ready on port ${PORT}."
    docker ps --filter "name=^/${CONTAINER_NAME}$"
    exit 0
  fi
  sleep 2
done

echo "Deploy did not become ready in time. Last container logs:" >&2
docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
exit 1
