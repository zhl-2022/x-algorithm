#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/zhl/qwen-qlora-lab}"
IMAGE="${IMAGE:-qwen-qlora-swift:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-17b-effective-lora-swift-deploy}"
PORT="${1:-${QWEN3_EFFECTIVE_PORT:-19082}}"
OUT_DIR="${OUT_DIR:-outputs/qwen3_17b_text_effective}"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

ckpt="$(find "$PROJECT_DIR/$OUT_DIR" -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true)"
if [ -z "$ckpt" ]; then
  echo "No checkpoint found under $PROJECT_DIR/$OUT_DIR" >&2
  exit 1
fi
adapter_rel="${ckpt#$PROJECT_DIR/}"

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

nvidia-smi > "$LOG_DIR/qwen3_17b_effective_deploy.before.nvidia-smi.txt"

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
    --adapters ${adapter_rel} \
    --model_type qwen3 \
    --template qwen3 \
    --torch_dtype bfloat16 \
    --infer_backend pt \
    --host 0.0.0.0 \
    --port ${PORT} \
    --served_model_name qwen3-1.7b-qlora-effective \
    --max_length 2048 \
    --max_new_tokens 512 \
    --temperature 0.2 \
    --stream false" \
  > "$LOG_DIR/qwen3_17b_effective_deploy.container_id.txt"

echo "Waiting for http://127.0.0.1:${PORT}/v1/models ..."
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "$LOG_DIR/qwen3_17b_effective_deploy.models.json"; then
    nvidia-smi > "$LOG_DIR/qwen3_17b_effective_deploy.after.nvidia-smi.txt"
    echo "Qwen3-1.7B effective LoRA deploy is ready on port ${PORT}."
    docker ps --filter "name=^/${CONTAINER_NAME}$"
    exit 0
  fi
  sleep 2
done

echo "Deploy did not become ready in time. Last container logs:" >&2
docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
exit 1
