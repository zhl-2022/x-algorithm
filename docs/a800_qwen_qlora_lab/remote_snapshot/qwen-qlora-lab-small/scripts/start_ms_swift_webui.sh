#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/zhl/qwen-qlora-lab}"
IMAGE="${IMAGE:-qwen-qlora-swift:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-qlora-swift-webui}"
HOST_PORT="${1:-${SWIFT_WEBUI_HOST_PORT:-17860}}"
CONTAINER_PORT="${CONTAINER_PORT:-7860}"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$LOG_DIR"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Docker image not found: $IMAGE" >&2
  echo "Build it first with: bash $PROJECT_DIR/scripts/build_image.sh" >&2
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

if ss -ltn | grep -q ":${HOST_PORT} "; then
  echo "Host port ${HOST_PORT} is already in use. Choose another port." >&2
  exit 1
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus device=0 \
  --ipc=host \
  --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e SWIFT_UI_LANG=zh \
  -e BROWSER=/bin/true \
  -e NO_PROXY=127.0.0.1,localhost,::1 \
  -e no_proxy=127.0.0.1,localhost,::1 \
  -e HF_HOME=/root/.cache/huggingface \
  -e MODELSCOPE_CACHE=/root/.cache/modelscope \
  -v "$PROJECT_DIR:/workspace/qwen-qlora-lab" \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/modelscope:/root/.cache/modelscope \
  -w /workspace/qwen-qlora-lab \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  "$IMAGE" \
  bash -lc "swift web-ui --lang zh --server_name 0.0.0.0 --server_port ${CONTAINER_PORT}" \
  > "$LOG_DIR/ms_swift_webui.container_id.txt"

echo "Waiting for ms-swift WebUI on http://127.0.0.1:${HOST_PORT} ..."
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${HOST_PORT}/" > "$LOG_DIR/ms_swift_webui.index.html"; then
    echo "ms-swift WebUI is ready on A800 port ${HOST_PORT}."
    docker ps --filter "name=^/${CONTAINER_NAME}$"
    exit 0
  fi
  sleep 2
done

echo "WebUI did not become ready in time. Last container logs:" >&2
docker logs --tail 120 "$CONTAINER_NAME" >&2 || true
exit 1
