#!/usr/bin/env bash
set -euo pipefail

MODE="interactive"
if [[ "${1:-}" == "--detached" || "${1:-}" == "-d" ]]; then
  MODE="detached"
fi

IMAGE="${XALGO_MLU_IMAGE:-cambricon-base/pytorch:v25.12.0-torch2.9.1-torchmlu1.30.2-ubuntu22.04-py310}"
DEVICES="${MLU_VISIBLE_DEVICES:-2,3}"
PROJECT_DIR="${XALGO_PROJECT_DIR:-/root/zhl/x-algorithm}"
DEV_CONTAINER="${XALGO_MLU_DEV_CONTAINER:-xalgorithm-mlu-dev}"
DETACHED_CONTAINER="${XALGO_MLU_CONTAINER:-xalgorithm-mlu}"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Project directory not found: $PROJECT_DIR" >&2
  echo "Sync or clone this repository to $PROJECT_DIR before starting the container." >&2
  exit 1
fi

echo "Checking MLU status before starting container..."
if command -v cnmon >/dev/null 2>&1; then
  cnmon | head -n 80
else
  echo "cnmon not found on host; continuing without device snapshot." >&2
fi

COMMON_ARGS=(
  --ipc host
  --privileged
  -v /etc/cambricon:/etc/cambricon:ro
  -v /usr/bin/cnmon:/usr/bin/cnmon:ro
  -v /root/zhl:/root/zhl
  -v /data1/model:/data1/model:ro
  -e "MLU_VISIBLE_DEVICES=$DEVICES"
  -e LANG=C.UTF-8
  -e TZ=Asia/Shanghai
  -e PYTHONUNBUFFERED=1
  -w "$PROJECT_DIR"
)

if [[ "$MODE" == "detached" ]]; then
  if docker ps -a --format '{{.Names}}' | grep -Fxq "$DETACHED_CONTAINER"; then
    echo "Container already exists: $DETACHED_CONTAINER" >&2
    echo "Use: docker exec -it $DETACHED_CONTAINER bash" >&2
    exit 1
  fi

  docker run -d \
    --name "$DETACHED_CONTAINER" \
    "${COMMON_ARGS[@]}" \
    "$IMAGE" \
    bash -lc "source /torch/venv3/pytorch/bin/activate 2>/dev/null || true; tail -f /dev/null"

  echo "Started detached container: $DETACHED_CONTAINER"
  echo "Enter with: docker exec -it $DETACHED_CONTAINER bash"
  echo "After entering, run: source /torch/venv3/pytorch/bin/activate"
else
  docker run -it --rm \
    --name "$DEV_CONTAINER" \
    "${COMMON_ARGS[@]}" \
    "$IMAGE" \
    bash -lc "source /torch/venv3/pytorch/bin/activate 2>/dev/null || true; exec bash"
fi
