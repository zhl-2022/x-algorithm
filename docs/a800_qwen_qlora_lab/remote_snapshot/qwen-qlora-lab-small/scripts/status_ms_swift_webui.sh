#!/usr/bin/env bash
set -euo pipefail

HOST_PORT="${1:-${SWIFT_WEBUI_HOST_PORT:-17860}}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-qlora-swift-webui}"

docker ps --filter "name=^/${CONTAINER_NAME}$"
echo
ss -ltnp | grep ":${HOST_PORT} " || true
echo
if curl -fsS "http://127.0.0.1:${HOST_PORT}/" >/dev/null; then
  echo "ready: http://127.0.0.1:${HOST_PORT}/"
else
  echo "not ready: http://127.0.0.1:${HOST_PORT}/"
fi
