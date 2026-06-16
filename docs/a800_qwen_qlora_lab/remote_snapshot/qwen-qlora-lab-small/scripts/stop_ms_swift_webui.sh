#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-qwen-qlora-swift-webui}"

if docker ps -q --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
  echo "Stopping ${CONTAINER_NAME} ..."
  docker stop "$CONTAINER_NAME" >/dev/null
fi

if docker ps -aq --filter "name=^/${CONTAINER_NAME}$" | grep -q .; then
  echo "Removing ${CONTAINER_NAME} ..."
  docker rm "$CONTAINER_NAME" >/dev/null
fi

echo "ms-swift WebUI container stopped."
