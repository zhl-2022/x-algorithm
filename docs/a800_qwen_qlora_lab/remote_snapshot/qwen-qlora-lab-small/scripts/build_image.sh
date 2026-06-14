#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/root/zhl/qwen-qlora-lab"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"
{
  echo "[$(date '+%F %T')] Building qwen-qlora-swift:latest"
  docker build -t qwen-qlora-swift:latest -f docker/Dockerfile .
  echo "[$(date '+%F %T')] Build complete"
} 2>&1 | tee "$LOG_DIR/build_image.log"