#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qwen-qlora-lab
mkdir -p logs
{
  echo "[$(date '+%F %T')] Pipeline started inside container"
  python scripts/generate_data.py
  bash scripts/download_models.sh
  bash scripts/train_rounds_inside.sh
  bash scripts/infer_rounds_inside.sh || true
  echo "[$(date '+%F %T')] Pipeline finished"
} 2>&1 | tee -a logs/pipeline_inside.log