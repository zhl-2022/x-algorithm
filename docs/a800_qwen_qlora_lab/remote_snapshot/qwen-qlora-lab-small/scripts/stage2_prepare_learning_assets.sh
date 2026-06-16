#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/zhl/qwen-qlora-lab}"
cd "$PROJECT_DIR"

mkdir -p data logs exports
python3 scripts/stage2_make_learning_data.py | tee logs/stage2_prepare_learning_assets.log

echo
echo "Stage 2 learning assets are ready:"
ls -lh data/stage2_* logs/stage2_prepare_learning_assets.log
