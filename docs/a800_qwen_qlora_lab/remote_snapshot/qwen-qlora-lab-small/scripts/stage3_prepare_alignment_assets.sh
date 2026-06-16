#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/zhl/qwen-qlora-lab}"
cd "$PROJECT_DIR"

mkdir -p data logs outputs exports
python3 scripts/stage3_make_alignment_data.py | tee logs/stage3_prepare_alignment_assets.log
python3 scripts/stage3_reward_plugin.py | tee logs/stage3_reward_selftest.log

echo
echo "Stage 3 alignment assets are ready:"
ls -lh \
  data/stage3_dpo_preferences.jsonl \
  data/stage3_grpo_prompts.jsonl \
  data/stage3_alignment_eval_cases.jsonl \
  logs/stage3_prepare_alignment_assets.log \
  logs/stage3_reward_selftest.log
