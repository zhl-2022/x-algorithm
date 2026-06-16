#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/zhl/qwen-qlora-lab}"
cd "$PROJECT_DIR"

echo "== Stage 3 readiness =="
echo "project: $PROJECT_DIR"
echo

echo "== disk =="
df -h /root
echo

echo "== gpu =="
nvidia-smi
echo

echo "== required files =="
required=(
  "models/Qwen3-1.7B/config.json"
  "data/stage3_dpo_preferences.jsonl"
  "data/stage3_grpo_prompts.jsonl"
  "data/stage3_alignment_eval_cases.jsonl"
  "scripts/stage3_reward_plugin.py"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "missing: $path"
    exit 1
  fi
  echo "ok: $path"
done
echo

echo "== latest effective SFT checkpoint =="
latest_ckpt="$(find outputs/qwen3_17b_text_effective -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true)"
if [[ -z "$latest_ckpt" ]]; then
  echo "missing effective SFT checkpoint under outputs/qwen3_17b_text_effective"
  exit 1
fi
echo "$latest_ckpt"
echo

echo "== swift rlhf parameter check =="
docker run --rm --gpus device=0 \
  -v "$PROJECT_DIR:/workspace/qwen-qlora-lab" \
  qwen-qlora-swift:latest \
  swift rlhf --help 2>/dev/null |
  grep -E -- '--(rlhf_type|tuner_type|reward_funcs|num_generations|use_vllm|beta|rpo_alpha)' | head -n 20
echo

echo "Stage 3 readiness check passed."
