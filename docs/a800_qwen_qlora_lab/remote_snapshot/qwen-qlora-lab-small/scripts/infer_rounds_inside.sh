#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/workspace/qwen-qlora-lab"
cd "$PROJECT_DIR"
mkdir -p data logs

latest_ckpt() {
  find "$1" -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true
}

write_infer_datasets() {
  python - <<'PY'
import json
from pathlib import Path

Path("data").mkdir(exist_ok=True)

text_prompts = [
    "请用三句话解释 QLoRA 为什么适合共享 A800 学习环境。",
    "请列出推荐系统离线实验需要记录的五类信息。",
    "如果训练时显存接近上限，应该优先调整哪些参数？",
]
with open("data/infer_text_prompts.jsonl", "w", encoding="utf-8") as f:
    for prompt in text_prompts:
        row = {"messages": [{"role": "user", "content": prompt}]}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

mm_prompts = [
    (
        "<image> 请描述这张推荐系统流程图，并指出训练闭环包含哪些环节。",
        "/workspace/qwen-qlora-lab/images/01_pipeline.png",
    ),
    (
        "<image> 请根据图中的指标表，说明离线评测应该关注什么。",
        "/workspace/qwen-qlora-lab/images/04_metrics.png",
    ),
]
with open("data/infer_mm_prompts.jsonl", "w", encoding="utf-8") as f:
    for prompt, image in mm_prompts:
        row = {"messages": [{"role": "user", "content": prompt}], "images": [image]}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

run_batch_infer() {
  local name="$1"
  local out_dir="$2"
  local dataset="$3"
  local suffix="$4"
  local ckpt
  ckpt=$(latest_ckpt "$out_dir")

  local md_log="logs/${name}.infer.md"
  local result_path="logs/${name}.${suffix}.result.jsonl"
  local command_log="logs/${name}.${suffix}.command.log"

  if [ -z "$ckpt" ]; then
    printf '# %s\n\nNo checkpoint found in %s\n' "$name" "$out_dir" >> "$md_log"
    return 0
  fi

  rm -f "$result_path" "$command_log"
  {
    echo
    echo "## ${suffix}"
    echo
    echo "checkpoint: $ckpt"
    echo "dataset: $dataset"
    echo "result_path: $result_path"
    echo "command_log: $command_log"
    echo
  } >> "$md_log"

  set +e
  timeout "${INFER_TIMEOUT_SECONDS:-900}" \
    swift infer \
      --adapters "$ckpt" \
      --val_dataset "$dataset" \
      --result_path "$result_path" \
      --stream false \
      --max_new_tokens 256 \
      --max_batch_size 1 \
      > "$command_log" 2>&1
  local rc=$?
  set -e

  {
    echo "exit_code: $rc"
    if [ -f "$result_path" ]; then
      echo "result_rows: $(wc -l < "$result_path")"
      echo
      echo "result_preview:"
      python - "$result_path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for idx, line in enumerate(f):
        if idx >= 2:
            break
        row = json.loads(line)
        response = row.get("response", "")
        response = response.replace("\n", " ")
        print(f"- row {idx + 1}: {response[:220]}")
PY
    else
      echo "result_rows: 0"
    fi
    echo
    echo "command_tail:"
    tail -n 20 "$command_log" || true
    echo
  } >> "$md_log"
}

write_infer_datasets

rm -f logs/round1_qwen3_17b_text.infer.md
rm -f logs/round2_qwen35_08b_text.infer.md
rm -f logs/round3_qwen35_2b_mm.infer.md
rm -f logs/round3_qwen35_08b_mm_fallback.infer.md

echo "# round1_qwen3_17b_text" > logs/round1_qwen3_17b_text.infer.md
run_batch_infer 'round1_qwen3_17b_text' 'outputs/qwen3_17b_text' 'data/infer_text_prompts.jsonl' 'text3'

echo "# round2_qwen35_08b_text" > logs/round2_qwen35_08b_text.infer.md
run_batch_infer 'round2_qwen35_08b_text' 'outputs/qwen35_08b_text' 'data/infer_text_prompts.jsonl' 'text3'

if [ -d 'outputs/qwen35_2b_mm' ]; then
  echo "# round3_qwen35_2b_mm" > logs/round3_qwen35_2b_mm.infer.md
  run_batch_infer 'round3_qwen35_2b_mm' 'outputs/qwen35_2b_mm' 'data/infer_text_prompts.jsonl' 'text3'
  run_batch_infer 'round3_qwen35_2b_mm' 'outputs/qwen35_2b_mm' 'data/infer_mm_prompts.jsonl' 'mm2'
elif [ -d 'outputs/qwen35_08b_mm_fallback' ]; then
  echo "# round3_qwen35_08b_mm_fallback" > logs/round3_qwen35_08b_mm_fallback.infer.md
  run_batch_infer 'round3_qwen35_08b_mm_fallback' 'outputs/qwen35_08b_mm_fallback' 'data/infer_text_prompts.jsonl' 'text3'
  run_batch_infer 'round3_qwen35_08b_mm_fallback' 'outputs/qwen35_08b_mm_fallback' 'data/infer_mm_prompts.jsonl' 'mm2'
fi
