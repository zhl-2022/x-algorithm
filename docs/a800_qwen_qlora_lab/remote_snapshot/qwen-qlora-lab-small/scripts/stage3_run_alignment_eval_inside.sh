#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_STAGE3_EVAL:-0}" != "1" ]]; then
  echo "Refuse to start Stage 3 fixed-prompt evaluation without CONFIRM_STAGE3_EVAL=1."
  echo "This guard prevents accidental A800 usage."
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/workspace/qwen-qlora-lab}"
cd "$PROJECT_DIR"
mkdir -p data logs

if command -v ldconfig >/dev/null 2>&1; then
  ldconfig || true
fi

python3 - <<'PY' >/dev/null 2>&1 || python3 -m pip install -q msgspec -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
import msgspec  # noqa: F401
PY

BASE_MODEL="${BASE_MODEL:-models/Qwen3-1.7B}"
INFER_DATASET="${INFER_DATASET:-data/stage3_alignment_infer_prompts.jsonl}"
EFFECTIVE_ADAPTER="${EFFECTIVE_ADAPTER:-outputs/qwen3_17b_text_effective/v0-20260615-102152/checkpoint-120}"
DPO_ADAPTER="${DPO_ADAPTER:-outputs/stage3_qwen3_17b_dpo/v0-20260616-031625/checkpoint-30}"
GRPO_ADAPTER="${GRPO_ADAPTER:-outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
INFER_TIMEOUT_SECONDS="${INFER_TIMEOUT_SECONDS:-1200}"

write_gpu_snapshot() {
  local out_file="$1"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$out_file"
  else
    python3 - <<'PY' > "$out_file"
import torch

print("nvidia-smi unavailable inside container")
print("torch_cuda_available:", torch.cuda.is_available())
print("torch_cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch_cuda_device_0:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info(0)
    print("torch_cuda_mem_free_mib:", round(free / 1024 / 1024, 2))
    print("torch_cuda_mem_total_mib:", round(total / 1024 / 1024, 2))
PY
  fi
}

run_infer() {
  local name="$1"
  local adapter="$2"
  local result_path="$3"
  local command_log="logs/stage3_alignment_eval_${name}.infer.log"

  if [[ ! -d "$adapter" ]]; then
    echo "Missing adapter for $name: $adapter" | tee -a logs/stage3_alignment_eval.run.log
    exit 1
  fi

  rm -f "$result_path" "$command_log"
  {
    echo
    echo "## $name"
    echo "base_model=$BASE_MODEL"
    echo "adapter=$adapter"
    echo "dataset=$INFER_DATASET"
    echo "result_path=$result_path"
    echo "max_new_tokens=$MAX_NEW_TOKENS"
    date -Is
  } | tee -a logs/stage3_alignment_eval.run.log

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" timeout "$INFER_TIMEOUT_SECONDS" \
    swift infer \
      --adapters "$adapter" \
      --val_dataset "$INFER_DATASET" \
      --result_path "$result_path" \
      --stream false \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --max_batch_size 1 \
      --temperature 0.2 \
      --top_p 0.8 \
      > "$command_log" 2>&1

  {
    echo "exit_code=0"
    echo "result_rows=$(wc -l < "$result_path")"
    echo "command_log=$command_log"
  } | tee -a logs/stage3_alignment_eval.run.log
}

rm -f logs/stage3_alignment_eval.run.log
echo "Stage 3 fixed-prompt evaluation" | tee logs/stage3_alignment_eval.run.log
echo "project_dir=$PROJECT_DIR" | tee -a logs/stage3_alignment_eval.run.log

python3 scripts/stage3_alignment_eval_compare.py --build_infer_dataset | tee -a logs/stage3_alignment_eval.run.log
write_gpu_snapshot logs/stage3_alignment_eval.before.nvidia-smi.txt

run_infer "effective_sft" "$EFFECTIVE_ADAPTER" "logs/stage3_alignment_eval_effective_sft.raw.jsonl"
run_infer "dpo_v1" "$DPO_ADAPTER" "logs/stage3_alignment_eval_dpo_v1.raw.jsonl"
run_infer "grpo_v1" "$GRPO_ADAPTER" "logs/stage3_alignment_eval_grpo_v1.raw.jsonl"

python3 scripts/stage3_alignment_eval_compare.py --compare | tee -a logs/stage3_alignment_eval.run.log
write_gpu_snapshot logs/stage3_alignment_eval.after.nvidia-smi.txt

echo "Stage 3 fixed-prompt evaluation complete." | tee -a logs/stage3_alignment_eval.run.log
