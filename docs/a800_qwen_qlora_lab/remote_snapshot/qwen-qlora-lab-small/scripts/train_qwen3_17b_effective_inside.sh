#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/workspace/qwen-qlora-lab"
cd "$PROJECT_DIR"
mkdir -p logs outputs data

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/qwen3_17b_effective_training.log; }
used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' '; }
latest_ckpt() {
  local dir="$1"
  if [ ! -d "$dir" ]; then return 0; fi
  find "$dir" -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true
}

python scripts/generate_effective_unique_data.py | tee -a logs/qwen3_17b_effective_training.log

DATASET="data/text_sft_effective_unique.jsonl"
EVAL_DATASET="data/text_sft_effective_eval_prompts.jsonl"
OUT_DIR="outputs/qwen3_17b_text_effective"
BASE_RESULT="logs/qwen3_17b_effective.base.result.jsonl"
LORA_RESULT="logs/qwen3_17b_effective.lora.result.jsonl"

nvidia-smi > logs/qwen3_17b_effective.before.nvidia-smi.txt

log "Run base inference for comparison"
set +e
timeout 900 swift infer \
  --model models/Qwen3-1.7B \
  --model_type qwen3 \
  --template qwen3 \
  --torch_dtype bfloat16 \
  --val_dataset "$EVAL_DATASET" \
  --result_path "$BASE_RESULT" \
  --stream false \
  --max_new_tokens 192 \
  --max_batch_size 1 \
  > logs/qwen3_17b_effective.base_infer.log 2>&1
base_rc=$?
set -e
log "Base inference rc=$base_rc result=$BASE_RESULT"

base_used=$(used_mib)
log "Start effective QLoRA training base_used_mib=$base_used"
( while true; do printf '%s,' "$(date '+%F %T')"; used_mib; sleep 2; done ) > logs/qwen3_17b_effective.gpu_sample.csv &
sampler_pid=$!

set +e
CUDA_VISIBLE_DEVICES=0 swift sft \
  --tuner_backend peft \
  --model models/Qwen3-1.7B \
  --model_type qwen3 \
  --template qwen3 \
  --quant_bits 4 \
  --dataset "$DATASET" \
  --torch_dtype bfloat16 \
  --num_train_epochs 8 \
  --max_steps 120 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0 \
  --target_modules all-linear \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_steps 40 \
  --save_total_limit 3 \
  --max_length 768 \
  --warmup_ratio 0.03 \
  --dataloader_num_workers 1 \
  --dataset_num_proc 1 \
  --output_dir "$OUT_DIR" \
  --save_only_model true \
  > logs/qwen3_17b_effective.train.log 2>&1
train_rc=$?
set -e

kill "$sampler_pid" >/dev/null 2>&1 || true
wait "$sampler_pid" >/dev/null 2>&1 || true
nvidia-smi > logs/qwen3_17b_effective.after_train.nvidia-smi.txt

max_used=$(awk -F, 'NF>=2 {gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' logs/qwen3_17b_effective.gpu_sample.csv)
delta=$((max_used - base_used))
ckpt="$(latest_ckpt "$OUT_DIR")"
log "Training rc=$train_rc max_used_mib=$max_used train_delta_mib=$delta latest_ckpt=$ckpt"
if [ "$train_rc" -ne 0 ]; then
  tail -n 80 logs/qwen3_17b_effective.train.log || true
  exit "$train_rc"
fi
if [ -z "$ckpt" ]; then
  log "No checkpoint found in $OUT_DIR"
  exit 1
fi

log "Run effective LoRA inference"
set +e
timeout 900 swift infer \
  --adapters "$ckpt" \
  --val_dataset "$EVAL_DATASET" \
  --result_path "$LORA_RESULT" \
  --stream false \
  --max_new_tokens 192 \
  --max_batch_size 1 \
  > logs/qwen3_17b_effective.lora_infer.log 2>&1
lora_rc=$?
set -e
log "LoRA inference rc=$lora_rc result=$LORA_RESULT"

python - <<'PY' | tee logs/qwen3_17b_effective.compare.md
import json
from pathlib import Path

base_path = Path("logs/qwen3_17b_effective.base.result.jsonl")
lora_path = Path("logs/qwen3_17b_effective.lora.result.jsonl")

def load(path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row.get("response", "").replace("\n", " ")[:260])
    return rows

base = load(base_path)
lora = load(lora_path)
print("# Qwen3-1.7B Effective QLoRA Compare")
print()
print(f"- base_rows: {len(base)}")
print(f"- lora_rows: {len(lora)}")
print()
for i, (b, l) in enumerate(zip(base, lora), 1):
    print(f"## Prompt {i}")
    print()
    print(f"- base: {b}")
    print(f"- lora: {l}")
    print()
PY

nvidia-smi > logs/qwen3_17b_effective.after_all.nvidia-smi.txt
log "Effective training workflow finished"
