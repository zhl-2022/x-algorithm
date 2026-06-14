#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/workspace/qwen-qlora-lab"
cd "$PROJECT_DIR"
mkdir -p logs outputs

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/training_status.log; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' '; }
used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' '; }
latest_ckpt() {
  local dir="$1"
  if [ ! -d "$dir" ]; then return 0; fi
  find "$dir" -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true
}

run_round() {
  local name="$1"
  local model_path="$2"
  local dataset="$3"
  local out_dir="$4"
  local default_max_len="$5"
  local existing
  existing=$(latest_ckpt "$out_dir")
  if [ -n "$existing" ] && [ "${FORCE_RERUN:-0}" != "1" ]; then
    log "SKIP $name existing_ckpt=$existing"
    return 0
  fi
  local max_len="$default_max_len"
  if [ -f logs/force_max_length_512 ]; then max_len=512; fi
  mkdir -p "$out_dir"
  local log_file="logs/${name}.train.log"
  local sample_file="logs/${name}.gpu_sample.csv"
  local base_used
  base_used=$(used_mib)
  log "START $name model=$model_path dataset=$dataset base_used_mib=$base_used max_length=$max_len"
  nvidia-smi > "logs/${name}.before.nvidia-smi.txt"
  ( while true; do printf '%s,' "$(date '+%F %T')"; used_mib; sleep 2; done ) > "$sample_file" &
  local sampler_pid=$!
  set +e
  CUDA_VISIBLE_DEVICES=0 swift sft \
    --tuner_backend peft \
    --model "$model_path" \
    --quant_bits 4 \
    --dataset "$dataset" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_steps 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --save_steps 10 \
    --save_total_limit 2 \
    --max_length "$max_len" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 1 \
    --output_dir "$out_dir" \
    --save_only_model true > "$log_file" 2>&1
  local rc=$?
  set -e
  kill "$sampler_pid" >/dev/null 2>&1 || true
  wait "$sampler_pid" >/dev/null 2>&1 || true
  nvidia-smi > "logs/${name}.after.nvidia-smi.txt"
  local max_used delta
  max_used=$(awk -F, 'NF>=2 {gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$sample_file")
  delta=$((max_used - base_used))
  log "END $name rc=$rc max_used_mib=$max_used train_delta_mib=$delta latest_ckpt=$(latest_ckpt "$out_dir")"
  if [ "$delta" -gt 16000 ]; then
    log "Memory delta exceeded 16000 MiB; forcing later rounds to max_length=512"
    touch logs/force_max_length_512
  fi
  if [ "$rc" -ne 0 ]; then
    log "FAILED $name; see $log_file"
    return "$rc"
  fi
}

free_now=$(free_mib)
log "Initial free memory MiB: $free_now"
if [ "$free_now" -lt 18000 ]; then
  log "ABORT: free memory below 18000 MiB. Set ALLOW_LOW_FREE=1 to override."
  if [ "${ALLOW_LOW_FREE:-0}" != "1" ]; then exit 1; fi
fi

run_round 'round1_qwen3_17b_text' 'models/Qwen3-1.7B' 'data/text_sft.jsonl' 'outputs/qwen3_17b_text' 1024
run_round 'round2_qwen35_08b_text' 'models/Qwen3.5-0.8B' 'data/text_sft.jsonl' 'outputs/qwen35_08b_text' 1024
MM_MODEL='models/Qwen3.5-2B'
MM_OUTPUT='outputs/qwen35_2b_mm'
if [ ! -f "$MM_MODEL/config.json" ] || [ -f 'models/Qwen3.5-2B.DOWNLOAD_FAILED' ]; then
  log 'Qwen3.5-2B unavailable; using Qwen3.5-0.8B for multimodal round.'
  MM_MODEL='models/Qwen3.5-0.8B'
  MM_OUTPUT='outputs/qwen35_08b_mm_fallback'
fi
run_round 'round3_qwen35_mm' "$MM_MODEL" 'data/mm_sft.jsonl' "$MM_OUTPUT" 1024
log 'All training rounds finished.'