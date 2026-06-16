#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_STAGE3_TRAIN:-0}" != "1" ]]; then
  echo "Refuse to start DPO training without CONFIRM_STAGE3_TRAIN=1."
  echo "This guard prevents accidental A800 usage."
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/workspace/qwen-qlora-lab}"
cd "$PROJECT_DIR"
mkdir -p logs outputs

DATASET="${DATASET:-data/stage3_dpo_preferences.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/stage3_qwen3_17b_dpo}"
MAX_STEPS="${MAX_STEPS:-30}"
BASE_MODEL="${BASE_MODEL:-models/Qwen3-1.7B}"
START_ADAPTER="${START_ADAPTER:-$(find outputs/qwen3_17b_text_effective -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true)}"

if [[ -z "$START_ADAPTER" || ! -d "$START_ADAPTER" ]]; then
  echo "Cannot find START_ADAPTER. Set START_ADAPTER or run effective SFT first."
  exit 1
fi

echo "DPO dataset: $DATASET" | tee logs/stage3_dpo.train.log
echo "Base model: $BASE_MODEL" | tee -a logs/stage3_dpo.train.log
echo "Start adapter: $START_ADAPTER" | tee -a logs/stage3_dpo.train.log
echo "Max steps: $MAX_STEPS" | tee -a logs/stage3_dpo.train.log
nvidia-smi > logs/stage3_dpo.before.nvidia-smi.txt

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" swift rlhf \
  --rlhf_type dpo \
  --tuner_backend peft \
  --tuner_type lora \
  --model "$BASE_MODEL" \
  --model_type qwen3 \
  --template qwen3 \
  --adapters "$START_ADAPTER" \
  --ref_adapters "$START_ADAPTER" \
  --quant_bits 4 \
  --dataset "$DATASET" \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 5e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0 \
  --target_modules all-linear \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_steps 15 \
  --save_total_limit 2 \
  --max_length 768 \
  --warmup_ratio 0.03 \
  --beta 0.1 \
  --rpo_alpha 0.1 \
  --dataloader_num_workers 1 \
  --dataset_num_proc 1 \
  --output_dir "$OUT_DIR" \
  --save_only_model true \
  2>&1 | tee -a logs/stage3_dpo.train.log

nvidia-smi > logs/stage3_dpo.after.nvidia-smi.txt
latest_ckpt="$(find "$OUT_DIR" -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true)"
echo "latest_dpo_checkpoint=$latest_ckpt" | tee -a logs/stage3_dpo.train.log
