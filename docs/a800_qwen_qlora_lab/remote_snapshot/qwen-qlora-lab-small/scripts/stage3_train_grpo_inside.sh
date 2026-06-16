#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONFIRM_STAGE3_TRAIN:-0}" != "1" ]]; then
  echo "Refuse to start GRPO training without CONFIRM_STAGE3_TRAIN=1."
  echo "This guard prevents accidental A800 usage."
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-/workspace/qwen-qlora-lab}"
cd "$PROJECT_DIR"
mkdir -p logs outputs

DATASET="${DATASET:-data/stage3_grpo_prompts.jsonl}"
OUT_DIR="${OUT_DIR:-outputs/stage3_qwen3_17b_grpo}"
MAX_STEPS="${MAX_STEPS:-20}"
BASE_MODEL="${BASE_MODEL:-models/Qwen3-1.7B}"
START_ADAPTER="${START_ADAPTER:-$(find outputs/stage3_qwen3_17b_dpo outputs/qwen3_17b_text_effective -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true)}"

if [[ -z "$START_ADAPTER" || ! -d "$START_ADAPTER" ]]; then
  echo "Cannot find START_ADAPTER. Set START_ADAPTER or run SFT/DPO first."
  exit 1
fi

echo "GRPO dataset: $DATASET" | tee logs/stage3_grpo.train.log
echo "Base model: $BASE_MODEL" | tee -a logs/stage3_grpo.train.log
echo "Start adapter: $START_ADAPTER" | tee -a logs/stage3_grpo.train.log
echo "Max steps: $MAX_STEPS" | tee -a logs/stage3_grpo.train.log
python3 scripts/stage3_reward_plugin.py | tee logs/stage3_reward_selftest.log
nvidia-smi > logs/stage3_grpo.before.nvidia-smi.txt

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" swift rlhf \
  --rlhf_type grpo \
  --tuner_backend peft \
  --tuner_type lora \
  --model "$BASE_MODEL" \
  --model_type qwen3 \
  --template qwen3 \
  --adapters "$START_ADAPTER" \
  --ref_adapters "$START_ADAPTER" \
  --quant_bits 4 \
  --dataset "$DATASET" \
  --external_plugins scripts/stage3_reward_plugin.py \
  --reward_funcs stage3_quality \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0 \
  --target_modules all-linear \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --save_steps 10 \
  --save_total_limit 2 \
  --max_length 768 \
  --max_completion_length 256 \
  --num_generations 4 \
  --temperature 0.7 \
  --top_p 0.9 \
  --beta 0.04 \
  --use_vllm false \
  --dataloader_num_workers 1 \
  --dataset_num_proc 1 \
  --output_dir "$OUT_DIR" \
  --save_only_model true \
  --log_completions true \
  2>&1 | tee -a logs/stage3_grpo.train.log

nvidia-smi > logs/stage3_grpo.after.nvidia-smi.txt
latest_ckpt="$(find "$OUT_DIR" -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true)"
echo "latest_grpo_checkpoint=$latest_ckpt" | tee -a logs/stage3_grpo.train.log
