#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/zhl/qwen-qlora-lab}"
IMAGE="${IMAGE:-qwen-qlora-swift:latest}"
OUT_DIR="${OUT_DIR:-outputs/qwen3_17b_text_effective}"
EXPORT_DIR="${EXPORT_DIR:-exports/qwen3_17b_effective_merged}"
LOG_DIR="$PROJECT_DIR/logs"

cd "$PROJECT_DIR"
mkdir -p "$LOG_DIR" exports

ckpt="$(find "$OUT_DIR" -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true)"
if [ -z "$ckpt" ]; then
  echo "No checkpoint found under $PROJECT_DIR/$OUT_DIR" >&2
  exit 1
fi

if [ "${CONFIRM_EXPORT:-0}" != "1" ]; then
  cat <<EOF
This script will merge the LoRA adapter into a full model:

  adapter: $ckpt
  output : $PROJECT_DIR/$EXPORT_DIR

Merged models are large and should not be synced to Git.
Run again with CONFIRM_EXPORT=1 if you really want to export.
EOF
  exit 2
fi

nvidia-smi > "$LOG_DIR/stage2_export.before.nvidia-smi.txt"
df -h /root > "$LOG_DIR/stage2_export.before.df.txt"

docker run --rm \
  --gpus device=0 \
  --ipc=host \
  --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HOME=/root/.cache/huggingface \
  -e MODELSCOPE_CACHE=/root/.cache/modelscope \
  -v "$PROJECT_DIR:/workspace/qwen-qlora-lab" \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /root/.cache/modelscope:/root/.cache/modelscope \
  -w /workspace/qwen-qlora-lab \
  "$IMAGE" \
  bash -lc "swift export \
    --model models/Qwen3-1.7B \
    --adapters ${ckpt#$PROJECT_DIR/} \
    --model_type qwen3 \
    --template qwen3 \
    --torch_dtype bfloat16 \
    --merge_lora true \
    --safe_serialization true \
    --max_shard_size 5GB \
    --output_dir ${EXPORT_DIR} \
    --exist_ok true" \
  2>&1 | tee "$LOG_DIR/stage2_export_effective_lora.log"

df -h /root > "$LOG_DIR/stage2_export.after.df.txt"
find "$EXPORT_DIR" -maxdepth 2 -type f | sort > "$LOG_DIR/stage2_export.files.txt"

echo
echo "Export finished: $PROJECT_DIR/$EXPORT_DIR"
echo "Do not commit or sync the exported model weights."
