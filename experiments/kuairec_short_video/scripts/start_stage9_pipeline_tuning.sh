#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/zhl/x-algorithm}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/torch/venv3/pytorch/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

LOG_DIR="experiments/kuairec_short_video/logs/stage9_pipeline_tuning"
CACHE_PATH="experiments/kuairec_short_video/data/cache/big_matrix_threshold08_prepared.pkl"
OUTPUTS_DIR="experiments/kuairec_short_video/outputs/stage9_pipeline_tuning"
REPORTS_DIR="experiments/kuairec_short_video/reports/stage9_pipeline_tuning"
mkdir -p "$LOG_DIR" "$(dirname "$CACHE_PATH")" "$OUTPUTS_DIR" "$REPORTS_DIR"

COMMON_ARGS=(
  --experiment distill_pipeline
  --matrix big_matrix.csv
  --positive-threshold 0.8
  --prepared-cache "$CACHE_PATH"
  --ranker-hard-negatives-per-user 80
  --ranker-hard-negative-pool-rows 3000000
  --candidate-ks 100,200
  --rerank-blend-alphas 0.5,0.75,1
  --auc-rows 500000
  --epochs 3
  --batch-size 8192
  --embedding-dim 64
  --tower-dim 64
  --hidden-dim 128
  --ranker-hidden-dims 256,128,64
  --ranker-positive-weight 4
  --outputs-dir "$OUTPUTS_DIR"
  --reports-dir "$REPORTS_DIR"
  --device auto
)

echo "[$(date '+%F %T')] stage9 pipeline tuning started"
if [ ! -f "$CACHE_PATH" ]; then
  echo "[$(date '+%F %T')] missing cache: $CACHE_PATH"
  echo "Build the cache with cache_kuairec_prepared_data.py before running stage9."
  exit 1
fi

run_job() {
  local name="$1"
  local train_rows="$2"
  local teacher_items="$3"
  local negative_items="$4"
  local log_file="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] starting $name train_rows=$train_rows teacher=$teacher_items negative=$negative_items"
  env MLU_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
    experiments/kuairec_short_video/scripts/run_upgrade_experiments.py \
    "${COMMON_ARGS[@]}" \
    --run-name "$name" \
    --train-rows "$train_rows" \
    --ranker-train-rows "$train_rows" \
    --teacher-items-per-user "$teacher_items" \
    --negative-items-per-user "$negative_items" \
    >"$log_file" 2>&1
  echo "[$(date '+%F %T')] completed $name log=$log_file"
}

run_job distill_pipeline_800k_t40n40 800000 40 40
run_job distill_pipeline_2m_t40n120 2000000 40 120
run_job distill_pipeline_2m_t120n40 2000000 120 40

echo "[$(date '+%F %T')] stage9 pipeline tuning completed"
