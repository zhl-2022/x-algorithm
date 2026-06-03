#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/zhl/x-algorithm}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/torch/venv3/pytorch/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

LOG_DIR="experiments/kuairec_short_video/logs/stage8_recall_boost"
PID_DIR="experiments/kuairec_short_video/outputs/stage8_recall_boost/pids"
CACHE_PATH="experiments/kuairec_short_video/data/cache/big_matrix_threshold08_prepared.pkl"
OUTPUTS_DIR="experiments/kuairec_short_video/outputs/stage8_recall_boost"
REPORTS_DIR="experiments/kuairec_short_video/reports/stage8_recall_boost"
mkdir -p "$LOG_DIR" "$PID_DIR" "$(dirname "$CACHE_PATH")" "$OUTPUTS_DIR" "$REPORTS_DIR"

COMMON_ARGS=(
  --matrix big_matrix.csv
  --positive-threshold 0.8
  --prepared-cache "$CACHE_PATH"
  --auc-rows 500000
  --batch-size 8192
  --embedding-dim 64
  --tower-dim 64
  --hidden-dim 128
  --outputs-dir "$OUTPUTS_DIR"
  --reports-dir "$REPORTS_DIR"
)

echo "[$(date '+%F %T')] stage8 recall boost started"
if [ ! -f "$CACHE_PATH" ]; then
  echo "[$(date '+%F %T')] building prepared cache: $CACHE_PATH"
  "$PYTHON_BIN" experiments/kuairec_short_video/scripts/cache_kuairec_prepared_data.py \
    --matrix big_matrix.csv \
    --positive-threshold 0.8 \
    --output-cache "$CACHE_PATH" \
    >"$LOG_DIR/cache_prepared_data.log" 2>&1
  tail -n 5 "$LOG_DIR/cache_prepared_data.log"
else
  echo "[$(date '+%F %T')] reusing prepared cache: $CACHE_PATH"
fi

start_job() {
  local name="$1"
  local visible_devices="$2"
  local device="$3"
  shift 3
  local log_file="$LOG_DIR/${name}.log"
  local pid_file="$PID_DIR/${name}.pid"
  echo "[$(date '+%F %T')] starting $name visible=$visible_devices device=$device"
  if [ "$visible_devices" = "none" ]; then
    env PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
      experiments/kuairec_short_video/scripts/run_upgrade_experiments.py \
      "${COMMON_ARGS[@]}" --device "$device" --run-name "$name" "$@" \
      >"$log_file" 2>&1 &
  else
    env MLU_VISIBLE_DEVICES="$visible_devices" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
      experiments/kuairec_short_video/scripts/run_upgrade_experiments.py \
      "${COMMON_ARGS[@]}" --device "$device" --run-name "$name" "$@" \
      >"$log_file" 2>&1 &
  fi
  echo $! >"$pid_file"
  echo "[$(date '+%F %T')] $name pid=$(cat "$pid_file") log=$log_file"
}

wait_for_jobs() {
  local failed=0
  for name in "$@"; do
    local pid
    pid="$(cat "$PID_DIR/${name}.pid")"
    if ! wait "$pid"; then
      echo "[$(date '+%F %T')] $name failed pid=$pid"
      failed=1
    else
      echo "[$(date '+%F %T')] $name completed pid=$pid"
    fi
  done
  return "$failed"
}

run_cpu_job() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] running CPU job $name"
  env PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
    experiments/kuairec_short_video/scripts/run_upgrade_experiments.py \
    "${COMMON_ARGS[@]}" --device cpu --run-name "$name" "$@" \
    >"$log_file" 2>&1
  echo "[$(date '+%F %T')] CPU job $name completed log=$log_file"
}

echo "[$(date '+%F %T')] batch1: distill 2M + fixed sequence + LightGCN tuning"
start_job distill_2m_t80n80 2 auto \
  --experiment distill_twotower \
  --train-rows 2000000 \
  --epochs 3 \
  --teacher-items-per-user 80 \
  --negative-items-per-user 80

start_job sequence_fixed_800k 3 auto \
  --experiment sequence_model \
  --train-rows 800000 \
  --epochs 2 \
  --sequence-length 20

for layers in 1 2 3; do
  for epochs in 2 5 10; do
    run_cpu_job "lightgcn_l${layers}_e${epochs}" \
      --experiment lightgcn \
      --train-rows 800000 \
      --epochs "$epochs" \
      --batch-size 32768 \
      --lightgcn-layers "$layers"
  done
done

wait_for_jobs distill_2m_t80n80 sequence_fixed_800k

echo "[$(date '+%F %T')] batch2: distill pipeline 2M"
start_job distill_pipeline_2m 2 auto \
  --experiment distill_pipeline \
  --train-rows 2000000 \
  --ranker-train-rows 2000000 \
  --ranker-hard-negatives-per-user 80 \
  --ranker-hard-negative-pool-rows 3000000 \
  --candidate-ks 100,200,500 \
  --rerank-blend-alphas 0,0.25,0.5,0.75,1 \
  --ranker-positive-weight 4 \
  --epochs 3 \
  --teacher-items-per-user 80 \
  --negative-items-per-user 80
wait_for_jobs distill_pipeline_2m

echo "[$(date '+%F %T')] stage8 recall boost completed"
