#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/zhl/x-algorithm}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/torch/venv3/pytorch/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

LOG_DIR="experiments/kuairec_short_video/logs/upgrade_experiments_batched"
PID_DIR="experiments/kuairec_short_video/outputs/upgrade_experiments_batched/pids"
CACHE_PATH="experiments/kuairec_short_video/data/cache/big_matrix_threshold08_prepared.pkl"
mkdir -p "$LOG_DIR" "$PID_DIR" "$(dirname "$CACHE_PATH")"

COMMON_ARGS=(
  --matrix big_matrix.csv
  --positive-threshold 0.8
  --prepared-cache "$CACHE_PATH"
  --train-rows 800000
  --auc-rows 300000
  --epochs 2
  --batch-size 4096
  --embedding-dim 64
  --tower-dim 64
  --hidden-dim 128
  --outputs-dir experiments/kuairec_short_video/outputs/upgrade_experiments_batched
  --reports-dir experiments/kuairec_short_video/reports/upgrade_experiments_batched
)

echo "[$(date '+%F %T')] building prepared cache: $CACHE_PATH"
"$PYTHON_BIN" experiments/kuairec_short_video/scripts/cache_kuairec_prepared_data.py \
  --matrix big_matrix.csv \
  --positive-threshold 0.8 \
  --output-cache "$CACHE_PATH" \
  >"$LOG_DIR/cache_prepared_data.log" 2>&1
tail -n 5 "$LOG_DIR/cache_prepared_data.log"

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
      "${COMMON_ARGS[@]}" --device "$device" "$@" \
      >"$log_file" 2>&1 &
  else
    env MLU_VISIBLE_DEVICES="$visible_devices" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
      experiments/kuairec_short_video/scripts/run_upgrade_experiments.py \
      "${COMMON_ARGS[@]}" --device "$device" "$@" \
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

echo "[$(date '+%F %T')] batch1: distill + sequence + lightgcn"
start_job distill_twotower 2 auto --experiment distill_twotower --teacher-items-per-user 40 --negative-items-per-user 40
start_job sequence_model 3 auto --experiment sequence_model --sequence-length 20
start_job lightgcn none cpu --experiment lightgcn --batch-size 32768 --lightgcn-layers 2
wait_for_jobs distill_twotower sequence_model lightgcn

echo "[$(date '+%F %T')] batch2: text encoder"
start_job text_encoder 2 auto --experiment text_encoder
wait_for_jobs text_encoder

echo "[$(date '+%F %T')] all upgrade experiments completed"
