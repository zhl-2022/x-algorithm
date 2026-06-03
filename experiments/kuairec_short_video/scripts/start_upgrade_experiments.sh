#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/root/zhl/x-algorithm}"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/torch/venv3/pytorch/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

LOG_DIR="experiments/kuairec_short_video/logs/upgrade_experiments"
PID_DIR="experiments/kuairec_short_video/outputs/upgrade_experiments/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

COMMON_ARGS=(
  --matrix big_matrix.csv
  --positive-threshold 0.8
  --train-rows 800000
  --auc-rows 300000
  --epochs 2
  --batch-size 4096
  --embedding-dim 64
  --tower-dim 64
  --hidden-dim 128
  --outputs-dir experiments/kuairec_short_video/outputs/upgrade_experiments
  --reports-dir experiments/kuairec_short_video/reports/upgrade_experiments
)

start_job() {
  local name="$1"
  local visible_devices="$2"
  local device="$3"
  shift 3
  local log_file="$LOG_DIR/${name}.log"
  local pid_file="$PID_DIR/${name}.pid"
  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running with pid $(cat "$pid_file")"
    return
  fi
  echo "starting $name on MLU_VISIBLE_DEVICES=$visible_devices device=$device"
  nohup env MLU_VISIBLE_DEVICES="$visible_devices" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" experiments/kuairec_short_video/scripts/run_upgrade_experiments.py \
      "${COMMON_ARGS[@]}" --device "$device" "$@" \
      >"$log_file" 2>&1 &
  echo $! >"$pid_file"
  echo "$name pid=$(cat "$pid_file") log=$log_file"
}

start_job distill_twotower 2 auto --experiment distill_twotower --teacher-items-per-user 40 --negative-items-per-user 40
start_job lightgcn cpu cpu --experiment lightgcn --batch-size 32768 --lightgcn-layers 2
start_job sequence_model 3 auto --experiment sequence_model --sequence-length 20
start_job text_encoder 2,3 auto --experiment text_encoder

echo "upgrade experiments submitted"
