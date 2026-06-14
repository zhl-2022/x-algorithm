#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="/root/zhl/qwen-qlora-lab"
echo '--- nvidia-smi ---'
nvidia-smi
echo '--- pipeline pid ---'
if [ -f "$PROJECT_DIR/logs/pipeline.pid" ]; then
  PID=$(cat "$PROJECT_DIR/logs/pipeline.pid")
  echo "PID=$PID"
  ps -p "$PID" -o pid,etime,pcpu,pmem,args || true
else
  echo 'no pipeline pid file'
fi
echo '--- latest pipeline log ---'
if [ -f "$PROJECT_DIR/logs/latest_pipeline_log.txt" ]; then
  LOG=$(cat "$PROJECT_DIR/logs/latest_pipeline_log.txt")
  echo "$LOG"
  tail -n 80 "$LOG" || true
else
  tail -n 80 "$PROJECT_DIR/logs/pipeline_inside.log" 2>/dev/null || true
fi
echo '--- training status ---'
tail -n 80 "$PROJECT_DIR/logs/training_status.log" 2>/dev/null || true
echo '--- models ---'
du -sh "$PROJECT_DIR/models"/* 2>/dev/null || true
echo '--- outputs ---'
find "$PROJECT_DIR/outputs" -maxdepth 3 -type d -name 'checkpoint-*' 2>/dev/null | sort -V || true