#!/usr/bin/env bash
set -euo pipefail
cd /workspace/qwen-qlora-lab
python - <<'PY'
import torch
print('python/torch check')
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')
PY
swift --help >/tmp/swift_help.txt 2>&1 || true
head -n 20 /tmp/swift_help.txt || true
swift sft --help | head -n 100
python scripts/generate_data.py
ls -lah data images | sed -n '1,80p'