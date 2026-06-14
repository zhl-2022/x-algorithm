#!/usr/bin/env bash
set -euo pipefail
cd /root/zhl/qwen-qlora-lab
bash scripts/docker_run.sh bash scripts/verify_env_inside.sh 2>&1 | tee logs/verify_env.log