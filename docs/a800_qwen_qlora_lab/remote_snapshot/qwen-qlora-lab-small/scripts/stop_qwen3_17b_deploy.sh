#!/usr/bin/env bash
set -euo pipefail

for name in qwen3-17b-base-swift-deploy qwen3-17b-lora-swift-deploy qwen3-17b-effective-lora-swift-deploy; do
  if docker ps -q --filter "name=^/${name}$" | grep -q .; then
    echo "Stopping ${name} ..."
    docker stop "$name" >/dev/null
  fi
  if docker ps -aq --filter "name=^/${name}$" | grep -q .; then
    echo "Removing ${name} ..."
    docker rm "$name" >/dev/null
  fi
done

nvidia-smi
