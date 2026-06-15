#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/zhl/qwen-qlora-lab}"
PORT="${1:-${QWEN3_BASE_PORT:-18080}}"
OUT="${OUT:-$PROJECT_DIR/logs/qwen3_17b_base_call.response.json}"

if [ -n "${PROMPT_B64:-}" ]; then
  PROMPT="$(python3 - <<'PY'
import base64
import os

print(base64.b64decode(os.environ["PROMPT_B64"]).decode("utf-8"))
PY
)"
else
  PROMPT="${2:-请用三句话解释 QLoRA 和 LoRA 的区别。/no_think}"
fi

mkdir -p "$(dirname "$OUT")"

python3 - "$PORT" "$PROMPT" "$OUT" <<'PY'
import json
import sys
import urllib.request

port, prompt, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
payload = {
    "model": "qwen3-1.7b-base",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.2,
    "max_tokens": 256,
    "stream": False,
}
data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
req = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    body = resp.read().decode("utf-8")

with open(out_path, "w", encoding="utf-8") as f:
    f.write(body)

row = json.loads(body)
content = row["choices"][0]["message"]["content"]
print(content)
print(f"\nresponse_json: {out_path}")
PY
