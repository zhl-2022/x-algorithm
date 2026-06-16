#!/usr/bin/env python3
"""Call an OpenAI-compatible local swift deploy endpoint for a prompt batch."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def prompt_from_row(row: dict[str, Any], idx: int) -> tuple[str, str]:
    row_id = str(row.get("id") or f"row_{idx + 1:03d}")
    if "prompt" in row:
        return row_id, str(row["prompt"])
    messages = row.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "user":
            return row_id, str(message.get("content", ""))
    return row_id, ""


def call_chat(
    *,
    port: int,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Stage 2 Batch Inference", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['id']}",
                "",
                f"- model: `{row['model']}`",
                f"- temperature: `{row['temperature']}`",
                f"- top_p: `{row['top_p']}`",
                "",
                "prompt:",
                "",
                "```text",
                row["prompt"],
                "```",
                "",
                "response:",
                "",
                "```text",
                row["response"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=19082)
    parser.add_argument("--model", default="qwen3-1.7b-qlora-effective")
    parser.add_argument("--input", default="data/stage2_eval_cases.jsonl")
    parser.add_argument("--output", default="logs/stage2_effective_lora_infer.jsonl")
    parser.add_argument("--markdown", default="logs/stage2_effective_lora_infer.md")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    markdown_path = Path(args.markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(read_jsonl(input_path)):
            row_id, prompt = prompt_from_row(row, idx)
            started = time.time()
            try:
                raw = call_chat(
                    port=args.port,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
                response = raw["choices"][0]["message"]["content"]
                error = None
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                raw = None
                response = ""
                error = repr(exc)
            result = {
                "id": row_id,
                "prompt": prompt,
                "response": response,
                "error": error,
                "model": args.model,
                "port": args.port,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "elapsed_seconds": round(time.time() - started, 3),
                "raw": raw,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            results.append(result)
            print(f"{row_id}: {len(response)} chars, error={error}")

    write_markdown(markdown_path, results)
    print(f"wrote {output_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
