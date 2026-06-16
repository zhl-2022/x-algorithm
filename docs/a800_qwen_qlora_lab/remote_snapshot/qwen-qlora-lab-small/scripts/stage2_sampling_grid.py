#!/usr/bin/env python3
"""Run a small temperature/top_p grid against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import time
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


def call_chat(
    *,
    port: int,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Stage 2 Sampling Grid", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['id']} temp={row['temperature']} top_p={row['top_p']} repeat={row['repeat']}",
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
    parser.add_argument("--input", default="data/stage2_sampling_prompts.jsonl")
    parser.add_argument("--output", default="logs/stage2_sampling_grid.jsonl")
    parser.add_argument("--markdown", default="logs/stage2_sampling_grid.md")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.2, 0.8])
    parser.add_argument("--top_ps", nargs="+", type=float, default=[0.8, 0.95])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    prompts = read_jsonl(Path(args.input))
    output_path = Path(args.output)
    markdown_path = Path(args.markdown)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as f:
        for prompt_row in prompts:
            for temperature in args.temperatures:
                for top_p in args.top_ps:
                    for repeat_idx in range(args.repeat):
                        started = time.time()
                        response = call_chat(
                            port=args.port,
                            model=args.model,
                            prompt=prompt_row["prompt"],
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                        )
                        row = {
                            "id": prompt_row["id"],
                            "prompt": prompt_row["prompt"],
                            "response": response,
                            "model": args.model,
                            "temperature": temperature,
                            "top_p": top_p,
                            "repeat": repeat_idx + 1,
                            "elapsed_seconds": round(time.time() - started, 3),
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        f.flush()
                        results.append(row)
                        print(f"{row['id']} temp={temperature} top_p={top_p}: {len(response)} chars")

    write_markdown(markdown_path, results)
    print(f"wrote {output_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
