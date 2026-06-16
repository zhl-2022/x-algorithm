#!/usr/bin/env python3
"""Tiny keyword-based evaluator for Stage 2 learning."""

from __future__ import annotations

import argparse
import csv
import json
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


def evaluate_case(case: dict[str, Any], pred: dict[str, Any]) -> dict[str, Any]:
    response = str(pred.get("response") or "")
    must_any = case.get("must_any") or []
    forbidden = case.get("forbidden") or []

    matched_groups: list[str] = []
    missed_groups: list[str] = []
    for group in must_any:
        group_terms = [str(term) for term in group]
        if any(term.lower() in response.lower() for term in group_terms):
            matched_groups.append(" / ".join(group_terms))
        else:
            missed_groups.append(" / ".join(group_terms))

    forbidden_hits = [term for term in forbidden if str(term).lower() in response.lower()]
    total = len(must_any)
    matched = len(matched_groups)
    score = matched / total if total else 0.0
    passed = score >= 0.75 and not forbidden_hits and not pred.get("error")
    return {
        "id": case["id"],
        "score": round(score, 4),
        "matched": matched,
        "total": total,
        "passed": passed,
        "matched_groups": matched_groups,
        "missed_groups": missed_groups,
        "forbidden_hits": forbidden_hits,
        "error": pred.get("error"),
        "prompt": case["prompt"],
        "response": response,
        "note": case.get("note", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "score",
                "matched",
                "total",
                "passed",
                "missed_groups",
                "forbidden_hits",
                "error",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "score": row["score"],
                    "matched": row["matched"],
                    "total": row["total"],
                    "passed": row["passed"],
                    "missed_groups": " | ".join(row["missed_groups"]),
                    "forbidden_hits": " | ".join(row["forbidden_hits"]),
                    "error": row["error"],
                    "note": row["note"],
                }
            )


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    passed = sum(1 for row in rows if row["passed"])
    avg_score = sum(row["score"] for row in rows) / len(rows) if rows else 0.0
    lines = [
        "# Stage 2 Keyword Evaluation",
        "",
        f"- cases: `{len(rows)}`",
        f"- passed: `{passed}`",
        f"- average_score: `{avg_score:.4f}`",
        "",
        "| id | score | passed | missed | forbidden |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        missed = "<br>".join(row["missed_groups"]) if row["missed_groups"] else ""
        forbidden = "<br>".join(row["forbidden_hits"]) if row["forbidden_hits"] else ""
        lines.append(f"| `{row['id']}` | `{row['score']}` | `{row['passed']}` | {missed} | {forbidden} |")

    lines.append("")
    for row in rows:
        lines.extend(
            [
                f"## {row['id']}",
                "",
                f"note: {row['note']}",
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
    parser.add_argument("--cases", default="data/stage2_eval_cases.jsonl")
    parser.add_argument("--pred", default="logs/stage2_effective_lora_infer.jsonl")
    parser.add_argument("--report", default="logs/stage2_effective_lora_eval.md")
    parser.add_argument("--csv", default="logs/stage2_effective_lora_eval.csv")
    args = parser.parse_args()

    cases = {row["id"]: row for row in read_jsonl(Path(args.cases))}
    preds = {row["id"]: row for row in read_jsonl(Path(args.pred))}
    rows = []
    for case_id, case in cases.items():
        pred = preds.get(case_id, {"id": case_id, "response": "", "error": "missing prediction"})
        rows.append(evaluate_case(case, pred))

    report_path = Path(args.report)
    csv_path = Path(args.csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report_path, rows)
    write_csv(csv_path, rows)
    print(f"wrote {report_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
