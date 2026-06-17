#!/usr/bin/env python3
"""Build Stage 3 fixed-prompt inference data and compare adapter outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SYSTEM = (
    "你是 A800 Qwen 微调学习教练。回答必须直接、准确、可复盘，"
    "优先说明风险、步骤、指标和验证方式。"
)

MODEL_NAMES = ["effective_sft", "dpo_v1", "grpo_v1"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_infer_dataset(cases_path: Path, output_path: Path) -> None:
    cases = read_jsonl(cases_path)
    rows = []
    for case in cases:
        rows.append(
            {
                "id": case["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": case["prompt"]},
                ],
            }
        )
    write_jsonl(output_path, rows)
    print(f"wrote {output_path} rows={len(rows)}")


def normalize_response(row: dict[str, Any]) -> str:
    response = row.get("response")
    if isinstance(response, str):
        return response.strip()
    messages = row.get("messages") or []
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "").strip()
    return ""


def repeated_sentence_count(text: str) -> int:
    parts = [p.strip() for p in re.split(r"[。！？!?；;，,\n]+", text) if len(p.strip()) >= 8]
    counts: dict[str, int] = {}
    for part in parts:
        counts[part] = counts.get(part, 0) + 1
    repeated_parts = sum(count - 1 for count in counts.values() if count > 1)

    repeated_phrases = 0
    for _match in re.finditer(r"(.{8,40}?)(?:\1){2,}", text):
        repeated_phrases += 1
    return repeated_parts + repeated_phrases


def evaluate_case(case: dict[str, Any], raw: dict[str, Any], model_name: str) -> dict[str, Any]:
    response = normalize_response(raw)
    response_lower = response.lower()
    must_any = case.get("must_any") or []
    forbidden = case.get("forbidden") or []

    matched_groups: list[str] = []
    missed_groups: list[str] = []
    for group in must_any:
        group_terms = [str(term) for term in group]
        if any(term.lower() in response_lower for term in group_terms):
            matched_groups.append(" / ".join(group_terms))
        else:
            missed_groups.append(" / ".join(group_terms))

    forbidden_hits = [str(term) for term in forbidden if str(term).lower() in response_lower]
    repeated = repeated_sentence_count(response)
    total = len(must_any)
    matched = len(matched_groups)
    keyword_score = matched / total if total else 0.0
    passed = keyword_score >= 0.75 and not forbidden_hits and bool(response)
    if repeated >= 2:
        passed = False

    failure_types: list[str] = []
    if missed_groups:
        failure_types.append("missing_required")
    if forbidden_hits:
        failure_types.append("forbidden_hit")
    if repeated >= 2:
        failure_types.append("repetition")
    if not response:
        failure_types.append("empty_response")
    if not failure_types and not passed:
        failure_types.append("quality_gap")

    return {
        "id": case["id"],
        "model": model_name,
        "prompt": case["prompt"],
        "response": response,
        "score": round(keyword_score, 4),
        "matched": matched,
        "total": total,
        "passed": passed,
        "matched_groups": matched_groups,
        "missed_groups": missed_groups,
        "forbidden_hits": forbidden_hits,
        "repeated_sentence_count": repeated,
        "failure_types": failure_types,
        "note": case.get("note", ""),
    }


def evaluate_model(cases_path: Path, raw_path: Path, model_name: str, output_path: Path) -> list[dict[str, Any]]:
    cases = read_jsonl(cases_path)
    raw_rows = read_jsonl(raw_path)
    rows = []
    for idx, case in enumerate(cases):
        raw = raw_rows[idx] if idx < len(raw_rows) else {"response": ""}
        rows.append(evaluate_case(case, raw, model_name))
    write_jsonl(output_path, rows)
    print(f"wrote {output_path} rows={len(rows)}")
    return rows


def model_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for row in rows if row["passed"])
    avg = sum(float(row["score"]) for row in rows) / len(rows) if rows else 0.0
    return {"cases": len(rows), "passed": passed, "average_score": round(avg, 4)}


def write_csv(path: Path, all_rows: dict[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "model",
                "score",
                "passed",
                "missed_groups",
                "forbidden_hits",
                "repeated_sentence_count",
                "failure_types",
            ],
        )
        writer.writeheader()
        for rows in all_rows.values():
            for row in rows:
                writer.writerow(
                    {
                        "id": row["id"],
                        "model": row["model"],
                        "score": row["score"],
                        "passed": row["passed"],
                        "missed_groups": " | ".join(row["missed_groups"]),
                        "forbidden_hits": " | ".join(row["forbidden_hits"]),
                        "repeated_sentence_count": row["repeated_sentence_count"],
                        "failure_types": " | ".join(row["failure_types"]),
                    }
                )


def write_markdown(path: Path, all_rows: dict[str, list[dict[str, Any]]]) -> None:
    summaries = {name: model_summary(rows) for name, rows in all_rows.items()}
    case_ids = [row["id"] for row in next(iter(all_rows.values()))] if all_rows else []
    row_by_model = {name: {row["id"]: row for row in rows} for name, rows in all_rows.items()}

    lines = [
        "# Stage 3 Alignment Fixed Prompt Evaluation",
        "",
        "同一批 `stage3_alignment_eval_cases.jsonl`，对比 effective SFT、DPO v1、GRPO v1。",
        "",
        "## Summary",
        "",
        "| model | cases | passed | average_score |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        lines.append(
            f"| `{name}` | `{summary['cases']}` | `{summary['passed']}` | `{summary['average_score']:.4f}` |"
        )

    lines.extend(["", "## Case Scores", "", "| case | effective_sft | dpo_v1 | grpo_v1 | GRPO verdict |", "|---|---:|---:|---:|---|"])
    for case_id in case_ids:
        grpo = row_by_model.get("grpo_v1", {}).get(case_id, {})
        verdict = "pass" if grpo.get("passed") else "fail: " + ", ".join(grpo.get("failure_types") or [])
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | {} |".format(
                case_id,
                row_by_model.get("effective_sft", {}).get(case_id, {}).get("score", ""),
                row_by_model.get("dpo_v1", {}).get(case_id, {}).get("score", ""),
                row_by_model.get("grpo_v1", {}).get(case_id, {}).get("score", ""),
                verdict,
            )
        )

    lines.extend(["", "## Per Case Outputs", ""])
    for case_id in case_ids:
        case_row = row_by_model["grpo_v1"][case_id]
        lines.extend([f"### {case_id}", "", f"note: {case_row['note']}", ""])
        lines.extend(["prompt:", "", "```text", case_row["prompt"], "```", ""])
        for name in MODEL_NAMES:
            row = row_by_model[name][case_id]
            lines.extend(
                [
                    f"#### {name}",
                    "",
                    f"- score: `{row['score']}`",
                    f"- passed: `{row['passed']}`",
                    f"- missed: `{', '.join(row['missed_groups'])}`",
                    f"- forbidden: `{', '.join(row['forbidden_hits'])}`",
                    "",
                    "```text",
                    row["response"],
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def write_failure_markdown(path: Path, grpo_rows: list[dict[str, Any]], dpo_rows: list[dict[str, Any]]) -> None:
    dpo_by_id = {row["id"]: row for row in dpo_rows}
    failures = []
    for row in grpo_rows:
        dpo_score = float(dpo_by_id.get(row["id"], {}).get("score", 0.0))
        if not row["passed"] or float(row["score"]) < dpo_score:
            failures.append((row, dpo_score))

    lines = [
        "# Stage 3 GRPO Failure Cases",
        "",
        "来源：`stage3_alignment_eval_cases.jsonl` 固定 prompt 对比 effective SFT、DPO v1、GRPO v1。",
        "",
        f"- failure_or_regression_cases: `{len(failures)}`",
        "",
    ]
    if not failures:
        lines.append("本轮规则评测没有发现 GRPO 失败或相对 DPO v1 的规则分回退，但仍需要人工审查内容质量。")
    for idx, (row, dpo_score) in enumerate(failures, 1):
        lines.extend(
            [
                f"## Case {idx}: {row['id']}",
                "",
                f"- model: `stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20`",
                f"- score: `{row['score']}`",
                f"- dpo_v1_score: `{dpo_score}`",
                f"- failure_type: `{', '.join(row['failure_types']) or 'regression'}`",
                f"- missed_required: `{', '.join(row['missed_groups'])}`",
                f"- forbidden_hits: `{', '.join(row['forbidden_hits'])}`",
                f"- root_cause: `需要人工复核；优先判断是 reward 关键词覆盖不足、GRPO 训练步数不足，还是回答内容真实缺陷。`",
                f"- next_action: `把该 case 加入下一轮 DPO/GRPO 数据，或调整 reward 中对应概念的同义词和惩罚项。`",
                "",
                "prompt:",
                "",
                "```text",
                row["prompt"],
                "```",
                "",
                "GRPO response:",
                "",
                "```text",
                row["response"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


def compare(cases_path: Path, logs_dir: Path) -> None:
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for model_name in MODEL_NAMES:
        raw_path = logs_dir / f"stage3_alignment_eval_{model_name}.raw.jsonl"
        scored_path = logs_dir / f"stage3_alignment_eval_{model_name}.scored.jsonl"
        all_rows[model_name] = evaluate_model(cases_path, raw_path, model_name, scored_path)

    write_csv(logs_dir / "stage3_alignment_eval_compare.csv", all_rows)
    write_markdown(logs_dir / "stage3_alignment_eval_compare.md", all_rows)
    write_failure_markdown(
        logs_dir / "stage3_alignment_eval_grpo_failures.md",
        all_rows["grpo_v1"],
        all_rows["dpo_v1"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="data/stage3_alignment_eval_cases.jsonl")
    parser.add_argument("--infer_dataset", default="data/stage3_alignment_infer_prompts.jsonl")
    parser.add_argument("--logs_dir", default="logs")
    parser.add_argument("--build_infer_dataset", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    logs_dir = Path(args.logs_dir)

    if args.build_infer_dataset:
        build_infer_dataset(cases_path, Path(args.infer_dataset))
    if args.compare:
        compare(cases_path, logs_dir)


if __name__ == "__main__":
    main()
