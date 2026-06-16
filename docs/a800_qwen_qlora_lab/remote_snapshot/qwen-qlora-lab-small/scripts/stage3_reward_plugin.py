#!/usr/bin/env python3
"""Small rule-based reward plugin for Stage 3 GRPO learning.

The plugin is intentionally simple. Its purpose is to make the reward signal
auditable while learning GRPO, not to replace a production reward model.
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from swift.rewards import ORM, orms
except Exception:  # pragma: no cover - allows local self-test without ms-swift.
    class ORM:  # type: ignore[no-redef]
        pass

    orms: dict[str, Any] = {}


def _as_sample_list(value: Any, n: int) -> list[Any]:
    if isinstance(value, list) and len(value) == n and (
        not value or isinstance(value[0], (list, tuple, dict, str))
    ):
        return value
    return [value for _ in range(n)]


def _normalize_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(item) for item in value.values()]
    if isinstance(value, (list, tuple, set)):
        terms: list[str] = []
        for item in value:
            if isinstance(item, (list, tuple, set)):
                terms.extend(str(x) for x in item)
            else:
                terms.append(str(item))
        return terms
    return [str(value)]


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _structure_score(text: str) -> float:
    bullet_like = len(re.findall(r"(^|\n)\s*(-|\d+[.)]|[一二三四五六七八九十]+[、.])", text))
    has_multiple_lines = text.count("\n") >= 2
    has_transition = _contains_any(text, ["第一", "第二", "首先", "然后", "最后", "步骤", "清单"])
    if bullet_like >= 2 or (has_multiple_lines and has_transition):
        return 0.2
    if has_transition:
        return 0.1
    return 0.0


class Stage3QualityORM(ORM):
    """Reward concise, structured, safe answers that mention required concepts."""

    def __call__(
        self,
        completions: list[str],
        required: Any = None,
        forbidden: Any = None,
        rubric: Any = None,
        **kwargs: Any,
    ) -> list[float]:
        required_rows = _as_sample_list(required, len(completions))
        forbidden_rows = _as_sample_list(forbidden, len(completions))
        rewards: list[float] = []

        for completion, req_value, forbid_value in zip(completions, required_rows, forbidden_rows):
            text = str(completion or "").strip()
            required_terms = _normalize_terms(req_value)
            forbidden_terms = _normalize_terms(forbid_value)

            score = 0.0
            if required_terms:
                matched = sum(1 for term in required_terms if term.lower() in text.lower())
                score += 0.45 * matched / len(required_terms)

            score += _structure_score(text)

            length = len(text)
            if 80 <= length <= 700:
                score += 0.15
            elif 40 <= length < 80 or 700 < length <= 1000:
                score += 0.07

            if _contains_any(text, ["风险", "检查", "验证", "记录", "不要", "不能", "共享"]):
                score += 0.2

            if _contains_any(text, forbidden_terms):
                score -= 0.35

            rewards.append(round(max(0.0, min(1.0, score)), 4))

        return rewards


orms["stage3_quality"] = Stage3QualityORM


def _selftest() -> None:
    reward = Stage3QualityORM()
    completions = [
        "第一，先运行 nvidia-smi 看显存和已有进程。第二，用 CUDA_VISIBLE_DEVICES 限定训练卡。最后记录日志，避免影响共享服务。",
        "直接停止其他服务，然后开始训练。",
    ]
    scores = reward(
        completions,
        required=[["nvidia-smi", "显存", "CUDA_VISIBLE_DEVICES", "日志"], ["nvidia-smi", "显存"]],
        forbidden=[["直接停止", "随便杀"], ["直接停止"]],
    )
    print(json.dumps({"scores": scores}, ensure_ascii=False))
    if not scores[0] > scores[1]:
        raise SystemExit("selftest failed: good answer must score higher than bad answer")


if __name__ == "__main__":
    _selftest()
