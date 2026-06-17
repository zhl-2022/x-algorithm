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


def _count_sentences(text: str) -> int:
    parts = [p.strip() for p in re.split(r"[。！？!?；;\n]+", text) if p.strip()]
    return len(parts)


def _repetition_penalty(text: str) -> float:
    """Penalize repeated phrases and low-information loops."""

    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 0.0

    penalty = 0.0

    # Repeated short Chinese phrases, e.g. "接口接口接口".
    if re.search(r"(.{2,12})\1{2,}", compact):
        penalty += 0.35

    # Repeated sentences with almost identical wording.
    sentences = [s.strip() for s in re.split(r"[。！？!?\n]+", text) if len(s.strip()) >= 8]
    if len(sentences) >= 3:
        normalized = [re.sub(r"\s+", "", s) for s in sentences]
        if len(set(normalized)) <= max(1, len(normalized) // 2):
            penalty += 0.3

    return min(0.5, penalty)


def _keyword_stuffing_penalty(text: str, required_terms: list[str]) -> float:
    """Penalize answers that list terms without enough explanatory context."""

    if not required_terms:
        return 0.0

    matched = sum(1 for term in required_terms if term.lower() in text.lower())
    if matched < max(2, len(required_terms) // 2):
        return 0.0

    sentence_count = _count_sentences(text)
    length = len(text)
    comma_count = text.count("，") + text.count(",") + text.count("、")

    if length < 90 and comma_count >= matched - 1 and sentence_count <= 2:
        return 0.25

    if matched >= 4 and sentence_count <= 1:
        return 0.25

    return 0.0


def _concept_error_penalty(text: str) -> float:
    """Penalize known wrong explanations observed in Stage 2/3 failures."""

    lowered = text.lower()
    penalty = 0.0

    wrong_top_p_patterns = [
        r"top[_-]?p.{0,16}(前\s*p\s*个|前\s*1\s*/\s*p\s*个|p\s*%\s*的?\s*token)",
        r"top[_-]?p.{0,20}(固定数量|固定个数)",
        r"前\s*p\s*个.{0,12}token",
        r"前\s*1\s*/\s*p\s*个.{0,12}token",
        r"p\s*%\s*的?\s*token",
    ]
    if any(re.search(pattern, lowered) for pattern in wrong_top_p_patterns):
        penalty += 0.35

    hallucinated_time_patterns = [
        "超过 24 小时会超出",
        "超过24小时会超出",
        "超过 12 小时会超出",
        "超过12小时会超出",
    ]
    if any(pattern in text for pattern in hallucinated_time_patterns):
        penalty += 0.35

    return min(0.6, penalty)


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

            score -= _repetition_penalty(text)
            score -= _keyword_stuffing_penalty(text, required_terms)
            score -= _concept_error_penalty(text)

            rewards.append(round(max(0.0, min(1.0, score)), 4))

        return rewards


orms["stage3_quality"] = Stage3QualityORM


def _selftest() -> None:
    reward = Stage3QualityORM()
    completions = [
        "第一，先运行 nvidia-smi 看显存和已有进程。第二，用 CUDA_VISIBLE_DEVICES 限定训练卡。最后记录日志，避免影响共享服务。",
        "直接停止其他服务，然后开始训练。",
        "nvidia-smi 显存 CUDA_VISIBLE_DEVICES 日志 风险 检查 验证 记录。",
        "QLoRA 只用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter。这样能减少显存。QLoRA 只用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter。这样能减少显存。QLoRA 只用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter。这样能减少显存。",
        "top_p 是 nucleus sampling 中选择概率最高的 p% 的 token。",
    ]
    scores = reward(
        completions,
        required=[
            ["nvidia-smi", "显存", "CUDA_VISIBLE_DEVICES", "日志"],
            ["nvidia-smi", "显存"],
            ["nvidia-smi", "显存", "CUDA_VISIBLE_DEVICES", "日志"],
            ["4bit", "LoRA", "adapter", "显存"],
            ["temperature", "top_p", "随机", "候选"],
        ],
        forbidden=[
            ["直接停止", "随便杀"],
            ["直接停止"],
            [],
            [],
            ["p% 的 token", "前 p 个 token", "前 1/p 个 token"],
        ],
    )
    print(json.dumps({"scores": scores}, ensure_ascii=False))
    if not scores[0] > scores[1]:
        raise SystemExit("selftest failed: good answer must score higher than bad answer")
    if not scores[0] > scores[2]:
        raise SystemExit("selftest failed: explanatory answer must beat keyword stuffing")
    if not scores[0] > scores[3]:
        raise SystemExit("selftest failed: explanatory answer must beat repetition")
    if scores[4] >= 0.5:
        raise SystemExit("selftest failed: wrong top_p answer must score low")


if __name__ == "__main__":
    _selftest()
