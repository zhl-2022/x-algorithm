#!/usr/bin/env python3
"""Create small inference/eval/sampling datasets for Stage 2 learning."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


EVAL_CASES = [
    {
        "id": "a800_shared_training",
        "prompt": "请用简洁中文解释：A800 共享训练。",
        "must_any": [
            ["nvidia-smi", "显存"],
            ["CUDA_VISIBLE_DEVICES", "指定 GPU", "限制 GPU"],
            ["batch", "max_length", "gradient_accumulation_steps"],
            ["日志", "checkpoint", "记录"],
        ],
        "forbidden": ["阿里云平台", "AIGC 平台"],
        "note": "检查模型是否学到共享 GPU 训练的安全流程。",
    },
    {
        "id": "adapter_inference",
        "prompt": "学习笔记里怎样总结：adapter 推理？",
        "must_any": [
            ["base", "基座"],
            ["adapter", "LoRA"],
            ["合并", "加载"],
            ["对比", "推理"],
        ],
        "forbidden": [],
        "note": "检查模型是否能讲清 adapter 推理不是重新训练。",
    },
    {
        "id": "qlora_memory",
        "prompt": "为什么 QLoRA 适合在剩余显存不多的 A800 上学习微调？",
        "must_any": [
            ["4bit", "量化"],
            ["LoRA", "adapter"],
            ["冻结", "基座"],
            ["显存"],
        ],
        "forbidden": [],
        "note": "检查 QLoRA 的核心动机。",
    },
    {
        "id": "training_log",
        "prompt": "训练日志里的 loss、token_acc、grad_norm 分别应该怎么看？",
        "must_any": [
            ["loss", "损失"],
            ["token_acc", "token"],
            ["grad_norm", "梯度"],
            ["过拟合", "验证集", "泛化"],
        ],
        "forbidden": [],
        "note": "检查训练日志指标理解。",
    },
    {
        "id": "eval_vs_train",
        "prompt": "为什么不能只看 train loss 判断微调效果？",
        "must_any": [
            ["训练集", "train"],
            ["验证集", "eval"],
            ["泛化", "过拟合"],
            ["固定评测", "人工评估", "benchmark"],
        ],
        "forbidden": [],
        "note": "检查评测意识。",
    },
    {
        "id": "sampling_params",
        "prompt": "推理采样里的 temperature 和 top_p 分别影响什么？",
        "must_any": [
            ["temperature", "温度"],
            ["top_p"],
            ["稳定", "随机"],
            ["发散", "多样"],
        ],
        "forbidden": [],
        "note": "检查采样参数理解。",
    },
]


SAMPLING_PROMPTS = [
    {
        "id": "sampling_a800_plan",
        "prompt": "请给我一份明天学习 QLoRA 推理、评测、导出的清单。",
    },
    {
        "id": "sampling_interview_answer",
        "prompt": "请生成一段面试回答：我如何在 A800 上完成 Qwen3 QLoRA 微调和推理对比。",
    },
    {
        "id": "sampling_short_explain",
        "prompt": "请用小白能懂的话解释：为什么同一个模型每次回答可能不一样？",
    },
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(DATA_DIR / "stage2_eval_cases.jsonl", EVAL_CASES)
    write_jsonl(DATA_DIR / "stage2_sampling_prompts.jsonl", SAMPLING_PROMPTS)
    write_jsonl(
        DATA_DIR / "stage2_infer_prompts.jsonl",
        [{"messages": [{"role": "user", "content": row["prompt"]}]} for row in EVAL_CASES],
    )
    print(f"wrote {DATA_DIR / 'stage2_eval_cases.jsonl'}")
    print(f"wrote {DATA_DIR / 'stage2_sampling_prompts.jsonl'}")
    print(f"wrote {DATA_DIR / 'stage2_infer_prompts.jsonl'}")


if __name__ == "__main__":
    main()
