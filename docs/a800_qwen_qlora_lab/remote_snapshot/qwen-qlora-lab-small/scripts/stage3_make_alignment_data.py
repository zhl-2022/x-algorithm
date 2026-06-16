#!/usr/bin/env python3
"""Create small DPO/GRPO learning datasets for Stage 3 alignment study."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

SYSTEM = (
    "你是 A800 Qwen 微调学习教练。回答必须直接、准确、可复盘，"
    "优先说明风险、步骤、指标和验证方式。"
)


DPO_ROWS = [
    {
        "id": "dpo_a800_safety",
        "prompt": "在共享 A800 上开始训练前，为什么不能直接启动任务？",
        "chosen": (
            "不能直接启动。第一步要看 nvidia-smi，确认显存、GPU 进程和已有服务；"
            "第二步用 CUDA_VISIBLE_DEVICES 限定自己的训练卡；第三步记录 df -h /root、训练命令和日志。"
            "共享机器上不允许为了训练随意停止 vLLM、Milvus、MinerU 等已有服务。"
        ),
        "rejected": "直接运行训练脚本就行，显存不够会自己报错，报错后再处理。",
        "reason": "chosen 有安全边界、资源检查和日志要求；rejected 忽略共享服务风险。",
    },
    {
        "id": "dpo_adapter_inference",
        "prompt": "请解释 LoRA adapter 推理和 merged model 推理的区别。",
        "chosen": (
            "adapter 推理是加载 base model，再额外加载 LoRA adapter；优点是体积小、可切换、适合实验对比。"
            "merged model 是把 adapter 合并进 base 权重，部署路径更简单，但磁盘占用更大，也不适合频繁切换多个 adapter。"
        ),
        "rejected": "二者没有区别，都是同一个模型文件，直接部署就行。",
        "reason": "chosen 区分了加载方式、优缺点和部署场景；rejected 是事实错误。",
    },
    {
        "id": "dpo_train_loss",
        "prompt": "为什么不能只看 train loss 判断微调效果？",
        "chosen": (
            "train loss 只说明模型对训练样本拟合得怎样，不代表真实问题会更好。"
            "还要看固定评测 prompt、人工错误分类、验证集指标和训练前后对比。"
            "如果 loss 降了但回答变死板、幻觉增加或只会背训练集，微调仍然失败。"
        ),
        "rejected": "train loss 越低就说明模型越好，只要多训几轮就可以。",
        "reason": "chosen 解释泛化和评测；rejected 把拟合误认为真实效果。",
    },
    {
        "id": "dpo_qlora_memory",
        "prompt": "为什么 QLoRA 适合用来在剩余显存有限的 A800 上学习？",
        "chosen": (
            "QLoRA 会把基座模型以 4bit 量化加载，并冻结大部分基座权重，只训练很小的 LoRA adapter。"
            "这样显存主要花在量化权重、激活、优化器状态和 adapter 上，比全参微调轻很多。"
            "但它不是零成本，max_length、batch size、num_generations 仍然会明显影响显存。"
        ),
        "rejected": "QLoRA 就是不占显存的训练方法，所以可以随便调大 batch size。",
        "reason": "chosen 说明节省机制和剩余风险；rejected 绝对化且危险。",
    },
    {
        "id": "dpo_dpo_data",
        "prompt": "DPO 数据里的 chosen 和 rejected 应该怎么构造？",
        "chosen": (
            "chosen 必须是你确实希望模型学习的回答，rejected 必须体现明确缺陷，例如事实错、遗漏风险、结构差或不符合项目规范。"
            "不要用两个都差不多的回答凑数，否则偏好信号很弱。每条样本都要能写出 reason，说明为什么 chosen 更好。"
        ),
        "rejected": "chosen 随便写一个长答案，rejected 随便写一个短答案，这样模型就会学会更详细。",
        "reason": "chosen 强调可解释偏好；rejected 把长度当质量。",
    },
    {
        "id": "dpo_grpo_reward",
        "prompt": "GRPO 训练里 reward function 为什么比 epoch 数更关键？",
        "chosen": (
            "GRPO 会让模型生成多条回答，再根据 reward 比较同组回答好坏。"
            "如果 reward 只奖励关键词堆砌，模型就会学会堆关键词；如果 reward 能覆盖正确性、格式、安全和简洁性，训练才可能朝目标移动。"
            "epoch 多只能放大奖励设计的优点或缺陷。"
        ),
        "rejected": "GRPO 只要把 epoch 调大就会自动对齐，不需要特别设计奖励函数。",
        "reason": "chosen 说明奖励塑形；rejected 忽略 GRPO 的核心机制。",
    },
    {
        "id": "dpo_sampling_params",
        "prompt": "temperature 和 top_p 在推理采样里分别影响什么？",
        "chosen": (
            "temperature 控制分布尖锐程度，越低越稳定，越高越发散；top_p 控制从累计概率前 p 的候选 token 里采样。"
            "学习和评测阶段建议低温度，方便复现；创意生成或采样对比才逐步提高 temperature 和 top_p。"
        ),
        "rejected": "temperature 和 top_p 都是让模型变聪明的参数，越大效果越好。",
        "reason": "chosen 说明采样机制和使用场景；rejected 是常见误解。",
    },
    {
        "id": "dpo_export_risk",
        "prompt": "为什么导出 merged model 之前要先确认磁盘和目的？",
        "chosen": (
            "merged model 会生成完整权重，体积远大于 LoRA adapter。"
            "导出前要确认 df -h /root、输出目录、是否真的需要简化部署，以及不会把模型权重同步到 Git。"
            "如果只是学习对比，优先保留 adapter 形式。"
        ),
        "rejected": "导出就是把日志打包一下，不会占多少空间，可以随时做。",
        "reason": "chosen 识别大权重风险；rejected 对导出含义错误。",
    },
    {
        "id": "dpo_webui_limit",
        "prompt": "ms-swift WebUI 适合学习什么，不适合替代什么？",
        "chosen": (
            "WebUI 适合学习参数入口、任务类型和小规模试跑。"
            "正式实验仍要沉淀脚本、固定数据、保存日志和写评测报告，因为脚本更可复现，也更容易审查训练参数。"
            "WebUI 上看不到的高级参数可以放在更多参数里，或者改用命令行。"
        ),
        "rejected": "有 WebUI 就不用写脚本了，界面点一次就算完成实验。",
        "reason": "chosen 区分学习入口和可复现实验；rejected 不符合工程要求。",
    },
    {
        "id": "dpo_failure_cases",
        "prompt": "为什么每轮对齐实验都要写 failure cases？",
        "chosen": (
            "failure cases 能告诉你模型到底在哪些问题上失败，例如漏掉安全检查、概念混淆、只会背训练句或输出太空。"
            "没有失败案例，就无法判断下一轮应该改数据、改 reward、改参数还是停止方向。"
        ),
        "rejected": "只要最终分数提高了，就没必要看失败案例。",
        "reason": "chosen 强调错误归因；rejected 只看单一分数。",
    },
    {
        "id": "dpo_alignment_goal",
        "prompt": "人类对齐是不是只是在训练模型更礼貌？",
        "chosen": (
            "不是。礼貌只是表层。对齐更关注模型是否按人的偏好和任务标准行动，"
            "例如遵守安全边界、承认不确定性、给出可执行步骤、避免幻觉、在多种答案里选择更有帮助的答案。"
        ),
        "rejected": "是的，对齐就是让模型多说礼貌用语，回答更客气。",
        "reason": "chosen 覆盖偏好、任务和安全；rejected 把对齐窄化成语气。",
    },
    {
        "id": "dpo_strict_learning",
        "prompt": "如果我要深入学习 DPO 和 GRPO，最低交付物是什么？",
        "chosen": (
            "最低交付物包括：数据文件、训练命令、奖励函数或偏好构造说明、训练日志、固定评测结果、失败案例和下一轮改进假设。"
            "只说跑通不合格；必须能解释为什么这样设计，以及结果支持或反驳了哪个假设。"
        ),
        "rejected": "只要能启动训练并得到 checkpoint，就说明已经学会了。",
        "reason": "chosen 提出严格闭环；rejected 停留在跑命令。",
    },
]


GRPO_ROWS = [
    {
        "id": "grpo_a800_safety",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "给出共享 A800 上启动训练前的安全检查清单。"}],
        "required": ["nvidia-smi", "显存", "CUDA_VISIBLE_DEVICES", "日志"],
        "forbidden": ["直接停止", "随便杀", "不用检查"],
        "rubric": "奖励安全检查、GPU 限定和日志意识。",
    },
    {
        "id": "grpo_dpo_format",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "用小白能懂的话解释 DPO 数据格式。"}],
        "required": ["chosen", "rejected", "messages", "偏好"],
        "forbidden": ["只要 prompt", "随便写"],
        "rubric": "奖励讲清 chosen/rejected 和偏好来源。",
    },
    {
        "id": "grpo_reward_design",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "设计一个 GRPO 奖励函数时，至少要考虑哪些维度？"}],
        "required": ["正确性", "格式", "安全", "简洁"],
        "forbidden": ["只看长度", "只看关键词"],
        "rubric": "奖励多维度 reward 思维，惩罚单一关键词思维。",
    },
    {
        "id": "grpo_eval_failure",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "为什么 GRPO 跑完后必须看失败案例？"}],
        "required": ["失败案例", "reward", "过拟合", "下一轮"],
        "forbidden": ["只看 loss", "不用复盘"],
        "rubric": "奖励能把失败案例和 reward 设计联系起来。",
    },
    {
        "id": "grpo_adapter_vs_merged",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "adapter 推理和 merged model 推理怎么取舍？"}],
        "required": ["base", "adapter", "合并", "磁盘"],
        "forbidden": ["没有区别", "都一样"],
        "rubric": "奖励部署形态和资源成本解释。",
    },
    {
        "id": "grpo_sampling",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "temperature 和 top_p 调参时怎么做对比实验？"}],
        "required": ["固定 prompt", "temperature", "top_p", "对比"],
        "forbidden": ["越大越好"],
        "rubric": "奖励固定变量和参数对比意识。",
    },
    {
        "id": "grpo_strict_report",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "一轮对齐实验的报告必须包含哪些内容？"}],
        "required": ["数据", "命令", "日志", "评测", "失败"],
        "forbidden": ["只写成功", "不用记录"],
        "rubric": "奖励完整实验闭环。",
    },
    {
        "id": "grpo_train_loss_limit",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "为什么 train loss 降低不等于对齐成功？"}],
        "required": ["泛化", "评测", "过拟合", "人工"],
        "forbidden": ["loss 越低越好"],
        "rubric": "奖励区分训练拟合和真实质量。",
    },
]


EVAL_CASES = [
    {
        "id": row["id"],
        "prompt": row["messages"][-1]["content"],
        "must_any": [[term] for term in row["required"]],
        "forbidden": row["forbidden"],
        "note": row["rubric"],
    }
    for row in GRPO_ROWS
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dpo_rows = []
    for row in DPO_ROWS:
        dpo_rows.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["chosen"]},
                ],
                "rejected_response": row["rejected"],
                "reason": row["reason"],
            }
        )
    write_jsonl(DATA_DIR / "stage3_dpo_preferences.jsonl", dpo_rows)
    write_jsonl(DATA_DIR / "stage3_grpo_prompts.jsonl", GRPO_ROWS)
    write_jsonl(DATA_DIR / "stage3_alignment_eval_cases.jsonl", EVAL_CASES)
    print(f"wrote {DATA_DIR / 'stage3_dpo_preferences.jsonl'} rows={len(dpo_rows)}")
    print(f"wrote {DATA_DIR / 'stage3_grpo_prompts.jsonl'} rows={len(GRPO_ROWS)}")
    print(f"wrote {DATA_DIR / 'stage3_alignment_eval_cases.jsonl'} rows={len(EVAL_CASES)}")


if __name__ == "__main__":
    main()
