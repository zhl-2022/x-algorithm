import json
from pathlib import Path


SYSTEM = "你是一个中文技术学习助手，回答必须简洁、具体、可执行。"

TOPICS = [
    (
        "A800 共享训练",
        "共享 GPU 上训练应先检查 nvidia-smi，确认显存、已有进程和空闲空间；再限制 CUDA_VISIBLE_DEVICES、batch size 和 max_length；最后保存命令、日志和显存记录，避免影响已有服务。",
    ),
    (
        "QLoRA 显存优化",
        "QLoRA 会用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter；这样能显著减少权重显存和优化器开销；适合共享 GPU 或单卡学习环境。",
    ),
    (
        "LoRA 参数含义",
        "LoRA rank 控制 adapter 容量，alpha 控制 LoRA 分支缩放强度，dropout 用于降低小数据过拟合；rank 越大表达能力越强，但显存和过拟合风险也更高。",
    ),
    (
        "推荐系统召回",
        "召回阶段负责从大规模候选池中快速找出可能相关的物品；常见方法包括 ItemCF、双塔向量召回和图模型召回；召回更关注覆盖率和速度。",
    ),
    (
        "推荐系统排序",
        "排序阶段会使用更丰富的用户、物品和上下文特征，对召回候选做精排；目标通常是点击率、完播率或综合收益；排序更关注最终相关性。",
    ),
    (
        "微调日志",
        "微调日志至少要记录训练命令、loss 曲线、学习率、checkpoint 路径和显存变化；没有验证集时不能只靠 train loss 判断效果；应固定测试 prompt 做对比。",
    ),
    (
        "adapter 推理",
        "LoRA adapter 不能脱离基座模型单独推理；部署时需要先加载原始模型，再通过 adapters 参数挂载 checkpoint；接口测试应保存请求、响应和服务日志。",
    ),
    (
        "max_length",
        "max_length 是单条样本 token 上限，不代表每条都补到这个长度；超过上限会截断；短样本通常只在同一 batch 内按最长样本动态 padding。",
    ),
]

PROMPT_TEMPLATES = [
    "请用简洁中文解释：{topic}。",
    "面试中如何说明：{topic}？",
    "学习笔记里怎样总结：{topic}？",
    "请给小白解释：{topic}。",
    "请用三句话说明：{topic}。",
    "请列出 {topic} 的关键注意点。",
    "真实项目里使用 {topic} 时要注意什么？",
    "请用可执行建议解释：{topic}。",
    "请说明 {topic} 和训练流程的关系。",
    "请用一段话总结：{topic}。",
    "请给出 {topic} 的排错要点。",
    "请解释 {topic} 为什么重要。",
]


def answer_for(topic: str, base: str, idx: int) -> str:
    prefixes = [
        "建议：",
        "可以这样理解：",
        "实践中要点是：",
        "简洁回答：",
    ]
    suffixes = [
        " 这条规则用于保证实验可复现、低风险。",
        " 面试时要强调资源约束、参数选择和验证闭环。",
        " 学习时不要只看命令，要同时看数据、参数、日志和输出。",
        " 如果效果不明显，应先检查数据质量和训练步数，而不是盲目换模型。",
    ]
    return f"{prefixes[idx % len(prefixes)]}{base}{suffixes[idx % len(suffixes)]}"


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    rows = []
    seen_prompts = set()
    idx = 0
    for topic, base in TOPICS:
        for tmpl in PROMPT_TEMPLATES:
            prompt = tmpl.format(topic=topic)
            if prompt in seen_prompts:
                raise RuntimeError(f"duplicate prompt: {prompt}")
            seen_prompts.add(prompt)
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer_for(topic, base, idx)},
                    ]
                }
            )
            idx += 1

    out_path = data_dir / "text_sft_effective_unique.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    eval_prompts = [
        "请用简洁中文解释：A800 共享训练。",
        "请用简洁中文解释：QLoRA 显存优化。",
        "请用简洁中文解释：LoRA 参数含义。",
        "请用简洁中文解释：推荐系统召回。",
        "真实项目里使用 adapter 推理 时要注意什么？",
        "请解释 max_length 为什么重要。",
    ]
    eval_path = data_dir / "text_sft_effective_eval_prompts.jsonl"
    with eval_path.open("w", encoding="utf-8") as f:
        for prompt in eval_prompts:
            f.write(json.dumps({"messages": [{"role": "user", "content": prompt}]}, ensure_ascii=False) + "\n")

    print(f"wrote {out_path} rows={len(rows)} unique_prompts={len(seen_prompts)}")
    print(f"wrote {eval_path} rows={len(eval_prompts)}")


if __name__ == "__main__":
    main()
