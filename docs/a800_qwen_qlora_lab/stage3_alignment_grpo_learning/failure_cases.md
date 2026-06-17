# Stage 3 Failure Cases

本文件记录 Stage 3 对齐实验中的真实失败案例。当前案例来自已同步到本地的 A800 日志：

- `remote_snapshot/qwen-qlora-lab-small/logs/stage3_dpo_stage2_eval_eval.md`
- `remote_snapshot/qwen-qlora-lab-small/logs/stage3_dpo_stage2_eval_infer.md`
- `remote_snapshot/qwen-qlora-lab-small/logs/stage3_dpo_correction_stage2_eval_eval.md`
- `remote_snapshot/qwen-qlora-lab-small/logs/stage3_dpo_correction_stage2_eval_infer.md`

当前结论：DPO v1 有局部提升，correction DPO 修正了部分概念，但引入了重复和幻觉；GRPO 训练尚未运行，所以这里暂不记录 GRPO 训练后的失败案例。

## Case 1: DPO v1 仍然讲错 `top_p`

- prompt: 推理采样里的 `temperature` 和 `top_p` 分别影响什么？
- model: `stage3_qwen3_17b_dpo/v0-20260616-031625/checkpoint-30`
- expected: `temperature` 控制 logits softmax 后分布的平滑程度；`top_p` 是 nucleus sampling，按概率排序后取累计概率达到 $p$ 的候选集合。
- actual: 模型说 `top_p` 代表 “softmax 前 1/p 个概率最高的 token” 或“前 p 个 token”。
- score: `0.75`
- failure_type: `reward_mismatch`, `concept_error`
- root_cause: 关键词评测只看到 `temperature`、`top_p` 等词出现，无法识别 `top_p` 的概念解释错误。
- next_action: 增加 DPO 负例，把“前 p 个 token”“前 1/p 个 token”“p% token”作为 rejected；在评测规则里增加错误短语惩罚。

## Case 2: DPO v1 对 train loss 和评测的解释不完整

- prompt: 为什么不能只看 train loss 判断微调效果？
- model: `stage3_qwen3_17b_dpo/v0-20260616-031625/checkpoint-30`
- expected: 说明 train loss 只代表训练集拟合，必须结合验证集、固定评测集、人工评估、benchmark 和实际推理样例。
- actual: 回答提到 test loss，但没有明确固定评测、人工评估、benchmark；还出现“train loss 会先下降后上升”等不稳定泛化说法。
- score: `0.5`
- failure_type: `missing_required`, `over_formatting`
- root_cause: 偏好数据中对“评测闭环”的覆盖不够，模型学到了一些训练 loss 常识，但没有稳定掌握本项目要求的固定 prompt 对比。
- next_action: 补充 chosen/rejected 数据，强制区分 `train loss`、验证集、固定评测 prompt、人工复盘和 benchmark。

## Case 3: Correction DPO 在 A800 共享训练问题上出现幻觉和重复

- prompt: 请用简洁中文解释：A800 共享训练。
- model: `stage3_qwen3_17b_dpo_corrections/v0-20260616-034853/checkpoint-40`
- expected: 先查 `nvidia-smi` 和磁盘，确认已有服务；用 `CUDA_VISIBLE_DEVICES`、batch size、`max_length`、日志和 checkpoint 限制风险；不停止共享服务。
- actual: 模型反复说“训练超过 24 小时会超出 nvidia-smi 的显存限制，超过 12 小时会超出 CUDA_VISIBLE_DEVICES 所见显存限制”。
- score: `0.5`
- failure_type: `hallucination`, `repetition`, `missing_required`
- root_cause: correction 数据只有 8 条且训练 40 steps，偏好修正过窄，导致局部过拟合；没有足够的反重复和反幻觉样本。
- next_action: 不继续使用这个 correction adapter 作为最佳模型；扩充到 30 到 50 条 correction 数据，加入反重复、反幻觉 rejected，再用更低步数或更低学习率重训。

## Case 4: Correction DPO 在 QLoRA 问题上关键词正确但机械复读

- prompt: 为什么 QLoRA 适合在剩余显存不多的 A800 上学习微调？
- model: `stage3_qwen3_17b_dpo_corrections/v0-20260616-034853/checkpoint-40`
- expected: 简洁说明 4bit 量化加载基座、冻结基座、只训练 LoRA adapter，从而降低权重显存和优化器开销。
- actual: 关键概念基本正确，但多次重复“只用 4bit 量化加载并冻结基座模型，只训练 LoRA adapter。这样能显著减少权重显存和优化器开销”。
- score: `1.0`
- failure_type: `reward_mismatch`, `repetition`
- root_cause: 当前关键词评测没有反重复惩罚，导致重复答案仍能拿满分。
- next_action: 在评测和 reward 中加入重复 n-gram 惩罚；把“正确但重复”的回答作为 rejected。

## Case 5: Correction DPO 仍然没有真正修正采样参数概念

- prompt: 推理采样里的 `temperature` 和 `top_p` 分别影响什么？
- model: `stage3_qwen3_17b_dpo_corrections/v0-20260616-034853/checkpoint-40`
- expected: `temperature` 控制分布平滑度和随机性；`top_p` 取累计概率达到 $p$ 的候选集合，不是固定数量或固定百分比的 token。
- actual: 模型说 `top_p` 是 “nucleus sampling 中选择概率最高的 p% 的 token”。
- score: `0.5`
- failure_type: `concept_error`, `missing_required`
- root_cause: correction 数据仍不足以压住常见错误说法，且评测缺少明确的错误短语禁止项。
- next_action: 追加 `top_p` 专项偏好样本，`forbidden` 加入“p% token”“前 p 个 token”“前 1/p 个 token”，并在人工复盘中单独标注采样概念。

## Failure Type Reference

| 类型 | 解释 |
|---|---|
| `missing_required` | 漏掉必须概念 |
| `forbidden_hit` | 出现危险表达 |
| `keyword_stuffing` | 堆关键词但没有解释 |
| `over_formatting` | 格式漂亮但内容空 |
| `hallucination` | 编造不存在命令、路径或结论 |
| `too_verbose` | 废话过多 |
| `too_short` | 回答过短，不可执行 |
| `reward_mismatch` | 人工判断和 reward 分数冲突 |
| `concept_error` | 关键词出现了，但概念解释错误 |
| `repetition` | 多次重复同一句或同一段 |

## Stage 3 GRPO 固定 prompt 失败案例

来源文件：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/logs/stage3_alignment_eval_grpo_failures.md
```

本轮固定 prompt 评测发现 `6` 个 GRPO 失败或需人工复核的 case：

| case | score | failure_type | 主要问题 | 下一步 |
|---|---:|---|---|---|
| `grpo_dpo_format` | `0.0` | `missing_required`, `repetition` | 没解释 `chosen/rejected/messages/偏好`，且重复“每条 prompt 会固定放 3 条”约 18 次 | 补 DPO 数据格式专项 preference，并把重复答案作为 rejected |
| `grpo_reward_design` | `0.0` | `missing_required` | 回答变成泛泛而谈，没有覆盖正确性、格式、安全、简洁 | 增加 reward 设计专项数据，明确多维 reward 不是只看关键词 |
| `grpo_eval_failure` | `0.25` | `missing_required` | 把问题带偏到推荐召回，漏掉 reward、过拟合、下一轮修正 | 增加“失败案例如何反推 reward/data”的 chosen 样本 |
| `grpo_adapter_vs_merged` | `0.25` | `missing_required` | 只讲 adapter 部署，没讲清 base、合并和磁盘取舍 | 补 adapter/merged/base 三者对比样本 |
| `grpo_strict_report` | `0.2` | `missing_required` | 报告内容不完整，没有覆盖数据、命令、日志、评测、失败闭环 | 用项目报告模板生成 chosen/rejected 对 |
| `grpo_train_loss_limit` | `0.0` | `missing_required` | 仍然没有稳定讲清泛化、评测、过拟合和人工复盘 | 补 loss 与真实效果分离的专项 preference |

这轮最重要的失败不是分数低，而是 `grpo_dpo_format` 出现了明显重复。这说明当前 GRPO reward 虽然已经加入一些反重复逻辑，但训练样本和 reward 覆盖仍不足，不能继续简单增加 `max_steps`。

下一轮进入 GRPO 前必须先完成：

1. 把以上 6 个失败 case 转成 DPO chosen/rejected 数据。
2. 在 reward 中显式惩罚重复 n-gram、无关术语和错误概念短语。
3. 继续用同一批 `stage3_alignment_eval_cases.jsonl` 做回归评测。
4. 只有当 GRPO 相比 DPO v1 不再出现重复退化，才允许扩大训练步数。
