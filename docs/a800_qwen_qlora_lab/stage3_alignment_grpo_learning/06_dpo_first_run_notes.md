# 06 DPO First Run Notes

本文件记录 Stage 3 第一轮 DPO 小训练结果。它不是最终效果报告，只是后续学习和复盘的起点。

## 1. 运行时间

| 项目 | 值 |
|---|---|
| 日期 | `2026-06-16` |
| 远端项目 | `/root/zhl/qwen-qlora-lab` |
| 本地快照 | `docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/` |
| 训练类型 | `swift rlhf --rlhf_type dpo` |
| 基座模型 | `models/Qwen3-1.7B` |
| 起始 adapter | `outputs/qwen3_17b_text_effective/v0-20260615-102152/checkpoint-120` |
| 数据 | `data/stage3_dpo_preferences.jsonl` |
| 数据条数 | `12` |
| 最大步数 | `30` |

## 2. 输出位置

远端：

```text
/root/zhl/qwen-qlora-lab/logs/stage3_dpo.train.log
/root/zhl/qwen-qlora-lab/outputs/stage3_qwen3_17b_dpo/v0-20260616-031625/checkpoint-30
```

本地轻量快照：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/logs/stage3_dpo.train.log
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/outputs/stage3_qwen3_17b_dpo/v0-20260616-031625/
```

注意：本地快照不包含 `adapter_model.safetensors`、`*.bin`、`*.pt` 权重文件。

## 3. 关键结果

| 指标 | 值 |
|---|---:|
| `global_step/max_steps` | `30/30` |
| `epoch` | `10` |
| `train_runtime` | `79.07s` |
| `train_samples_per_second` | `1.518` |
| `train_steps_per_second` | `0.379` |
| `train_loss` | `0.4317` |
| 训练日志中的 `memory(GiB)` | `4.29` |
| 最终 checkpoint | `checkpoint-30` |

为什么 `epoch=10`：

本轮只有 12 条 DPO 样本，`max_steps=30`、`gradient_accumulation_steps=4`，所以小数据会被反复采样，多轮经过数据集。这个设置只适合学习 DPO 流程，不代表生产训练配置。

## 4. 训练日志里最值得看的字段

| 字段 | 解释 |
|---|---|
| `loss` | DPO 当前 step 的训练损失 |
| `rewards/chosen` | DPO 隐式奖励下 chosen 的得分 |
| `rewards/rejected` | DPO 隐式奖励下 rejected 的得分 |
| `rewards/margins` | chosen 和 rejected 的奖励差距 |
| `rewards/accuracies` | chosen 是否优于 rejected 的比例 |
| `logps/chosen` | 模型给 chosen 的 log probability |
| `logps/rejected` | 模型给 rejected 的 log probability |
| `nll_loss` | 混入的 SFT/NLL 部分，和 `rpo_alpha` 有关 |
| `grad_norm` | 梯度范数，用来看训练是否异常 |
| `learning_rate` | 当前学习率，最后降到 0 |

最后一步日志：

```text
loss=0.242
rewards/chosen=31.43
rewards/rejected=4.985
rewards/margins=26.44
rewards/accuracies=1
train_loss=0.4317
```

初步解释：

1. `rewards/accuracies=1` 说明当前 batch 中 chosen 都比 rejected 更受模型偏好。
2. `rewards/margins` 明显为正，说明 DPO 正在拉开 chosen/rejected 差距。
3. 这不等于真实效果已经变好，还必须做固定 prompt 推理对比。

## 5. 本轮不能下的结论

现在不能说：

```text
DPO 已经让模型变好了。
```

因为还没做：

1. SFT adapter 与 DPO adapter 的同 prompt 对比。
2. Stage 3 eval cases 的规则评测。
3. 至少 3 个 failure cases。
4. 判断是否出现模板化或关键词化。

## 6. 下一步严格要求

进入 GRPO 前，必须完成：

- [ ] 用同一批 prompt 对比 effective SFT 和 DPO 输出。
- [ ] 填写至少 3 个 `failure_cases.md`。
- [ ] 判断 DPO 是否只是加强了模板表达。
- [ ] 判断 rejected 类型错误是否真的减少。
- [ ] 决定 GRPO 从 DPO checkpoint 起步，还是从 effective SFT checkpoint 起步。

未完成这些，不建议直接跑 GRPO。
