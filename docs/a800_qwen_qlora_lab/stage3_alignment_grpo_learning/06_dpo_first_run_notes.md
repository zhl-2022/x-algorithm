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

- [x] 用同一批 prompt 对比 effective SFT 和 DPO 输出。
- [x] 填写至少 3 个 `failure_cases.md`。
- [x] 判断 DPO 是否只是加强了模板表达。
- [x] 判断 rejected 类型错误是否真的减少。
- [x] 决定 GRPO 从 DPO checkpoint 起步，还是从 effective SFT checkpoint 起步。

未完成这些，不建议直接跑 GRPO。

## 7. GRPO 起点决策

当前决策：

> 第一轮 GRPO 学习实验从 DPO v1 checkpoint 起步，不从 correction DPO checkpoint 起步；effective SFT adapter 继续作为对照基线。

具体路径：

| 角色 | checkpoint |
|---|---|
| GRPO 起点 adapter | `outputs/stage3_qwen3_17b_dpo/v0-20260616-031625/checkpoint-30` |
| 不作为起点 | `outputs/stage3_qwen3_17b_dpo_corrections/v0-20260616-034853/checkpoint-40` |
| 对照基线 | `outputs/qwen3_17b_text_effective/v0-20260615-102152/checkpoint-120` |

原因：

1. DPO v1 在同一批 Stage2 prompt 上从 effective SFT 的 `0.7500` 提升到 `0.8333`，说明它有阶段性收益。
2. correction DPO 虽然修正了 `training_log` 和 `eval_vs_train`，但引入了 A800 共享训练幻觉、QLoRA 重复和 `top_p` 概念残留错误。
3. GRPO 的 reward v1 依赖关键词、结构和长度，有关键词堆砌风险；如果从已经重复的 correction DPO 起步，风险会叠加。
4. 从 DPO v1 起步能继续利用偏好对齐收益，同时避免 correction DPO 的明显回退。

进入 GRPO 前的额外保护：

1. reward 必须加入反重复、反关键词堆砌和 `top_p` 错误解释惩罚。
2. reward selftest 必须证明好答案高分、危险答案低分、重复答案低分、关键词堆砌低分。
3. 第一轮只跑小步数，默认 `max_steps=20`，不追求最终效果，只验证 GRPO 流程和失败模式。
4. 训练后必须继续固定同一批 prompt 对比 effective SFT、DPO v1 和 GRPO。

## 8. GRPO 首跑状态

上述 GRPO 起点决策已经执行。

| 项目 | 值 |
|---|---|
| 运行日期 | `2026-06-17` |
| 训练入口 | `.\scripts\a800\run_stage3_grpo_train.ps1 -ConfirmTrain -MaxSteps 20` |
| 同步入口 | `.\scripts\a800\sync_stage3_grpo_results.ps1` |
| 输出目录 | `outputs/stage3_qwen3_17b_grpo/v1-20260617-063053` |
| 最终 checkpoint | `checkpoint-20` |
| `global_step/max_steps` | `20/20` |
| `reward` 范围 | `0.25` 到 `1.0` |
| `kl` 最大值 | `0.02504611` |

现在进入下一阶段：固定 prompt 推理和评测。不能只根据 GRPO 训练日志说模型已经变好，必须把 effective SFT、DPO v1、GRPO 的回答放到同一批问题上比较。
