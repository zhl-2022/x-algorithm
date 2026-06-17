# Stage 3 人类对齐与 GRPO 深入学习入口

本阶段承接已经完成的 `Qwen3-1.7B effective QLoRA`。目标不是再堆一次训练，而是把你从“会微调”推进到“能解释对齐实验为什么成立”。

严格顺序：

1. 先学 `DPO`：理解偏好数据、参考模型、`chosen/rejected`。
2. 再学 `reward function`：理解奖励到底在鼓励什么。
3. 最后学 `GRPO`：理解多采样、组内相对优势、KL 约束和失败复盘。

## 你必须先接受的标准

只跑出 checkpoint 不算完成。每一轮对齐实验至少要交付：

| 交付物 | 合格标准 |
|---|---|
| 数据文件 | 每条样本能解释为什么这样构造 |
| 训练命令 | 能说清每个关键参数的作用 |
| 日志 | 能指出 loss、reward、KL、学习率等指标的含义 |
| 推理对比 | 固定 prompt 对比训练前后 |
| 评测报告 | 至少有规则评测和人工复盘 |
| 失败案例 | 至少列 3 个失败案例和下一轮改法 |

## 本阶段新增文件

| 文件 | 用途 |
|---|---|
| `01_dpo_vs_sft_vs_grpo.md` | 区分 SFT、DPO、GRPO 的训练目标 |
| `02_dpo_data_format_and_examples.md` | 逐条解释 DPO 数据格式和样本质量标准 |
| `03_reward_function_design.md` | 解释 Stage 3 的规则奖励函数 |
| `04_grpo_training_walkthrough.md` | 解释 GRPO 训练脚本、参数和执行顺序 |
| `05_eval_and_failure_analysis.md` | 训练后评测、失败归因和复盘模板 |
| `06_dpo_first_run_notes.md` | 记录本次 DPO 首跑结果和进入 GRPO 前的检查项 |
| `07_stage2_sampling_and_dpo_correction_results.md` | 解释 `temperature/top_p` 原理，并复盘 DPO 修正是否真正变好 |
| `08_grpo_first_run_notes.md` | 记录本次 GRPO 首跑结果、日志指标和下一步评测要求 |
| `failure_cases.md` | 后续填写失败案例 |

新增远端快照脚本：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage3_make_alignment_data.py
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage3_reward_plugin.py
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage3_prepare_alignment_assets.sh
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage3_readiness_check.sh
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage3_train_dpo_inside.sh
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage3_train_grpo_inside.sh
```

新增本地 PowerShell 入口：

```powershell
.\scripts\a800\sync_stage3_alignment_assets.ps1
.\scripts\a800\prepare_stage3_alignment_assets.ps1
.\scripts\a800\check_stage3_alignment_ready.ps1
.\scripts\a800\run_stage3_reward_selftest.ps1
.\scripts\a800\run_stage3_dpo_train.ps1
.\scripts\a800\run_stage3_grpo_train.ps1
.\scripts\a800\sync_stage3_grpo_results.ps1
```

## 第一轮建议执行顺序

先只准备和检查，不训练：

```powershell
.\scripts\a800\sync_stage3_alignment_assets.ps1
.\scripts\a800\prepare_stage3_alignment_assets.ps1
.\scripts\a800\check_stage3_alignment_ready.ps1
.\scripts\a800\run_stage3_reward_selftest.ps1
```

确认你能解释数据和 reward 后，再跑 DPO：

```powershell
.\scripts\a800\run_stage3_dpo_train.ps1 -ConfirmTrain -MaxSteps 30
```

DPO 复盘完成后，再跑 GRPO：

```powershell
.\scripts\a800\run_stage3_grpo_train.ps1 -ConfirmTrain -MaxSteps 20
.\scripts\a800\sync_stage3_grpo_results.ps1
```

## 关键学习边界

1. DPO 数据使用 `messages + rejected_response`，不是你之前 SFT 的单答案格式。
2. GRPO 数据可以只有 prompt，但本阶段额外放了 `required/forbidden/rubric`，用于自定义 reward。
3. Stage 3 的 reward 是规则函数，目的是学习可解释闭环，不代表生产级奖励模型。
4. GRPO 默认关闭 `vLLM`，因为当前目标是理解流程；正式大规模训练再考虑 vLLM colocate/server 模式。
5. 所有训练脚本都有显式确认开关，避免误占 A800。

## 参考来源

- ms-swift RLHF 文档：https://swift.readthedocs.io/en/latest/Instruction/RLHF.html
- ms-swift 自定义数据格式：https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html
- Qwen ms-swift 训练文档：https://qwen.readthedocs.io/en/latest/training/ms_swift.html
- ms-swift GRPO 说明：https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md
