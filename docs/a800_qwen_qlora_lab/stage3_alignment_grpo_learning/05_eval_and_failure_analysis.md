# 05 评测和失败案例分析

对齐训练的复盘标准要比 SFT 更严格。你不能只说“模型回答看起来更好了”，必须说明：

1. 哪些问题变好了。
2. 哪些问题变差了。
3. reward 是否真的奖励了你想要的行为。
4. 下一轮应该改数据、改 reward、改参数，还是停止这个方向。

## 1. 固定评测集

Stage 3 固定评测集：

```text
data/stage3_alignment_eval_cases.jsonl
```

它来自 `stage3_grpo_prompts.jsonl`，保留了：

| 字段 | 用途 |
|---|---|
| `id` | 案例编号 |
| `prompt` | 固定评测问题 |
| `must_any` | 必须覆盖的概念 |
| `forbidden` | 禁止出现的表达 |
| `note` | 人工复盘说明 |

这类规则评测很粗糙，但能做 sanity check。

## 2. 训练前后必须对比同一批 prompt

你至少要对比 3 个模型状态：

| 状态 | 说明 |
|---|---|
| effective SFT adapter | 你上一轮 96 条数据训练出来的基线 |
| DPO adapter | 偏好对齐后的结果 |
| GRPO adapter | 规则 reward 强化后的结果 |

同一个 prompt 必须固定，否则不能比较。

错误做法：

```text
训练前问 A 问题，训练后问 B 问题，然后说训练后更好。
```

正确做法：

```text
同一个 prompt:
- SFT 回答
- DPO 回答
- GRPO 回答
- 人工判断
- 规则评分
- 失败原因
```

## 3. 失败案例模板

每轮训练后新建或补充：

```text
docs/a800_qwen_qlora_lab/stage3_alignment_grpo_learning/failure_cases.md
```

模板：

```markdown
## Case 1

- prompt:
- model:
- expected:
- actual:
- score:
- failure_type:
- root_cause:
- next_action:
```

失败类型建议：

| 类型 | 解释 |
|---|---|
| `missing_required` | 漏掉必须概念 |
| `forbidden_hit` | 出现危险表达 |
| `keyword_stuffing` | 堆关键词但没解释 |
| `over_formatting` | 分点很多但内容空 |
| `hallucination` | 编造不存在命令、路径或结论 |
| `too_verbose` | 废话过多 |
| `too_short` | 回答过短，不可执行 |
| `reward_mismatch` | 人觉得好，但 reward 分低，或反过来 |

## 4. 如何判断是数据问题还是 reward 问题

| 现象 | 更可能的问题 | 处理方式 |
|---|---|---|
| 模型反复漏掉某个概念 | 数据覆盖不够 | 增加 DPO/SFT 样本 |
| 模型堆关键词 | reward 过度依赖关键词 | 增加结构、语义和人工规则 |
| 模型回答越来越模板化 | 格式奖励过强 | 降低结构分，增加内容分 |
| 模型出现危险建议 | forbidden 不够或数据偏好错 | 加强 forbidden 和 DPO 负例 |
| loss 正常但回答变差 | 评测目标和训练目标不一致 | 检查数据和 reward |
| reward 高但人工觉得差 | reward mismatch | 重写 reward |

## 5. DPO 复盘问题

DPO 训练后，你必须回答：

1. DPO 是否让模型更倾向 chosen 的表达？
2. 有没有把 chosen 的口吻学成模板？
3. rejected 的错误是否真的减少？
4. 哪些 prompt 没有改善？
5. `beta=0.1` 是否约束太强或太弱？
6. `rpo_alpha=0.1` 是否让训练更稳定？

如果答不上来，说明你只是跑了 DPO，还没有学懂 DPO。

### 5.1 当前 DPO 复盘答案

当前已同步到本地的 A800 结果包括：

| 文件 | 说明 |
|---|---|
| `logs/stage3_dpo_stage2_eval_eval.md` | DPO v1 在 Stage2 固定问题上的规则评测 |
| `logs/stage3_dpo_stage2_eval_infer.md` | DPO v1 的实际回答 |
| `logs/stage3_dpo_correction_stage2_eval_eval.md` | correction DPO 的规则评测 |
| `logs/stage3_dpo_correction_stage2_eval_infer.md` | correction DPO 的实际回答 |
| `logs/stage3_reward_selftest.log` | reward 插件自检，分数为 `[0.82, 0.0]` |

远端没有发现单独整理好的 `failure_cases` 文件；本地已根据这些日志补充 `failure_cases.md`。

| 问题 | 当前答案 |
|---|---|
| DPO 是否让模型更倾向 chosen 的表达？ | 部分是。DPO v1 的规则评测从 effective SFT 的 `0.7500` 提升到 `0.8333`，说明它学到了一部分偏好方向。 |
| 有没有把 chosen 的口吻学成模板？ | correction DPO 有明显风险。它在 QLoRA 和 A800 共享训练问题上出现重复句，说明小数据 + 多步 correction 容易把局部表达学硬。 |
| rejected 的错误是否真的减少？ | 部分减少。`training_log` 和 `eval_vs_train` 在 correction DPO 中明显改善，但 `top_p` 概念错误没有真正修正。 |
| 哪些 prompt 没有改善？ | `sampling_params` 仍然讲错 `top_p`；correction DPO 的 `a800_shared_training` 出现 12/24 小时显存限制幻觉；`qlora_memory` 虽关键词正确但重复。 |
| `beta=0.1` 是否约束太强或太弱？ | 仅凭当前实验不能下定论。但从 correction DPO 出现重复和幻觉看，`beta=0.1` 没能单独防止小数据过拟合；下一轮应优先降低训练强度并扩大数据，而不是只调 `beta`。 |
| `rpo_alpha=0.1` 是否让训练更稳定？ | 训练过程没有崩溃，DPO 指标如 `rewards/accuracies` 和 `rewards/margins` 能拉开；但输出质量仍有回退，所以“训练稳定”不等于“模型变好”。需要 ablation 才能证明它的独立作用。 |

当前严格结论：

1. DPO v1 是比 effective SFT 更好的阶段性 adapter。
2. correction DPO 不是更好的最终 adapter，它只是暴露了小数据 correction 的过拟合问题。
3. 下一轮 DPO 应先扩充 failure-driven preference 数据，再减少 correction 强度。

## 6. GRPO 复盘问题

GRPO 训练后，你必须回答：

1. `num_generations=4` 生成的同组回答差异是否足够？
2. reward 分数有没有拉开差距？
3. 高 reward 回答是否真的更好？
4. 低 reward 回答低在哪里？
5. KL 是否过大，模型是否偏离 reference？
6. 模型有没有学会投机，例如堆关键词？
7. 下一轮 reward 应该怎么改？

### 6.1 当前 GRPO 复盘答案

当前 A800 已完成第一轮小步 GRPO，结果已经同步到本地轻量快照：

| 文件 | 当前状态 |
|---|---|
| `data/stage3_grpo_prompts.jsonl` | 已准备，8 条 prompt |
| `data/stage3_alignment_eval_cases.jsonl` | 已准备，8 条固定评测案例 |
| `scripts/stage3_reward_plugin.py` | 已加入反重复、反关键词堆砌和概念错误惩罚 |
| `logs/stage3_reward_selftest.log` | 已同步，自检分数 `[0.82, 0.0, 0.47, 0.3, 0.0]` |
| `logs/stage3_grpo.train.log` | 已同步，GRPO 训练完整跑完 |
| `outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/logging.jsonl` | 已同步，包含逐 step 指标 |
| `outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20/` | 已同步 checkpoint 元数据，本地不含权重 |

关键结果：

| 指标 | 值 |
|---|---:|
| `global_step/max_steps` | `20/20` |
| `train_runtime` | `241.5763s` |
| `train_loss` | `0.00024549` |
| `reward` 最小值 | `0.25` |
| `reward` 最大值 | `1.0` |
| `reward` 平均值 | `0.4884` |
| `reward_std=0` 步数 | `3/20` |
| `kl` 最大值 | `0.02504611` |
| `kl` 平均值 | `0.006138` |
| `completions/clipped_ratio` 最大值 | `0.25` |
| 日志显存 | `4.35 GiB` |

| 问题 | 当前答案 |
|---|---|
| `num_generations=4` 生成的同组回答差异是否足够？ | 部分足够。多数 step 的 `reward_std` 大于 0，说明组内 4 条回答能被 reward 拉开；但有 `3/20` 个 step 的 `reward_std=0`，这些 step 的组内回答没有形成有效相对优势。 |
| reward 分数有没有拉开差距？ | 有，但不稳定。训练中 `reward` 从 `0.25` 到 `1.0`，平均 `0.4884`；这说明 reward 有训练信号，但还不代表回答质量一定提升。 |
| 高 reward 回答是否真的更好？ | 尚未完成固定 prompt 人工抽查，所以不能下最终结论。现在只能说规则 reward 认为部分回答更好。 |
| 低 reward 回答低在哪里？ | selftest 显示低分主要来自 dangerous action、重复解释、关键词堆砌和 `top_p` 概念错误；真实生成还要从 `log_completions` 或后续推理结果中抽查。 |
| KL 是否过大，模型是否偏离 reference？ | 没有明显爆炸。最大 `kl=0.02504611`，平均 `0.006138`，第一轮看起来较保守。 |
| 模型有没有学会投机，例如堆关键词？ | 尚未证明。reward 已加关键词堆砌惩罚，但必须部署 GRPO checkpoint 后用固定 prompt 检查。 |
| 下一轮 reward 应该怎么改？ | 先不要急着改。下一步必须对比 effective SFT、DPO v1 和 GRPO 的固定 prompt 输出；如果发现堆关键词，再降低关键词覆盖权重，并增加概念解释质量规则。 |

当前严格结论：

1. GRPO 训练链路已经跑通。
2. reward 插件已经真实参与训练，日志字段为 `rewards/Stage3QualityORM/mean`。
3. KL 暂时稳定，没有明显偏离 reference。
4. 还不能声明 GRPO 效果优于 DPO v1。
5. 下一步必须做固定 prompt 推理、规则评测和人工失败案例分析。

### 6.2 固定 prompt 推理评测结果

已用同一批 `data/stage3_alignment_eval_cases.jsonl` 对比 3 个 adapter：

| 模型状态 | checkpoint | cases | passed | average_score |
|---|---|---:|---:|---:|
| effective SFT | `outputs/qwen3_17b_text_effective/v0-20260615-102152/checkpoint-120` | `8` | `1` | `0.2125` |
| DPO v1 | `outputs/stage3_qwen3_17b_dpo/v0-20260616-031625/checkpoint-30` | `8` | `1` | `0.2188` |
| GRPO v1 | `outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20` | `8` | `2` | `0.3063` |

本轮输出文件：

| 文件 | 用途 |
|---|---|
| `logs/stage3_alignment_eval_compare.md` | 三个 adapter 的逐 prompt 回答对比 |
| `logs/stage3_alignment_eval_compare.csv` | 规则评分表，适合后续做表格分析 |
| `logs/stage3_alignment_eval_grpo_failures.md` | GRPO 失败和相对 DPO 回退案例 |
| `logs/stage3_alignment_eval_*_raw.jsonl` | `swift infer` 原始输出 |
| `logs/stage3_alignment_eval_*_scored.jsonl` | 每个模型的结构化评分结果 |

逐 case 结论：

| case | effective SFT | DPO v1 | GRPO v1 | 判断 |
|---|---:|---:|---:|---|
| `grpo_a800_safety` | `1.0` | `1.0` | `1.0` | 三者都能回答 A800 安全检查 |
| `grpo_dpo_format` | `0.0` | `0.0` | `0.0` | GRPO 出现严重重复，且没讲清 `chosen/rejected/messages/偏好` |
| `grpo_reward_design` | `0.0` | `0.0` | `0.0` | 三者都没覆盖正确性、格式、安全、简洁等核心维度 |
| `grpo_eval_failure` | `0.0` | `0.25` | `0.25` | GRPO 没真正讲清 reward、过拟合和下一轮修正 |
| `grpo_adapter_vs_merged` | `0.25` | `0.25` | `0.25` | GRPO 提到 adapter，但没讲清 base、合并、磁盘取舍 |
| `grpo_sampling` | `0.25` | `0.25` | `0.75` | GRPO 有明显改善，但仍有 `pillar` 等无关/错误表达 |
| `grpo_strict_report` | `0.2` | `0.0` | `0.2` | GRPO 仍没完整覆盖数据、命令、日志、评测、失败 |
| `grpo_train_loss_limit` | `0.0` | `0.0` | `0.0` | 三者都没稳定讲清泛化、评测、过拟合、人工复盘 |

严格结论：

1. GRPO v1 不是一次合格的质量提升，只是把 `temperature/top_p` 这个固定 case 从 `0.25` 提升到 `0.75`。
2. GRPO v1 引入了严重重复风险，典型 case 是 `grpo_dpo_format`，同一句“每条 prompt 会固定放 3 条”重复约 18 次。
3. GRPO v1 在 `grpo_reward_design`、`grpo_train_loss_limit` 等核心概念上仍然失败，说明 reward 和训练数据没有覆盖到真正要学的概念解释。
4. 不能通过“继续加 GRPO 步数”解决当前问题；否则重复和错误概念可能被继续强化。
5. 下一轮应优先扩充 failure-driven DPO/GRPO 数据，并修改 reward：降低纯关键词覆盖权重，增加反重复、反无关术语、概念解释质量和错误短语惩罚。

下一轮最小改动建议：

| 问题 | 处理方式 |
|---|---|
| DPO 格式解释失败 | 补 5 到 10 条 `chosen/rejected/messages/偏好` 专项 DPO preference |
| reward 设计解释失败 | 把 `正确性/格式/安全/简洁/可验证` 写成必答维度，并加入同义词 |
| GRPO 输出重复 | 在 reward 中加入重复 n-gram 惩罚，并把重复答案作为 rejected |
| `top_p` 仍有错误表达 | `forbidden` 加入 `p% token`、`前 p 个 token`、`前 1/p 个 token` 等错误短语 |
| train loss 解释失败 | 补充“loss 只代表训练目标，不代表泛化和人类偏好”的 preference 数据 |

## 7. 面试级回答标准

如果面试官问：

> 你做过 DPO/GRPO 吗？

不合格回答：

```text
我用 ms-swift 跑过 DPO 和 GRPO，配置了数据和参数。
```

合格回答：

```text
我先在 Qwen3-1.7B 的 SFT adapter 上做 DPO，用 messages + rejected_response 构造偏好数据。
每条样本都有 chosen/rejected 的偏好理由，例如共享 A800 训练时 chosen 会包含 nvidia-smi、CUDA_VISIBLE_DEVICES 和不影响已有服务，而 rejected 会故意遗漏这些安全边界。
之后我做了一个小型 GRPO 实验，用规则 reward 奖励 required 概念、结构化表达和安全验证意识，同时惩罚危险表达。
我没有只看 loss，而是固定 prompt 比较 SFT、DPO、GRPO 输出，并记录 reward mismatch 和关键词堆砌等失败案例。
```

这个回答才说明你理解了实验闭环。

## 8. 下一轮改进计划模板

每轮结束后写：

```markdown
# Stage 3 Alignment Iteration N

## Hypothesis

这轮我想验证什么？

## Data Change

我改了哪些数据？为什么？

## Reward Change

我改了哪些 reward？预期会影响什么？

## Training Command

完整命令。

## Result

关键日志、评测分数和样例。

## Failure Cases

至少 3 个失败案例。

## Decision

下一轮继续、修改方向，还是停止？
```

没有这个复盘，不允许进入下一轮。
