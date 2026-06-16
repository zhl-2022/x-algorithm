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

## 6. GRPO 复盘问题

GRPO 训练后，你必须回答：

1. `num_generations=4` 生成的同组回答差异是否足够？
2. reward 分数有没有拉开差距？
3. 高 reward 回答是否真的更好？
4. 低 reward 回答低在哪里？
5. KL 是否过大，模型是否偏离 reference？
6. 模型有没有学会投机，例如堆关键词？
7. 下一轮 reward 应该怎么改？

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
