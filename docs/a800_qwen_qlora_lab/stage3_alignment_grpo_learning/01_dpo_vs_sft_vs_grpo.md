# 01 SFT、DPO、GRPO 到底在训练什么

如果你要学深，第一件事是把训练目标分清楚。SFT、DPO、GRPO 都可能叫“微调”，但它们优化的对象不同。

## 1. SFT：模仿标准答案

SFT 的数据长这样：

```json
{"messages":[{"role":"user","content":"什么是 QLoRA？"},{"role":"assistant","content":"QLoRA 是..."}]}
```

它训练的是：

> 给定用户输入，让模型更像数据里的 assistant 回答。

你之前的 `text_sft_effective_unique.jsonl` 就属于这一类。

SFT 的优点：

| 优点 | 解释 |
|---|---|
| 简单 | 只要有问题和标准回答 |
| 稳定 | loss 比较容易理解 |
| 适合入门 | 能快速让模型学某种表达和知识 |

SFT 的缺点：

| 缺点 | 解释 |
|---|---|
| 不知道哪个回答更好 | 它只看一个目标回答 |
| 容易学表面格式 | 数据差时会背模板 |
| 不能直接表达偏好 | 例如“这个回答比另一个更安全” |

## 2. DPO：学习偏好差距

DPO 的数据不是单答案，而是一对答案：

```json
{
  "messages": [
    {"role": "user", "content": "为什么训练前要看 nvidia-smi？"},
    {"role": "assistant", "content": "因为要确认显存、进程和已有服务，避免影响共享机器..."}
  ],
  "rejected_response": "直接训练就行，报错再说。"
}
```

在 `ms-swift` 里，`messages` 里的最后一个 assistant 是 preferred answer，也就是 `chosen`；`rejected_response` 是 rejected answer。

DPO 的核心不是“模仿 chosen”，而是让模型更偏向 chosen、远离 rejected。

简化理解：

$$
\text{DPO 目标} =
\text{提高 chosen 概率}
-
\text{提高 rejected 概率}
-
\text{偏离 reference 的惩罚}
$$

更接近论文形式的表达是：

$$
\mathcal{L}_{DPO}
=
-\log \sigma
\left(
\beta
\left[
\log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}
-
\log \frac{\pi_{ref}(y_w|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
$$

你现在只需要抓住三个符号：

| 符号 | 含义 |
|---|---|
| $x$ | 用户问题 |
| $y_w$ | chosen，好答案 |
| $y_l$ | rejected，差答案 |
| $\pi_\theta$ | 正在训练的模型 |
| $\pi_{ref}$ | 参考模型，防止模型偏太远 |
| $\beta$ | 偏离参考模型的约束强度 |

## 3. GRPO：按奖励比较一组回答

GRPO 的数据通常只有 prompt：

```json
{"messages":[{"role":"user","content":"解释 adapter 推理。"}]}
```

但训练时模型会对同一个 prompt 生成多条回答，例如 `num_generations=4`：

```text
回答 1: ...
回答 2: ...
回答 3: ...
回答 4: ...
```

然后 reward function 给每条回答打分：

```text
回答 1: 0.9
回答 2: 0.2
回答 3: 0.7
回答 4: 0.4
```

GRPO 不需要像 PPO 那样单独训练 value model。它用同组回答的相对分数计算 advantage：

$$
\hat A_i =
\frac{R_i - \text{mean}(R)}
{\text{std}(R) + \epsilon}
$$

直觉上：

| 情况 | GRPO 会怎么学 |
|---|---|
| 某条回答高于组平均 | 增强这种回答的概率 |
| 某条回答低于组平均 | 降低这种回答的概率 |
| 所有回答分数都差不多 | 学习信号很弱 |
| reward 设计错了 | 模型会朝错误方向优化 |

## 4. 三者对比

| 方法 | 数据 | 训练目标 | 适合你现在学什么 |
|---|---|---|---|
| SFT | prompt + answer | 模仿答案 | 学微调基本流程 |
| DPO | prompt + chosen + rejected | 学偏好差距 | 学“什么答案更好” |
| GRPO | prompt + reward | 按奖励强化行为 | 学奖励设计和 RL 对齐 |

## 5. 严格判断标准

你不能只说：

> SFT 是监督微调，DPO 是偏好对齐，GRPO 是强化学习。

这还不够。

你必须能回答：

1. 为什么 DPO 需要 reference model？
2. 为什么 DPO 数据必须能解释 chosen 比 rejected 好在哪里？
3. 为什么 GRPO 的 reward 设计比训练轮数更重要？
4. 为什么 GRPO 要生成多条回答，而不是只生成一条？
5. 为什么 GRPO 的 loss 波动不能像 SFT loss 那样简单理解？

## 6. 你当前项目里的对应文件

| 阶段 | 文件 |
|---|---|
| SFT 数据 | `remote_snapshot/qwen-qlora-lab-small/data/text_sft_effective_unique.jsonl` |
| DPO 数据 | `remote_snapshot/qwen-qlora-lab-small/data/stage3_dpo_preferences.jsonl` |
| GRPO prompt 数据 | `remote_snapshot/qwen-qlora-lab-small/data/stage3_grpo_prompts.jsonl` |
| GRPO reward | `remote_snapshot/qwen-qlora-lab-small/scripts/stage3_reward_plugin.py` |
| DPO 训练脚本 | `remote_snapshot/qwen-qlora-lab-small/scripts/stage3_train_dpo_inside.sh` |
| GRPO 训练脚本 | `remote_snapshot/qwen-qlora-lab-small/scripts/stage3_train_grpo_inside.sh` |
