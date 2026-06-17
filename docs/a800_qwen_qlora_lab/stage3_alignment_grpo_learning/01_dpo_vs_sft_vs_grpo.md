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

下面是这 5 个问题的合格回答。你后续面试或复盘时，不要只背结论，要能说清楚背后的训练信号。

### 5.1 为什么 DPO 需要 reference model？

一句话回答：

> DPO 需要 reference model，是为了让模型学习 chosen 相对 rejected 更好，同时不要为了迎合偏好数据而偏离原来的语言能力太远。

DPO 不是单纯让模型提高 `chosen` 的概率。它真正比较的是：当前模型对 `chosen/rejected` 的偏好差距，相比参考模型有没有变大。

公式里对应的是这一项：

$$
\log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}
-
\log \frac{\pi_{ref}(y_w|x)}{\pi_{ref}(y_l|x)}
$$

你可以这样理解：

| 对象 | 作用 |
|---|---|
| 当前模型 $\pi_\theta$ | 要被训练得更偏向 `chosen` |
| reference model $\pi_{ref}$ | 提供“原来模型怎么判断”的基准 |
| 二者差值 | 只奖励相对 reference 更合理的偏好变化 |

如果没有 reference model，DPO 很容易变成“过度追 chosen”。小数据尤其危险：模型可能为了某几条偏好样本牺牲通用能力，出现重复、过度自信、格式僵硬、知识偏移等问题。

在你当前项目里，第一次 DPO 的 reference adapter 是 SFT 后的 `checkpoint-120`。这代表我们希望 DPO 从 SFT 模型出发修正偏好，而不是把模型推到一个完全陌生的分布。

常见错误回答：

| 错误说法 | 问题 |
|---|---|
| reference model 是用来推理的 | 不准确。它主要参与训练目标，约束偏移 |
| reference model 越强越好 | 不一定。通常应和当前起点一致或非常接近 |
| 有 chosen/rejected 就不需要 reference | 错。偏好方向和偏离约束是两件事 |

### 5.2 为什么 DPO 数据必须能解释 chosen 比 rejected 好在哪里？

一句话回答：

> DPO 学的是偏好差距。如果 chosen 和 rejected 的优劣原因不清楚，模型学到的就可能是噪声、格式偏见或偶然措辞，而不是真正的质量标准。

DPO 数据不是随便找一个“好回答”和一个“坏回答”拼在一起。每一条样本都应该有明确的偏好理由，例如：

| 偏好理由 | chosen 应该体现什么 | rejected 常见问题 |
|---|---|---|
| 概念正确 | 准确解释 `top_p`、`temperature`、`LoRA` 等概念 | 把 `top_p` 说成“前 p 个 token” |
| 可操作 | 给出具体命令、路径、检查项 | 只说“检查一下配置” |
| 不幻觉 | 不编造不存在的训练结果 | 编造显存、服务、训练结论 |
| 不重复 | 表达完整但不机械复读 | 出现“接口接口接口”这类重复 |
| 符合项目约束 | 不停止业务服务，不同步大模型权重 | 建议直接停掉共享服务 |

如果 chosen 和 rejected 的差别只是“chosen 更长”或“chosen 更像模板”，DPO 可能会学成：

1. 回答越长越好。
2. 固定格式越多越好。
3. 遇到不确定问题也要强行给结论。
4. 把训练集中偶然出现的措辞当成质量标准。

这也是你上一轮 correction DPO 出现回退的根本原因之一：8 条修正数据太少，而且偏好信号覆盖不均。它修正了 `training_log` 和 `eval_vs_train` 这类目标概念，但也带来了重复和局部幻觉。

合格的 DPO 数据应该能回答这三个问题：

| 检查问题 | 合格标准 |
|---|---|
| chosen 好在哪里？ | 能指出具体质量点，不是泛泛说“更好” |
| rejected 错在哪里？ | 能指出事实错、逻辑错、遗漏或重复 |
| 这个偏好能泛化吗？ | 换一个相近问题时，仍然是合理偏好 |

### 5.3 为什么 GRPO 的 reward 设计比训练轮数更重要？

一句话回答：

> GRPO 是按 reward 强化行为。reward 如果错了，训练轮数越多，模型越会稳定地学错。

SFT 的目标来自标准答案，DPO 的目标来自 `chosen/rejected` 偏好对，而 GRPO 的目标来自 reward function。也就是说，reward function 本身就是 GRPO 的“老师”。

如果 reward 设计得好，模型会被推向你真正想要的行为：

| 目标行为 | reward 应该奖励 |
|---|---|
| 概念准确 | 命中正确关键概念，并避免错误解释 |
| 结构清晰 | 先结论，再原因，再实践建议 |
| 不重复 | 惩罚连续重复、模板复读 |
| 不幻觉 | 惩罚编造路径、命令、指标 |
| 可验证 | 奖励能被规则或测试检查的答案 |

如果 reward 设计得差，增加训练轮数通常不会解决问题，反而会放大问题。例如 reward 只检查是否出现“top_p”这个词，不检查解释是否正确，那么模型可能学会高频堆叠关键词，却仍然把 `top_p` 讲错。

这叫 reward hacking：模型不是学会了真正任务，而是学会了钻 reward 的空子。

所以 GRPO 的顺序应该是：

1. 先设计可解释、可测试的 reward。
2. 再用少量样本做 reward selftest。
3. 再跑小步数训练。
4. 最后看输出是否真的变好。

不要反过来先堆训练轮数。

### 5.4 为什么 GRPO 要生成多条回答，而不是只生成一条？

一句话回答：

> GRPO 需要同一个 prompt 下的多条回答，才能比较组内相对好坏，并计算相对 advantage。

GRPO 的核心是 group relative，也就是“组内相对比较”。同一个问题生成多条回答后，reward function 会给每条回答打分：

```text
prompt: 解释 top_p 和 temperature
回答 A: 0.9
回答 B: 0.6
回答 C: 0.2
回答 D: 0.7
```

然后 GRPO 会看每条回答相对组平均分是高还是低：

$$
\hat A_i =
\frac{R_i - \text{mean}(R)}
{\text{std}(R) + \epsilon}
$$

这带来几个好处：

| 好处 | 解释 |
|---|---|
| 不需要单独 value model | 用组内平均和标准差估计相对优势 |
| 学习信号更稳定 | 同一个 prompt 下直接比较不同回答质量 |
| 能优化生成策略 | 奖励高的回答方向被增强，奖励低的回答方向被压低 |

如果只生成一条回答，就没有“组内比较”。只有一个分数时，你很难判断它相对这个 prompt 的其它可能回答到底好多少，也无法可靠计算 group-relative advantage。

这也是为什么 `num_generations` 是 GRPO 的关键参数。它太小，比较信号弱；它太大，显存和推理成本会上升。你当前学习阶段用 `num_generations=4` 是为了在 A800 剩余显存和学习效果之间折中。

### 5.5 为什么 GRPO 的 loss 波动不能像 SFT loss 那样简单理解？

一句话回答：

> SFT loss 主要表示模型离固定标准答案有多远；GRPO loss 同时受采样回答、reward 分布、KL 约束和组内相对优势影响，所以波动不等于模型一定变差。

SFT 的训练目标比较直观：给定输入和标准答案，模型预测下一个 token 越准，loss 越低。虽然 SFT loss 也不能完全代表真实效果，但它至少和“模仿标准答案”直接相关。

GRPO 不一样。它每一步都可能重新采样多条回答，再由 reward function 打分。loss 会受到这些因素影响：

| 因素 | 对 loss 的影响 |
|---|---|
| 采样随机性 | 同一个 prompt 不同轮可能生成不同回答 |
| reward 分布 | 如果一组回答分数接近，学习信号会变弱 |
| reward 噪声 | reward 判断错会让 loss 方向变得不可靠 |
| KL 约束 | 模型既要提高奖励，又不能偏离参考策略太远 |
| 组内标准化 | advantage 来自相对分数，不是绝对正确率 |

所以 GRPO 不能只看 loss 是否单调下降。更严格的判断方式是同时看：

1. 固定评测集上的任务指标是否提升。
2. 高温采样下是否减少重复和幻觉。
3. reward 分数是否提升，并且人工抽查确实更好。
4. KL 是否失控，输出是否偏离原模型能力。
5. 多轮采样的方差是否下降。

对你当前项目来说，GRPO 是否有效，不能只问“loss 降了吗”，而要问：

1. `top_p`、`temperature`、`DPO`、`GRPO` 这些概念是否讲对。
2. 是否还出现“接口接口接口”这类重复。
3. 是否还编造 A800 服务状态、训练结果或不存在的文件。
4. 对同一批 `stage3_alignment_eval_cases.jsonl`，是否比 SFT 和 DPO adapter 更稳。

## 6. 你当前项目里的对应文件

| 阶段 | 文件 |
|---|---|
| SFT 数据 | `remote_snapshot/qwen-qlora-lab-small/data/text_sft_effective_unique.jsonl` |
| DPO 数据 | `remote_snapshot/qwen-qlora-lab-small/data/stage3_dpo_preferences.jsonl` |
| GRPO prompt 数据 | `remote_snapshot/qwen-qlora-lab-small/data/stage3_grpo_prompts.jsonl` |
| GRPO reward | `remote_snapshot/qwen-qlora-lab-small/scripts/stage3_reward_plugin.py` |
| DPO 训练脚本 | `remote_snapshot/qwen-qlora-lab-small/scripts/stage3_train_dpo_inside.sh` |
| GRPO 训练脚本 | `remote_snapshot/qwen-qlora-lab-small/scripts/stage3_train_grpo_inside.sh` |
