# 07 Stage 2 采样问题与 DPO 修正结果

本文件回答两个问题：

1. 为什么 `temperature=0.8`、`top_p=0.95` 容易出现重复、跑偏。
2. 把 Stage 2 失败输出整理成 DPO 数据后，模型是否相比原来有进步。

## 1. temperature 的原理

模型每一步都会输出所有候选 token 的 logits。先经过 softmax 变成概率：

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

加入 temperature 后：

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

其中 $T$ 就是 `temperature`。

| temperature | 效果 |
|---:|---|
| 接近 `0` | 分布变尖，几乎只选最高概率 token，输出稳定 |
| `0.2` | 仍然偏稳定，适合评测和学习 |
| `0.8` | 分布更平，低概率 token 更容易被采到 |
| 大于 `1` | 更发散，更容易跑偏、重复、幻觉 |

直觉：

```text
低温度 = 更保守
高温度 = 更开放，但更不可靠
```

## 2. top_p 的原理

`top_p` 也叫 nucleus sampling。它不是“取前 p 个 token”，也不是“取前 1/p 个 token”。

它的过程是：

1. 把 token 按概率从高到低排序。
2. 从最高概率 token 开始累加概率。
3. 取累计概率刚好达到 `p` 的最小 token 集合。
4. 只在这个集合里采样。

例如：

| token | 概率 | 累计概率 |
|---|---:|---:|
| A | `0.45` | `0.45` |
| B | `0.25` | `0.70` |
| C | `0.15` | `0.85` |
| D | `0.08` | `0.93` |
| E | `0.04` | `0.97` |

如果 `top_p=0.8`，候选可能是 `A/B/C`。

如果 `top_p=0.95`，候选可能变成 `A/B/C/D/E`。

所以：

```text
top_p 越大，候选范围越宽。
top_p 越小，候选范围越窄。
```

## 3. 为什么 `temp=0.8 top_p=0.95` 会出现“接口接口接口”

Stage 2 采样里出现了：

```text
接口接口接口接口接口...
```

主要原因不是模型“重新训练”，而是生成阶段的采样路径失控：

| 原因 | 解释 |
|---|---|
| 高 temperature | 让低概率 token 更容易被采到 |
| 大 top_p | 保留更宽的候选集合 |
| 没有足够重复惩罚 | 已经重复的 token 没有被强力压低 |
| 数据有模板化倾向 | 模型学会了固定尾句和固定结构 |
| max tokens 较长 | 重复一旦开始，会有更长空间继续重复 |

解决方向：

| 场景 | 建议 |
|---|---|
| 严肃评测 | `temperature=0.2`、`top_p=0.8` |
| 面试答案生成 | `temperature=0.2~0.4`、`top_p=0.8~0.9` |
| 创意采样 | 可以升高，但必须人工筛选 |
| 防重复 | 加 `repetition_penalty=1.1~1.2`，并降低 `max_new_tokens` |

## 4. 三版结果对比

同一批 Stage 2 评测题，共 6 条。

| 版本 | 数据/训练 | passed | average_score | 判断 |
|---|---|---:|---:|---|
| effective SFT | 96 条 SFT 数据 | `4/6` | `0.7500` | 有项目词汇，但概念错误较多 |
| DPO v1 | 12 条通用 DPO 偏好数据，30 step | `5/6` | `0.8333` | 关键词分数提升，但仍有事实错误 |
| DPO correction | 8 条 Stage 2 错误修正数据，40 step | `4/6` | `0.7917` | 修正部分概念，但引入重复和幻觉 |

## 5. DPO v1 的实际变化

DPO v1 的结果：

```text
passed: 5/6
average_score: 0.8333
```

进步：

| 项 | 变化 |
|---|---|
| `training_log` | 从 `0.5` 提升到 `0.75` |
| `eval_vs_train` | 从 `0.25` 提升到 `0.5` |
| `a800_shared_training` | 保持 `1.0` |
| `adapter_inference` | 保持 `1.0` |
| `qlora_memory` | 保持 `1.0` |

但仍有错误：

1. `top_p` 仍解释错。
2. `train loss` 仍有“先下降后上升”等不严谨表述。
3. 只是关键词分数提升，不代表概念完全正确。

结论：DPO v1 有局部进步，但不合格。

## 6. DPO correction 的实际变化

Correction 数据文件：

```text
data/stage3_dpo_stage2_corrections.jsonl
```

训练输出：

```text
outputs/stage3_qwen3_17b_dpo_corrections/v0-20260616-034853/checkpoint-40
```

结果：

```text
passed: 4/6
average_score: 0.7917
```

进步：

| 项 | 变化 |
|---|---|
| `training_log` | 提升到 `1.0` |
| `eval_vs_train` | 提升到 `0.75` |

退化：

| 项 | 问题 |
|---|---|
| `a800_shared_training` | 出现“超过 24 小时会超出 nvidia-smi 显存限制”等幻觉，并重复多次 |
| `qlora_memory` | 重复“4bit 量化加载并冻结基座模型” |
| `sampling_params` | 仍把 `top_p` 说成概率最高的 `p% token`，不够准确 |

结论：Correction DPO 修正了目标错误，但过拟合和重复更明显，整体不能作为更好的 adapter。

## 7. 为什么 correction DPO 会退化

主要原因：

| 原因 | 解释 |
|---|---|
| 数据太少 | 只有 8 条 correction 偏好数据 |
| 训练步数偏强 | 40 step 对 8 条数据相当于反复看很多轮 |
| 目标过窄 | 数据集中在几个概念错误，容易牺牲通用回答 |
| 缺少重复惩罚样本 | 没有足够 DPO 样本专门惩罚重复输出 |
| 只用关键词评测 | 规则分数不能发现所有事实错误 |

这说明：DPO 不是“只要加 rejected 就会变好”。偏好数据太少、太窄、训练太猛，会造成局部修正和全局退化同时发生。

## 8. 下一轮应该怎么改

下一轮不要继续直接加 step。应该改数据和训练策略：

1. 把 correction 数据扩到 `30~50` 条，而不是 8 条。
2. 每类错误至少 5 条变体，例如：
   - `train loss` 与泛化。
   - `token_acc` 与真实效果。
   - `grad_norm` 与训练稳定性。
   - `temperature/top_p` 原理。
   - 高温重复输出。
   - 关键词评测局限。
3. 加入“反重复” DPO 样本：
   - chosen：简洁回答。
   - rejected：连续重复同一句或同一个词。
4. 降低 correction DPO 强度：
   - `MaxSteps=15~20`
   - 或降低学习率到 `1e-5~2e-5`
5. 固定同一批 prompt 做三方对比：
   - effective SFT
   - DPO v1
   - DPO correction

当前推荐保留 `DPO v1` 作为阶段性结果，不推荐把 `DPO correction checkpoint-40` 当成更优模型。

## 9. 严格结论

这轮实验的真正收获不是“DPO 修好了模型”，而是：

1. DPO v1 证明偏好数据可以改善部分评测项。
2. Correction DPO 证明少量纠错数据会带来过拟合和重复风险。
3. 下一轮必须扩大 correction 数据覆盖面，并显式惩罚重复和幻觉。
4. 评估不能只看关键词分数，必须看原始回答。
