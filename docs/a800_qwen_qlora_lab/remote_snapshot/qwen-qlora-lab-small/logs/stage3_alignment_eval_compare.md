# Stage 3 Alignment Fixed Prompt Evaluation

同一批 `stage3_alignment_eval_cases.jsonl`，对比 effective SFT、DPO v1、GRPO v1。

## Summary

| model | cases | passed | average_score |
|---|---:|---:|---:|
| `effective_sft` | `8` | `1` | `0.2125` |
| `dpo_v1` | `8` | `1` | `0.2188` |
| `grpo_v1` | `8` | `2` | `0.3063` |

## Case Scores

| case | effective_sft | dpo_v1 | grpo_v1 | GRPO verdict |
|---|---:|---:|---:|---|
| `grpo_a800_safety` | `1.0` | `1.0` | `1.0` | pass |
| `grpo_dpo_format` | `0.0` | `0.0` | `0.0` | fail: missing_required, repetition |
| `grpo_reward_design` | `0.0` | `0.0` | `0.0` | fail: missing_required |
| `grpo_eval_failure` | `0.0` | `0.25` | `0.25` | fail: missing_required |
| `grpo_adapter_vs_merged` | `0.25` | `0.25` | `0.25` | fail: missing_required |
| `grpo_sampling` | `0.25` | `0.25` | `0.75` | pass |
| `grpo_strict_report` | `0.2` | `0.0` | `0.2` | fail: missing_required |
| `grpo_train_loss_limit` | `0.0` | `0.0` | `0.0` | fail: missing_required |

## Per Case Outputs

### grpo_a800_safety

note: 奖励安全检查、GPU 限定和日志意识。

prompt:

```text
给出共享 A800 上启动训练前的安全检查清单。
```

#### effective_sft

- score: `1.0`
- passed: `True`
- missed: ``
- forbidden: ``

```text
<think>

</think>

可以这样理解：共享 GPU 上训练应先检查 nvidia-smi，确认显存、已有进程和空闲空间；再限制 CUDA_VISIBLE_DEVICES、batch size 和 max_length；最后保存命令、日志和显存记录，避免影响已有服务。 面试时要强调资源约束、参数选择和验证闭环。
```

#### dpo_v1

- score: `1.0`
- passed: `True`
- missed: ``
- forbidden: ``

```text
<think>

</think>

建议：共享 GPU 上训练应先检查 nvidia-smi，确认显存、已有进程、空闲空间和显存利用率；再限制 CUDA_VISIBLE_DEVICES、batch size 和 max_length；最后保存命令、日志和显存记录，避免影响已有服务。 这条规则用于保证实验可复现、低风险。
```

#### grpo_v1

- score: `1.0`
- passed: `True`
- missed: ``
- forbidden: ``

```text
<think>

</think>

建议：共享 GPU 上训练应先检查 nvidia-smi，确认显存、已有进程、空闲空间和显存利用率；再限制 CUDA_VISIBLE_DEVICES、batch size 和 max_length；最后保存命令、日志和显存记录，避免影响已有服务。 这条规则用于保证实验可复现、低风险。
```

### grpo_dpo_format

note: 奖励讲清 chosen/rejected 和偏好来源。

prompt:

```text
用小白能懂的话解释 DPO 数据格式。
```

#### effective_sft

- score: `0.0`
- passed: `False`
- missed: `chosen, rejected, messages, 偏好`
- forbidden: ``

```text
<think>

</think>

简洁回答：DPO 通常只有一条测试样例，格式是 [prompt, response]；验证重点是 cost、accuracy 和相关性；不能只靠单条实验判断因果。 如果效果不明显，应先检查数据质量和训练步数，而不是盲目换模型。
```

#### dpo_v1

- score: `0.0`
- passed: `False`
- missed: `chosen, rejected, messages, 偏好`
- forbidden: ``

```text
<think>

</think>

简洁回答：DPO 通常只支持单条样本、固定 prompt、原始回复三要素；每条样本至少要包含 prompt、reward、logit、原始回复这四个字段；每条样本只能有一个 reward。 如果效果不明显，应先检查数据质量和训练步数，而不是盲目换模型。
```

#### grpo_v1

- score: `0.0`
- passed: `False`
- missed: `chosen, rejected, messages, 偏好`
- forbidden: ``

```text
<think>

</think>

建议：DPO 通常只在同一 batch 内对单条提示做改写，而不是对已有训练集做重新洗牌；每条 prompt 会固定放 3 条，每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放
```

### grpo_reward_design

note: 奖励多维度 reward 思维，惩罚单一关键词思维。

prompt:

```text
设计一个 GRPO 奖励函数时，至少要考虑哪些维度？
```

#### effective_sft

- score: `0.0`
- passed: `False`
- missed: `正确性, 格式, 安全, 简洁`
- forbidden: ``

```text
<think>

</think>

实践中要点是：奖励不能只靠一个样例学习，应先检查数据质量和相关性，再尝试固定参数、单条测试或随机移除样本；如果没有显著提升，应先检查数据质量和训练步数，而不是盲目换模型。 学习时不要只看命令，要同时看数据、参数、日志和输出。
```

#### dpo_v1

- score: `0.0`
- passed: `False`
- missed: `正确性, 格式, 安全, 简洁`
- forbidden: ``

```text
<think>

</think>

至少要从任务目标、数据质量和训练步数、奖励缩放、参数选择、固定测试点、固定训练步数、固定学习率、固定 batch size、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、固定 max_length、
```

#### grpo_v1

- score: `0.0`
- passed: `False`
- missed: `正确性, 格式, 安全, 简洁`
- forbidden: ``

```text
<think>

</think>

至少要从任务目标、奖励表达、训练步数、参数选择、数据质量和训练验证这几个方面来综合判断。奖励函数不能只靠单条样本说话，必须保证每条样本都至少有 3 条以上固定轨迹可以对比。
```

### grpo_eval_failure

note: 奖励能把失败案例和 reward 设计联系起来。

prompt:

```text
为什么 GRPO 跑完后必须看失败案例？
```

#### effective_sft

- score: `0.0`
- passed: `False`
- missed: `失败案例, reward, 过拟合, 下一轮`
- forbidden: ``

```text
<think>

</think>

实践中要点是：GRPO 不能只靠参数随机试错，必须先检查数据质量和相关性，再固定变量做单变量分析；否则容易得出不严谨的结论。 学习时不要只看命令，要同时看数据、参数、日志和输出。
```

#### dpo_v1

- score: `0.25`
- passed: `False`
- missed: `reward, 过拟合, 下一轮`
- forbidden: ``

```text
<think>

</think>

建议：跑 GRPO 时至少要记录命令、日志、loss 曲线和显存变化；只看命令不能判断问题。如果没有明显失败案例，不能只靠一个 batch 做结论。 这条规则用于保证实验可复现、低风险。
```

#### grpo_v1

- score: `0.25`
- passed: `False`
- missed: `reward, 过拟合, 下一轮`
- forbidden: ``

```text
<think>

</think>

GRPO 跑完后必须看失败案例，因为单条召回可能没有明显特征，但会看召回覆盖率、召回物品种类、召回覆盖率、召回物品相似度、召回物品数量、召回物品多样性、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率
```

### grpo_adapter_vs_merged

note: 奖励部署形态和资源成本解释。

prompt:

```text
adapter 推理和 merged model 推理怎么取舍？
```

#### effective_sft

- score: `0.25`
- passed: `False`
- missed: `base, 合并, 磁盘`
- forbidden: ``

```text
<think>

</think>

实践中要点是：LoRA adapter 不能脱离基座模型单独推理；部署时需要先加载原始模型，再通过 adapters 参数挂载 checkpoint；接口测试应保存请求、响应和服务日志。 学习时不要只看命令，要同时看数据、参数、日志和输出。
```

#### dpo_v1

- score: `0.25`
- passed: `False`
- missed: `base, 合并, 磁盘`
- forbidden: ``

```text
<think>

</think>

可以这样理解：LoRA adapter 不能脱离基座模型单独推理；部署时需要先加载原始模型，再通过 adapters 参数挂载 checkpoint；接口测试至少要上传测试 prompt、保存响应和服务端日志。 面试时要强调资源约束、参数选择和验证闭环。
```

#### grpo_v1

- score: `0.25`
- passed: `False`
- missed: `base, 合并, 磁盘`
- forbidden: ``

```text
<think>

</think>

可以这样理解：LoRA adapter 不能脱离基座模型单独推理；部署时需要先加载原始模型，再通过 adapters 参数挂载 checkpoint；接口测试至少要上传测试 prompt、保存响应和服务日志。 面试时要强调资源约束、参数选择和验证闭环。
```

### grpo_sampling

note: 奖励固定变量和参数对比意识。

prompt:

```text
temperature 和 top_p 调参时怎么做对比实验？
```

#### effective_sft

- score: `0.25`
- passed: `False`
- missed: `固定 prompt, temperature, 对比`
- forbidden: ``

```text
<think>

</think>

实践中要点是：温度控制的是生成多样性，top_p 是基于概率排序的约束采样，适合中英数据；低温度会更保守，适合先试错再优化；高温度会显著增加输出多样性。 学习时不要只看命令，要同时看数据、参数、日志和输出。
```

#### dpo_v1

- score: `0.25`
- passed: `False`
- missed: `固定 prompt, temperature, 对比`
- forbidden: ``

```text
<think>

</think>

可以这样理解：top_p 控制的是 n 维 softmax 上取 top p% 做排序，p 越大，输出越随机；p 越小，输出越集中。温度控制的是整体表达能力，温度越高表达越泛化，温度越低表达越保守。 面试时要强调资源约束、参数选择、问题验证和可复现性。
```

#### grpo_v1

- score: `0.75`
- passed: `True`
- missed: `对比`
- forbidden: ``

```text
<think>

</think>

可以这样理解：top_p 控制的是 n 个 token 中按概率排序取前 p 的 token；temperature 控制的是输出 token 之间的温度，温度越高表达越随机；pillar 控制的是单条样本上固定 prompt 生成多个 token 的数量。实践中要点是：top_p 不能脱离 n 个 token 说好就好；温度不能只看一次；pillar 不能只看一条样本。
```

### grpo_strict_report

note: 奖励完整实验闭环。

prompt:

```text
一轮对齐实验的报告必须包含哪些内容？
```

#### effective_sft

- score: `0.2`
- passed: `False`
- missed: `数据, 日志, 评测, 失败`
- forbidden: ``

```text
<think>

</think>

可以这样理解：一轮对齐实验至少要记录训练命令、loss 曲线、checkpoint 路径和显存变化；没有验证集时不能只靠 train loss 判断效果；应固定测试 prompt 做对比。 面试时要强调资源约束、参数选择和验证闭环。
```

#### dpo_v1

- score: `0.0`
- passed: `False`
- missed: `数据, 命令, 日志, 评测, 失败`
- forbidden: ``

```text
<think>

</think>

建议：一轮对齐实验至少要保证样本质量、对齐质量和基座模型质量；要记录每条样本的原始 prompt、对齐 token 和输出；要固定对比维度，比如只对比 token 0 上的输出；要至少记录三组实验对比。 这条规则用于保证实验可复现、低风险。
```

#### grpo_v1

- score: `0.2`
- passed: `False`
- missed: `数据, 日志, 评测, 失败`
- forbidden: ``

```text
<think>

</think>

一轮对齐实验至少要记录训练命令、loss 曲线、checkpoint 路径、显存变化、固定 token 响应和对比 prompt 响应。没有对比基准，不能说明方法有效性；没有显存、loss、token 响应对比，不能说明方法改进点。
```

### grpo_train_loss_limit

note: 奖励区分训练拟合和真实质量。

prompt:

```text
为什么 train loss 降低不等于对齐成功？
```

#### effective_sft

- score: `0.0`
- passed: `False`
- missed: `泛化, 评测, 过拟合, 人工`
- forbidden: ``

```text
<think>

</think>

简洁回答：Train loss 降低不代表对齐成功，可能原因包括数据质量和训练步数、checkpoint 版本不一致、或单条样本训练效果差。实践中应固定测试 prompt 做对比，而不是只靠现象判断模型能力。
```

#### dpo_v1

- score: `0.0`
- passed: `False`
- missed: `泛化, 评测, 过拟合, 人工`
- forbidden: ``

```text
<think>

</think>

train loss 降低通常意味着模型学习到了更优的参数，但对齐成功需要同时满足：loss 更低、更稳定；更关键的是要能看到 reward 和 action 之间有因果关系。如果只靠 loss 来判断对齐效果，会比较片面。
```

#### grpo_v1

- score: `0.0`
- passed: `False`
- missed: `泛化, 评测, 过拟合, 人工`
- forbidden: ``

```text
<think>

</think>

train loss 降低通常意味着模型学习更充分，但不能单独用来判断对齐质量。需要同时看固定样本回复、loss 曲线、reward curve、token 游戏日志和人类评估。如果效果不明显，应先检查数据质量和训练步数，而不是盲目换模型。
```
