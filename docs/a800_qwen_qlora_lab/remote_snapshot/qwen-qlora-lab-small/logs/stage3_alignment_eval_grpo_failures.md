# Stage 3 GRPO Failure Cases

来源：`stage3_alignment_eval_cases.jsonl` 固定 prompt 对比 effective SFT、DPO v1、GRPO v1。

- failure_or_regression_cases: `6`

## Case 1: grpo_dpo_format

- model: `stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20`
- score: `0.0`
- dpo_v1_score: `0.0`
- failure_type: `missing_required, repetition`
- missed_required: `chosen, rejected, messages, 偏好`
- forbidden_hits: ``
- root_cause: `需要人工复核；优先判断是 reward 关键词覆盖不足、GRPO 训练步数不足，还是回答内容真实缺陷。`
- next_action: `把该 case 加入下一轮 DPO/GRPO 数据，或调整 reward 中对应概念的同义词和惩罚项。`

prompt:

```text
用小白能懂的话解释 DPO 数据格式。
```

GRPO response:

```text
<think>

</think>

建议：DPO 通常只在同一 batch 内对单条提示做改写，而不是对已有训练集做重新洗牌；每条 prompt 会固定放 3 条，每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放 3 条；每条 prompt 会固定放
```

## Case 2: grpo_reward_design

- model: `stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20`
- score: `0.0`
- dpo_v1_score: `0.0`
- failure_type: `missing_required`
- missed_required: `正确性, 格式, 安全, 简洁`
- forbidden_hits: ``
- root_cause: `需要人工复核；优先判断是 reward 关键词覆盖不足、GRPO 训练步数不足，还是回答内容真实缺陷。`
- next_action: `把该 case 加入下一轮 DPO/GRPO 数据，或调整 reward 中对应概念的同义词和惩罚项。`

prompt:

```text
设计一个 GRPO 奖励函数时，至少要考虑哪些维度？
```

GRPO response:

```text
<think>

</think>

至少要从任务目标、奖励表达、训练步数、参数选择、数据质量和训练验证这几个方面来综合判断。奖励函数不能只靠单条样本说话，必须保证每条样本都至少有 3 条以上固定轨迹可以对比。
```

## Case 3: grpo_eval_failure

- model: `stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20`
- score: `0.25`
- dpo_v1_score: `0.25`
- failure_type: `missing_required`
- missed_required: `reward, 过拟合, 下一轮`
- forbidden_hits: ``
- root_cause: `需要人工复核；优先判断是 reward 关键词覆盖不足、GRPO 训练步数不足，还是回答内容真实缺陷。`
- next_action: `把该 case 加入下一轮 DPO/GRPO 数据，或调整 reward 中对应概念的同义词和惩罚项。`

prompt:

```text
为什么 GRPO 跑完后必须看失败案例？
```

GRPO response:

```text
<think>

</think>

GRPO 跑完后必须看失败案例，因为单条召回可能没有明显特征，但会看召回覆盖率、召回物品种类、召回覆盖率、召回物品相似度、召回物品数量、召回物品多样性、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率、召回物品召回覆盖率
```

## Case 4: grpo_adapter_vs_merged

- model: `stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20`
- score: `0.25`
- dpo_v1_score: `0.25`
- failure_type: `missing_required`
- missed_required: `base, 合并, 磁盘`
- forbidden_hits: ``
- root_cause: `需要人工复核；优先判断是 reward 关键词覆盖不足、GRPO 训练步数不足，还是回答内容真实缺陷。`
- next_action: `把该 case 加入下一轮 DPO/GRPO 数据，或调整 reward 中对应概念的同义词和惩罚项。`

prompt:

```text
adapter 推理和 merged model 推理怎么取舍？
```

GRPO response:

```text
<think>

</think>

可以这样理解：LoRA adapter 不能脱离基座模型单独推理；部署时需要先加载原始模型，再通过 adapters 参数挂载 checkpoint；接口测试至少要上传测试 prompt、保存响应和服务日志。 面试时要强调资源约束、参数选择和验证闭环。
```

## Case 5: grpo_strict_report

- model: `stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20`
- score: `0.2`
- dpo_v1_score: `0.0`
- failure_type: `missing_required`
- missed_required: `数据, 日志, 评测, 失败`
- forbidden_hits: ``
- root_cause: `需要人工复核；优先判断是 reward 关键词覆盖不足、GRPO 训练步数不足，还是回答内容真实缺陷。`
- next_action: `把该 case 加入下一轮 DPO/GRPO 数据，或调整 reward 中对应概念的同义词和惩罚项。`

prompt:

```text
一轮对齐实验的报告必须包含哪些内容？
```

GRPO response:

```text
<think>

</think>

一轮对齐实验至少要记录训练命令、loss 曲线、checkpoint 路径、显存变化、固定 token 响应和对比 prompt 响应。没有对比基准，不能说明方法有效性；没有显存、loss、token 响应对比，不能说明方法改进点。
```

## Case 6: grpo_train_loss_limit

- model: `stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20`
- score: `0.0`
- dpo_v1_score: `0.0`
- failure_type: `missing_required`
- missed_required: `泛化, 评测, 过拟合, 人工`
- forbidden_hits: ``
- root_cause: `需要人工复核；优先判断是 reward 关键词覆盖不足、GRPO 训练步数不足，还是回答内容真实缺陷。`
- next_action: `把该 case 加入下一轮 DPO/GRPO 数据，或调整 reward 中对应概念的同义词和惩罚项。`

prompt:

```text
为什么 train loss 降低不等于对齐成功？
```

GRPO response:

```text
<think>

</think>

train loss 降低通常意味着模型学习更充分，但不能单独用来判断对齐质量。需要同时看固定样本回复、loss 曲线、reward curve、token 游戏日志和人类评估。如果效果不明显，应先检查数据质量和训练步数，而不是盲目换模型。
```
