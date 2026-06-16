# 02 DPO 数据格式和样本质量标准

DPO 的难点不是命令，而是数据。偏好数据写得差，训练越久越糟。

## 1. ms-swift 标准 DPO 格式

本阶段使用 `ms-swift` 推荐的标准格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是 A800 Qwen 微调学习教练..."},
    {"role": "user", "content": "在共享 A800 上开始训练前，为什么不能直接启动任务？"},
    {"role": "assistant", "content": "不能直接启动。第一步要看 nvidia-smi..."}
  ],
  "rejected_response": "直接运行训练脚本就行，显存不够会自己报错，报错后再处理。",
  "reason": "chosen 有安全边界、资源检查和日志要求；rejected 忽略共享服务风险。"
}
```

字段解释：

| 字段 | 作用 |
|---|---|
| `messages` | 对话上下文 |
| `messages[-1]` | chosen，也就是希望模型更偏向的回答 |
| `rejected_response` | rejected，也就是希望模型远离的回答 |
| `reason` | 给你自己看的数据审查理由，训练时不是核心字段 |

注意：这里没有单独写 `chosen` 字段，因为 `ms-swift` 的标准格式是把 chosen 放在 `messages` 的最后一个 assistant 里。

## 2. 当前数据在哪里

本地快照路径：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/data/stage3_dpo_preferences.jsonl
```

远端 A800 路径：

```text
/root/zhl/qwen-qlora-lab/data/stage3_dpo_preferences.jsonl
```

生成脚本：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage3_make_alignment_data.py
```

## 3. 逐条审查一条样本

以 `dpo_a800_safety` 为例。

prompt：

```text
在共享 A800 上开始训练前，为什么不能直接启动任务？
```

chosen 的关键点：

| 关键点 | 为什么重要 |
|---|---|
| `nvidia-smi` | 训练前先查 GPU 进程和显存 |
| `CUDA_VISIBLE_DEVICES` | 限制自己的训练只看指定卡 |
| `df -h /root` | 防止磁盘写满 |
| 不停止已有服务 | 符合共享服务器边界 |
| 记录命令和日志 | 可复盘、可排错 |

rejected 的问题：

| 问题 | 类型 |
|---|---|
| “直接运行训练脚本就行” | 安全流程错误 |
| “显存不够会自己报错” | 被动处理风险 |
| 没提已有服务 | 共享环境风险 |
| 没提日志 | 不可复盘 |

这是一条合格 DPO 样本，因为 chosen 和 rejected 有清楚差异。

## 4. 什么是不合格 DPO 样本

不合格样本一：

```json
{
  "chosen": "A800 训练前要看显存。",
  "rejected_response": "A800 训练前需要检查显存。"
}
```

问题：两个答案差不多，偏好信号太弱。

不合格样本二：

```json
{
  "chosen": "训练前要看显存、磁盘、已有服务、日志、模型路径、数据路径、batch size、学习率、环境变量、Docker 版本...",
  "rejected_response": "训练前要检查显存。"
}
```

问题：可能只是 chosen 更长，不一定更好。

不合格样本三：

```json
{
  "chosen": "直接杀掉其他进程释放显存。",
  "rejected_response": "先看 nvidia-smi，不影响已有服务。"
}
```

问题：偏好方向反了，会教坏模型。

## 5. 你审查 DPO 数据时必须问的 6 个问题

每条样本都要过这 6 个问题：

1. chosen 是否事实正确？
2. rejected 是否有明确缺陷？
3. 缺陷是事实错、遗漏风险、格式差，还是价值偏好差？
4. chosen 是否只是更长，而不是真的更好？
5. 这条样本是否符合当前项目规范？
6. 如果面试官问“为什么这条 chosen 更好”，你能不能 30 秒讲清？

## 6. 当前 12 条 DPO 样本覆盖面

| 样本 | 训练偏好 |
|---|---|
| `dpo_a800_safety` | 共享 GPU 安全边界 |
| `dpo_adapter_inference` | adapter 与 merged model 区别 |
| `dpo_train_loss` | 不只看 train loss |
| `dpo_qlora_memory` | QLoRA 节省显存但不是零成本 |
| `dpo_dpo_data` | 偏好数据质量 |
| `dpo_grpo_reward` | reward 设计优先于 epoch |
| `dpo_sampling_params` | 采样参数解释 |
| `dpo_export_risk` | merged model 导出风险 |
| `dpo_webui_limit` | WebUI 不能替代脚本化实验 |
| `dpo_failure_cases` | 失败案例复盘 |
| `dpo_alignment_goal` | 对齐不只是礼貌 |
| `dpo_strict_learning` | 深入学习的交付标准 |

## 7. 你下一步要做的练习

打开数据文件后，逐条写旁注：

```text
id:
chosen 好在哪里:
rejected 错在哪里:
如果要改进，我会怎么改:
这条样本可能导致什么副作用:
```

这一步比直接跑 DPO 更重要。你如果不能审数据，后面看训练日志没有意义。
