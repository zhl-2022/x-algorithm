# 04 DPO 和 GRPO 训练脚本 walkthrough

本阶段训练入口分成两类：

1. 本地 PowerShell 脚本：负责从 Windows 发起远端命令。
2. 远端 bash 脚本：负责在 A800 Docker 容器内真正执行训练。

你必须先看懂脚本，再训练。

## 1. 先同步和准备资产

第一步：

```powershell
.\scripts\a800\sync_stage3_alignment_assets.ps1
```

它做 4 件事：

| 步骤 | 作用 |
|---|---|
| 本地运行 `stage3_make_alignment_data.py` | 生成本地快照里的 Stage 3 数据 |
| 打包 `data/stage3_*` 和 `scripts/stage3_*` | 只同步必要学习资产 |
| 先复制到 `srv4` | 因为本地不能直接到 A800 内网地址 |
| 再从 `srv4` 复制到 A800 | 落到 `/root/zhl/qwen-qlora-lab` |

第二步：

```powershell
.\scripts\a800\prepare_stage3_alignment_assets.ps1
```

它会在 A800 上运行：

```bash
bash /root/zhl/qwen-qlora-lab/scripts/stage3_prepare_alignment_assets.sh
```

输出文件：

```text
/root/zhl/qwen-qlora-lab/data/stage3_dpo_preferences.jsonl
/root/zhl/qwen-qlora-lab/data/stage3_grpo_prompts.jsonl
/root/zhl/qwen-qlora-lab/data/stage3_alignment_eval_cases.jsonl
/root/zhl/qwen-qlora-lab/logs/stage3_prepare_alignment_assets.log
/root/zhl/qwen-qlora-lab/logs/stage3_reward_selftest.log
```

## 2. 就绪检查

执行：

```powershell
.\scripts\a800\check_stage3_alignment_ready.ps1
```

它会检查：

| 检查项 | 为什么必须看 |
|---|---|
| `df -h /root` | 防止训练或导出写满磁盘 |
| `nvidia-smi` | 防止影响已有服务 |
| `models/Qwen3-1.7B/config.json` | 确认基座模型存在 |
| Stage 3 数据文件 | 确认训练数据已经生成 |
| effective SFT checkpoint | DPO/GRPO 从已有 SFT adapter 起步 |
| `swift rlhf --help` | 确认服务器版本支持关键参数 |

如果这个检查失败，不要训练。

## 3. DPO 训练入口

PowerShell：

```powershell
.\scripts\a800\run_stage3_dpo_train.ps1 -ConfirmTrain -MaxSteps 30
```

真正执行的是容器内脚本：

```text
/workspace/qwen-qlora-lab/scripts/stage3_train_dpo_inside.sh
```

关键命令结构：

```bash
swift rlhf \
  --rlhf_type dpo \
  --model models/Qwen3-1.7B \
  --adapters outputs/qwen3_17b_text_effective/.../checkpoint-120 \
  --ref_adapters outputs/qwen3_17b_text_effective/.../checkpoint-120 \
  --tuner_type lora \
  --quant_bits 4 \
  --dataset data/stage3_dpo_preferences.jsonl \
  --beta 0.1 \
  --rpo_alpha 0.1
```

关键参数解释：

| 参数 | 当前值 | 解释 |
|---|---:|---|
| `--rlhf_type` | `dpo` | 选择 DPO 对齐算法 |
| `--adapters` | effective SFT checkpoint | 从已经学过项目知识的 adapter 起步 |
| `--ref_adapters` | 同一个 checkpoint | reference model，用来约束偏离 |
| `--tuner_type` | `lora` | 继续用 LoRA 方式训练 |
| `--quant_bits` | `4` | QLoRA 加载，节省显存 |
| `--dataset` | DPO 数据 | 使用 `messages + rejected_response` |
| `--beta` | `0.1` | KL/偏离参考模型的约束强度 |
| `--rpo_alpha` | `0.1` | 混入一定 SFT loss，提高稳定性 |
| `--max_steps` | `30` | 第一轮只做学习试跑 |

为什么先跑 DPO：

1. DPO 比 GRPO 更容易解释。
2. DPO 不需要先设计复杂 reward。
3. DPO 可以检查你是否真的理解“偏好数据”。

## 4. GRPO 训练入口

PowerShell：

```powershell
.\scripts\a800\run_stage3_grpo_train.ps1 -ConfirmTrain -MaxSteps 20
```

真正执行的是：

```text
/workspace/qwen-qlora-lab/scripts/stage3_train_grpo_inside.sh
```

关键命令结构：

```bash
swift rlhf \
  --rlhf_type grpo \
  --model models/Qwen3-1.7B \
  --adapters <DPO checkpoint 或 effective SFT checkpoint> \
  --ref_adapters <同一个起点 checkpoint> \
  --dataset data/stage3_grpo_prompts.jsonl \
  --external_plugins scripts/stage3_reward_plugin.py \
  --reward_funcs stage3_quality \
  --num_generations 4 \
  --max_completion_length 256 \
  --use_vllm false
```

关键参数解释：

| 参数 | 当前值 | 解释 |
|---|---:|---|
| `--rlhf_type` | `grpo` | 选择 GRPO |
| `--external_plugins` | `stage3_reward_plugin.py` | 加载自定义 reward |
| `--reward_funcs` | `stage3_quality` | 使用已注册的规则奖励函数 |
| `--num_generations` | `4` | 每个 prompt 生成 4 条回答做组内比较 |
| `--max_completion_length` | `256` | 控制单次生成长度和显存 |
| `--temperature` | `0.7` | 保持一定探索 |
| `--top_p` | `0.9` | 限制采样候选范围 |
| `--beta` | `0.04` | KL 约束，防止偏离参考模型太远 |
| `--use_vllm` | `false` | 第一轮学习优先简单稳定，不引入 vLLM 复杂度 |

## 5. 为什么训练脚本要求确认

脚本里有硬保护：

```bash
if [[ "${CONFIRM_STAGE3_TRAIN:-0}" != "1" ]]; then
  echo "Refuse to start training..."
  exit 2
fi
```

原因：

1. A800 是共享服务器。
2. GRPO 会生成多条回答，比 SFT 更吃显存和时间。
3. 训练脚本不应该被误点运行。
4. 你要先完成数据审查和 reward 自检。

## 6. 训练后看哪些日志

DPO：

```text
/root/zhl/qwen-qlora-lab/logs/stage3_dpo.train.log
/root/zhl/qwen-qlora-lab/logs/stage3_dpo.before.nvidia-smi.txt
/root/zhl/qwen-qlora-lab/logs/stage3_dpo.after.nvidia-smi.txt
/root/zhl/qwen-qlora-lab/outputs/stage3_qwen3_17b_dpo/
```

GRPO：

```text
/root/zhl/qwen-qlora-lab/logs/stage3_grpo.train.log
/root/zhl/qwen-qlora-lab/logs/stage3_reward_selftest.log
/root/zhl/qwen-qlora-lab/logs/stage3_grpo.before.nvidia-smi.txt
/root/zhl/qwen-qlora-lab/logs/stage3_grpo.after.nvidia-smi.txt
/root/zhl/qwen-qlora-lab/outputs/stage3_qwen3_17b_grpo/
```

你要重点找：

| 日志内容 | 你要判断什么 |
|---|---|
| dataset rows | 数据是否正确加载 |
| trainable params | 是否真的只训练 LoRA |
| loss | 是否出现 NaN 或异常爆炸 |
| reward | GRPO 是否有区分度 |
| KL | 是否偏离 reference 太强 |
| completions | 模型生成的回答是否被 reward 正确区分 |
| checkpoint | 是否按预期保存 |

## 7. 第一轮不追求高分

第一轮 DPO/GRPO 的目标是：

1. 数据能加载。
2. 训练能开始。
3. 日志能解释。
4. reward 能区分好坏回答。
5. 能产出失败案例。

如果你只追求“效果变好”，你会忽略对齐实验最重要的部分：错误归因。
