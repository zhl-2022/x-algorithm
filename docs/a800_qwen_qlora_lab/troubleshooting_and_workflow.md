# A800 Qwen QLoRA 实操难题复盘与复用 Workflow

## 背景

本次任务是在 A800 `test3` 的 `/root/zhl/qwen-qlora-lab` 下搭建一个独立 Qwen QLoRA 学习项目，跳过第一轮测试，直接完成三轮训练：

| 轮次 | 模型 | 数据 | 目标 |
|---|---|---|---|
| Round 1 | `Qwen/Qwen3-1.7B` | 文本 SFT | 跑通 QLoRA 文本微调 |
| Round 2 | `Qwen/Qwen3.5-0.8B` | 文本 SFT | 验证 Qwen3.5 小模型文本微调 |
| Round 3 | `Qwen/Qwen3.5-2B` | 图文 SFT | 验证多模态 QLoRA 训练和图片输入推理 |

最终三轮训练和推理验证均已完成，核心产物在远端：

- 项目目录：`/root/zhl/qwen-qlora-lab`
- Docker 镜像：`qwen-qlora-swift:latest`
- 训练框架：`ms-swift 4.3.0`
- 最终 checkpoint：
  - `outputs/qwen3_17b_text/v1-20260613-211241/checkpoint-20`
  - `outputs/qwen35_08b_text/v1-20260613-212200/checkpoint-20`
  - `outputs/qwen35_2b_mm/v0-20260613-213024/checkpoint-20`

## 遇到的主要难题

### 1. A800 不是完全空闲机器，不能影响现有服务

**现象**

A800 上已有多个常驻服务占用显存，包括 vLLM、MinerU、Milvus 等。训练前显存基线大约是：

| 项目 | 数值 |
|---|---:|
| 总显存 | `81920 MiB` |
| 已用显存 | `57981 MiB` |
| 剩余显存 | `23175 MiB` |

**风险**

如果直接启动训练，可能挤占现有服务显存，导致业务容器 OOM 或推理服务异常。

**处理方式**

1. 训练前后都保存 `nvidia-smi`。
2. 所有训练容器固定 `--gpus device=0` 和 `CUDA_VISIBLE_DEVICES=0`。
3. 脚本里增加显存门槛：可用显存低于约 `18000 MiB` 时不自动启动训练。
4. 只停止自己启动的验证容器，不停止任何已有业务服务。

**结果**

训练结束后显存回到基线：

| 状态 | 显存 |
|---|---:|
| 训练前基线 | `57981 MiB used / 23175 MiB free` |
| 最终状态 | `57981 MiB used / 23175 MiB free` |

### 2. Windows PowerShell、双层 SSH 和远端 Bash 的引号冲突

**现象**

从本地 PowerShell 经 `srv4` 跳到 A800 时，复杂命令中的 `$()`、`$!`、管道和引号容易被本地 PowerShell 提前解析，典型错误包括：

- 本地执行了 `Get-Date`，而不是远端执行 `date`。
- 本地尝试执行 `grep`、`head`、`docker`。
- 远端变量 `$d` 被本地展开为空。
- `cd /root/zhl/qwen-qlora-lab` 没有按预期生效。

**处理方式**

1. 简单命令优先使用 A800 工作流入口。
2. 复杂命令尽量使用绝对路径，少依赖远端 `cd`。
3. 避免在一条 PowerShell 命令里嵌套太多 `$()`。
4. 必须使用远端变量时，转义 `$`，例如 `` `$! ``。
5. 关键检查用显式路径重跑，避免误读错误目录。

**推荐命令模板**

```powershell
ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 'nvidia-smi'"
```

需要访问项目文件时，优先写绝对路径：

```powershell
ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 'bash /root/zhl/qwen-qlora-lab/scripts/status.sh'"
```

### 3. `ms-swift`、`transformers` 和基础镜像 Torch 版本不匹配

**现象**

初始镜像从 `hiyouga/llamafactory:latest` 派生。安装最新版 `ms-swift` 和 `transformers` 后，遇到两个方向的兼容问题：

| 尝试 | 问题 |
|---|---|
| `transformers 4.57.1` | `Qwen3-1.7B` 可跑，但 `Qwen3.5` 的 `model_type=qwen3_5` 不被识别 |
| `transformers 5.10.2` | 能识别 Qwen3.5，但基础 Torch `2.6.0+cu124` 缺少 `torch.float8_e8m0fnu` |

**处理方式**

最终采用：

- `ms-swift 4.3.0`
- `transformers 5.10.2`
- `accelerate`
- `peft`
- `bitsandbytes`
- `modelscope`
- `decord`

并在容器内加了一个轻量 `sitecustomize.py` 兼容 shim：如果当前 Torch 没有 `torch.float8_e8m0fnu`，就临时映射到已有 float8 dtype，保证 Transformers 5 能启动。

**注意**

这个 shim 适合学习实验和环境打通，不建议直接视为生产级长期方案。生产环境更好的做法是升级到官方匹配的 Torch、CUDA、Transformers 组合。

### 4. `swift --help` 顶层命令异常，但子命令可用

**现象**

容器里执行 `swift --help` 报：

```text
KeyError: '--help'
```

但下面的子命令正常：

```bash
swift sft --help
swift infer --help
```

**处理方式**

没有把顶层 `swift --help` 作为环境失败条件，而是以 `swift sft --help`、`swift infer --help` 和实际训练命令作为有效验证。

**经验**

框架 CLI 的顶层帮助异常不一定代表训练功能不可用。验证时要测真正需要用的子命令。

### 5. `ms-swift 4.3.0` 参数和旧文档不完全一致

**现象**

早期训练命令使用了类似参数：

```bash
--train_type lora
--trust_remote_code true
```

在当前 `ms-swift 4.3.0` 中被判定为未识别参数。

**处理方式**

改成当前版本可接受的 LoRA/QLoRA 参数组合：

```bash
--tuner_backend peft
--quant_bits 4
--lora_rank 8
--lora_alpha 32
--target_modules all-linear
```

**经验**

大模型微调框架更新很快。不要完全依赖旧命令，必须用当前容器里的 `swift sft --help` 和实际报错修正参数。

### 6. ModelScope 下载成功，但必须做文件级验证

**现象**

ModelScope 能正常下载三个模型，但仅看到下载完成日志还不够，需要确认本地目录可被训练框架识别。

**处理方式**

每个模型目录都检查：

- `config.json`
- tokenizer 文件，例如 `tokenizer.json`、`tokenizer_config.json`
- safetensors 权重
- `du -sh` 目录大小

**最终模型大小**

| 模型目录 | 大小 |
|---|---:|
| `models/Qwen3-1.7B` | `3.8G` |
| `models/Qwen3.5-0.8B` | `1.7G` |
| `models/Qwen3.5-2B` | `4.3G` |

### 7. checkpoint 检测在空目录时会让脚本提前退出

**现象**

训练脚本使用 `set -euo pipefail`，如果直接执行：

```bash
find outputs/... -type d -name 'checkpoint-*'
```

当目录不存在时，`find` 返回非零，脚本可能提前退出。

**处理方式**

把 checkpoint 查找函数改成容错形式：

```bash
latest_ckpt() {
  find "$1" -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n1 || true
}
```

**经验**

训练流水线里“第一次运行”和“断点续跑”必须同时考虑。空目录不是失败，只是还没有 checkpoint。

### 8. `swift infer` 交互模式不适合自动化验证

**现象**

最初用 here-string 或 pipe 给 `swift infer` 喂 prompt，遇到两个问题：

1. 有时只处理第一条输入后 `EOFError`。
2. 有时进入交互循环等待下一条输入，导致验证容器一直占用显存。

**处理方式**

1. 先确认卡住的是自己启动的 `qwen-qlora-swift:latest` 容器。
2. 只停止这个验证容器，不影响其他业务容器。
3. 把推理验证改成批量模式：

```bash
swift infer \
  --adapters <checkpoint> \
  --val_dataset data/infer_text_prompts.jsonl \
  --result_path logs/<round>.result.jsonl \
  --stream false \
  --max_new_tokens 256 \
  --max_batch_size 1
```

**结果**

批量推理稳定生成结果：

| 轮次 | 推理结果 |
|---|---|
| Round 1 | `logs/round1_qwen3_17b_text.text3.result.jsonl` |
| Round 2 | `logs/round2_qwen35_08b_text.text3.result.jsonl` |
| Round 3 文本 | `logs/round3_qwen35_2b_mm.text3.result.jsonl` |
| Round 3 图片 | `logs/round3_qwen35_2b_mm.mm2.result.jsonl` |

### 9. 多模态验证不能只看训练完成，还要看图片字段是否被框架识别

**现象**

第三轮使用 `data/mm_sft.jsonl`，里面包含：

- `messages`
- `images`
- `<image>` 占位符

训练完成并不自动证明图片输入链路在推理阶段也可用。

**处理方式**

额外创建 `data/infer_mm_prompts.jsonl`，用两张本地图片做批量推理验证。

推理日志显示：

```text
features: ['messages', 'images']
num_rows: 2
```

并成功生成 `logs/round3_qwen35_2b_mm.mm2.result.jsonl`。

**经验**

多模态训练至少要验证三层：

1. 数据集 JSONL 能被读取。
2. `images` 字段能进入框架特征。
3. adapter 加载后能对图片任务生成响应。

## 训练结果摘要

| 轮次 | 状态 | 最大显存增量 | 最终 checkpoint |
|---|---|---:|---|
| Round 1 | 成功 | `4406 MiB` | `outputs/qwen3_17b_text/v1-20260613-211241/checkpoint-20` |
| Round 2 | 成功 | `2734 MiB` | `outputs/qwen35_08b_text/v1-20260613-212200/checkpoint-20` |
| Round 3 | 成功 | `6420 MiB` | `outputs/qwen35_2b_mm/v0-20260613-213024/checkpoint-20` |

## 可复用 Workflow

### Step 1. 进入 A800 并确认资源

```powershell
ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 'hostname; nvidia-smi; df -h /root'"
```

确认项：

- 主机是 `test3`。
- 剩余显存建议大于 `20GB`。
- `/root` 剩余磁盘建议大于 `60GB`。
- 记录当前 GPU 进程，不停止已有服务。

### Step 2. 准备项目目录

```bash
mkdir -p /root/zhl/qwen-qlora-lab/{data,images,models,outputs,logs,scripts,docker}
```

### Step 3. 构建或复用 Docker 镜像

优先复用：

```bash
docker images | grep qwen-qlora-swift
```

如果要重建，保留以下原则：

- 使用国内 pip 源。
- 固定 `ms-swift`、`transformers`、`peft`、`accelerate` 版本。
- 对 Qwen3.5 先验证 `swift sft --help` 和一个最小训练命令。

### Step 4. 下载模型并做文件级验证

ModelScope 优先：

```bash
modelscope download --model Qwen/Qwen3-1.7B --local_dir /root/zhl/qwen-qlora-lab/models/Qwen3-1.7B
modelscope download --model Qwen/Qwen3.5-0.8B --local_dir /root/zhl/qwen-qlora-lab/models/Qwen3.5-0.8B
modelscope download --model Qwen/Qwen3.5-2B --local_dir /root/zhl/qwen-qlora-lab/models/Qwen3.5-2B
```

验证：

```bash
ls models/Qwen3-1.7B/config.json
ls models/Qwen3-1.7B/*token*
ls models/Qwen3-1.7B/*.safetensors
du -sh models/*
```

### Step 5. 低风险训练参数

共享 A800 上默认使用低显存参数：

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --model models/Qwen3-1.7B \
  --dataset data/text_sft.jsonl \
  --tuner_backend peft \
  --quant_bits 4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --max_length 1024 \
  --output_dir outputs/qwen3_17b_text
```

如果显存峰值超过 `16GB`，下一轮优先降：

1. `max_length=512`
2. 保持 `per_device_train_batch_size=1`
3. 减少样本数或 `save_steps`
4. 必要时换更小模型

### Step 6. 推理验证使用 batch 模式

不要用交互模式自动化验证。推荐：

```bash
swift infer \
  --adapters outputs/qwen3_17b_text/v1-20260613-211241/checkpoint-20 \
  --val_dataset data/infer_text_prompts.jsonl \
  --result_path logs/round1_qwen3_17b_text.text3.result.jsonl \
  --stream false \
  --max_new_tokens 256 \
  --max_batch_size 1
```

多模态验证：

```bash
swift infer \
  --adapters outputs/qwen35_2b_mm/v0-20260613-213024/checkpoint-20 \
  --val_dataset data/infer_mm_prompts.jsonl \
  --result_path logs/round3_qwen35_2b_mm.mm2.result.jsonl \
  --stream false \
  --max_new_tokens 256 \
  --max_batch_size 1
```

### Step 7. 最终状态检查

```bash
bash /root/zhl/qwen-qlora-lab/scripts/status.sh
```

必须确认：

- 没有残留的训练/验证容器。
- 显存回到训练前基线附近。
- checkpoint 存在。
- infer `.result.jsonl` 存在。
- 日志里有训练返回码和显存峰值。

## 后续是否要抽成 Skill

这次我先沉淀成项目文档，而不是直接写入全局 Codex skill，原因是：

1. 这里包含了公司服务器路径、A800 跳转方式和具体模型实验记录，更像项目内 workflow。
2. 直接修改 `C:\Users\zhl\.codex\skills` 会影响全局行为，最好等流程稳定后再抽象。
3. 当前文档已经能作为 skill 的蓝本。

如果后续要抽成 skill，建议命名：

- `a800-qwen-qlora`

建议触发场景：

- 用户要求在 A800 上训练 Qwen、Qwen3、Qwen3.5。
- 用户要求使用 ModelScope 下载模型。
- 用户要求用 `ms-swift` 做 QLoRA/SFT。
- 用户要求保护共享 GPU 上已有服务。

建议 skill 核心规则：

1. 先查 `nvidia-smi` 和 `df -h /root`。
2. 不停止已有业务服务。
3. 训练使用低显存默认参数。
4. 下载模型后必须检查 `config.json`、tokenizer 和 safetensors。
5. 推理验证使用 `--val_dataset --result_path`，不使用交互式 `swift infer`。
6. 遇到 PowerShell 双层 SSH 命令时，优先用绝对路径并避免未转义 `$()`。
