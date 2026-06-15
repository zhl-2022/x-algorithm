# `train_rounds_inside.sh` 逐行讲解

原始文件：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/train_rounds_inside.sh
```

这个脚本负责在容器内部依次跑三轮训练。虽然你现在重点看 Round 1，但脚本本身包含 Round 1、Round 2、Round 3 的统一训练函数。

## 整体结构

| 行号范围 | 作用 |
|---|---|
| 1-5 | Bash 安全设置和进入项目目录 |
| 7-14 | 定义日志、显存、checkpoint 辅助函数 |
| 16-81 | 定义 `run_round`：真正执行单轮训练 |
| 83-88 | 训练前显存门槛检查 |
| 90-100 | 依次启动三轮训练 |

## 逐行讲解

| 行号 | 原代码 | 小白解释 |
|---:|---|---|
| 1 | `#!/usr/bin/env bash` | 指定用 `bash` 执行这个脚本。服务器看到这行后知道不要用 `sh` 或其他 shell。 |
| 2 | `set -euo pipefail` | Bash 安全模式：命令失败就退出，未定义变量就报错，管道中任一命令失败也算失败。训练脚本里很重要，避免错误被吞掉。 |
| 3 | `PROJECT_DIR="/workspace/qwen-qlora-lab"` | 容器内项目目录。Docker 运行时会把远端 `/root/zhl/qwen-qlora-lab` 挂载到这个路径。 |
| 4 | `cd "$PROJECT_DIR"` | 进入项目目录。后面所有相对路径，如 `data/text_sft.jsonl`、`logs/`、`outputs/` 都基于这里。 |
| 5 | `mkdir -p logs outputs` | 确保日志目录和输出目录存在。`-p` 表示如果已经存在也不报错。 |
| 6 | 空行 | 分隔代码块，提高可读性。 |
| 7 | `log() { ... }` | 定义日志函数。每条日志前加时间戳，同时打印到屏幕并追加到 `logs/training_status.log`。 |
| 8 | `free_mib() { ... }` | 定义查询 GPU 剩余显存的函数。单位是 MiB，只取第一张 GPU。 |
| 9 | `used_mib() { ... }` | 定义查询 GPU 已用显存的函数。训练前后和训练过程中都用它估算本轮显存增量。 |
| 10 | `latest_ckpt() {` | 定义查找最新 checkpoint 的函数。 |
| 11 | `local dir="$1"` | 取函数第一个参数，表示要查找的输出目录。`local` 表示变量只在函数内有效。 |
| 12 | `if [ ! -d "$dir" ]; then return 0; fi` | 如果目录不存在，直接返回。这里返回 0 是为了“不把没 checkpoint 当成脚本失败”。 |
| 13 | `find "$dir" ...` | 找到所有 `checkpoint-*` 目录，按版本号排序，取最后一个，也就是最新 checkpoint。 |
| 14 | `}` | 结束 `latest_ckpt` 函数。 |
| 15 | 空行 | 分隔函数。 |
| 16 | `run_round() {` | 定义单轮训练函数。Round 1、Round 2、Round 3 都调用它。 |
| 17 | `local name="$1"` | 第一个参数：本轮名称，例如 `round1_qwen3_17b_text`。这个名字会用于日志文件名。 |
| 18 | `local model_path="$2"` | 第二个参数：模型路径，例如 `models/Qwen3-1.7B`。 |
| 19 | `local dataset="$3"` | 第三个参数：数据集路径，例如 `data/text_sft.jsonl`。 |
| 20 | `local out_dir="$4"` | 第四个参数：输出目录，例如 `outputs/qwen3_17b_text`。 |
| 21 | `local default_max_len="$5"` | 第五个参数：默认最大长度，例如 `1024`。 |
| 22 | `local existing` | 声明一个变量，用来保存已有 checkpoint 路径。 |
| 23 | `existing=$(latest_ckpt "$out_dir")` | 在输出目录里查找是否已经有 checkpoint。 |
| 24 | `if [ -n "$existing" ] && ...` | 如果已经有 checkpoint，并且没有设置 `FORCE_RERUN=1`，就跳过本轮。 |
| 25 | `log "SKIP ..."` | 记录跳过原因：已有 checkpoint。 |
| 26 | `return 0` | 直接结束本轮函数，表示成功跳过。 |
| 27 | `fi` | 结束 checkpoint 跳过判断。 |
| 28 | `local max_len="$default_max_len"` | 本轮默认使用传入的最大长度。 |
| 29 | `if [ -f logs/force_max_length_512 ]; then max_len=512; fi` | 如果之前某轮显存太高，脚本会创建这个标记文件；后续轮次强制把 `max_length` 降到 512。 |
| 30 | `mkdir -p "$out_dir"` | 确保本轮输出目录存在。 |
| 31 | `local log_file="logs/${name}.train.log"` | 本轮完整训练 stdout/stderr 日志文件。Round 1 对应 `logs/round1_qwen3_17b_text.train.log`。 |
| 32 | `local sample_file="logs/${name}.gpu_sample.csv"` | 本轮显存采样文件，每 2 秒记录一次已用显存。 |
| 33 | `local base_used` | 声明训练开始前已用显存变量。 |
| 34 | `base_used=$(used_mib)` | 记录训练开始前 GPU 已用显存。 |
| 35 | `log "START ..."` | 记录本轮开始，包含模型、数据、训练前显存、最大长度。 |
| 36 | `nvidia-smi > ...before...` | 把训练前完整 `nvidia-smi` 保存下来，方便复盘当时 GPU 上有哪些进程。 |
| 37 | `( while true; do ... ) > "$sample_file" &` | 后台启动一个无限循环，每 2 秒记录一次显存，用 `&` 放到后台。 |
| 38 | `local sampler_pid=$!` | `$!` 表示刚刚后台进程的 PID。保存它是为了训练结束后杀掉采样进程。 |
| 39 | `set +e` | 暂时关闭“命令失败就退出”。原因是我们要捕获 `swift sft` 的 exit code，而不是让脚本立刻中断。 |
| 40 | `CUDA_VISIBLE_DEVICES=0 swift sft \` | 只让训练看到第 0 张 GPU，并启动 `ms-swift` 的 SFT 训练命令。 |
| 41 | `--tuner_backend peft` | 使用 PEFT 作为 adapter/LoRA 后端。 |
| 42 | `--model "$model_path"` | 传入模型路径。Round 1 是 `models/Qwen3-1.7B`。 |
| 43 | `--quant_bits 4` | 用 4bit 量化加载基座模型，这是 QLoRA 的关键。 |
| 44 | `--dataset "$dataset"` | 传入训练数据。Round 1 是 `data/text_sft.jsonl`。 |
| 45 | `--torch_dtype bfloat16` | 训练计算精度使用 `bfloat16`。A800 支持 bf16，通常比 fp32 省显存。 |
| 46 | `--num_train_epochs 1` | 最多训练 1 个 epoch。这里还设置了 `max_steps=20`，实际会先被步数限制截断。 |
| 47 | `--max_steps 20` | 最多训练 20 步。学习实验用短跑，不追求生产效果。 |
| 48 | `--per_device_train_batch_size 1` | 每张 GPU 每次只放 1 条样本，保守省显存。 |
| 49 | `--per_device_eval_batch_size 1` | 评估时每张 GPU batch size 也是 1。虽然本轮没有验证集，但参数保留。 |
| 50 | `--learning_rate 1e-4` | 学习率是 `0.0001`，控制 adapter 参数更新幅度。 |
| 51 | `--lora_rank 8` | LoRA rank 为 8，表示低秩 adapter 的容量。 |
| 52 | `--lora_alpha 32` | LoRA alpha 为 32，控制 LoRA 更新缩放强度。 |
| 53 | `--target_modules all-linear` | 对模型中的线性层插入 LoRA。后续会在 `adapter_config.json` 展开成具体模块。 |
| 54 | `--gradient_accumulation_steps 8` | 梯度累积 8 次再更新一次，相当于有效 batch size 变成 8。 |
| 55 | `--logging_steps 1` | 每一步都记录训练指标。 |
| 56 | `--save_steps 10` | 每 10 步保存一次 checkpoint。20 步训练会保存 `checkpoint-10` 和 `checkpoint-20`。 |
| 57 | `--save_total_limit 2` | 最多保留 2 个 checkpoint，防止输出目录无限变大。 |
| 58 | `--max_length "$max_len"` | 输入和输出拼接后的最大 token 长度。Round 1 默认 1024。 |
| 59 | `--warmup_ratio 0.05` | 前 5% 训练步做学习率 warmup，避免一开始更新过猛。 |
| 60 | `--dataloader_num_workers 1` | DataLoader 使用 1 个 worker 读取数据。小数据够用，也更稳。 |
| 61 | `--dataset_num_proc 1` | 数据预处理使用 1 个进程。避免共享服务器上过度并发。 |
| 62 | `--output_dir "$out_dir"` | 输出目录。Round 1 是 `outputs/qwen3_17b_text`，框架会在里面创建版本目录。 |
| 63 | `--save_only_model true > "$log_file" 2>&1` | 只保存模型相关内容，并把标准输出和错误输出都写入训练日志。 |
| 64 | `local rc=$?` | `$?` 是上一条命令的 exit code。`0` 表示成功，非 0 表示失败。 |
| 65 | `set -e` | 恢复“命令失败就退出”的安全模式。 |
| 66 | `kill "$sampler_pid" ...` | 停止后台显存采样进程。失败也不报错，因为采样进程可能已经结束。 |
| 67 | `wait "$sampler_pid" ...` | 等待采样进程真正退出，避免残留后台进程。 |
| 68 | `nvidia-smi > ...after...` | 保存训练后的完整 `nvidia-smi`。 |
| 69 | `local max_used delta` | 声明两个变量：训练期间最大已用显存和显存增量。 |
| 70 | `max_used=$(awk ...)` | 从显存采样 CSV 中找最大已用显存。 |
| 71 | `delta=$((max_used - base_used))` | 本轮新增显存约等于最高已用显存减去训练前已用显存。 |
| 72 | `log "END ..."` | 记录本轮结束：exit code、最大显存、显存增量、最新 checkpoint。 |
| 73 | `if [ "$delta" -gt 16000 ]; then` | 如果本轮新增显存超过 16GB，就触发降级。 |
| 74 | `log "Memory delta exceeded ..."` | 记录显存过高原因。 |
| 75 | `touch logs/force_max_length_512` | 创建标记文件，后续轮次看到它就把 `max_length` 降到 512。 |
| 76 | `fi` | 结束显存判断。 |
| 77 | `if [ "$rc" -ne 0 ]; then` | 如果训练命令失败。 |
| 78 | `log "FAILED ..."` | 记录失败，并提示看完整训练日志。 |
| 79 | `return "$rc"` | 把训练失败的 exit code 返回给调用者。 |
| 80 | `fi` | 结束失败判断。 |
| 81 | `}` | 结束 `run_round` 函数。 |
| 82 | 空行 | 分隔函数和主流程。 |
| 83 | `free_now=$(free_mib)` | 读取训练开始前 GPU 剩余显存。 |
| 84 | `log "Initial free memory MiB: $free_now"` | 记录初始剩余显存。 |
| 85 | `if [ "$free_now" -lt 18000 ]; then` | 如果剩余显存低于 18GB，进入保护逻辑。 |
| 86 | `log "ABORT ..."` | 记录为什么要中止：剩余显存太低。 |
| 87 | `if [ "${ALLOW_LOW_FREE:-0}" != "1" ]; then exit 1; fi` | 默认直接退出；只有显式设置 `ALLOW_LOW_FREE=1` 才允许低显存强行跑。 |
| 88 | `fi` | 结束显存门槛判断。 |
| 89 | 空行 | 分隔主流程。 |
| 90 | `run_round 'round1_qwen3_17b_text' ...` | 启动 Round 1：Qwen3-1.7B 文本 QLoRA。 |
| 91 | `run_round 'round2_qwen35_08b_text' ...` | 启动 Round 2：Qwen3.5-0.8B 文本 QLoRA。 |
| 92 | `MM_MODEL='models/Qwen3.5-2B'` | 多模态轮次默认使用 Qwen3.5-2B。 |
| 93 | `MM_OUTPUT='outputs/qwen35_2b_mm'` | 多模态轮次默认输出目录。 |
| 94 | `if [ ! -f "$MM_MODEL/config.json" ] || ...` | 如果 Qwen3.5-2B 没下载成功或被标记下载失败，就走 fallback。 |
| 95 | `log 'Qwen3.5-2B unavailable...'` | 记录 fallback 原因。 |
| 96 | `MM_MODEL='models/Qwen3.5-0.8B'` | fallback 使用 Qwen3.5-0.8B。 |
| 97 | `MM_OUTPUT='outputs/qwen35_08b_mm_fallback'` | fallback 输出目录也改名，避免和 2B 结果混淆。 |
| 98 | `fi` | 结束 fallback 判断。 |
| 99 | `run_round 'round3_qwen35_mm' ...` | 启动 Round 3：多模态 QLoRA。 |
| 100 | `log 'All training rounds finished.'` | 所有轮次结束后写入最终日志。 |

## Round 1 实际调用

第 90 行是你现在重点关注的任务：

```bash
run_round 'round1_qwen3_17b_text' 'models/Qwen3-1.7B' 'data/text_sft.jsonl' 'outputs/qwen3_17b_text' 1024
```

展开后含义是：

| 参数位置 | 值 | 含义 |
|---:|---|---|
| `$1` | `round1_qwen3_17b_text` | 本轮训练名字 |
| `$2` | `models/Qwen3-1.7B` | 基座模型路径 |
| `$3` | `data/text_sft.jsonl` | 训练数据 |
| `$4` | `outputs/qwen3_17b_text` | 输出目录 |
| `$5` | `1024` | 默认最大 token 长度 |

## 这份脚本体现的工程思路

这不是只会写一条 `swift sft` 命令，而是做了完整工程保护：

1. 训练前检查显存。
2. 训练前后保存 `nvidia-smi`。
3. 训练中每 2 秒采样显存。
4. 已有 checkpoint 时自动跳过，避免重复训练。
5. 训练失败时保留日志并返回错误码。
6. 显存过高时自动降低后续轮次 `max_length`。

面试时你可以说：

```text
我没有裸跑训练命令，而是把训练封装成可复用函数，加入显存门槛、日志记录、checkpoint 跳过、失败返回和显存降级逻辑，适合共享 A800 环境。
```

