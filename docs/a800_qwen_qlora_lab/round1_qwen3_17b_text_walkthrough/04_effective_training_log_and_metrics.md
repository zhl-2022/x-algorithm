# Effective 训练日志、曲线和指标问答

本文记录 `qwen3_17b_text_effective` 这轮训练相关的学习问答。对应文件主要是：

| 文件 | 作用 |
|---|---|
| `remote_snapshot/qwen-qlora-lab-small/logs/qwen3_17b_effective.train.log` | `swift sft` 的完整控制台日志 |
| `remote_snapshot/qwen-qlora-lab-small/outputs/qwen3_17b_text_effective/v0-20260615-102152/logging.jsonl` | 每一步训练指标和最终汇总 |
| `remote_snapshot/qwen-qlora-lab-small/outputs/qwen3_17b_text_effective/v0-20260615-102152/images/` | ms-swift 生成的训练曲线图片 |
| `remote_snapshot/qwen-qlora-lab-small/data/text_sft_effective_unique.jsonl` | 本轮实际训练数据 |

## 1. Linux 内核偏旧是 Docker 容器里的内核偏旧吗

不是。Docker 容器本身没有独立的 Linux kernel，容器共享宿主机 kernel。

日志里的 warning 是：

```text
Detected kernel version 4.19.90, which is below the recommended minimum of 5.5.0; this can cause the process to hang.
```

这句话里的 `4.19.90` 通常来自 A800 宿主机，也就是 `test3` 这台机器的内核版本。容器镜像可以提供不同的用户态环境，例如 Ubuntu、Python、CUDA、PyTorch、ms-swift，但不能单独改变 kernel。

你可以这样理解：

| 层级 | 能不能被 Docker 镜像替换 | 例子 |
|---|---:|---|
| 宿主机 kernel | 不能 | `uname -r` 看到的 Linux kernel |
| 容器用户态系统 | 可以 | Ubuntu 文件系统、bash、glibc |
| Python 环境 | 可以 | `torch`、`transformers`、`ms-swift` |
| CUDA 用户态库 | 可以 | 容器内 CUDA runtime、cuDNN |
| NVIDIA 驱动内核模块 | 主要在宿主机 | `nvidia-smi` 依赖宿主机驱动 |

所以换一个 Docker 镜像通常不能解决这个 warning。真正的解决方式是升级宿主机 Linux kernel，或者换到内核更高的机器上运行。

但这次训练已经完成，没有因为这个 warning 卡住。它对我们当前小规模学习实验不是阻塞问题，只是后续跑长时间训练时需要注意。

建议后续长任务这样降低风险：

1. 保持 `save_steps`，不要几个小时不保存 checkpoint。
2. 训练前记录 `uname -r`、`nvidia-smi`、`df -h /root`。
3. 长任务优先使用后台日志和可恢复 checkpoint。
4. 如果出现无报错但进程长时间不动，再把 kernel warning 纳入排查。

## 2. 训练曲线图片怎么看

图片目录是：

```text
outputs/qwen3_17b_text_effective/v0-20260615-102152/images/
```

里面每张图都来自 `logging.jsonl` 里的字段。

| 图片 | 看什么 | 本轮现象 | 怎么解读 |
|---|---|---|---|
| `train_loss.png` | 每 step 的训练 loss | 从 `7.95` 很快降到接近 `0.001` | 训练集被快速拟合，说明这批 96 条样本基本被模型记住 |
| `train_token_acc.png` | token 级准确率 | 从约 `0.36` 升到 `1.0` | 训练答案里的 token 基本都能预测出来 |
| `train_grad_norm.png` | 梯度范数 | 前期较大，后期接近 0 | 训练从大幅更新进入收敛状态，没有看到梯度爆炸 |
| `train_learning_rate.png` | 学习率变化 | warmup 到 `2e-4`，最后衰减到 `0` | 学习率调度正常 |
| `train_epoch.png` | 数据遍历轮数 | 最后到 `5.0` | 实际完整看过训练集 5 遍 |
| `train_total_flos.png` | 累计浮点计算量 | 单调增长到 `4.587e14` | 训练计算量随 step 累加，主要用于估算成本 |
| `train_train_loss.png` | 最终汇总 loss | 最终汇总约 `0.6592` | 这是全程平均，不等于最后一步 loss |
| `train_train_runtime.png` | 训练耗时 | 约 `217.7s` | 纯训练耗时约 3 分 38 秒 |
| `train_train_samples_per_second.png` | 样本吞吐 | 约 `2.204 samples/s` | 每秒处理约 2.2 条样本 |
| `train_train_steps_per_second.png` | step 吞吐 | 约 `0.551 steps/s` | 每个优化 step 约 1.81 秒 |

正式大规模微调时，更希望看到：

1. `train_loss` 平滑下降，但不要过早接近 0。
2. 有 `eval_loss` 或业务评测指标，并且验证集不明显恶化。
3. `grad_norm` 没有长期爆炸、没有 NaN。
4. `learning_rate` 按预期 warmup 和 decay。
5. 吞吐在长时间训练中基本稳定。
6. 能根据验证集选择 `best_model_checkpoint`，而不是只有最后一个 checkpoint。

本轮没有验证集，所以曲线只能说明“训练集拟合成功”，不能单独证明泛化能力。

## 3. `total_flos=4.587e14` 是什么水平

本轮最终汇总里有：

```text
total_flos = 458744172042240
```

也就是大约：

```text
4.587e14 FLOPs
= 458.7 万亿次浮点运算
= 0.459 PFLOPs
```

这对个人学习训练来说已经是一个真实 GPU 训练任务；但对大模型预训练来说非常小。

一个粗略对比：

| 场景 | 量级理解 |
|---|---|
| 本轮 QLoRA 小样本训练 | `4.587e14 FLOPs`，约 `0.459 PFLOPs` |
| 训练 1.7B 模型看 10 亿 token 的粗略预训练 | 约 `1e19 FLOPs` 量级 |
| 真正大模型预训练 | 往往是 `1e21` 到 `1e24 FLOPs` 量级 |

所以它的定位是：

```text
适合学习微调流程和观察 adapter 效果；
不属于大规模预训练；
也不代表正式业务微调的总计算量。
```

还要注意：`total_flos` 是框架估算的累计浮点计算量，不等于 GPU 实际满血跑出的理论峰值。因为本轮数据很短、batch 很小、QLoRA 有量化和 adapter 逻辑，实际 wall time 主要还受到框架开销、数据处理、kernel 调度和保存 checkpoint 影响。

## 4. `qwen3_17b_effective.train.log` 怎么完整理解

这份日志是 `swift sft` 的 stdout/stderr。它比 `logging.jsonl` 更完整，因为它不仅有每步指标，还有启动参数、环境信息、模型结构、LoRA 配置、warning 和 checkpoint 保存信息。

### 4.1 启动命令

日志第一行是完整命令，核心参数如下：

| 参数 | 当前值 | 含义 |
|---|---|---|
| `--model` | `models/Qwen3-1.7B` | 基座模型 |
| `--dataset` | `data/text_sft_effective_unique.jsonl` | 96 条去重训练数据 |
| `--quant_bits` | `4` | 4bit QLoRA 加载基座模型 |
| `--tuner_backend` | `peft` | 用 PEFT 后端实现 LoRA |
| `--lora_rank` | `16` | LoRA adapter 容量 |
| `--lora_alpha` | `32` | LoRA 缩放系数 |
| `--lora_dropout` | `0` | 不做 LoRA dropout |
| `--gradient_accumulation_steps` | `4` | 每 4 条样本累计一次梯度 |
| `--max_steps` | `120` | 最多训练 120 个优化 step |
| `--num_train_epochs` | `8` | 最多 8 轮，但被 `max_steps` 提前截断 |
| `--save_steps` | `40` | 第 40、80、120 步保存 checkpoint |
| `--max_length` | `768` | 单条样本最大 token 长度 |

### 4.2 数据集

日志显示：

```text
train_dataset: num_rows: 96
val_dataset: None
Dataset Token Length: 111.635417±7.871520, min=98, max=128
```

意思是：

1. 本轮训练集只有 96 条。
2. 没有验证集。
3. 样本都很短，平均约 112 token，最长约 128 token。
4. `max_length=768` 不会成为瓶颈。

因此本轮很容易被模型记住，loss 快速下降是符合预期的。

### 4.3 LoRA/QLoRA 配置

日志显示：

```text
PeftModelForCausalLM: 1738.0076M Params (17.4326M Trainable [1.0030%])
```

含义是：

| 项目 | 数值 | 解释 |
|---|---:|---|
| 总参数 | 约 `17.38` 亿 | Qwen3-1.7B 基座主体 |
| 可训练参数 | 约 `1743` 万 | LoRA adapter |
| 可训练比例 | `1.0030%` | 只训练约 1% 参数 |

`target_modules` 包括：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

也就是 attention 和 MLP 里的主要线性层。

### 4.4 每步指标

关键变化如下：

| step | loss | token_acc | 解读 |
|---:|---:|---:|---|
| 1 | `7.9485` | `0.3577` | 刚开始不会答，正常 |
| 24 | `0.6703` | `0.8467` | 约 1 个 epoch 后已经明显拟合 |
| 48 | `0.0211` | `0.9924` | 约 2 个 epoch 后基本记住训练集 |
| 96 | `0.0012` | `1.0` | 第 4 个 epoch 时几乎完全拟合 |
| 120 | `0.0011` | `1.0` | 最终训练完成 |

这里的 `token_acc=1.0` 不代表模型真实能力满分，只代表训练样本答案里的 token 基本都能被预测出来。对于 96 条固定样本，这是过拟合或强记忆的信号。

### 4.5 checkpoint

日志末尾显示：

```text
last_model_checkpoint: .../checkpoint-120
best_model_checkpoint: None
```

解释：

| 字段 | 含义 |
|---|---|
| `last_model_checkpoint` | 最后保存的 checkpoint，是本轮部署使用的 adapter |
| `best_model_checkpoint` | 没有验证集，所以无法选出最佳 checkpoint |

本轮可用于推理对比的是：

```text
outputs/qwen3_17b_text_effective/v0-20260615-102152/checkpoint-120
```

### 4.6 最终汇总

日志最终汇总是：

| 指标 | 数值 | 含义 |
|---|---:|---|
| `train_runtime` | `217.7458s` | 纯训练约 3 分 38 秒 |
| `train_samples_per_second` | `2.204` | 每秒约 2.2 条样本 |
| `train_steps_per_second` | `0.551` | 每秒约 0.55 个优化 step |
| `train_loss` | `0.6592` | 全程平均 loss，不是最后一步 loss |
| `epoch` | `5.0` | 实际完整看过训练集 5 遍 |

## 5. 面试和项目复盘怎么讲

可以这样说：

```text
我用 ms-swift + PEFT 在 A800 上对 Qwen3-1.7B 做了一轮 4bit QLoRA 学习实验。训练数据是 96 条去重 SFT 样本，LoRA rank 设置为 16，只训练约 1743 万参数，占总参数约 1%。训练跑满 120 step，实际约 5 个 epoch，loss 从 7.95 降到 0.001 左右，token_acc 达到 1.0，说明训练集被充分拟合。由于没有验证集和更大规模评测，这个结果主要证明 QLoRA 流程、adapter 保存和部署对比链路跑通，不能直接当成生产泛化能力证明。
```

这段话的重点是：既说清楚自己真的做过训练，也不会把小样本拟合夸大成生产效果。
