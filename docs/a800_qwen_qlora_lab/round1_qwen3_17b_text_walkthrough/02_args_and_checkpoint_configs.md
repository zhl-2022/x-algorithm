# Round 1 参数与 checkpoint 配置讲解

本文件讲解你点名的这些配置：

```text
outputs/qwen3_17b_text/v1-20260613-211241/args.json
outputs/qwen3_17b_text/v1-20260613-211241/checkpoint-10/adapter_config.json
outputs/qwen3_17b_text/v1-20260613-211241/checkpoint-10/additional_config.json
outputs/qwen3_17b_text/v1-20260613-211241/checkpoint-10/args.json
outputs/qwen3_17b_text/v1-20260613-211241/checkpoint-10/trainer_state.json
```

## 先看它们分别是什么

| 文件 | 谁生成 | 作用 |
|---|---|---|
| `args.json` | `ms-swift` | 记录本轮训练最终参数快照 |
| `checkpoint-10/args.json` | `ms-swift` | checkpoint 内复制的一份参数快照 |
| `adapter_config.json` | `peft` | 记录 LoRA adapter 怎么挂到基座模型上 |
| `additional_config.json` | `ms-swift`/LoRA 扩展 | 记录少量额外 LoRA 学习率参数 |
| `trainer_state.json` | Hugging Face Trainer | 记录保存 checkpoint 时训练器状态 |

我已经校验过：

```text
根目录 args.json 与 checkpoint-10/args.json 内容完全一致。
```

所以你只需要重点读一份 `args.json`，知道 checkpoint 里也保留了一份用于独立复现。

## `args.json` 怎么读

`args.json` 有 367 行。不要逐字背。它是框架把“显式传入参数 + 自动推断参数 + 默认值”全部保存下来的结果。

小白读它时分三层：

1. 必须掌握：模型、数据、训练、LoRA、QLoRA、保存输出。
2. 需要知道：精度、日志、优化器、学习率调度、验证集。
3. 暂时忽略：未启用的高级功能，例如 FSDP、Ray、GaLore、ReFT、SwanLab 等。

### 1. 输出和保存相关

| 字段 | 当前值 | 解释 |
|---|---|---|
| `output_dir` | `/workspace/qwen-qlora-lab/outputs/qwen3_17b_text/v1-20260613-211241` | 本次训练最终输出目录。框架在 `outputs/qwen3_17b_text` 下自动创建了版本目录。 |
| `overwrite_output_dir` | `false` | 不覆盖已有输出目录，降低误删风险。 |
| `save_strategy` | `steps` | 按训练步数保存 checkpoint。 |
| `save_steps` | `10.0` | 每 10 步保存一次。对应 `checkpoint-10` 和 `checkpoint-20`。 |
| `save_total_limit` | `2` | 最多保留 2 个 checkpoint。 |
| `save_safetensors` | `true` | 使用 safetensors 格式保存权重，通常更安全。 |
| `save_only_model` | `true` | 只保存模型相关内容，不保存完整训练状态，节省空间。 |
| `logging_dir` | `.../runs` | TensorBoard 日志目录。 |
| `run_name` | 同 `output_dir` | 本次训练 run 名称。 |

你要理解：

```text
训练不是只产生一个文件，而是一个输出目录，里面有参数、日志、checkpoint 和图表。
```

### 2. batch 和训练步数相关

| 字段 | 当前值 | 解释 |
|---|---|---|
| `per_device_train_batch_size` | `1` | 每张 GPU 每次训练 1 条样本。省显存。 |
| `per_device_eval_batch_size` | `1` | 评估 batch size。当前无验证集，影响不大。 |
| `gradient_accumulation_steps` | `8` | 累积 8 次梯度再更新一次参数。 |
| `num_train_epochs` | `1.0` | 最多训练 1 个 epoch。 |
| `max_steps` | `20` | 最多训练 20 步，优先级高于 epoch。 |
| `train_batch_size` | 在 `trainer_state.json` 中为 `1` | Trainer 记录的 per-device batch；有效 batch 要结合梯度累积。 |

有效 batch size：

$$
1 \times 8 \times 1 = 8
$$

意思是：每次只放 1 条样本进显存，但累积 8 次后再更新参数。

### 3. 学习率和优化器相关

| 字段 | 当前值 | 解释 |
|---|---|---|
| `learning_rate` | `0.0001` | 学习率，也就是 adapter 参数每次更新的幅度。 |
| `weight_decay` | `0.1` | 权重衰减，帮助正则化。 |
| `adam_beta1` | `0.9` | AdamW 优化器动量参数。 |
| `adam_beta2` | `0.95` | AdamW 二阶动量参数。 |
| `adam_epsilon` | `1e-08` | 防止除零的小数。 |
| `max_grad_norm` | `1.0` | 梯度裁剪阈值，防止梯度爆炸。 |
| `optim` | `adamw_torch` | 使用 PyTorch 自带 AdamW。 |
| `lr_scheduler_type` | `cosine` | 学习率使用 cosine 调度，逐渐下降。 |
| `warmup_ratio` | `0.05` | 前 5% 步数 warmup。 |
| `warmup_steps` | `0` | 没手动指定 warmup 步数，由比例决定。 |

你在 `logging.jsonl` 里能看到 `learning_rate` 从 `0.0001` 逐渐降到 `0.0`，这就是 scheduler 在工作。

### 4. 日志相关

| 字段 | 当前值 | 解释 |
|---|---|---|
| `logging_strategy` | `steps` | 按步数记录日志。 |
| `logging_first_step` | `true` | 第一步就记录日志。 |
| `logging_steps` | `1` | 每一步都记录日志。 |
| `logging_nan_inf_filter` | `true` | 过滤 NaN/Inf 日志值，避免日志异常。 |
| `report_to` | `["tensorboard"]` | 同时写 TensorBoard 日志。 |

对应产物：

```text
logging.jsonl
```

每行就是一步训练日志。

### 5. 精度和设备相关

| 字段 | 当前值 | 解释 |
|---|---|---|
| `no_cuda` | `false` | 不禁用 CUDA，所以会使用 GPU。 |
| `use_cpu` | `false` | 不使用 CPU 训练。 |
| `bf16` | `true` | 使用 bfloat16。A800 支持 bf16。 |
| `fp16` | `false` | 不使用 fp16。 |
| `torch_dtype` | `bfloat16` | 模型计算精度是 bf16。 |
| `local_rank` | `-1` 或训练时内部为 `0` | 单进程/单卡训练，不是多卡 DDP。 |
| `global_world_size` | `1` | 总训练进程数为 1。 |
| `local_world_size` | `1` | 本机训练进程数为 1。 |

你要理解：

```text
这次是单卡 A800 上的 bf16 + 4bit QLoRA 训练。
```

### 6. 模型和模板相关

| 字段 | 当前值 | 解释 |
|---|---|---|
| `model` | `models/Qwen3-1.7B` | 命令行传入的模型路径。 |
| `model_dir` | `/workspace/qwen-qlora-lab/models/Qwen3-1.7B` | 框架解析后的绝对模型目录。 |
| `model_type` | `qwen3` | 框架识别出的模型类型。 |
| `task_type` | `causal_lm` | 因果语言模型，即根据前文预测后文。 |
| `template` | `qwen3` | 使用 Qwen3 对话模板。 |
| `template_backend` | `swift` | 模板由 ms-swift 处理。 |
| `use_chat_template` | `true` | 使用聊天模板把 `messages` 转成模型输入。 |
| `system` | `null` | 命令行没有单独传 system，system 来自数据里的 `messages`。 |
| `model_suffix` | `Qwen3-1.7B` | 框架提取的模型后缀。 |

模板很重要。训练数据是：

```json
{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
```

模型不能直接吃 JSON。模板负责把它变成 Qwen3 能理解的 prompt 格式。

### 7. 数据相关

| 字段 | 当前值 | 解释 |
|---|---|---|
| `dataset` | `["data/text_sft.jsonl"]` | 本轮训练数据。 |
| `val_dataset` | `[]` | 没有单独验证集。 |
| `split_dataset_ratio` | `0.0` | 不从训练集拆验证集。 |
| `_val_dataset_exists` | `false` | 框架确认没有验证集。 |
| `dataset_num_proc` | `1` | 数据预处理使用 1 个进程。 |
| `dataset_shuffle` | `true` | 训练前打乱数据。 |
| `val_dataset_shuffle` | `false` | 验证集不打乱。当前无验证集。 |
| `load_from_cache_file` | `false` | 不强依赖缓存。 |
| `strict` | `false` | 数据解析不是最严格模式。 |

这解释了为什么 `logging.jsonl` 没有 eval loss：因为本轮没有验证集。

### 8. 序列长度和截断相关

| 字段 | 当前值 | 解释 |
|---|---|---|
| `max_length` | `1024` | 单条样本 token 总长度上限。 |
| `truncation_strategy` | `delete` | 超长时删除截断。 |
| `padding_side` | `right` | padding 加在右边。 |
| `padding_free` | `false` | 没启用 padding-free 优化。 |
| `packing` | `false` | 没把多条短样本打包成一条长序列。 |

小白先记住：

```text
max_length 越大，能容纳的上下文越长，但显存越高。
```

### `padding_side=right` 到底是什么意思

`padding_side` 只决定“如果需要补齐 token，补在左边还是右边”。它不是 `max_length`，也不表示每条样本一定补到 1024。

用一个简化例子看：

```text
原始样本 token:
[101, 102, 103, 104]

如果当前 batch 需要补到长度 6，并且 padding_side=right:
[101, 102, 103, 104, PAD, PAD]

如果 padding_side=left:
[PAD, PAD, 101, 102, 103, 104]
```

你问“如果一条样本没达到 1024，是不是在右边补齐 0？”更准确的答案是：

| 问题 | 答案 |
|---|---|
| 一定补到 1024 吗？ | 不一定。`max_length=1024` 是长度上限，常见训练 collator 会动态补到当前 batch 中最长样本，而不是固定补到 1024。 |
| 当前 batch size 是 1，还会补很多吗？ | 通常不会补很多，因为一个 batch 里只有这一条样本。除非框架为了对齐硬件计算，把长度补到某个倍数。 |
| 右边补的是数字 `0` 吗？ | 不一定。补的是 tokenizer 定义的 `pad_token_id`。有的模型 `pad_token_id` 是 `0`，有的会用 `eos_token_id` 或其它特殊 token。 |
| 模型会把补齐部分当正文学习吗？ | 正常不会。输入里会有 `attention_mask`，真实 token 是 `1`，padding 位置是 `0`；训练标签里 padding 位置通常会被设成 `-100`，loss 会忽略它。 |
| 为什么训练常用 right padding？ | SFT 训练时右补齐更自然：真实文本从左到右连续排列，padding 放在末尾。批量生成推理时 decoder-only 模型常更偏向 left padding，这是另一个场景。 |

所以本轮你应该这样理解：

```text
max_length=1024:
  单条样本最多允许 1024 个 token，超过就截断。

padding_side=right:
  如果 batch 内需要补齐长度，把 padding token 放在真实文本右边。

padding_free=false:
  使用普通 padding 方式，没有启用更高级的无 padding 训练优化。
```

### 针对这份 `text_sft.jsonl`，什么时候会 padding

这份训练数据本身不会提前 padding。`text_sft.jsonl` 里保存的是原始 `messages`：

```text
system / user / assistant
```

真正 padding 发生在训练时组 batch 的阶段：

```text
text_sft.jsonl 原始 messages
  -> 套用 qwen3 chat template
  -> tokenizer 转成 token ids
  -> DataLoader/collator 组 batch
  -> 如果同一个 batch 里样本长度不一样，短样本才 padding
```

本轮真实配置是：

```json
{
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "max_length": 1024,
  "padding_side": "right",
  "packing": false,
  "group_by_length": false
}
```

最关键的是 `per_device_train_batch_size=1`。这表示每次 forward 实际只拿 1 条样本，通常不需要为了对齐其它样本而 padding 到 1024。

举例：

```text
样本 A tokenize 后长度 90
batch_size=1

这个 batch 只有 A:
[90 个真实 token]
```

通常不会变成：

```text
[90 个真实 token + 934 个 PAD]
```

如果以后把 batch size 改成 4，padding 才会更明显：

```text
同一个 batch 里四条样本长度:
80, 120, 300, 150

batch 最长是 300
padding_side=right 后大概变成:
80  -> 300
120 -> 300
300 -> 300
150 -> 300
```

如果某条样本超过 `max_length=1024`：

```text
1300 -> 先截断到 1024
其它短样本再按 batch 最长长度补齐
```

还要注意：`gradient_accumulation_steps=8` 不等于一次把 8 条样本拼成一个 batch。它是连续跑 8 次 `batch_size=1` 的 forward/backward，梯度先累积，最后再更新一次参数。所以这 8 条样本之间不会互相 padding。

结论：

```text
这份 180 条短问答语料在当前配置下，真正 padding 很少。
max_length=1024 主要是防止超长样本，而不是要求每条都补到 1024。
```

### 9. QLoRA 量化相关

| 字段 | 当前值 | 解释 |
|---|---|---|
| `quant_bits` | `4` | 使用 4bit 量化加载基座模型。 |
| `quant_method` | `null` | 没显式指定其它量化方法。 |
| `bnb_4bit_compute_dtype` | `bfloat16` | bitsandbytes 4bit 计算使用 bf16。 |
| `bnb_4bit_quant_type` | `nf4` | 4bit 量化类型是 NF4，QLoRA 常见配置。 |
| `bnb_4bit_use_double_quant` | `true` | 启用 double quant，进一步省显存。 |

这几个字段说明：这不是普通 LoRA，而是 QLoRA。

### 4bit 量化是什么意思

模型参数本质上是一大堆数字。正常加载模型时，这些数字通常用 `float32`、`float16` 或 `bfloat16` 存储。

4bit 量化的意思是：把基座模型的大部分权重用 4bit 表示。4bit 只有 16 个离散档位，所以它不是“无损压缩”，而是用更少的位宽近似原来的权重。

以本轮 `Qwen3-1.7B` 为例，日志里记录：

```text
model_parameter: 1729.2913M
trainable_parameter: 8.7163M
trainable_percentage: 0.5040%
```

只看权重存储的粗略量级：

| 加载/存储方式 | 每个参数约占 | 1.729B 参数粗略权重大小 | 说明 |
|---|---:|---:|---|
| `float32` | 32 bit = 4 byte | 约 6.9 GB | 最占显存，训练稳定但成本高。 |
| `float16` / `bfloat16` | 16 bit = 2 byte | 约 3.5 GB | 大模型训练/推理常用精度。 |
| 8bit | 8 bit = 1 byte | 约 1.7 GB | 常用于省显存推理，也可配合 PEFT。 |
| 4bit | 4 bit = 0.5 byte | 约 0.86 GB | QLoRA 常用，实际还会有 scale、zero point、metadata 等额外开销。 |

注意：这只是“基座权重”的粗略大小。真实训练显存还包括：

- 激活值，也就是 forward 中间结果。
- LoRA adapter 参数和梯度。
- optimizer state。
- CUDA kernel、缓存和框架运行开销。
- batch size、`max_length`、gradient checkpointing 等引入的额外影响。

QLoRA 的关键思路是：

```text
基座模型:
  4bit 量化加载，并冻结不训练。

LoRA adapter:
  用 bf16/fp16 这类较高精度训练，只训练很少一部分新增参数。
```

这就是为什么本轮 1.7B 模型总参数很多，但真正训练的参数只有约 `0.5040%`。

### 当前常见量化类型怎么分

学习时先按用途分成三层：

| 类型 | 常见选项 | 主要用途 | 你当前是否使用 |
|---|---|---|---|
| 常规精度，不算量化 | `fp32`、`fp16`、`bf16` | 训练/推理的普通浮点精度 | 使用 `bf16` 做计算精度 |
| bitsandbytes 量化 | 8bit `LLM.int8()`、4bit `FP4`、4bit `NF4` | Hugging Face/PEFT/QLoRA 常用 | 使用 4bit `NF4` |
| 离线/部署量化 | GPTQ、AWQ、GGUF、INT4/INT8 engine 等 | 更多用于推理部署 | 本轮没有使用 |

本轮属于：

```text
bitsandbytes 4bit NF4 QLoRA
```

它的目标不是把最终模型导出成部署引擎，而是在有限显存下完成微调学习。

### 三个 `bnb_4bit_*` 字段逐个解释

| 字段 | 当前值 | 控制什么 | 改大/改小或改值的影响 |
|---|---|---|---|
| `bnb_4bit_compute_dtype` | `bfloat16` | 4bit 权重参与矩阵计算时使用的计算 dtype。权重存得很低位，但计算不能真的全程只用 4bit。 | `bf16` 在 A800 上速度和稳定性较好；改成 `float32` 更占显存更慢；改成 `float16` 也可用，但 bf16 数值范围更大。 |
| `bnb_4bit_quant_type` | `nf4` | 4bit 权重量化格式。常见是 `fp4` 和 `nf4`。 | `nf4` 是 QLoRA 论文常用方案，适合近似正态分布的预训练权重；`fp4` 也能用，但 QLoRA 微调一般优先 `nf4`。 |
| `bnb_4bit_use_double_quant` | `true` | 是否对第一次量化产生的量化常数再做一次量化，也叫 nested quantization。 | `true` 会进一步省显存，通常几乎不增加额外成本；`false` 稍微更简单，但显存占用更高一点。 |

你可以把它们理解成三层：

```text
quant_bits=4:
  我要用 4bit 装基座模型。

bnb_4bit_quant_type=nf4:
  4bit 档位怎么设计。

bnb_4bit_compute_dtype=bfloat16:
  真正做矩阵乘法时用什么计算精度。

bnb_4bit_use_double_quant=true:
  量化参数本身再压缩一层，继续省显存。
```

### 为什么本轮按这些值配置

本轮目标是“在不影响服务器其它服务的前提下，用 A800 剩余显存跑通 Qwen3-1.7B QLoRA 学习训练”。所以配置选择偏保守：

| 配置 | 选择原因 |
|---|---|
| `quant_bits=4` | A800 当时不是整卡空闲，4bit 可以显著降低基座模型显存，让训练更稳。 |
| `bnb_4bit_compute_dtype=bfloat16` | A800 支持 bf16，bf16 比 fp32 省显存、速度更好，比 fp16 有更大的数值范围。 |
| `bnb_4bit_quant_type=nf4` | QLoRA 训练 4bit base model 的常见推荐配置，适合预训练模型权重分布。 |
| `bnb_4bit_use_double_quant=true` | 进一步省显存，适合学习实验和显存预算不宽裕的服务器环境。 |
| `lora_rank=8`、`lora_alpha=32` | adapter 容量适中，训练参数少，先保证流程跑通。 |
| `max_length=1024` | 兼顾上下文长度和显存。第一轮学习不追求长上下文效果。 |

如果你以后显存非常充足，并且追求更高质量，可以尝试：

| 目标 | 可尝试改动 |
|---|---|
| 想减少量化误差 | 不用 4bit，改成 bf16 LoRA 或全参/部分参数训练。 |
| 想容纳更长样本 | 把 `max_length` 从 `1024` 提到 `2048` 或更高，但要监控显存。 |
| 想提升 adapter 表达能力 | 把 `lora_rank` 从 `8` 提到 `16` 或 `32`，显存和训练时间会上升。 |
| 想做生产级实验 | 加验证集，观察 eval loss 和下游任务指标，不只看 train loss。 |

### 官方语义参考

- Hugging Face tokenizer 文档：<https://huggingface.co/docs/transformers/en/main_classes/tokenizer>
  - `padding_side` 可选 `right` 或 `left`，决定 padding 加在哪一侧。
- Hugging Face Transformers `BitsAndBytesConfig` 文档：<https://huggingface.co/docs/transformers/main_classes/quantization>
  - `load_in_4bit` 会把线性层替换为 bitsandbytes 的 FP4/NF4 4bit 层。
  - `bnb_4bit_compute_dtype` 控制计算 dtype。
  - `bnb_4bit_quant_type` 可选 `fp4`/`nf4`。
  - `bnb_4bit_use_double_quant` 控制 nested quantization。
- Hugging Face bitsandbytes 量化说明：<https://huggingface.co/docs/transformers/quantization/bitsandbytes>
  - NF4、double quant 和 4bit QLoRA 的常见用法可以从这里继续看。

### 10. LoRA 相关

| 字段 | 当前值 | 解释 |
|---|---|---|
| `tuner_backend` | `peft` | 使用 PEFT 后端。 |
| `tuner_type` | `lora` | 微调方式是 LoRA。 |
| `target_modules` | `["all-linear"]` | 命令层面指定所有线性层。 |
| `lora_rank` | `8` | LoRA rank，控制 adapter 容量。 |
| `lora_alpha` | `32` | LoRA 缩放系数。 |
| `lora_dropout` | `0.05` | LoRA dropout，轻度正则化。 |
| `lora_bias` | `none` | 不训练 bias。 |
| `modules_to_save` | `[]` | 不额外保存其它模块。 |
| `use_rslora` | `false` | 没启用 RsLoRA。 |
| `use_dora` | `false` | 没启用 DoRA。 |

命令里写的是：

```bash
--target_modules all-linear
```

而 `adapter_config.json` 里会展开成具体模块：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

### `--tuner_backend peft` 是什么意思

`tuner_backend` 表示“微调 adapter 由哪个后端库来管理”。本轮是：

```bash
--tuner_backend peft
```

意思是：让 ms-swift 调用 Hugging Face 生态里的 PEFT 库来创建、训练、保存和加载 LoRA adapter。

你可以把关系理解成：

```text
ms-swift:
  负责训练命令、数据读取、模板、日志、推理封装。

PEFT:
  负责 LoRA adapter 的具体实现。
  例如在哪些模块插入 LoRA、rank/alpha/dropout 怎么生效、adapter_config.json 怎么保存。

Transformers:
  负责加载 Qwen3 基座模型和 tokenizer。

bitsandbytes:
  负责 4bit 量化加载，也就是 QLoRA 的量化部分。
```

所以 `--tuner_backend peft` 不是训练指标，也不是模型效果指标，而是一个“实现后端选择”。它决定 LoRA adapter 用 PEFT 这套成熟格式保存，因此后续推理和部署也能通过：

```bash
swift infer --adapters <checkpoint>
swift deploy --adapters <checkpoint>
```

把微调后的 LoRA adapter 挂回基座模型。

如果不用 `peft`，ms-swift 也可能支持其它后端，例如 `unsloth`。但本轮用 `peft` 更稳妥，因为：

| 原因 | 说明 |
|---|---|
| 生态成熟 | PEFT 是 Hugging Face 常用 adapter 库。 |
| checkpoint 通用 | 保存出来有标准 `adapter_config.json` 和 `adapter_model.safetensors`。 |
| 和 QLoRA 配合常见 | `transformers + peft + bitsandbytes` 是常见 QLoRA 组合。 |
| 学习成本低 | 你后面看 LoRA、QLoRA、adapter 部署资料时，大部分都基于 PEFT。 |

### `lora_rank` 和 `lora_alpha` 是什么意思

LoRA 的核心可以先这样理解：

$$
y = W x + \frac{\alpha}{r} B A x
$$

其中：

| 符号 | 含义 |
|---|---|
| $W$ | 原始大模型线性层权重，本轮冻结不训练。 |
| $A$、$B$ | LoRA 新增的两个小矩阵，训练时只更新它们。 |
| $r$ | `lora_rank`，也就是低秩维度。 |
| $\alpha$ | `lora_alpha`，控制 LoRA 分支缩放强度。 |
| $\alpha / r$ | LoRA 分支真正乘上的缩放系数。 |

你当前配置是：

```text
lora_rank = 8
lora_alpha = 32
scale = 32 / 8 = 4
```

`lora_rank=8` 控制 adapter 容量。rank 越大，LoRA 新增参数越多，表达能力越强，但显存、训练时间和过拟合风险也会上升。

| rank | 直观特点 |
|---:|---|
| `4` | 很省显存，适合快速验证，能力较弱。 |
| `8` | 常见入门配置，省显存，也足够学习流程。 |
| `16` | 表达能力更强，适合稍认真一点的 SFT。 |
| `32+` | 能力更强，但小数据上更容易过拟合，也更耗资源。 |

`lora_alpha=32` 不直接增加参数量，而是控制 LoRA 分支对原模型输出的影响强度。

| alpha 相对 rank | 直观影响 |
|---|---|
| 太小 | LoRA 影响弱，模型学得慢，微调效果不明显。 |
| 适中 | 学习比较稳定，常用。 |
| 太大 | LoRA 更新过猛，可能训练不稳定或过拟合。 |

所以本轮可以这样理解：

```text
rank=8:
  adapter 容量不大，适合小数据学习和省显存。

alpha=32:
  scale=4，让 LoRA 分支有足够影响力。

dropout=0.05:
  做轻微正则化，降低小数据过拟合。
```

面试里可以这样说：

```text
LoRA 不直接更新原始大模型权重，而是在目标线性层上增加低秩矩阵 BA。
rank 控制低秩矩阵的容量和参数量，alpha 控制 LoRA 更新的缩放强度。
rank 越大表达能力越强，但更耗显存、更容易过拟合；
alpha 越大 LoRA 对原模型的影响越强，但过大可能导致训练不稳定。
我的实验里用 rank=8、alpha=32，是一个偏保守的 QLoRA 学习配置。
```

### Qwen3-1.7B 大致有哪些层

根据 Qwen/Qwen3-1.7B 的 `config.json`，这个模型是 `Qwen3ForCausalLM`，核心配置如下：

| 字段 | 值 | 含义 |
|---|---:|---|
| `num_hidden_layers` | `28` | 28 个 Transformer decoder layer。 |
| `hidden_size` | `2048` | 主干隐藏维度。 |
| `intermediate_size` | `6144` | MLP 中间层维度。 |
| `num_attention_heads` | `16` | attention query 头数。 |
| `num_key_value_heads` | `8` | key/value 头数，说明使用了 GQA/MQA 类结构。 |
| `vocab_size` | `151936` | 词表大小。 |
| `tie_word_embeddings` | `true` | 输入 embedding 和输出 lm head 权重绑定。 |

可以把模型主干理解成：

```text
Qwen3ForCausalLM
  ├─ embed_tokens
  ├─ layers[0..27]
  │   ├─ input_layernorm
  │   ├─ self_attn
  │   │   ├─ q_proj
  │   │   ├─ k_proj
  │   │   ├─ v_proj
  │   │   └─ o_proj
  │   ├─ post_attention_layernorm
  │   └─ mlp
  │       ├─ gate_proj
  │       ├─ up_proj
  │       ├─ down_proj
  │       └─ silu
  ├─ norm
  └─ lm_head
```

这次 QLoRA 不是训练整个模型，也不是直接改原始线性层权重。更准确地说是：

```text
冻结 4bit 基座模型原始参数。
只在指定线性层旁边挂 LoRA adapter。
训练这些新增的 LoRA adapter 参数。
```

从 `adapter_config.json` 看，实际被 LoRA 命中的模块是：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

所以每个 Transformer layer 里，attention 的 4 个线性投影和 MLP 的 3 个线性投影都加了 LoRA。因为有 28 层，所以大约是：

```text
28 层 * 7 个目标模块 = 196 个 LoRA 注入位置
```

但这些位置训练的是新增 LoRA 小矩阵，不是原始大矩阵本身。本轮没有训练：

```text
embed_tokens
layernorm / RMSNorm
lm_head
bias
原始 q/k/v/o/gate/up/down 权重
```

一句话总结：

```text
这次微调只训练 28 层里这些线性模块上的 LoRA adapter；
基座模型的原始参数，包括原始线性层权重，都是冻结的。
```

参考：Qwen/Qwen3-1.7B `config.json`：<https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json>

### 11. 多模态相关字段为什么也出现了

Round 1 是文本模型，但 `args.json` 里仍有：

| 字段 | 当前值 | 解释 |
|---|---|---|
| `freeze_vit` | `true` | 如果是多模态模型，冻结视觉塔。文本模型里基本不影响。 |
| `freeze_aligner` | `true` | 如果是多模态模型，冻结视觉-文本对齐层。文本模型里基本不影响。 |
| `vit_gradient_checkpointing` | `false` | 视觉塔梯度检查点。文本模型里不重点看。 |

原因是 ms-swift 的参数体系同时覆盖文本和多模态，文本任务也会保存这些默认字段。

### 12. 高级功能字段怎么处理

`args.json` 中还有很多高级字段，例如：

```text
fsdp
deepspeed
use_ray
use_galore
reft_rank
swanlab_project
torch_compile
use_liger_kernel
push_to_hub
```

这些在本轮大多是 `false`、`null` 或空列表，说明没有启用。小白阶段只需要知道：

```text
这些是框架支持的高级能力，本次实验没有使用。
```

不要把时间花在逐个深挖未启用字段上。

## `adapter_config.json` 逐字段讲解

原始文件：

```text
checkpoint-10/adapter_config.json
```

这个文件不是训练过程参数，而是 LoRA adapter 的结构配置。

| 行号/字段 | 当前值 | 含义 |
|---|---|---|
| `base_model_name_or_path` | `/workspace/qwen-qlora-lab/models/Qwen3-1.7B` | adapter 必须依附的基座模型。没有基座模型，adapter 不能单独推理。 |
| `bias` | `none` | 不训练 bias 参数。 |
| `fan_in_fan_out` | `false` | 线性层权重方向配置。Qwen 这类模型通常保持 false。 |
| `inference_mode` | `true` | 当前保存的 adapter 配置可用于推理。 |
| `init_lora_weights` | `true` | 训练开始时初始化 LoRA 权重。 |
| `lora_alpha` | `32` | 和训练命令里的 `--lora_alpha 32` 对应。 |
| `lora_dropout` | `0.05` | LoRA dropout。 |
| `peft_type` | `LORA` | PEFT 类型是 LoRA。 |
| `peft_version` | `0.19.1` | 生成该 adapter 的 PEFT 版本。 |
| `r` | `8` | 和训练命令里的 `--lora_rank 8` 对应。 |
| `target_modules` | `q_proj` 等 7 个模块 | 框架把 `all-linear` 展开后的具体线性层。 |
| `task_type` | `CAUSAL_LM` | 任务是因果语言模型。 |
| `use_dora` | `false` | 没启用 DoRA。 |
| `use_qalora` | `false` | 这里不是 QALoRA。注意 QLoRA 的 4bit 信息在 `args.json` 中。 |
| `use_rslora` | `false` | 没启用 RsLoRA。 |

### `target_modules` 为什么重要

`target_modules` 是 adapter 真正插入的位置：

| 模块 | 大致含义 |
|---|---|
| `q_proj` | attention query 投影 |
| `k_proj` | attention key 投影 |
| `v_proj` | attention value 投影 |
| `o_proj` | attention 输出投影 |
| `gate_proj` | MLP 门控投影 |
| `up_proj` | MLP 升维投影 |
| `down_proj` | MLP 降维投影 |

你可以这样理解：

```text
训练时不是改整个 Qwen3，而是在注意力层和 MLP 的线性投影旁边挂 LoRA 小矩阵。
```

## `additional_config.json` 讲解

原始内容只有一行：

```json
{"lora_dtype": null, "lorap_lr_ratio": null, "lorap_emb_lr": 1e-06}
```

| 字段 | 当前值 | 含义 |
|---|---|---|
| `lora_dtype` | `null` | 没有额外指定 LoRA dtype，使用框架默认。 |
| `lorap_lr_ratio` | `null` | 没启用 LoRA+ 的特殊学习率比例。 |
| `lorap_emb_lr` | `1e-06` | LoRA+ embedding 学习率默认值/保留值。本轮没有重点使用。 |

小白理解：

```text
这个文件不是本轮核心。它是框架为了兼容 LoRA 扩展功能保存的额外配置。
```

## `trainer_state.json` 逐字段讲解

原始文件：

```text
checkpoint-10/trainer_state.json
```

这是第 10 步保存 checkpoint 时的 Trainer 状态。它描述“训练器当时走到哪里了”。

| 字段 | 当前值 | 含义 |
|---|---|---|
| `best_global_step` | `null` | 没有最佳 step，因为本轮没有验证集。 |
| `best_metric` | `null` | 没有最佳指标，因为没有 eval metric。 |
| `best_model_checkpoint` | `null` | 没有最佳模型 checkpoint。 |
| `epoch` | `0.4444` | 第 10 步时约跑了 0.44 个 epoch。 |
| `eval_steps` | `10.0` | 评估步数配置是 10，但本轮没有验证集。 |
| `global_step` | `10` | 当前 checkpoint 是第 10 步保存的。 |
| `is_hyper_param_search` | `false` | 没在做超参数搜索。 |
| `is_local_process_zero` | `true` | 当前进程是本机主进程。 |
| `is_world_process_zero` | `true` | 当前进程是全局主进程。单卡训练下就是 true。 |
| `log_history` | 10 条记录 | 第 1 步到第 10 步的 loss、学习率、token_acc 等。 |
| `logging_steps` | `1` | 每一步记录日志。 |
| `max_steps` | `20` | 本轮计划最多训练 20 步。 |
| `num_input_tokens_seen` | `0` | Trainer 没统计输入 token 数。 |
| `num_train_epochs` | `1` | 配置最多训练 1 个 epoch。 |
| `save_steps` | `10` | 每 10 步保存一次。 |
| `stateful_callbacks` | `TrainerControl` | Trainer 内部控制状态。 |
| `total_flos` | `68971992477696.0` | 到第 10 步为止估算的浮点计算量。 |
| `train_batch_size` | `1` | Trainer 记录的单步 batch size。 |
| `trial_name` | `null` | 没有超参搜索 trial 名称。 |
| `trial_params` | `null` | 没有超参搜索参数。 |

### `log_history` 怎么看

`trainer_state.json` 里的 `log_history` 和 `logging.jsonl` 前 10 步内容对应。

例如第 1 步：

| 字段 | 含义 |
|---|---|
| `step` | 当前是第几步 |
| `loss` | 这一步训练 loss |
| `grad_norm` | 梯度范数 |
| `learning_rate` | 当前学习率 |
| `token_acc` | token 级准确率 |
| `epoch` | 数据集进度 |

第 10 步保存 checkpoint 时，`TrainerControl` 里：

```json
"should_save": true
```

说明 Trainer 知道此时应该保存 checkpoint。这和 `save_steps=10` 对应。

## 这些文件怎么一起解释一次训练

你可以用下面顺序讲：

1. `train_rounds_inside.sh` 里第 90 行启动 Round 1。
2. 第 40-63 行实际执行 `swift sft`。
3. `args.json` 记录这条命令展开后的完整参数。
4. `logging.jsonl` 记录第 1-20 步的训练指标。
5. 第 10 步时，`save_steps=10` 触发 `checkpoint-10`。
6. `checkpoint-10/trainer_state.json` 记录当时 `global_step=10`。
7. `checkpoint-10/adapter_config.json` 记录 LoRA adapter 怎么挂到 Qwen3-1.7B 上。
8. 第 20 步时保存 `checkpoint-20`，它是本轮最终 checkpoint。

面试表达：

```text
训练脚本定义了训练命令和显存保护逻辑，args.json 记录本轮完整参数，logging.jsonl 记录每步训练曲线，trainer_state.json 记录 checkpoint 保存时训练器状态，adapter_config.json 记录 LoRA adapter 与基座模型的绑定关系。
```
