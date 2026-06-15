# 大模型训练与微调小白学习指南

## 这份文档怎么用

这份文档不是让你背 `ms-swift` WebUI 的每一个字段，而是带你用当前目录里的真实文件，理解一次 Qwen QLoRA 微调从 0 到 1 的完整闭环：

```text
环境准备 -> 数据准备 -> 模型下载 -> 训练参数 -> QLoRA/LoRA adapter -> 日志判断 -> checkpoint -> 推理验证 -> 面试表达
```

你后续学习时建议按顺序执行：

1. 第一次读：只看每章的“你要理解什么”和“打开哪个文件”。
2. 第二次读：照着“练习任务”在 VS Code 里逐个打开文件，自己解释字段。
3. 第三次读：打开本地 WebUI，把文档里的参数在界面中找到。
4. 第四次读：准备面试表达，把每章最后的问题自己讲一遍。

## 当前学习材料在哪里

本地轻量快照目录：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/
```

这个目录不是完整训练工程，它是远端 A800 上 `/root/zhl/qwen-qlora-lab` 的轻量学习快照。它保留了脚本、数据、日志、推理结果、`args.json`、`adapter_config.json` 等适合学习的文本文件；没有保留大模型权重和 adapter 权重。

## 一句话理解微调

大模型微调不是“重新训练一个大模型”，而是：

1. 选一个已有基座模型，例如 `Qwen3-1.7B`。
2. 准备一批目标领域样本，例如推荐系统、服务器运维、QLoRA 学习问答。
3. 用 SFT 让模型模仿这些样本的回答风格和知识表达。
4. 用 LoRA/QLoRA 只训练很少量 adapter 参数，降低显存和存储成本。
5. 用 checkpoint 做推理，检查模型是否更符合你的任务。

在这次项目中，完整闭环是：

| 阶段    | 当前文件                                                     | 你要理解什么                                |
| ------- | ------------------------------------------------------------ | ------------------------------------------- |
| 环境    | `docker/Dockerfile`                                        | 训练需要哪些 Python 包、为什么要做兼容处理  |
| 数据    | `scripts/generate_data.py`、`data/*.jsonl`               | 文本 SFT 和多模态 SFT 样本长什么样          |
| 模型    | `scripts/download_models.sh`、`logs/model_downloads.log` | 模型怎么下载、怎么校验                      |
| 训练    | `scripts/train_rounds_inside.sh`                           | 一次 `swift sft` 训练命令包含哪些关键参数 |
| 结果    | `outputs/*/args.json`、`outputs/*/logging.jsonl`         | 实际训练参数和每步训练状态                  |
| adapter | `outputs/*/checkpoint-*/adapter_config.json`               | LoRA 训练了哪些模块、rank/alpha 是什么      |
| 推理    | `scripts/infer_rounds_inside.sh`、`logs/*.infer.md`      | 如何用 adapter 批量验证效果                 |
| 运维    | `logs/training_status.log`、`logs/*.nvidia-smi.txt`      | 如何确认没有影响共享 GPU 服务               |

## 你不需要一开始掌握什么

你不需要一开始理解所有高级训练概念。

| 暂时不用深挖                             | 原因                                       |
| ---------------------------------------- | ------------------------------------------ |
| 预训练细节                               | 本项目做的是 SFT/QLoRA，不是从零预训练     |
| 分布式训练、ZeRO、FSDP                   | 本次只用 A800 单卡保守训练                 |
| 每个 WebUI 字段                          | 很多字段是高级功能或默认值，先掌握核心字段 |
| `adapter_model.safetensors` 二进制内容 | 它是权重文件，不能用肉眼学习               |
| 大规模评测平台                           | 当前目标是学习闭环，不是生产级评测平台     |

你要先掌握的是：数据格式、训练命令、LoRA 参数、日志判断、推理验证。

## 第一课：看懂项目目录

打开目录：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/
```

你会看到这些子目录：

| 目录         | 含义                           | 怎么看                                                 |
| ------------ | ------------------------------ | ------------------------------------------------------ |
| `data/`    | 训练和推理用的 JSONL 数据      | 先看 `text_sft.jsonl` 和 `mm_sft.jsonl`            |
| `images/`  | 多模态训练用的合成图片         | 打开 `01_pipeline.png` 等图片                        |
| `scripts/` | 数据生成、下载、训练、推理脚本 | 重点看 `train_rounds_inside.sh`                      |
| `docker/`  | 远端训练环境镜像               | 看它安装了哪些框架和依赖                               |
| `logs/`    | 训练、下载、推理、显存日志     | 看成功/失败原因和显存占用                              |
| `outputs/` | 训练输出目录                   | 看 `args.json`、`logging.jsonl`、checkpoint 元数据 |

练习任务：

- [X] 在 VS Code 里展开 `remote_snapshot/qwen-qlora-lab-small/`。
- [X] 找到 `data/`、`scripts/`、`logs/`、`outputs/`。
- [X] 用一句话解释每个目录的作用。

面试表达：

```text
我把一次微调实验拆成了环境、数据、模型、训练、日志、checkpoint 和推理验证几个部分，每个部分都有可追溯文件，便于复现和排错。
```

## 第二课：看懂文本 SFT 数据

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/data/text_sft.jsonl
```

你会看到每一行都是一个 JSON：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个帮助学习推荐系统、服务器运维和大模型微调的中文技术助手。"
    },
    {
      "role": "user",
      "content": "请用简洁中文解释：推荐系统召回。"
    },
    {
      "role": "assistant",
      "content": "召回阶段负责从大规模候选池中快速找出可能相关的物品，常见方法包括 ItemCF、双塔向量召回和图模型召回。"
    }
  ]
}
```

### `messages` 是什么

`messages` 是聊天模型最常见的监督微调格式。它表示一次对话。

| 字段          | 含义                       | 当前例子         |
| ------------- | -------------------------- | ---------------- |
| `system`    | 给模型设定身份、风格和边界 | 中文技术助手     |
| `user`      | 用户问题                   | 解释推荐系统召回 |
| `assistant` | 期望模型学会的回答         | 召回阶段的解释   |

训练时，模型会看到 `system + user`，然后学习生成 `assistant` 的内容。

### 这批数据为什么这么设计

当前 `text_sft.jsonl` 不是生产数据，而是学习数据。它围绕这些主题：

- 推荐系统召回。
- 推荐系统排序。
- QLoRA 显存优化。
- LoRA 参数含义。
- A800 共享训练。
- ModelScope 下载。
- 多模态数据格式。
- 训练日志记录。
- 推荐模型评估。
- 显存保护策略。

这样设计的目的不是追求模型效果，而是让你用小数据跑通流程。

### 好数据和坏数据的区别

| 类型   | 表现                                               |
| ------ | -------------------------------------------------- |
| 好数据 | 问题明确、回答准确、风格稳定、和目标任务一致       |
| 坏数据 | 回答含糊、事实错误、格式混乱、同一问题答案互相矛盾 |

在真实项目里，数据质量通常比调参更重要。微调不是把模型变聪明，而是把你的数据分布、任务格式和回答偏好灌进模型。

练习任务：

- [X] 打开 `text_sft.jsonl`。
- [X] 复制一条样本，标出 `system`、`user`、`assistant`。
- [X] 自己写 3 条推荐系统相关样本，保持同样格式。
- [X] 判断这批数据适合让模型学什么，不适合让模型学什么。

面试表达：

```text
我使用 messages 格式做 SFT，system 约束模型身份，user 是输入问题，assistant 是期望输出。训练目标是让模型在目标领域中更稳定地生成指定风格的回答。
```

## 第三课：看懂多模态数据

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/data/mm_sft.jsonl
```

你会看到每条样本多了一个 `images` 字段：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个能读懂技术流程图的中文助手。"
    },
    {
      "role": "user",
      "content": "<image> 请描述这张技术图的核心含义。"
    },
    {
      "role": "assistant",
      "content": "这张图展示了离线推荐系统从日志到评测报告的三段流程。"
    }
  ],
  "images": [
    "/workspace/qwen-qlora-lab/images/01_pipeline.png"
  ]
}
```

### 多模态样本比文本多了什么

| 字段         | 作用                 |
| ------------ | -------------------- |
| `messages` | 仍然表示对话         |
| `<image>`  | 告诉模板这里插入图片 |
| `images`   | 给出图片路径         |

这类数据用于训练或微调能读图的模型。文本模型只看文字，多模态模型还要把图片编码成视觉特征。

### 当前图片从哪里来

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/images/
```

这些图片由脚本生成：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/generate_data.py
```

当前图片包括：

| 图片                | 含义                   |
| ------------------- | ---------------------- |
| `01_pipeline.png` | 推荐系统离线 pipeline  |
| `02_gpu.png`      | A800 共享 GPU 使用策略 |
| `03_qlora.png`    | QLoRA 训练闭环         |
| `04_metrics.png`  | 推荐系统指标           |
| `05_download.png` | 模型下载回退策略       |
| `06_safety.png`   | 低风险训练原则         |

练习任务：

- [X] 打开 `mm_sft.jsonl`。
- [X] 打开 `images/01_pipeline.png`。
- [X] 检查 JSONL 里的图片路径和图片文件是否对应。
- [X] 自己解释 `<image>` 的作用。

面试表达：

```text
多模态 SFT 样本在 messages 外还包含 images 字段，文本中用 <image> 标记图片插入位置。模型训练时同时学习图片内容和文本回答之间的对应关系。
```

## 第四课：看懂数据生成脚本

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/generate_data.py
```

这个脚本做了两件事：

1. 生成 `data/text_sft.jsonl`。
2. 生成 `data/mm_sft.jsonl` 和 `images/*.png`。

### 为什么先看数据生成脚本

因为它能回答三个基础问题：

| 问题               | 对应代码                                              |
| ------------------ | ----------------------------------------------------- |
| 训练数据有多少条   | `for i in range(180)`                               |
| 每条样本是什么格式 | `rows.append({"messages": [...]})`                  |
| 多模态图片怎么来的 | `Image.new`、`ImageDraw.Draw`、`img.save(path)` |

### 当前数据规模意味着什么

文本数据有 `180` 条，多模态数据有 `6` 条。这个规模只适合学习流程，不适合生产效果。

如果你面试时讲这个实验，要说清楚：

```text
这次数据规模很小，目标是学习和验证 QLoRA/SFT 工程闭环，而不是追求生产模型质量。
```

练习任务：

- [X] 找到 `text_topics`，看这批文本样本覆盖哪些主题。
- [X] 找到 `mm_specs`，看每张图对应什么回答。
- [X] 修改一条样本主题，思考训练结果会被怎样影响。

## 第五课：看懂环境 Dockerfile

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/docker/Dockerfile
```

关键内容：

```dockerfile
FROM hiyouga/llamafactory:latest
```

这表示镜像从 LLaMA-Factory 的基础镜像派生。

```dockerfile
RUN python -m pip install -U \
      ms-swift==4.3.0 transformers==5.10.2 accelerate==1.14.0 peft==0.19.1 \
      bitsandbytes qwen-vl-utils modelscope ...
```

这表示主训练框架是 `ms-swift`，同时安装了：

| 包                | 用途                   |
| ----------------- | ---------------------- |
| `ms-swift`      | 训练、推理、导出主框架 |
| `transformers`  | 模型结构和 tokenizer   |
| `accelerate`    | 训练加速和设备管理     |
| `peft`          | LoRA/QLoRA adapter     |
| `bitsandbytes`  | 4bit/8bit 量化         |
| `qwen-vl-utils` | Qwen 多模态处理工具    |
| `modelscope`    | 国内模型下载           |

### 为什么有兼容 shim

Dockerfile 里有一段 `sitecustomize.py`：

```python
if not hasattr(torch, "float8_e8m0fnu"):
    ...
```

这是为了处理 `transformers 5.x` 和基础镜像里的 `torch 2.6.0+cu124` 之间的兼容问题。它是学习实验中的局部兼容补丁，不是生产环境最佳实践。

练习任务：

- [X] 找到 `ms-swift==4.3.0`。
- [X] 找到 `bitsandbytes`。
- [X] 找到 `modelscope`。
- [X] 用一句话说明 Dockerfile 为什么重要。

面试表达：

```text
我没有直接在裸机环境里混装依赖，而是用 Docker 固化训练环境，并在日志中记录了框架版本，降低复现实验时的环境漂移风险。
```

## 第六课：看懂模型下载

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/download_models.sh
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/logs/model_downloads.log
```

脚本核心逻辑：

1. 优先用 ModelScope 下载。
2. 如果失败，再用 Hugging Face fallback。
3. 下载后检查 `config.json`、tokenizer 文件、权重文件。
4. 记录模型目录大小。

当前下载结果：

| 模型             | 目录大小 |
| ---------------- | -------: |
| `Qwen3-1.7B`   | `3.8G` |
| `Qwen3.5-0.8B` | `1.7G` |
| `Qwen3.5-2B`   | `4.3G` |

### 为什么下载后要校验

只看命令返回成功不够。真实服务器经常出现：

- 下载中断。
- tokenizer 缺失。
- 权重文件没下载完整。
- 模型目录路径写错。

所以脚本检查：

```bash
config.json
*.safetensors 或 model*.bin
*tokenizer* 或 vocab* 或 merges.txt
```

练习任务：

- [X] 打开 `download_models.sh`。
- [X] 找到 `verify_model_dir` 函数。
- [X] 打开 `model_downloads.log`，确认三个模型大小。

面试表达：

```text
模型下载后我不会只看命令是否成功，还会校验 config、tokenizer 和权重文件，并记录目录大小，避免训练时才发现模型不完整。
```

## 第七课：看懂训练脚本主线

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/train_rounds_inside.sh
```

这个脚本是整份学习材料最重要的文件之一。

它做了这些事：

1. 进入项目目录。
2. 定义日志函数。
3. 定义显存读取函数。
4. 定义查找最新 checkpoint 的函数。
5. 定义 `run_round` 函数。
6. 检查初始剩余显存。
7. 依次运行三轮训练。

三轮训练是：

| 轮次    | 模型                    | 数据                    | 输出目录                    |
| ------- | ----------------------- | ----------------------- | --------------------------- |
| Round 1 | `models/Qwen3-1.7B`   | `data/text_sft.jsonl` | `outputs/qwen3_17b_text`  |
| Round 2 | `models/Qwen3.5-0.8B` | `data/text_sft.jsonl` | `outputs/qwen35_08b_text` |
| Round 3 | `models/Qwen3.5-2B`   | `data/mm_sft.jsonl`   | `outputs/qwen35_2b_mm`    |

### 显存保护逻辑

脚本先查显存：

```bash
free_now=$(free_mib)
if [ "$free_now" -lt 18000 ]; then
  exit 1
fi
```

含义是：共享 A800 上剩余显存太低时，不要硬跑训练，避免影响已有服务。

训练中还采样显存：

```bash
( while true; do printf '%s,' "$(date '+%F %T')"; used_mib; sleep 2; done ) > "$sample_file" &
```

训练结束后计算显存峰值：

```bash
max_used=$(awk -F, ...)
delta=$((max_used - base_used))
```

如果显存增量超过 `16000 MiB`，后续轮次强制降到 `max_length=512`。

练习任务：

- [X] 找到 `free_mib` 函数。
- [X] 找到 `used_mib` 函数。
- [X] 找到 `logs/force_max_length_512` 的逻辑。
- [X] 解释为什么共享 GPU 上要先检查显存。

面试表达：

```text
因为 A800 是共享服务器，我在训练脚本里做了显存门槛、训练前后 nvidia-smi、训练过程显存采样和 OOM 降级策略，避免影响已有推理服务。
```

## 第八课：看懂 `swift sft` 训练命令

`train_rounds_inside.sh` 里最核心的是这段：

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
  --tuner_backend peft \
  --model "$model_path" \
  --quant_bits 4 \
  --dataset "$dataset" \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --max_steps 20 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 8 \
  --logging_steps 1 \
  --save_steps 10 \
  --max_length "$max_len" \
  --output_dir "$out_dir" \
  --save_only_model true
```

### 参数分类理解

| 分类        | 参数                              | 当前值          | 小白解释                               |
| ----------- | --------------------------------- | --------------- | -------------------------------------- |
| 设备        | `CUDA_VISIBLE_DEVICES`          | `0`           | 只用第 0 张可见 GPU                    |
| 框架        | `swift sft`                     | SFT 训练        | 做监督微调                             |
| 模型        | `--model`                       | 模型目录        | 选择要微调的基座模型                   |
| 数据        | `--dataset`                     | JSONL 文件      | 选择训练样本                           |
| 量化        | `--quant_bits`                  | `4`           | 用 4bit 加载基座模型，降低显存         |
| 精度        | `--torch_dtype`                 | `bfloat16`    | 训练计算用 bf16                        |
| 步数        | `--max_steps`                   | `20`          | 只训练 20 步，学习实验用               |
| batch       | `--per_device_train_batch_size` | `1`           | 单步每卡只放 1 条样本                  |
| 梯度累积    | `--gradient_accumulation_steps` | `8`           | 累积 8 次再更新，相当于有效 batch 变大 |
| 学习率      | `--learning_rate`               | `1e-4`        | 控制每次参数更新幅度                   |
| LoRA 容量   | `--lora_rank`                   | `8`           | adapter 低秩矩阵大小                   |
| LoRA 缩放   | `--lora_alpha`                  | `32`          | adapter 更新强度                       |
| LoRA 插入层 | `--target_modules`              | `all-linear`  | 对线性层插入 LoRA                      |
| 序列长度    | `--max_length`                  | `1024`        | 输入输出总 token 上限                  |
| 保存        | `--save_steps`                  | `10`          | 每 10 步保存一次                       |
| 输出        | `--output_dir`                  | `outputs/...` | 保存训练产物                           |

### 有效 batch size 怎么算

本次训练：

```text
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
GPU 数量 = 1
```

有效 batch size：

$$
1 \times 8 \times 1 = 8
$$

这表示模型每累积 8 个小 batch 后更新一次参数。

### 你调参时先动哪些

| 目标           | 优先调整                                   |
| -------------- | ------------------------------------------ |
| 显存不够       | 降 `max_length`，保持 batch size 为 1    |
| 训练太慢       | 降模型大小或 `max_length`                |
| 过拟合         | 减少 epoch/steps，增加验证集，清洗重复数据 |
| 效果不明显     | 增加高质量数据，适当增加 steps             |
| 输出风格不稳定 | 提高数据一致性，检查模板和 system prompt   |

练习任务：

- [X] 在 WebUI 里找到 `model`、`dataset`、`max_length`、`learning_rate`。
- [X] 在 WebUI 里找到 `lora_rank`、`lora_alpha`、`target_modules`。
- [X] 用自己的话解释有效 batch size。

面试表达：

```text
我使用 QLoRA 方式加载 4bit 基座模型，并通过 PEFT LoRA 只训练 adapter。为了控制共享 A800 的显存风险，我把单卡 batch size 固定为 1，用 gradient accumulation 扩大有效 batch。
```

## 第九课：看懂 LoRA 和 QLoRA

### LoRA 是什么

LoRA 的思想是：不直接改大模型所有参数，而是在部分线性层旁边插入两个小矩阵。训练时只更新这两个小矩阵。

你可以粗略理解为：

```text
原模型权重 W 不动
新增一小块可训练参数 A 和 B
训练时学习 A 和 B
推理时 W + LoRA adapter 一起工作
```

### QLoRA 是什么

QLoRA 是 LoRA 加上量化加载：

```text
基座模型用 4bit 量化加载，减少显存
LoRA adapter 用较高精度训练
最终只保存 adapter
```

本项目里对应参数：

| 概念         | 参数                                                        |
| ------------ | ----------------------------------------------------------- |
| 使用 QLoRA   | `--quant_bits 4`                                          |
| 使用 LoRA    | `--tuner_backend peft`、`--lora_rank`、`--lora_alpha` |
| 插入哪些层   | `--target_modules all-linear`                             |
| adapter 输出 | `outputs/*/checkpoint-*`                                  |

## 第十课：看懂 `adapter_config.json`

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/outputs/qwen3_17b_text/v1-20260613-211241/checkpoint-10/adapter_config.json
```

重点字段：

| 字段                        | 当前值                                          | 含义                     |
| --------------------------- | ----------------------------------------------- | ------------------------ |
| `base_model_name_or_path` | `/workspace/qwen-qlora-lab/models/Qwen3-1.7B` | adapter 对应哪个基座模型 |
| `peft_type`               | `LORA`                                        | 这是 LoRA adapter        |
| `task_type`               | `CAUSAL_LM`                                   | 因果语言模型任务         |
| `r`                       | `8`                                           | LoRA rank                |
| `lora_alpha`              | `32`                                          | LoRA 缩放系数            |
| `lora_dropout`            | `0.05`                                        | adapter dropout          |
| `target_modules`          | `q_proj`、`v_proj`、`gate_proj` 等        | 实际插入 LoRA 的模块     |
| `inference_mode`          | `true`                                        | 当前 checkpoint 用于推理 |

### 为什么训练命令里是 `all-linear`，这里变成具体模块

训练命令写：

```bash
--target_modules all-linear
```

这是一个简写。框架会根据模型结构，把它展开成具体线性层，例如：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

这些名字来自 Transformer 层内部结构。

练习任务：

- [X] 打开 `adapter_config.json`。
- [X] 找到 `r` 和 `lora_alpha`。
- [X] 找到 `target_modules`。
- [X] 解释为什么 adapter 必须知道 `base_model_name_or_path`。

面试表达：

```text
LoRA adapter 不是独立模型，它必须和对应的基座模型一起使用。adapter_config.json 记录了基座模型路径、LoRA rank、alpha 和实际注入的 target modules。
```

## 第十一课：看懂训练输出目录

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/outputs/qwen3_17b_text/v1-20260613-211241/
```

重点文件：

| 文件               | 作用                      |
| ------------------ | ------------------------- |
| `args.json`      | 本次训练的完整参数快照    |
| `logging.jsonl`  | 每一步训练指标            |
| `checkpoint-10/` | 第 10 步保存的 checkpoint |
| `checkpoint-20/` | 第 20 步保存的 checkpoint |
| `images/`        | 训练曲线图                |

### `args.json` 怎么看

`args.json` 很长，不要从头背到尾。先找这些字段：

| 字段                            | 当前例子                | 你要理解什么         |
| ------------------------------- | ----------------------- | -------------------- |
| `model`                       | `models/Qwen3-1.7B`   | 用了哪个基座模型     |
| `model_type`                  | `qwen3`               | 框架识别的模型类型   |
| `template`                    | `qwen3`               | 使用哪个 prompt 模板 |
| `dataset`                     | `data/text_sft.jsonl` | 训练数据             |
| `quant_bits`                  | `4`                   | 是否 QLoRA           |
| `max_length`                  | `1024`                | 序列长度上限         |
| `per_device_train_batch_size` | `1`                   | 单卡 batch           |
| `gradient_accumulation_steps` | `8`                   | 梯度累积             |
| `learning_rate`               | `0.0001`              | 学习率               |
| `lora_rank`                   | `8`                   | LoRA rank            |
| `lora_alpha`                  | `32`                  | LoRA alpha           |
| `target_modules`              | `all-linear`          | LoRA 目标模块        |
| `swift_version`               | `4.3.0`               | 框架版本             |

### 为什么 `args.json` 很重要

因为它记录了“实际训练发生了什么”。命令行写过什么不一定可靠，训练中框架可能有默认值和自动推断，`args.json` 是最终参数快照。

练习任务：

- [X] 打开 `args.json`。
- [X] 搜索 `model`。
- [X] 搜索 `dataset`。
- [X] 搜索 `quant_bits`。
- [X] 搜索 `lora_rank`。
- [X] 搜索 `target_modules`。

## 第十二课：看懂训练日志指标

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/outputs/qwen3_17b_text/v1-20260613-211241/logging.jsonl
```

前几行类似：

```json
{"loss": 7.39256144, "grad_norm": 22.3578167, "learning_rate": 0.0001, "token_acc": 0.34549356, "global_step/max_steps": "1/20", "memory(GiB)": 3.82}
{"loss": 7.10582113, "grad_norm": 20.37049866, "learning_rate": 9.932e-05, "token_acc": 0.35574837, "global_step/max_steps": "2/20", "memory(GiB)": 3.82}
```

### 指标怎么理解

| 指标                      | 含义                     | 怎么判断                          |
| ------------------------- | ------------------------ | --------------------------------- |
| `loss`                  | 模型预测和目标答案的差距 | 通常希望整体下降，但不能只看 loss |
| `grad_norm`             | 梯度范数                 | 过大可能不稳定，异常 NaN 需要警惕 |
| `learning_rate`         | 当前学习率               | 本次用 cosine 调度，逐步下降      |
| `token_acc`             | token 级预测准确率       | 可辅助观察训练是否学到东西        |
| `epoch`                 | 训练数据遍历进度         | 小实验未必完整跑完 1 epoch        |
| `global_step/max_steps` | 当前训练步数             | 本次最多 20 步                    |
| `memory(GiB)`           | 框架记录的显存           | 只作参考，结合 `nvidia-smi`     |
| `train_speed(s/it)`     | 每步耗时                 | 用来比较训练速度                  |

### 当前 Round 1 日志说明什么

Round 1 里 `loss` 从约 `7.39` 降到约 `2.63`，说明模型在这批小数据上快速拟合。但这不代表模型能力全面提升，因为数据很小、步数很少、没有严格验证集。

正确结论是：

```text
训练流程跑通，模型能拟合小样本；还不能证明生产效果好。
```

练习任务：

- [ ] 打开 `logging.jsonl`。
- [ ] 找到第 1 步 loss。
- [ ] 找到第 20 步 loss。
- [ ] 判断 loss 是否整体下降。
- [ ] 写一句“这个结果能说明什么，不能说明什么”。

面试表达：

```text
我不会只用训练 loss 判断微调成功。loss 下降说明模型在训练集上拟合了样本，但还需要独立验证集、固定推理问题和业务指标判断真实收益。
```

## 第十三课：看懂训练状态日志

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/logs/training_status.log
```

你会看到：

```text
[2026-06-13 21:12:26] START round1_qwen3_17b_text ...
[2026-06-13 21:14:12] END round1_qwen3_17b_text rc=0 max_used_mib=62387 train_delta_mib=4406 latest_ckpt=...
```

### 关键字段

| 字段                | 含义                     |
| ------------------- | ------------------------ |
| `START`           | 某轮训练开始             |
| `END`             | 某轮训练结束             |
| `rc=0`            | exit code 为 0，表示成功 |
| `rc=1`            | exit code 非 0，表示失败 |
| `base_used_mib`   | 训练前 GPU 已用显存      |
| `max_used_mib`    | 训练期间最高已用显存     |
| `train_delta_mib` | 本轮新增显存峰值         |
| `latest_ckpt`     | 最新 checkpoint 路径     |

### 当前三轮结果

| 轮次    | 结果 |     显存增量 | checkpoint                                    |
| ------- | ---- | -----------: | --------------------------------------------- |
| Round 1 | 成功 | `4406 MiB` | `outputs/qwen3_17b_text/.../checkpoint-20`  |
| Round 2 | 成功 | `2734 MiB` | `outputs/qwen35_08b_text/.../checkpoint-20` |
| Round 3 | 成功 | `6420 MiB` | `outputs/qwen35_2b_mm/.../checkpoint-20`    |

注意：日志里也保留了失败记录，例如 Round 1 和 Round 2 曾经 `rc=1`。这很正常，真实工程不是一次成功，重要的是能定位、修复、复跑，并保留记录。

练习任务：

- [ ] 打开 `training_status.log`。
- [ ] 找到第一次失败。
- [ ] 找到第一次成功。
- [ ] 找出三轮最终 checkpoint。
- [ ] 用一句话说明显存增量是否在可接受范围内。

## 第十四课：看懂完整训练日志

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/logs/round1_qwen3_17b_text.train.log
```

这个文件是完整 stdout/stderr，信息很多。小白不用全部看，先看四类信息：

| 信息            | 怎么找                                            |
| --------------- | ------------------------------------------------- |
| 训练进度        | 搜索 `Train:`                                   |
| 每步指标        | 搜索 `'loss'`                                   |
| 保存 checkpoint | 搜索 `Saving model checkpoint`                  |
| 最终结果        | 搜索 `train_runtime`、`last_model_checkpoint` |

当前 Round 1 最终摘要：

```text
train_runtime: 78.3589
train_samples_per_second: 2.042
train_steps_per_second: 0.255
train_loss: 3.81233135
last_model_checkpoint: .../checkpoint-20
```

### 为什么 `train_loss` 不等于最后一步 loss

最后一步 loss 是当前 step 的 loss，`train_loss` 通常是训练期间平均或汇总后的 loss。面试时不用纠结公式细节，但要知道它们不是同一个值。

练习任务：

- [ ] 打开 `round1_qwen3_17b_text.train.log`。
- [ ] 搜索 `Saving model checkpoint`。
- [ ] 搜索 `train_runtime`。
- [ ] 找到 `last_model_checkpoint`。

## 第十五课：看懂推理验证

训练完成后不能只说“有 checkpoint 了”，还要用固定问题测试输出。

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/infer_rounds_inside.sh
```

核心命令：

```bash
swift infer \
  --adapters "$ckpt" \
  --val_dataset "$dataset" \
  --result_path "$result_path" \
  --stream false \
  --max_new_tokens 256 \
  --max_batch_size 1
```

### 推理参数解释

| 参数                     | 含义                      |
| ------------------------ | ------------------------- |
| `--adapters`           | 加载哪个 LoRA checkpoint  |
| `--val_dataset`        | 用哪个 JSONL 做验证输入   |
| `--result_path`        | 把输出保存到哪里          |
| `--stream false`       | 非流式输出，适合批量验证  |
| `--max_new_tokens 256` | 最多生成 256 个新 token   |
| `--max_batch_size 1`   | 每次推理 1 条，保守省显存 |

### 看推理结果

打开：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/logs/round1_qwen3_17b_text.infer.md
```

你会看到：

```text
result_rows: 3
result_preview:
- row 1: ...
- row 2: ...
```

这说明脚本用 3 个固定问题跑了批量推理，并保存了结果。

### 怎么判断推理好不好

先看三个层次：

| 层次   | 判断问题                          |
| ------ | --------------------------------- |
| 格式   | 是否能正常输出中文，是否跑完 3 条 |
| 相关性 | 回答是否围绕问题                  |
| 稳定性 | 是否啰嗦、幻觉、遗漏重点          |

真实项目还要做更严格评测：

- 人工标注集。
- 自动指标。
- 和 base 模型对比。
- 和 prompt/RAG 方案对比。
- 线上 A/B 或离线业务指标。

练习任务：

- [ ] 打开 `infer_rounds_inside.sh`。
- [ ] 找到 `--adapters`。
- [ ] 打开 `round1_qwen3_17b_text.infer.md`。
- [ ] 读两条 `result_preview`。
- [ ] 写一句“这次推理验证有什么不足”。

面试表达：

```text
我用固定验证问题和 batch infer 方式验证 adapter，不依赖交互式手动测试。推理结果会保存成 JSONL 和 Markdown 摘要，便于复盘和对比。
```

## 第十六课：把 WebUI 字段和文件对应起来

你已经打开了 `ms-swift` WebUI。你不需要把每个字段背下来，先把界面字段和训练脚本对应起来。

### 顶部标签页

| WebUI 标签         | 含义                        | 当前是否重点 |
| ------------------ | --------------------------- | ------------ |
| `LLM预训练/微调` | SFT、LoRA、QLoRA 等训练入口 | 重点         |
| `LLM人类对齐`    | DPO、偏好对齐等             | 暂时了解     |
| `LLM GRPO`       | 强化学习/GRPO 相关          | 暂时不学     |
| `LLM推理`        | 加载模型或 adapter 做推理   | 第二重点     |
| `LLM导出`        | 合并/导出模型               | 后续学习     |
| `LLM评测`        | 自动评测                    | 后续学习     |
| `LLM采样`        | 生成/采样数据               | 后续学习     |

### 模型设置

| WebUI 字段      | 命令参数                       | 当前例子              | 解释                         |
| --------------- | ------------------------------ | --------------------- | ---------------------------- |
| 模型 id 或路径  | `--model`                    | `models/Qwen3-1.7B` | 基座模型目录                 |
| 模型类型        | `--model_type`               | `qwen3`             | 框架识别模型结构             |
| Prompt 模板类型 | `--template`                 | `qwen3`             | 把 messages 转成模型实际输入 |
| System 字段     | `--system` 或数据里的 system | 当前在数据里          | 指定助手身份                 |

模板很重要。模型和模板不匹配，轻则回答奇怪，重则训练数据拼接错误。

### 数据集设置

| WebUI 字段     | 命令参数                  | 当前例子                | 解释                |
| -------------- | ------------------------- | ----------------------- | ------------------- |
| 数据集名称     | `--dataset`             | `data/text_sft.jsonl` | 训练样本            |
| 验证集拆分比例 | `--split_dataset_ratio` | `0.0`                 | 小实验不拆验证集    |
| 句子最大长度   | `--max_length`          | `1024`                | 输入输出 token 上限 |
| 无填充批处理   | `--padding_free`        | `false`               | 高级优化，先不碰    |

### 训练参数设置

| WebUI 字段 | 命令参数                          | 当前例子     | 解释            |
| ---------- | --------------------------------- | ------------ | --------------- |
| 训练 Stage | 隐含 SFT                          | `sft`      | 监督微调        |
| 训练方式   | `--tuner_type`                  | `lora`     | 用 LoRA 微调    |
| 随机数种子 | `--seed`                        | `42`       | 保持可复现      |
| 训练精度   | `--torch_dtype`                 | `bfloat16` | A800 上常用     |
| batch size | `--per_device_train_batch_size` | `1`        | 省显存          |
| 梯度累积   | `--gradient_accumulation_steps` | `8`        | 有效 batch      |
| 学习率     | `--learning_rate`               | `1e-4`     | 更新幅度        |
| 最大步数   | `--max_steps`                   | `20`       | 学习实验短跑    |
| 保存步数   | `--save_steps`                  | `10`       | 保存 checkpoint |

### LoRA/QLoRA 设置

| WebUI 字段     | 命令参数             | 当前例子       | 解释          |
| -------------- | -------------------- | -------------- | ------------- |
| 量化 bit       | `--quant_bits`     | `4`          | QLoRA 关键    |
| LoRA rank      | `--lora_rank`      | `8`          | adapter 容量  |
| LoRA alpha     | `--lora_alpha`     | `32`         | adapter 缩放  |
| Target modules | `--target_modules` | `all-linear` | 哪些层加 LoRA |

### WebUI 学习方式

你现在打开 WebUI 后，按下面顺序做：

1. 不要先点开始训练。
2. 先把 `train_rounds_inside.sh` 里的参数在 WebUI 中找到。
3. 找到一个参数，就回到文档里看解释。
4. 再打开 `args.json`，确认训练后实际记录的字段。
5. 最后再看 `logging.jsonl`，理解训练过程中这些参数带来的结果。

练习任务：

- [ ] 在 WebUI 找到 `模型 id 或路径`。
- [ ] 在 WebUI 找到 `数据集名称`。
- [ ] 在 WebUI 找到 `max_length`。
- [ ] 在 WebUI 找到 `lora_rank`。
- [ ] 在 WebUI 找到 `quant_bits`。
- [ ] 不看文档，自己解释这 5 个字段。

## 第十七课：后续怎么实践

### 阶段 1：只读文件，不训练

目标：看懂完整闭环。

任务：

- [ ] 读 `data/text_sft.jsonl`，解释 `messages`。
- [ ] 读 `scripts/train_rounds_inside.sh`，解释 10 个核心参数。
- [ ] 读 `outputs/qwen3_17b_text/.../args.json`，找到实际参数。
- [ ] 读 `outputs/qwen3_17b_text/.../checkpoint-10/adapter_config.json`，解释 LoRA。
- [ ] 读 `logs/round1_qwen3_17b_text.infer.md`，解释推理验证。

完成标准：

```text
你能用 3 分钟讲清楚一次 QLoRA 训练从数据到 adapter 再到推理验证的过程。
```

### 阶段 2：在 WebUI 中复现参数，不启动训练

目标：把命令行参数和界面字段对应起来。

任务：

- [ ] 打开 `ms-swift` WebUI。
- [ ] 在 `LLM预训练/微调` 页找到模型设置。
- [ ] 对照 `train_rounds_inside.sh` 填入相同参数。
- [ ] 不点击训练，只截图或记录你填了哪些字段。

完成标准：

```text
你能把 WebUI 中的关键字段映射到 swift sft 命令参数。
```

### 阶段 3：本地做极小样例，不追求效果

当前 WSL 是 CPU 版 PyTorch，不建议训练 Qwen3/Qwen3.5。你可以做的是：

- 只用 WebUI 学参数。
- 用 tiny 模型或极小模型做流程验证。
- 不在本地下载 1B/2B 级模型权重。

如果以后要本地试跑，应新建独立 tiny 环境，不要破坏现在两套 WebUI 环境。

完成标准：

```text
你知道本地 Windows/WSL 主要用于学习 UI 和流程，正式训练仍放 A800。
```

### 阶段 4：在 A800 上做一次自己的小实验

目标：把示例数据换成你自己的数据。

建议任务：

1. 新建 `data/my_text_sft.jsonl`。
2. 写 30 条你真正想让模型学习的问答。
3. 保持 `messages` 格式。
4. 复制一份训练命令，只改：
   - `--dataset data/my_text_sft.jsonl`
   - `--output_dir outputs/my_qwen_text`
   - `--max_steps 20`
5. 跑完后固定 5 个问题做推理验证。

不要一开始追求大规模训练。先把“自定义数据 -> adapter -> 推理验证”跑通。

完成标准：

```text
你能用自己的数据训练出一个 adapter，并能说明它相比 base 模型在什么问题上更符合预期。
```

### 阶段 5：做真实项目需要补什么

真实项目不只跑通训练，还要补：

| 模块 | 要补什么                               |
| ---- | -------------------------------------- |
| 数据 | 数据来源、清洗规则、去重、质量抽检     |
| 评测 | 固定评测集、人工标准答案、自动评分     |
| 对比 | base 模型 vs prompt vs RAG vs 微调     |
| 安全 | 敏感数据、越权输出、幻觉、拒答         |
| 部署 | adapter 加载、导出、服务接口、回滚     |
| 成本 | 训练显存、训练时长、推理延迟、存储大小 |

## 第十八课：面试怎么讲

### 一分钟版本

```text
我在 A800 上用 ms-swift 跑通了 Qwen/Qwen3.5 的 QLoRA 学习实验。流程包括 ModelScope 下载模型、构造 messages 格式的文本和多模态 SFT 数据、用 4bit QLoRA 训练 LoRA adapter、记录训练日志和显存采样、保存 checkpoint，最后用 batch infer 对固定问题做推理验证。这个实验重点是学习和复现微调工程闭环，不是追求生产级模型效果。
```

### 三分钟版本

```text
这个实验我主要关注三个点。

第一是数据格式。我用 messages JSONL 组织文本 SFT 数据，多模态样本额外包含 images 字段，并用 <image> 标记图片位置。

第二是低风险训练。因为 A800 是共享环境，我在训练前检查 nvidia-smi 和磁盘空间，训练中采样显存，限制 batch size、max_length 和 max_steps，并用 QLoRA 降低显存占用。

第三是可复现验证。训练参数保存在 args.json，LoRA adapter 配置保存在 adapter_config.json，每步 loss 和 token_acc 保存在 logging.jsonl，最终用固定验证集通过 swift infer 批量推理，并保存结果。
```

### 面试官可能追问

| 问题                           | 回答要点                                                                         |
| ------------------------------ | -------------------------------------------------------------------------------- |
| 为什么用 QLoRA                 | 4bit 加载基座模型，显存低，只训练 adapter                                        |
| LoRA rank 是什么               | 低秩矩阵容量，越大表达能力越强但参数更多                                         |
| 为什么不只看 loss              | loss 只说明训练集拟合，真实效果要看验证集和业务任务                              |
| 为什么要记录显存               | 共享 GPU 环境要避免影响已有服务                                                  |
| adapter 能单独用吗             | 不能，必须和对应基座模型一起加载                                                 |
| 数据少有什么问题               | 容易过拟合，只能验证流程，不代表生产效果                                         |
| 模板为什么重要                 | messages 需要被模板转成模型认识的 prompt 格式                                    |
| ms-swift 和 LLaMA-Factory 区别 | ms-swift 更贴 Qwen/ModelScope/多模态工程，LLaMA-Factory WebUI 更适合入门理解流程 |

## 第十九课：你应该掌握到什么程度

### 入门合格

- [ ] 能解释 `messages` 数据格式。
- [ ] 能解释 SFT、LoRA、QLoRA。
- [ ] 能读懂 `swift sft` 的核心参数。
- [ ] 能找到训练产物和 checkpoint。
- [ ] 能看懂 loss、token_acc、显存日志。
- [ ] 能用 adapter 做批量推理验证。

### 面试可讲

- [ ] 能讲清楚为什么这次只用 20 步小数据。
- [ ] 能讲清楚共享 A800 上如何低风险训练。
- [ ] 能讲清楚为什么训练成功不等于模型效果好。
- [ ] 能讲清楚后续如何做真实评测。
- [ ] 能讲清楚 ms-swift 和 LLaMA-Factory 的适用场景。

### 项目可用

- [ ] 能构造自己的高质量 SFT 数据。
- [ ] 能设计固定验证集。
- [ ] 能对比 base 模型和微调模型。
- [ ] 能记录完整实验参数和日志。
- [ ] 能把 adapter 加载到推理服务里。
- [ ] 能评估训练成本和推理成本。

## 第二十课：推荐学习顺序清单

按这个顺序打开文件：

1. `README.md`
2. `data/text_sft.jsonl`
3. `data/mm_sft.jsonl`
4. `scripts/generate_data.py`
5. `docker/Dockerfile`
6. `scripts/download_models.sh`
7. `logs/model_downloads.log`
8. `scripts/train_rounds_inside.sh`
9. `logs/training_status.log`
10. `outputs/qwen3_17b_text/v1-20260613-211241/args.json`
11. `outputs/qwen3_17b_text/v1-20260613-211241/logging.jsonl`
12. `outputs/qwen3_17b_text/v1-20260613-211241/checkpoint-10/adapter_config.json`
13. `scripts/infer_rounds_inside.sh`
14. `logs/round1_qwen3_17b_text.infer.md`
15. `wsl_local_finetune_setup.md`

每打开一个文件，回答三件事：

1. 这个文件在训练流程中处于哪个阶段？
2. 这个文件最重要的 3 个字段或命令是什么？
3. 如果这个文件出错，会导致什么问题？

## 最后：你现在应该怎么做

你已经打开了 WebUI，下一步不要急着训练。按下面做：

1. 在 VS Code 左边打开 `train_rounds_inside.sh`。
2. 在浏览器 WebUI 打开 `LLM预训练/微调`。
3. 从脚本里找一个参数，就在 WebUI 中找到对应字段。
4. 先完成这 10 个字段：
   - `model`
   - `model_type`
   - `template`
   - `dataset`
   - `max_length`
   - `torch_dtype`
   - `per_device_train_batch_size`
   - `gradient_accumulation_steps`
   - `learning_rate`
   - `lora_rank`
   - `lora_alpha`
   - `quant_bits`
5. 打开 `args.json`，确认这些字段训练后如何被记录。
6. 打开 `logging.jsonl`，看训练中 loss 和 token_acc 如何变化。
7. 打开 `adapter_config.json`，看 LoRA adapter 最终配置。
8. 打开 `infer.md`，看推理验证是否真的跑过。

完成这一步后，你就不是“看过 WebUI”，而是理解了微调闭环。
