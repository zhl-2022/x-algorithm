# Round 1 Qwen3-1.7B 文本 QLoRA 文件逐行讲解

## 这个目录讲什么

这个目录专门解释 Round 1：

```text
round1_qwen3_17b_text
```

对应训练任务：

| 项目 | 值 |
|---|---|
| 模型 | `models/Qwen3-1.7B` |
| 数据 | `data/text_sft.jsonl` |
| 输出目录 | `outputs/qwen3_17b_text` |
| 实际版本目录 | `outputs/qwen3_17b_text/v1-20260613-211241` |
| 中间 checkpoint | `checkpoint-10` |
| 最终 checkpoint | `checkpoint-20` |

这不是泛泛讲微调，而是围绕你点名的这些文件逐个解释：

| 原始文件 | 讲解文档 |
|---|---|
| `scripts/train_rounds_inside.sh` | `01_train_script_line_by_line.md` |
| `outputs/qwen3_17b_text/v1-20260613-211241/args.json` | `02_args_and_checkpoint_configs.md` |
| `outputs/qwen3_17b_text/v1-20260613-211241/logging.jsonl` | `03_logging_jsonl_line_by_line.md` |
| `checkpoint-10/adapter_config.json` | `02_args_and_checkpoint_configs.md` |
| `checkpoint-10/additional_config.json` | `02_args_and_checkpoint_configs.md` |
| `checkpoint-10/args.json` | `02_args_and_checkpoint_configs.md` |
| `checkpoint-10/trainer_state.json` | `02_args_and_checkpoint_configs.md` |

## 先理解这些文件之间的关系

Round 1 的流程可以这样看：

```text
train_rounds_inside.sh
  -> 调用 swift sft
    -> 读取 models/Qwen3-1.7B
    -> 读取 data/text_sft.jsonl
    -> 每步写 logging.jsonl
    -> 第 10 步保存 checkpoint-10
    -> 第 20 步保存 checkpoint-20
```

训练产物分两层：

| 层级 | 文件 | 作用 |
|---|---|---|
| 本轮输出目录 | `args.json` | 记录本轮训练的完整参数 |
| 本轮输出目录 | `logging.jsonl` | 记录每一步训练指标 |
| checkpoint 目录 | `adapter_config.json` | 记录 LoRA adapter 配置 |
| checkpoint 目录 | `additional_config.json` | 记录少量额外 LoRA 参数 |
| checkpoint 目录 | `args.json` | checkpoint 内复制的一份训练参数 |
| checkpoint 目录 | `trainer_state.json` | 记录保存 checkpoint 时 Trainer 的状态 |

我已经检查过：

```text
v1-20260613-211241/args.json
checkpoint-10/args.json
```

这两个文件内容完全一致。你可以理解为：根目录的 `args.json` 是本轮实验总参数，checkpoint 里的 `args.json` 是为了让这个 checkpoint 单独拿出来时也能知道它当时怎么训练出来的。

## 学习顺序

建议你按这个顺序看：

1. 先看 `01_train_script_line_by_line.md`，理解训练脚本每一行在干什么。
2. 再看 `03_logging_jsonl_line_by_line.md`，理解训练过程中每一步发生了什么。
3. 最后看 `02_args_and_checkpoint_configs.md`，理解框架自动保存的配置和 checkpoint 元数据。

不要先硬啃 367 行 `args.json`。它是框架快照，不是手写配置文件。先掌握核心字段，再知道其他字段大多是默认值或高级功能。

