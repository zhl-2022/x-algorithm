# A800 Qwen QLoRA 本地快照学习指南

## 快照位置

本地已同步一个轻量快照：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/
```

这个快照保留了脚本、数据、图片、训练日志、推理结果和 checkpoint 元数据；没有同步：

- `models/` 基座模型权重。
- `adapter_model.safetensors` LoRA 权重文件。
- `*.pt`、`*.bin` 这类训练状态大文件。

原因是本地主要用于观察训练流程、参数、日志和数据格式；真正复跑训练仍应在 A800 上完成。

## 推荐学习顺序

### 1. 先看项目总览

| 文件 | 重点 |
|---|---|
| `README.md` | 项目目录、三轮训练目标、基本启动方式 |
| `docker/Dockerfile` | 为什么从 LLaMA-Factory 镜像派生、安装了哪些包、兼容 shim 怎么处理 |

### 2. 再看脚本入口

| 文件 | 重点 |
|---|---|
| `scripts/start_pipeline.sh` | 后台启动方式、显存门槛、日志路径 |
| `scripts/run_pipeline_inside.sh` | 数据生成、模型下载、训练、推理的流水线顺序 |
| `scripts/docker_run.sh` | 容器如何挂载项目目录、缓存目录和 GPU |
| `scripts/status.sh` | 如何汇总显存、模型、checkpoint、日志 |

### 3. 学训练前准备

| 文件 | 重点 |
|---|---|
| `scripts/generate_data.py` | 文本 SFT 和多模态 SFT 的 JSONL 格式 |
| `scripts/download_models.sh` | ModelScope 优先、Hugging Face 备用、模型文件校验 |
| `scripts/verify_env.sh` | 容器外环境验证流程 |
| `scripts/verify_env_inside.sh` | 容器内 `torch`、`cuda`、`swift`、数据文件验证 |

### 4. 重点看训练脚本

| 文件 | 重点 |
|---|---|
| `scripts/train_rounds_inside.sh` | 三轮训练参数、QLoRA 配置、显存采样、checkpoint 检测 |
| `outputs/*/args.json` | ms-swift 实际记录下来的训练参数 |
| `outputs/*/logging.jsonl` | 每步训练 loss、学习率、吞吐等记录 |
| `outputs/*/images/*.png` | 训练曲线图，例如 loss、learning rate、token accuracy |

重点关注这些参数：

- `--quant_bits 4`
- `--tuner_backend peft`
- `--lora_rank 8`
- `--lora_alpha 32`
- `--target_modules all-linear`
- `--per_device_train_batch_size 1`
- `--gradient_accumulation_steps 8`
- `--max_length 1024`

### 5. 看数据格式

| 文件 | 重点 |
|---|---|
| `data/text_sft.jsonl` | 文本 SFT 的 `messages` 格式 |
| `data/mm_sft.jsonl` | 多模态 SFT 的 `messages + images` 格式 |
| `data/infer_text_prompts.jsonl` | 批量文本推理验证数据 |
| `data/infer_mm_prompts.jsonl` | 批量图片推理验证数据 |
| `images/*.png` | 本地合成图表如何参与多模态训练和推理 |

### 6. 看推理验证

| 文件 | 重点 |
|---|---|
| `scripts/infer_rounds_inside.sh` | 为什么不用交互式 `swift infer`，而用 `--val_dataset --result_path` |
| `logs/round1_qwen3_17b_text.infer.md` | Round 1 推理摘要 |
| `logs/round2_qwen35_08b_text.infer.md` | Round 2 推理摘要 |
| `logs/round3_qwen35_2b_mm.infer.md` | Round 3 文本和图片推理摘要 |
| `logs/*.result.jsonl` | 每条样本的模型输出 |

重点理解：

```bash
swift infer \
  --adapters <checkpoint> \
  --val_dataset <jsonl> \
  --result_path <result.jsonl> \
  --stream false \
  --max_new_tokens 256 \
  --max_batch_size 1
```

这比交互式 `swift infer` 更适合服务器自动验证。

### 7. 看训练日志和显存日志

| 文件 | 重点 |
|---|---|
| `logs/training_status.log` | 每轮训练是否成功、显存峰值、最终 checkpoint |
| `logs/model_downloads.log` | 模型下载和目录大小 |
| `logs/round*.before.nvidia-smi.txt` | 每轮训练前显存状态 |
| `logs/round*.after.nvidia-smi.txt` | 每轮训练后显存状态 |
| `logs/round*.gpu_sample.csv` | 训练过程中显存采样 |
| `logs/round*.train.log` | ms-swift 训练完整 stdout/stderr |

## 不建议优先看的内容

| 内容 | 原因 |
|---|---|
| `adapter_model.safetensors` | 是 LoRA 权重，二进制文件，本地阅读价值低；已从轻量快照排除 |
| 基座模型 `models/` | 体积大，且本地无法直接训练 A800 任务 |
| Docker 镜像层 | 体积大，不适合放进项目仓库 |
| `pipeline_inside.log` 和长下载进度日志 | 主要是 ModelScope 进度条，已从轻量快照排除；学习时看 `model_downloads.log` 即可 |

## 如果以后要同步完整 adapter

完整包在 A800 上已经生成过：

```text
/tmp/qwen-qlora-lab-study.zip
```

大小约 `160M`，包含 LoRA `adapter_model.safetensors`。如果只是学习流程，不需要它；如果要本地保留完整 adapter 产物，再单独同步。

## 本地目录导航

```text
docs/a800_qwen_qlora_lab/
├── troubleshooting_and_workflow.md
├── local_snapshot_learning_guide.md
├── local_webui_setup.md
├── a800_qwen_qlora_workflow.md
└── remote_snapshot/
    └── qwen-qlora-lab-small/
        ├── README.md
        ├── docker/
        ├── scripts/
        ├── data/
        ├── images/
        ├── logs/
        └── outputs/
```
