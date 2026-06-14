# 本地 Windows 学习 Qwen 微调 WebUI 指南

## 结论

本地 Windows 可以打开微调框架的 WebUI 学习参数和流程，但不建议直接在原生 Windows 上做正式 QLoRA 训练。

推荐优先级：

1. **只学习 UI 和参数**：Windows 原生或 WSL 都可以。
2. **本地轻量试跑**：优先 WSL2 + NVIDIA CUDA。
3. **正式训练 Qwen/Qwen3.5**：仍建议在 A800 上跑。

原因：

- Windows 原生环境对 `bitsandbytes`、CUDA、PyTorch、FlashAttention、多模态依赖的兼容性更容易出问题。
- WSL2 更接近 Linux 服务器环境，和 A800 Docker/脚本习惯一致。
- 你的本地机器即使能打开 WebUI，也未必有足够显存做 QLoRA；但可以用它学习参数含义和数据格式。

## ms-swift WebUI

官方入口：

- 文档：<https://swift.readthedocs.io/en/latest/GetStarted/Web-UI.html>
- GitHub：<https://github.com/modelscope/ms-swift>

启动命令：

```bash
swift web-ui --lang zh
```

或：

```bash
SWIFT_UI_LANG=zh swift web-ui
```

官方说明里有两个要点：

1. WebUI 是命令行的高层封装，界面里的参数和 `swift sft`、`swift infer` 基本对应。
2. WebUI 启动训练/部署时会在系统里启动独立后台进程；关闭 UI 不一定会停止后台任务。

### 适合学习什么

| 学习点 | 对应本项目文件 |
|---|---|
| 模型路径怎么填 | `remote_snapshot/qwen-qlora-lab-small/outputs/*/args.json` |
| 数据集格式 | `remote_snapshot/qwen-qlora-lab-small/data/*.jsonl` |
| QLoRA 参数 | `scripts/train_rounds_inside.sh` |
| 推理验证参数 | `scripts/infer_rounds_inside.sh` |
| 输出目录和日志 | `logs/training_status.log`、`outputs/*/logging.jsonl` |

## LLaMA-Factory WebUI

官方入口：

- 文档：<https://llamafactory.readthedocs.io/en/latest/getting_started/webui.html>
- GitHub：<https://github.com/hiyouga/LLaMA-Factory>

启动命令：

```bash
llamafactory-cli webui
```

LLaMA-Factory 的 WebUI 主要分成：

- Training
- Evaluation & Prediction
- Chat
- Export

它更适合初学者理解“模型、数据集、训练方法、超参、输出目录、导出”之间的关系。

## 两个框架怎么选

| 场景 | 建议 |
|---|---|
| 学习微调界面和参数含义 | 先看 LLaMA-Factory |
| 学 Qwen3.5、多模态、ModelScope 生态 | 看 ms-swift |
| 在 A800 上复用这次实验 | 用 ms-swift |
| 文本 SFT 入门、调参可视化 | LLaMA-Factory 更顺手 |
| 多模态 QLoRA 和新 Qwen 适配 | ms-swift 优先 |

## Windows 原生安装建议

Windows 原生只建议用于打开 WebUI 和阅读参数，不建议正式训练。

### ms-swift 轻量安装

```powershell
conda create -n swift-ui python=3.11 -y
conda activate swift-ui
pip install -U ms-swift modelscope gradio
swift web-ui --lang zh
```

如果遇到 CUDA、`bitsandbytes`、`torch` 相关报错，不要在 Windows 原生环境里继续深挖，切到 WSL。

### LLaMA-Factory 轻量安装

```powershell
conda create -n llamafactory-ui python=3.11 -y
conda activate llamafactory-ui
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
llamafactory-cli webui
```

如果只是学习 WebUI，可以先不下载大模型。

## WSL2 安装建议

WSL2 更接近服务器环境，推荐作为本地学习环境。

### 1. 检查 NVIDIA 透传

在 WSL 里执行：

```bash
nvidia-smi
```

能看到显卡后再继续。

### 2. 创建 Python 环境

```bash
conda create -n swift-ui python=3.11 -y
conda activate swift-ui
```

### 3. 安装 ms-swift

```bash
pip install -U ms-swift modelscope gradio
swift web-ui --lang zh
```

### 4. 安装 LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
llamafactory-cli webui
```

## 本地学习时不要做的事

1. 不要把 A800 上的 `models/` 基座模型目录完整拉到本地。
2. 不要在本地 WebUI 里误填 A800 的绝对路径，例如 `/root/zhl/qwen-qlora-lab/models/...`。
3. 不要把本地 WebUI 当成正式训练环境；本地主要用于理解参数和数据流。
4. 不要把 `adapter_model.safetensors`、基座权重、Docker 镜像层提交到 Git。

## 推荐练习

1. 打开 `ms-swift` WebUI，只观察字段，不启动训练。
2. 对照 `scripts/train_rounds_inside.sh`，在 UI 里找到这些参数：
   - `model`
   - `dataset`
   - `tuner_backend`
   - `quant_bits`
   - `lora_rank`
   - `lora_alpha`
   - `max_length`
   - `output_dir`
3. 打开 LLaMA-Factory WebUI，观察 Training、Chat、Export 三个页面。
4. 对照 `data/text_sft.jsonl` 和 `data/mm_sft.jsonl`，理解数据集注册和消息格式。
