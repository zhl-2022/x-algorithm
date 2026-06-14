# WSL 本地微调 WebUI 环境实装指南

## 结论

本地 WSL 已安装两套独立学习环境：

| 环境 | 路径 | 用途 |
|---|---|---|
| `swift-env` | `/home/zhl/finetune-webui/swift-env` | 学习和复现 A800 上的 `ms-swift` Qwen/Qwen3.5 微调流程 |
| `llamafactory-env` | `/home/zhl/finetune-webui/llamafactory-env` | 打开 LLaMA-Factory 的 LlamaBoard WebUI，学习更直观的训练参数组织 |
| 源码目录 | `/home/zhl/finetune-webui/LLaMA-Factory` | LLaMA-Factory 官方仓库源码 |

这两套环境默认都是 CPU 版 PyTorch，适合本地打开 WebUI、学习参数、阅读数据格式和做极小样例试跑；不建议在本地 GTX 1650 4GB 上训练 Qwen3/Qwen3.5。正式 QLoRA 仍应放到 A800。

## 当前实装状态

| 项目 | 状态 |
|---|---|
| WSL 发行版 | Ubuntu 24.04.3 LTS |
| WSL 用户 | `zhl` |
| 本地 GPU 透传 | `NVIDIA GeForce GTX 1650, 4096 MiB` 可被 `nvidia-smi` 看到 |
| `ms-swift` | `4.3.0` |
| `ms-swift` 环境 PyTorch | `torch 2.12.0+cpu`，`torch.cuda.is_available() == False` |
| `ms-swift` 环境 Transformers | `5.10.2` |
| `ms-swift` 环境 Gradio | `5.50.0` |
| LLaMA-Factory | `0.9.6.dev0`，来自官方源码 |
| LLaMA-Factory 环境 PyTorch | `torch 2.12.0+cpu` |
| LLaMA-Factory 环境 Transformers | `5.6.0` |
| LLaMA-Factory 环境 Gradio | `5.50.0` |
| LLaMA-Factory 环境 Torchaudio | `2.11.0+cpu` |

## 从 Windows 一键检查环境

在本地 PowerShell 的仓库根目录执行：

```powershell
powershell -NoProfile -File scripts\wsl\check_finetune_webui_env.ps1
```

这个脚本会检查：

- WSL 基础信息。
- `nvidia-smi` 是否可见。
- `ms-swift`、`transformers`、`gradio`、`torch` 版本。
- `swift web-ui --help` 是否能正常返回。
- LLaMA-Factory 的 `llamafactory-cli env` 是否能正常返回。

## 打开 ms-swift WebUI

推荐从 Windows PowerShell 启动：

```powershell
powershell -NoProfile -File scripts\wsl\start_ms_swift_webui.ps1
```

然后在 Windows 浏览器打开：

```text
http://127.0.0.1:7860
```

如果想换端口：

```powershell
powershell -NoProfile -File scripts\wsl\start_ms_swift_webui.ps1 -Port 7870
```

如果默认 `7860` 已被占用，脚本会自动在 `7860-7899` 里寻找下一个 WSL 空闲端口，并在终端输出实际 URL。

等价的 WSL 手动命令：

```bash
cd ~/finetune-webui
source swift-env/bin/activate
swift web-ui --lang zh --server_name 127.0.0.1 --server_port 7860
```

## 打开 LLaMA-Factory WebUI

推荐从 Windows PowerShell 启动：

```powershell
powershell -NoProfile -File scripts\wsl\start_llamafactory_webui.ps1
```

然后在 Windows 浏览器打开：

```text
http://127.0.0.1:7861
```

如果想换端口：

```powershell
powershell -NoProfile -File scripts\wsl\start_llamafactory_webui.ps1 -Port 7871
```

如果默认 `7861` 已被占用，脚本会自动在 `7861-7899` 里寻找下一个 WSL 空闲端口，并在终端输出实际 URL。

等价的 WSL 手动命令：

```bash
cd ~/finetune-webui
source llamafactory-env/bin/activate
export GRADIO_SERVER_NAME=127.0.0.1
export GRADIO_SERVER_PORT=7861
llamafactory-cli webui
```

注意：LLaMA-Factory 的 WebUI 命令没有普通 `--help` 行为，执行 `llamafactory-cli webui` 会直接进入界面启动流程。

## 本地如何学习微调

建议按这个顺序学习：

1. 先打开 LLaMA-Factory WebUI，看 `Train`、`Evaluate & Predict`、`Chat`、`Export` 四类页面如何组织。
2. 再打开 `ms-swift` WebUI，对照 A800 训练脚本理解 Qwen/Qwen3.5 的参数。
3. 不下载大模型，只使用本仓库里的轻量快照阅读数据和日志。
4. 如果一定要本地试跑，只用 tiny 模型和几十条样例；CPU 会很慢，不要把它当成正式训练结果。

重点对照这些文件：

| 学习内容 | 本地文件 |
|---|---|
| 文本 SFT 数据格式 | `docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/data/text_sft.jsonl` |
| 多模态 SFT 数据格式 | `docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/data/mm_sft.jsonl` |
| A800 三轮训练命令 | `docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/train_rounds_inside.sh` |
| A800 推理验证命令 | `docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/infer_rounds_inside.sh` |
| 训练过程问题复盘 | `docs/a800_qwen_qlora_lab/troubleshooting_and_workflow.md` |
| 后续远端 workflow | `docs/a800_qwen_qlora_lab/a800_qwen_qlora_workflow.md` |

WSL 里访问本仓库路径：

```bash
cd /mnt/e/programs/x-algorithm
```

## 本地不建议做的事

| 不建议 | 原因 |
|---|---|
| 在本地下载完整 Qwen3/Qwen3.5 权重 | 占用磁盘且本地显存不足以正式 QLoRA |
| 在 WebUI 里直接填 A800 的 `/root/zhl/...` 绝对路径 | WSL 本地没有这些远端路径 |
| 用本地 GTX 1650 4GB 微调 1.7B/2B Qwen | 4GB 显存不足，且当前环境是 CPU PyTorch |
| 把 `models/`、`outputs/`、adapter 权重提交到 Git | 权重大、不可复现、已经被项目规则排除 |
| 在仓库根目录长期运行 LLaMA-Factory WebUI | 它可能生成 `llamaboard_cache/`，脚本已改为在 `~/finetune-webui` 下运行 |

## 已处理的安装问题

| 问题 | 现象 | 处理 |
|---|---|---|
| PyTorch CUDA wheel 过大 | `ms-swift` 初次解析时尝试下载 CUDA 版 `torch` 和 `nvidia-*` 依赖 | 改为 CPU PyTorch，WebUI 学习优先 |
| `ms-swift 2.2.5` 与数据集依赖不兼容 | `cannot import name 'LargeList' from 'datasets'` | 显式升级到 `ms-swift==4.3.0` |
| `torchvision::nms` 缺失 | `torch 2.12.0+cpu` 与旧 `torchvision 0.25.0+cpu` 不匹配 | 升级为 `torchvision 0.27.0+cpu` |
| Gradio 与 Hugging Face Hub 不兼容 | `cannot import name 'HfFolder'` | 升级为 `gradio 5.50.0` |
| LLaMA-Factory WebUI 导入 CUDA 版 torchaudio | 缺少 `libcudart.so.13` | 替换为 `torchaudio 2.11.0+cpu` |
| 复制 venv 后激活路径错误 | `activate` 仍指向旧 `swift-env` | 修正 `llamafactory-env/bin` 中的 venv 路径 |
| WebUI 默认端口被占用 | `Cannot find empty port in range: 7860-7860` | 关闭占用进程，或使用脚本自动切到空闲端口 |
| Gradio startup-events 返回 502 | `http://127.0.0.1:7860/gradio_api/startup-events failed (code 502)` | 在启动脚本中为 `127.0.0.1`、`localhost`、`::1` 显式设置 `NO_PROXY/no_proxy` |
| WSL 内自动打开浏览器失败 | `gio: http://127.0.0.1:7860/: Operation not supported` | 在启动脚本中设置 `BROWSER=/bin/true`，改用 Windows 浏览器打开 URL |

### 端口占用排查

如果看到类似错误：

```text
OSError: Cannot find empty port in range: 7860-7860
```

说明 WSL 里已经有进程监听这个端口。查看方式：

```powershell
wsl -- ss -ltnp
```

只看常用 WebUI 端口：

```powershell
wsl -- bash -lc "ss -ltnp | grep -E ':(7860|7861|7862|7870)\b' || true"
```

如果确认是旧的测试进程，可以在 WSL 里按 PID 停掉：

```bash
kill <PID>
```

也可以直接启动脚本并让它自动选择空闲端口：

```powershell
powershell -NoProfile -File scripts\wsl\start_ms_swift_webui.ps1
```

### Gradio 502 排查

如果 WebUI 已经打印：

```text
* Running on local URL:  http://127.0.0.1:7860
```

随后又报：

```text
Couldn't start the app because 'http://127.0.0.1:7860/gradio_api/startup-events' failed (code 502)
```

通常是 WSL 里的代理变量把 `127.0.0.1` 请求转发到了代理端口。检查方式：

```powershell
wsl -- bash -lc "env | grep -iE 'proxy|no_proxy'"
```

启动脚本已经显式设置：

```bash
export NO_PROXY="127.0.0.1,localhost,::1,$NO_PROXY"
export no_proxy="127.0.0.1,localhost,::1,$no_proxy"
export BROWSER=/bin/true
```

因此直接重新运行脚本即可：

```powershell
powershell -NoProfile -File scripts\wsl\start_ms_swift_webui.ps1
```

## 以后如果要改成本地 GPU 训练

当前没有安装本地 CUDA 训练环境，这是有意保守选择。若后续确实要用 WSL + GTX 1650 做小模型试跑，应新建第三个环境，不要覆盖这两套 WebUI 学习环境：

```bash
cd ~/finetune-webui
uv venv tiny-gpu-env --python 3.11
source tiny-gpu-env/bin/activate
```

然后只选择极小模型或 tiny random 模型验证流程。Qwen3/Qwen3.5 的真实训练继续使用 A800。

## 参考入口

- ms-swift WebUI：<https://swift.readthedocs.io/en/latest/GetStarted/Web-UI.html>
- ms-swift GitHub：<https://github.com/modelscope/ms-swift>
- LLaMA-Factory WebUI：<https://llamafactory.readthedocs.io/en/latest/getting_started/webui.html>
- LLaMA-Factory GitHub：<https://github.com/hiyouga/LLaMA-Factory>
