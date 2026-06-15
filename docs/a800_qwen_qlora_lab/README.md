# A800 Qwen QLoRA Lab

这个目录记录 A800 上 Qwen/Qwen3.5 QLoRA 学习训练的完整复盘、可复用 workflow、本地 WebUI 学习方式和远端轻量快照。

## 文件索引

| 文件 | 用途 |
|---|---|
| `a800_qwen_qlora_workflow.md` | 后续执行 A800 Qwen QLoRA 任务时优先遵循的项目级 workflow |
| `troubleshooting_and_workflow.md` | 本次训练过程中遇到的难题、根因和解决方式 |
| `beginner_finetune_learning_guide.md` | 面向小白的训练/微调完整学习路线，逐文件讲解数据、参数、日志、adapter 和推理 |
| `round1_qwen3_17b_text_walkthrough/` | Round 1 训练脚本、`args.json`、`logging.jsonl`、checkpoint 配置的逐行/逐字段讲解 |
| `local_webui_setup.md` | 在 Windows/WSL 本地打开 ms-swift 或 LLaMA-Factory WebUI 的学习指南 |
| `wsl_local_finetune_setup.md` | 本地 WSL 已安装微调 WebUI 环境的路径、启动命令和排错记录 |
| `local_snapshot_learning_guide.md` | 如何阅读本地同步的远端训练轻量快照 |
| `remote_snapshot/qwen-qlora-lab-small/` | A800 `/root/zhl/qwen-qlora-lab` 的轻量快照，不包含模型权重和 adapter 权重 |

## 远端项目

```text
/root/zhl/qwen-qlora-lab
```

远端已经完成三轮训练：

| 轮次 | 模型 | 结果 |
|---|---|---|
| Round 1 | `Qwen/Qwen3-1.7B` | 文本 QLoRA 成功 |
| Round 2 | `Qwen/Qwen3.5-0.8B` | 文本 QLoRA 成功 |
| Round 3 | `Qwen/Qwen3.5-2B` | 图文 QLoRA 成功 |

## 本地快照边界

本地快照保留：

- 脚本
- 数据样例
- 合成图片
- 训练日志
- 推理结果
- `args.json`
- 训练曲线图片
- checkpoint 元数据

本地快照不保留：

- 基座模型权重
- `adapter_model.safetensors`
- `*.pt`
- `*.bin`
- Docker 镜像层

## 后续任务入口

遇到新的 A800/Qwen/QLoRA 任务时，先读：

```text
docs/a800_qwen_qlora_lab/a800_qwen_qlora_workflow.md
```

如果要本地打开 WebUI 学习，读：

```text
docs/a800_qwen_qlora_lab/beginner_finetune_learning_guide.md
docs/a800_qwen_qlora_lab/local_webui_setup.md
docs/a800_qwen_qlora_lab/wsl_local_finetune_setup.md
```
