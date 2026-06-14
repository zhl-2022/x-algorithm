# A800 Qwen QLoRA 项目级 Workflow

## 触发场景

当任务包含以下关键词时，优先使用本 workflow：

- A800
- Qwen、Qwen3、Qwen3.5
- QLoRA、LoRA、SFT
- ms-swift
- ModelScope 下载模型
- 多模态微调
- 共享 GPU 服务器训练

## 先读文件

每次执行前先读：

1. `docs/a800_qwen_qlora_lab/troubleshooting_and_workflow.md`
2. `docs/a800_qwen_qlora_lab/local_snapshot_learning_guide.md`
3. `docs/a800_qwen_qlora_lab/local_webui_setup.md`

如果任务涉及远端已有产物，再看：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/
```

## 远端边界

当前 A800 实验目录：

```text
/root/zhl/qwen-qlora-lab
```

本地只保留轻量快照：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/
```

不要把以下内容同步或提交到 Git：

- `/root/zhl/qwen-qlora-lab/models/`
- `adapter_model.safetensors`
- `*.pt`
- `*.bin`
- Docker 镜像层
- 原始大模型权重

## A800 连接方式

优先使用本地 workflow：

```powershell
powershell -NoProfile -File C:\Users\zhl\Documents\WindowsPowerShell\CodexServerWorkflow\Invoke-KlbServer.ps1 -Target a800 -Command "hostname"
```

复杂 SSH 命令可使用已验证的双层 SSH：

```powershell
ssh srv4 "ssh -o StrictHostKeyChecking=no root@10.100.1.3 'hostname; nvidia-smi'"
```

注意 PowerShell 引号陷阱：

- 避免未转义 `$()`。
- 避免未转义 `$!`。
- 远端变量容易被本地 PowerShell 提前展开。
- 复杂命令优先写成远端脚本，或使用绝对路径分步执行。

## 训练前检查

必须执行：

```bash
nvidia-smi
df -h /root
```

判断标准：

- 剩余显存建议大于 `20GB`。
- `/root` 剩余磁盘建议大于 `60GB`。
- 记录现有 GPU 进程。
- 不停止已有 vLLM、MinerU、Milvus 等业务服务。

## 默认训练原则

共享 A800 上默认保守配置：

```bash
CUDA_VISIBLE_DEVICES=0
--quant_bits 4
--tuner_backend peft
--lora_rank 8
--lora_alpha 32
--target_modules all-linear
--torch_dtype bfloat16
--per_device_train_batch_size 1
--gradient_accumulation_steps 8
--max_length 1024
```

如果显存峰值超过 `16GB`，下一轮优先改：

1. `--max_length 512`
2. 保持 `--per_device_train_batch_size 1`
3. 减少样本数
4. 换更小模型

## 模型下载规则

ModelScope 优先，Hugging Face 备用。

下载后必须检查：

```bash
config.json
tokenizer.json 或 tokenizer_config.json
*.safetensors
du -sh <model_dir>
```

## 训练验证规则

每轮至少保留：

- 训练命令
- stdout/stderr 日志
- 训练前 `nvidia-smi`
- 训练后 `nvidia-smi`
- 显存采样 CSV
- 最终 checkpoint 路径

远端现有汇总脚本：

```bash
bash /root/zhl/qwen-qlora-lab/scripts/status.sh
```

## 推理验证规则

不要用交互式 `swift infer` 做自动化验证，容易卡住。

使用 batch 模式：

```bash
swift infer \
  --adapters <checkpoint> \
  --val_dataset <jsonl> \
  --result_path <result.jsonl> \
  --stream false \
  --max_new_tokens 256 \
  --max_batch_size 1
```

多模态验证必须确认日志里出现：

```text
features: ['messages', 'images']
```

## 本地学习规则

本地 Windows/WSL 用于学习 WebUI 和参数，不作为正式训练环境。

推荐：

- ms-swift：`swift web-ui --lang zh`
- LLaMA-Factory：`llamafactory-cli webui`

正式训练仍放在 A800。

## 完成后报告

最终回复必须包含：

1. 训练是否完成。
2. checkpoint 路径。
3. 显存峰值或显存增量。
4. 当前显存是否回到基线。
5. 推理验证结果路径。
6. 未做或失败的验证项。
