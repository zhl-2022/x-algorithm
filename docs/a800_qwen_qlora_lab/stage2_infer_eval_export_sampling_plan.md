# Stage 2 推理、评测、导出和采样学习计划

这份计划承接已经完成的 `Qwen3-1.7B effective QLoRA` 训练。目标不是继续训练，而是把训练后的 adapter 放进完整使用链路：

```text
checkpoint -> 推理 -> 固定评测 -> 采样对比 -> 可选导出 -> 复盘
```

## 0. 明天开始前先做就绪检查

在本地 PowerShell 执行：

```powershell
.\scripts\a800\check_stage2_learning_ready.ps1
```

这个命令会在 A800 上检查：

1. `/root/zhl/qwen-qlora-lab` 是否存在。
2. `models/Qwen3-1.7B/config.json` 是否存在。
3. `outputs/qwen3_17b_text_effective/.../checkpoint-*` 是否存在。
4. `qwen-qlora-swift:latest` 镜像是否存在。
5. 当前 `nvidia-smi` 和 `/root` 磁盘空间。

如果这个检查失败，先不要做推理和导出。

## 1. 准备明天学习用数据文件

在本地 PowerShell 执行：

```powershell
.\scripts\a800\prepare_stage2_learning_assets.ps1
```

它会在远端生成：

| 文件 | 用途 |
|---|---|
| `/root/zhl/qwen-qlora-lab/data/stage2_infer_prompts.jsonl` | 通用批量推理 prompts |
| `/root/zhl/qwen-qlora-lab/data/stage2_eval_cases.jsonl` | 带关键词规则的评测样例 |
| `/root/zhl/qwen-qlora-lab/data/stage2_sampling_prompts.jsonl` | 采样参数对比 prompts |

这些文件不是新的训练数据，而是明天推理和评测用的问题集。

## 2. 推理学习

先确保没有上次残留服务：

```powershell
.\scripts\a800\stop_qwen3_17b_deploy.ps1
```

启动 effective LoRA 推理服务：

```powershell
.\scripts\a800\start_qwen3_17b_effective_lora_deploy.ps1 -Port 19082
```

单条测试：

```powershell
.\scripts\a800\call_qwen3_17b_effective_lora.ps1 -Port 19082 -Prompt "请用简洁中文解释：A800 共享训练。"
```

批量测试：

```powershell
.\scripts\a800\run_stage2_openai_batch.ps1 -Port 19082 -Model qwen3-1.7b-qlora-effective
```

默认输出：

```text
/root/zhl/qwen-qlora-lab/logs/stage2_effective_lora_infer.jsonl
/root/zhl/qwen-qlora-lab/logs/stage2_effective_lora_infer.md
```

重点学习：

1. `messages` 格式如何变成模型输入。
2. `temperature=0.2` 时回答为什么更稳定。
3. 同一批 prompt 如何用于 base 和 LoRA 对比。
4. 推理服务为什么要记录端口、模型名、prompt、response 和参数。

## 3. 评测学习

运行关键词规则评测：

```powershell
.\scripts\a800\run_stage2_keyword_eval.ps1
```

默认读取：

```text
/root/zhl/qwen-qlora-lab/data/stage2_eval_cases.jsonl
/root/zhl/qwen-qlora-lab/logs/stage2_effective_lora_infer.jsonl
```

默认输出：

```text
/root/zhl/qwen-qlora-lab/logs/stage2_effective_lora_eval.md
/root/zhl/qwen-qlora-lab/logs/stage2_effective_lora_eval.csv
```

重点学习：

| 问题 | 你要理解什么 |
|---|---|
| 为什么不能只看 `train_loss` | 训练集拟合不等于泛化 |
| 为什么要固定评测 prompt | 否则每次对比对象不一致 |
| 为什么关键词评测很粗糙 | 它只能做入门 sanity check，不能替代人工和真实 benchmark |
| 为什么还要保存原始 response | 方便人工复盘错误类型 |

## 4. 采样学习

运行采样参数网格：

```powershell
.\scripts\a800\run_stage2_sampling_grid.ps1 -Port 19082 -Model qwen3-1.7b-qlora-effective
```

默认会测试多组参数，例如：

| 参数 | 值 |
|---|---|
| `temperature` | `0.0`、`0.2`、`0.8` |
| `top_p` | `0.8`、`0.95` |
| `max_tokens` | `256` |

默认输出：

```text
/root/zhl/qwen-qlora-lab/logs/stage2_sampling_grid.jsonl
/root/zhl/qwen-qlora-lab/logs/stage2_sampling_grid.md
```

重点学习：

1. `temperature` 越低，回答越稳定。
2. `temperature` 越高，表达更发散，但也更可能跑偏。
3. `top_p` 控制候选 token 的累计概率范围。
4. 学习/评测阶段优先用低温度，创作/头脑风暴才考虑高温度。

## 5. 导出学习

导出不是必须步骤。它会生成 merged model，磁盘占用明显大于 LoRA adapter。本地快照和 Git 都不要保存导出的模型权重。

如果只是学习概念，先看这个命令但不要执行：

```powershell
.\scripts\a800\export_qwen3_17b_effective_lora.ps1
```

它会提示需要确认参数。确认要导出时再执行：

```powershell
.\scripts\a800\export_qwen3_17b_effective_lora.ps1 -ConfirmExport
```

默认远端输出：

```text
/root/zhl/qwen-qlora-lab/exports/qwen3_17b_effective_merged/
```

你要理解三种形态：

| 形态 | 内容 | 适合场景 |
|---|---|---|
| base model | 原始 Qwen3-1.7B | 通用推理、作为基座 |
| LoRA adapter | 小体积增量权重 | 学习、实验、多 adapter 切换 |
| merged model | base + adapter 合并后的完整模型 | 部署简化，但占用更大 |

## 6. 建议的明天学习顺序

按这个顺序来：

1. `check_stage2_learning_ready.ps1`：确认环境。
2. `prepare_stage2_learning_assets.ps1`：生成学习 prompts。
3. 启动 effective LoRA 推理服务。
4. 单条 prompt 调用，确认服务可用。
5. 批量推理，保存结果。
6. 关键词评测，理解为什么评测不等于训练 loss。
7. 采样网格，对比不同 `temperature/top_p`。
8. 看导出脚本，理解 adapter 和 merged model 的区别。
9. 如果磁盘和显存都允许，再执行导出。
10. 停止推理容器，释放显存。

最后释放服务：

```powershell
.\scripts\a800\stop_qwen3_17b_deploy.ps1
```

## 7. 可选：把 A800 上的 ms-swift WebUI 映射到本地

如果你想直接打开服务器上的 `ms-swift WebUI`，使用：

```powershell
.\scripts\a800\start_a800_ms_swift_webui.ps1
```

默认会做两件事：

1. 在 A800 上启动 `qwen-qlora-swift-webui` 容器，远端端口是 `17860`。
2. 在本地 Windows 启动 SSH tunnel，把它映射到 `http://127.0.0.1:7861`。

查看状态：

```powershell
.\scripts\a800\status_a800_ms_swift_webui.ps1
```

停止 WebUI 和本地 tunnel：

```powershell
.\scripts\a800\stop_a800_ms_swift_webui.ps1
```

注意：WebUI 页面本身通常不明显占显存，但如果你在 WebUI 里启动训练、推理或导出，就会占用 A800。使用前仍然要先看 `nvidia-smi`，不要影响已有服务。

如果本地映射失败，并且 SSH 输出类似：

```text
channel 0: open failed: administratively prohibited: open failed
```

说明跳板机 `srv4` 禁用了 SSH TCP forwarding。这不是 `ms-swift` 的问题，也不是 Docker 端口问题。此时远端 WebUI 可以在 A800 上启动，但不能通过普通 `ssh -L` 映射到本地浏览器。需要管理员开放跳板机 TCP forwarding、安全网关端口，或者继续使用本地 WSL WebUI 学界面、A800 脚本跑训练/推理。

## 8. 明天看结果时优先打开哪些文件

远端文件：

```text
/root/zhl/qwen-qlora-lab/logs/stage2_effective_lora_infer.md
/root/zhl/qwen-qlora-lab/logs/stage2_effective_lora_eval.md
/root/zhl/qwen-qlora-lab/logs/stage2_sampling_grid.md
```

本地代码和说明：

```text
docs/a800_qwen_qlora_lab/stage2_infer_eval_export_sampling_plan.md
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage2_openai_batch.py
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage2_keyword_eval.py
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage2_sampling_grid.py
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/scripts/stage2_export_effective_lora.sh
```

不要把 `exports/` 里的模型权重同步到本地或提交到 Git。
