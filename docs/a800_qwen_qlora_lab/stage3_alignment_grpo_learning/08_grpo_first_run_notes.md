# 08 GRPO First Run Notes

本文件记录 Stage 3 第一轮 GRPO 小训练结果。它的定位是“链路验证 + 指标学习”，不是最终对齐效果报告。

## 1. 运行信息

| 项目 | 值 |
|---|---|
| 日期 | `2026-06-17` |
| 远端项目 | `/root/zhl/qwen-qlora-lab` |
| 本地快照 | `docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/` |
| 训练类型 | `swift rlhf --rlhf_type grpo` |
| 基座模型 | `models/Qwen3-1.7B` |
| 起始 adapter | `outputs/stage3_qwen3_17b_dpo/v0-20260616-031625/checkpoint-30` |
| reference adapter | `outputs/stage3_qwen3_17b_dpo/v0-20260616-031625/checkpoint-30` |
| 数据 | `data/stage3_grpo_prompts.jsonl` |
| 数据条数 | `8` |
| 最大步数 | `20` |
| 每个 prompt 生成数 | `num_generations=4` |

本轮特意从 DPO v1 起步，而不是从 correction DPO 起步，因为 correction DPO 已经暴露出重复、幻觉和 `top_p` 概念错误残留。

## 2. 输出位置

远端：

```text
/root/zhl/qwen-qlora-lab/logs/stage3_grpo.train.log
/root/zhl/qwen-qlora-lab/logs/stage3_reward_selftest.log
/root/zhl/qwen-qlora-lab/outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20
```

本地轻量快照：

```text
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/logs/stage3_grpo.train.log
docs/a800_qwen_qlora_lab/remote_snapshot/qwen-qlora-lab-small/outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/
```

注意：本地快照只同步了日志、`args.json`、`logging.jsonl`、`trainer_state.json`、`adapter_config.json` 和曲线图，不包含 `adapter_model.safetensors`。

## 3. 本轮解决的环境问题

| 问题 | 处理方式 |
|---|---|
| A800 Docker 默认 NVIDIA runtime 创建新容器失败 | 改用 `--runtime=runc`，手动挂载 `/dev/nvidia*` 和必要驱动库 |
| 容器内 `bitsandbytes/triton` 找不到 `libcuda.so` | 在容器启动后执行 `ldconfig` |
| 容器内缺少 `msgspec` | 在训练脚本里用阿里云 pip 源做最小依赖安装 |
| 容器内没有 `nvidia-smi` | 训练脚本改用 `torch.cuda.mem_get_info()` 写显存快照 |

这些是共享服务器环境问题，不是 GRPO 算法问题。不能通过重启 Docker 或停止其他服务来解决，因为 A800 上还有已有业务服务。

## 4. 关键训练参数

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `--rlhf_type` | `grpo` | 使用 GRPO 后训练算法 |
| `--adapters` | DPO v1 checkpoint | 当前被训练的策略模型起点 |
| `--ref_adapters` | DPO v1 checkpoint | reference，用于 KL 约束 |
| `--external_plugins` | `scripts/stage3_reward_plugin.py` | 加载自定义规则 reward |
| `--reward_funcs` | `stage3_quality` | 使用本阶段自定义评分函数 |
| `--num_generations` | `4` | 每个 prompt 生成 4 条回答做组内比较 |
| `--temperature` | `0.7` | 保留一定探索，但不让采样过散 |
| `--top_p` | `0.9` | nucleus sampling 候选集合阈值 |
| `--beta` | `0.04` | 约束策略不要偏离 reference 太远 |
| `--learning_rate` | `1e-5` | 比 SFT 更保守 |
| `--lora_rank` | `16` | 本轮 GRPO adapter 容量 |
| `--max_completion_length` | `256` | 限制生成长度和显存 |
| `--use_vllm` | `false` | 首轮优先稳定，不引入 vLLM 复杂度 |

## 5. Reward 自检结果

本轮训练前 reward selftest：

```json
{"scores": [0.82, 0.0, 0.47, 0.3, 0.0]}
```

这 5 个分数对应：

| 样例 | 分数 | 解释 |
|---|---:|---|
| 好的 A800 安全回答 | `0.82` | 覆盖检查、隔离和不影响服务 |
| 危险地建议直接停止其他服务 | `0.0` | 命中 forbidden |
| 关键词堆砌 | `0.47` | 有关键词但解释质量不足 |
| 重复解释 QLoRA | `0.3` | 被反重复惩罚压低 |
| 错误解释 `top_p` | `0.0` | 命中概念错误惩罚 |

最低标准已经满足：reward 能区分好答案、危险答案、重复答案和概念错误答案。

## 6. 关键日志结果

| 指标 | 本轮结果 | 解读 |
|---|---:|---|
| `global_step/max_steps` | `20/20` | 小步训练完整跑完 |
| `epoch` | `2.5` | 8 条数据被重复采样多轮 |
| `train_runtime` | `241.5763s` | 约 4 分钟 |
| `train_loss` | `0.00024549` | GRPO loss 不能按 SFT loss 直接理解 |
| `reward` 最小值 | `0.25` | 训练中最低平均 reward |
| `reward` 最大值 | `1.0` | 有 batch 生成了规则高分回答 |
| `reward` 平均值 | `0.4884` | 本轮 reward 中等，未证明效果变好 |
| `reward_std=0` 步数 | `3/20` | 有 3 个 step 组内回答无区分度 |
| `kl` 最大值 | `0.02504611` | 没有看到明显 KL 爆炸 |
| `kl` 平均值 | `0.006138` | 策略整体没有大幅偏离 reference |
| `completions/clipped_ratio` 最大值 | `0.25` | 第 9 step 有 1/4 回答触顶 256 token |
| `completions/mean_length` 平均值 | `74.72` | 生成长度整体可控 |
| 日志显存 | `4.35 GiB` | 训练日志记录的额外显存规模较小 |
| 最终 checkpoint | `checkpoint-20` | 远端存在，本地不保存权重 |

本轮能下的结论：

1. GRPO 训练链路已经跑通。
2. 自定义 reward 已经进入训练日志，字段为 `rewards/Stage3QualityORM/mean`。
3. KL 没有异常爆炸。
4. 部分 step 的 `reward_std=0` 说明有些 prompt 的 4 条回答没有被 reward 拉开，这会削弱 GRPO 学习信号。
5. 仅凭本轮日志不能证明 GRPO 后的模型比 DPO v1 更好，必须做固定 prompt 推理评测。

## 7. 你应该重点看哪些文件

| 文件 | 学习重点 |
|---|---|
| `logs/stage3_grpo.train.log` | 训练命令、模型加载、reward/loss/KL 逐步输出 |
| `logs/stage3_reward_selftest.log` | reward 规则是否能区分好坏 |
| `outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/args.json` | 所有 GRPO 参数最终展开值 |
| `outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/logging.jsonl` | 每一步结构化指标 |
| `outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/checkpoint-20/trainer_state.json` | 最终训练状态 |
| `outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/images/train_reward.png` | reward 曲线 |
| `outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/images/train_kl.png` | KL 曲线 |
| `outputs/stage3_qwen3_17b_grpo/v1-20260617-063053/images/train_completions_mean_length.png` | 生成长度变化 |

## 8. 下一步严格要求

现在不能直接说“GRPO 有提升”。下一步必须完成：

- [x] 部署或批量推理 GRPO `checkpoint-20`。
- [x] 用同一批 `stage3_alignment_eval_cases.jsonl` 对比 effective SFT、DPO v1、GRPO。
- [x] 至少抽查 8 个 prompt，每个 prompt 记录人工判断。
- [x] 找出 3 个 GRPO 失败案例，补充到 `failure_cases.md`。
- [x] 判断失败来自数据、reward、采样参数还是训练步数。
- [ ] 如果出现关键词堆砌，继续降低关键词覆盖权重，并提高概念解释质量规则。

本轮固定 prompt 评测已经完成，结论是：

| 模型状态 | cases | passed | average_score |
|---|---:|---:|---:|
| effective SFT | `8` | `1` | `0.2125` |
| DPO v1 | `8` | `1` | `0.2188` |
| GRPO v1 | `8` | `2` | `0.3063` |

严格判断：

1. GRPO v1 只在 `grpo_sampling` 上有明显改善。
2. GRPO v1 在 `grpo_dpo_format` 上出现严重重复，说明它不是稳定提升。
3. 下一步不能直接加大 GRPO 训练步数，应先扩充 failure-driven preference 数据，并加强反重复和概念质量 reward。

进入下一轮 GRPO 前，你要能回答：

1. 哪些 prompt 的 `reward_std=0`，为什么 reward 没拉开？
2. 高 reward 回答是否真的比低 reward 回答更好？
3. `top_p`、QLoRA、A800 共享训练这些概念错误是否减少？
4. GRPO 是否引入新的模板化或重复问题？
5. 下一轮是改 reward，还是先扩充 DPO/GRPO 数据？
