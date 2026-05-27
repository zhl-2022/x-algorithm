# 项目协作说明

## 初始目的

本项目用于系统学习 X/Twitter 开源推荐系统相关代码，并在公开数据集上复现一个
可解释、可评测、可写入简历的推荐系统训练闭环。

目标不是完整复刻 X 的生产推荐系统，而是围绕其多阶段推荐架构，逐步完成：

1. 数据处理与行为样本构造。
2. 召回、排序、评测的最小可运行 pipeline。
3. MovieLens、MIND、KuaiRec 或 Tenrec 等公开数据集实验。
4. Popularity、ItemCF、MF、Two-Tower、DNN Ranker 等模型对比。
5. 在公司 MLU 服务器上完成训练适配、性能记录和实验分析。
6. 沉淀可复现实验脚本、指标报告和最终简历项目描述。

## 当前阶段

当前重点是 `experiments/movielens_recall/`：

- 使用 MovieLens 1M 作为入门数据集。
- 将 `rating >= 4` 定义为正反馈。
- 按用户时间序列切分训练集、验证集和测试集。
- 已完成 Popularity baseline 与 `Recall@20` 评测。

## 服务器与 MLU 训练环境

本地 PowerShell 中 `srv` 是连接公司寒武纪服务器的快捷函数，当前等价于 `srv4`。
在 Codex 非交互检查中优先使用下面的非交互命令：

```powershell
powershell -NoProfile -File C:\Users\zhl\Documents\WindowsPowerShell\CodexServerWorkflow\Invoke-KlbServer.ps1 -Target srv4 -Command "hostname"
```

已验证的服务器状态：

| 项目 | 当前信息 |
|---|---|
| 目标 | `srv4` |
| 主机名 | `node2` |
| 用户 | `root` |
| 设备 | 8 张 `MLU590-H8`，单卡 80GB |
| 设备监控 | `/usr/bin/cnmon` |
| Docker 镜像 | `cambricon-base/pytorch:v25.12.0-torch2.9.1-torchmlu1.30.2-ubuntu22.04-py310` |
| PyTorch/torch_mlu | 容器内 `torch 2.9.1`、`torch_mlu 1.30.2` |
| 当前建议训练卡 | `MLU_VISIBLE_DEVICES=2,3` |
| 项目远端路径 | `/root/zhl/x-algorithm` |

2026-05-27 检查时，Card `2` 和 Card `3` 显存占用均为 `0 MiB/81920 MiB`，
适合作为当前两卡训练目标。Card `0`、`1`、`4`、`5`、`6`、`7` 上有推理或
embedding 服务占用，训练前必须重新运行 `cnmon` 确认空闲状态。

当前项目默认使用新版寒武纪 PyTorch 镜像，不默认使用 `klb/llamafactory-mlu:v1`。
后者更适合 LLaMAFactory 大模型微调，而当前项目主要训练推荐模型。

本地检查命令：

```powershell
powershell -NoProfile -File scripts\mlu\check_srv4_mlu.ps1
```

服务器上推荐的交互式训练容器入口：

```bash
cd /root/zhl/x-algorithm
bash scripts/mlu/start_xalgorithm_mlu.sh
```

如果需要保留后台开发容器：

```bash
cd /root/zhl/x-algorithm
bash scripts/mlu/start_xalgorithm_mlu.sh --detached
docker exec -it xalgorithm-mlu bash
source /torch/venv3/pytorch/bin/activate
```

训练使用建议：

1. 训练前执行 `cnmon`，确认 Card `2`、`3` 仍然空闲。
2. 使用 `MLU_VISIBLE_DEVICES=2,3` 限定训练进程只看到两张空闲卡。
3. 当前 MovieLens Popularity baseline 不需要 MLU；等实现 MF 或 Two-Tower 后再迁移训练。
4. 先做单卡 MLU 跑通，再做两卡 `torchrun --nproc_per_node=2` 或 DDP。
5. 训练日志至少记录：命令、模型、数据集、batch size、embedding dim、训练耗时、显存占用、`Recall@K`、`NDCG@K`。
6. `srv3` 当前存在 SSH host key changed 警告，不自动清理 `known_hosts`，除非人工确认后再使用。
7. 容器方案细节记录在 `docs/mlu_training_container.md`。

## 协作规则

1. 文档默认使用中文，保留必要英文术语，例如 `Recall@20`、`Two-Tower`、`ItemCF`。
2. 创建或编辑 Markdown 时遵循 `markdown-format-standard`。
3. 涉及代码、脚本、实验流程改动时遵循 `engineering-workflow-standard`。
4. 做代码审查、风险评估或质量门禁时遵循 `code-review-findings-first`。
5. 不随意改动上游 X algorithm 源码，优先在 `experiments/` 和 `docs/` 下做学习实验和记录。
6. 不提交原始数据集、处理后的大文件和模型产物；这些文件应保留在被忽略目录中。
7. 每次声称实验成功前必须实际运行命令，并在回复中列出关键验证结果。

## 提交工作流

默认不做无提示自动 commit，避免把用户临时改动、数据文件或异常工作区状态误提交。
当用户明确要求“提交”或“commit”时，按以下流程执行：

1. 运行 `git status --short`，确认工作区范围。
2. 只暂存本轮任务相关文件，不把无关改动、原始数据、缓存和模型产物纳入提交。
3. 提交前运行本轮任务相关的最小验证命令。
4. 使用简洁的 Conventional Commits 风格提交信息，例如 `chore: add recsys baseline and MLU workflow`。
5. 提交后运行 `git status --short` 和 `git log --oneline -3`，确认工作区干净并记录 commit hash。

如果用户要求推送 GitHub 或创建 PR，再使用 GitHub 相关工作流；本地普通提交不默认 push。

## 推荐执行顺序

1. MovieLens 1M：Popularity baseline。
2. MovieLens 1M：ItemCF。
3. MovieLens 1M：Matrix Factorization。
4. MovieLens 1M：Two-Tower 召回。
5. MIND-small：新闻推荐迁移实验。
6. MLU 单卡训练适配。
7. MLU 双卡训练和性能对比。
8. 整理最终项目 README、实验报告和简历描述。
