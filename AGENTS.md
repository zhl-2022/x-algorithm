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

当前已完成两批数据集实验：

1. `experiments/movielens_recall/`：MovieLens 1M 入门推荐闭环。
2. `experiments/mind_news/`：MIND 新闻推荐、MLU 放大训练和内容感知双塔实验。

当前进入第三批数据集：

- `experiments/kuairec_short_video/`
- 数据集：KuaiRec 短视频推荐数据。
- 目标：从电影推荐、新闻推荐进一步迁移到短视频/信息流推荐，重点研究观看时长、完播率、
  类别、多模态文本描述和高密度曝光矩阵下的离线评测。
- 2026-06-02 已在 srv4 的 `/root/zhl/x-algorithm` 下载并解压 KuaiRec，原始数据保存在
  `experiments/kuairec_short_video/data/raw/`，该目录已被 `.gitignore` 忽略。
- 第一版标签口径：`watch_ratio >= 1.0` 作为正反馈；第一轮 baseline 建议先用
  `small_matrix.csv` 跑通，再扩展到 `big_matrix.csv`。
- KuaiRec `small_matrix.csv` 第一轮完整训练已完成：Popularity、Category、ItemCF、MF、
  Two-Tower、DNN Ranker、Two-Tower + DNN Ranker pipeline。当前最佳 `NDCG@20`
  为 Two-Tower 的 `0.143577`。
- KuaiRec 阶段二已完成：标签阈值消融、`small_matrix.csv` 全量训练、`big_matrix.csv`
  采样放大、Two-Tower 与 Ranker 分数融合重排。当前阶段二最佳是
  `watch_ratio >= 0.8` 下 Two-Tower 的 `NDCG@20=0.153744`。
- KuaiRec 阶段三已完成 Ranker hard negative 优化：追加 141,100 条 Two-Tower 高分难负样本后，
  `DNNRanker NDCG@20=0.240050`，`TwoTower+DNN-Rerank@200 NDCG@20=0.203215`。
  当前小矩阵上的 Ranker 重排问题已解决，并已在后续阶段迁移到 `big_matrix.csv`。
- KuaiRec 阶段四到六已完成：`big_matrix.csv` hard negative、in-batch negative 和 MLU 单卡/双卡 benchmark。
  big 上 best pipeline `NDCG@20=0.005245`，仍低于 ItemCF `0.065921`；双卡吞吐 `908,159 samples/s`
  高于单卡 `723,335 samples/s`。
- 总体项目路线记录在 `docs/project_roadmap.md`，KuaiRec 阶段二总结记录在
  `experiments/kuairec_short_video/reports/stage2_summary_report.md`。
  KuaiRec 阶段三总结记录在
  `experiments/kuairec_short_video/reports/stage3_ranker_optimization_report.md`。
  KuaiRec 阶段四到六总结记录在
  `experiments/kuairec_short_video/reports/stage4_big_hardneg_report.md`、
  `experiments/kuairec_short_video/reports/stage5_big_inbatch_report.md` 和
  `experiments/kuairec_short_video/reports/stage6_mlu_ddp_report.md`。
- 当前进入阶段性项目收尾：总实验报告记录在 `docs/project_summary_report.md`，简历材料记录在
  `docs/resume_project_writeup.md`，面试问答记录在 `docs/interview_qa.md`。
- KuaiRec big 后续不建议继续盲目堆同类 MLP，默认优先设计 ItemCF 蒸馏 Two-Tower；
  详细方案记录在 `experiments/kuairec_short_video/reports/big_matrix_improvement_plan.md`。
- KuaiRec 第七轮补强实验已完成：ItemCF 蒸馏 Two-Tower、LightGCN、GRU 序列兴趣模型和
  TextCNN 双塔均已按“预处理缓存 + 分批启动”方式跑通。当前最有效的是 ItemCF 蒸馏 Two-Tower，
  `NDCG@20=0.033562`，高于 stage5 best pipeline `0.005245`，但仍低于 big ItemCF `0.065921`。
  后续若继续训练，优先扩大蒸馏样本和优化 teacher 权重。

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
3. 统计类 baseline 不需要 MLU；MF、Two-Tower、Ranker 和文本 encoder 训练优先迁移到 MLU。
4. 当前 MIND 训练脚本已跑通单进程 MLU；如需体现双卡能力，再做两卡
   `torchrun --nproc_per_node=2` 或 DDP。
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

1. MovieLens 1M：Popularity、ItemCF、MF、Two-Tower、DNN Ranker、两阶段 pipeline。
2. MIND-small：Popularity、Category、DNNRanker、ContentTwoTower、TwoTower+DNN-Rerank、MLU 放大实验。
3. KuaiRec：短视频推荐数据准备、观看反馈建模、Popularity/Category/Two-Tower/Ranker 迁移。
4. KuaiRec 阶段二：标签阈值、训练规模、候选集和融合重排消融。
5. KuaiRec 阶段三：Ranker hard negative 优化，使 `TwoTower+DNN-Rerank` 的 `NDCG@20` 超过单独 Two-Tower。
6. KuaiRec 阶段四到六：迁移 hard negative 到 `big_matrix.csv`、验证 in-batch negative、完成 MLU 双卡 benchmark。
7. KuaiRec 阶段七：ItemCF 蒸馏 Two-Tower、LightGCN、序列兴趣模型和轻量文本 encoder 中等规模验证。
8. 如需要继续优化 KuaiRec big 场景，优先扩大 ItemCF 蒸馏 Two-Tower，再考虑图召回和序列模型深调。
9. 如需要继续扩展，再考虑 Tenrec 或 KuaiRand。
10. 整理最终项目 README、实验报告、架构图、简历描述和面试问答。
