# 推荐系统简历项目总路线

## 1. 项目定位

本项目不是简单跑一个推荐模型，而是围绕 X/Twitter 推荐系统的多阶段思想，复现一个可解释、
可评测、可迁移到公司 MLU 服务器的推荐系统训练闭环。

最终简历卖点应该是：

> 基于 X/Twitter 开源推荐系统架构，完成 MovieLens、MIND、KuaiRec 三类公开数据集上的推荐实验，
> 覆盖数据处理、召回、排序、两阶段 pipeline、离线评测、MLU 训练适配和实验分析。

## 2. 当前不是在训练 Grok/LLM 基座

当前项目的“底座模型”不是 Transformer，也不是 Grok/LLM 类大语言模型，而是推荐系统里的表征学习底座：

| 模型 | 架构 | 作用 |
|---|---|---|
| `MF` | 用户 embedding + 物品 embedding 点积 + bias | 学习最基础的用户-物品协同信号 |
| `Two-Tower` | 用户塔 MLP + 内容塔 MLP + 向量点积 | 企业推荐中常见的向量召回底座 |
| `DNNRanker` | ID embedding + 类别/文本/统计特征 + MLP | 对候选内容做点击或完播概率排序 |
| `TwoTower+DNN-Rerank` | 双塔召回 TopK + Ranker 重排 | 模拟真实推荐系统的多阶段 pipeline |

如果后续想结合 Grok/X 方向，合理路线不是从零训练 LLM，而是把新闻标题、短视频 caption、
用户历史行为编码成推荐表征，构造一个小型内容推荐基座模型。

## 3. 数据集路线

| 阶段 | 数据集 | 场景 | 当前状态 | 核心价值 |
|---|---|---|---|---|
| 1 | MovieLens 1M | 电影推荐 | 已完成 | 跑通推荐系统最小闭环 |
| 2 | MIND-small | 新闻推荐 | 已完成 | 引入曝光候选、文本特征和内容推荐 |
| 3 | KuaiRec `small_matrix` | 短视频推荐 | 已完成两轮 | 引入观看时长、完播率、类别和 caption |
| 4 | KuaiRec `big_matrix` | 大规模短视频推荐 | 已完成多轮补强实验 | 观察大候选池、训练规模和神经召回瓶颈 |
| 5 | Tenrec / KuaiRand | 更大规模信息流 | 待定 | 用于最终规模化证明 |

## 4. 已完成关键结论

| 阶段 | 最重要结论 |
|---|---|
| MovieLens | 两阶段 pipeline 已超过 Popularity、ItemCF、MF 和单独 Two-Tower，说明召回 + 排序闭环有效。 |
| MIND | 内容感知 Two-Tower 在 `NDCG@10` 上强于 DNNRanker，说明新闻文本和类别特征对内容推荐有效。 |
| KuaiRec 第一轮 | `Two-Tower` 在 `small_matrix.csv` 上达到当前最佳 `NDCG@20=0.143577`，短视频双塔召回有效。 |
| KuaiRec 阶段二 | `watch_ratio >= 0.8` 和全量 `small_matrix` 训练都提升了双塔，但 Ranker 重排仍未稳定超过 Two-Tower。 |
| KuaiRec 阶段三 | Ranker hard negative 训练生效，`TwoTower+DNN-Rerank@200 NDCG@20=0.203215`，已超过单独 Two-Tower。 |
| KuaiRec `big_matrix` | hard negative 与 in-batch 都有局部收益，但 ItemCF 仍最强，说明需要图召回、序列建模或蒸馏。 |
| KuaiRec 升级实验 | ItemCF 蒸馏 Two-Tower 在 big 上达到 `NDCG@20=0.033562`，明显高于 stage5 best pipeline `0.005245`。 |
| KuaiRec 阶段八 | 蒸馏双塔接 DNNRanker 后达到 `NDCG@20=0.044560`，优于单独蒸馏双塔，pipeline 精调有效。 |
| KuaiRec 阶段九 | `2m_t40n120` 达到 `NDCG@20=0.048158`，说明降低 teacher 占比、增加随机负样本比单纯加 teacher 更有效。 |
| KuaiRec 最终收尾 | soft label 蒸馏精调达到 `NDCG@20=0.055883`，换 seed 复跑 `0.052947`，KuaiRec 模型实验收尾。 |
| MLU 工程 | 单卡/双卡 DDP benchmark 已跑通，双卡吞吐约比单卡高 25.6%。 |

## 5. KuaiRec 后续路线

当前优先级不是继续无目的加模型，而是解决三个明确问题：

1. **让 Ranker 重排超过 Two-Tower**
   - 已完成：通过 Two-Tower hard negative 训练，`TwoTower+DNN-Rerank@200` 的 `NDCG@20`
     达到 `0.203215`，超过单独 `Two-Tower` 的 `0.159630`。

2. **提升 `big_matrix.csv` 上的神经 TopK**
   - 已尝试 hard negative 和 in-batch negative，best pipeline `NDCG@20=0.005245`，仍低于 ItemCF `0.065921`。
   - 已完成 ItemCF 蒸馏 Two-Tower、LightGCN、序列模型和轻量文本 encoder 中等规模验证。
   - 阶段八显示，2M 单独蒸馏 `NDCG@20=0.027320` 低于 800k 蒸馏，但蒸馏 pipeline 达到 `0.044560`。
   - 阶段九显示，`2m_t40n120` pipeline 达到 `0.048158`，后续应围绕负样本比例和 teacher soft label 继续精调。
   - 阶段十和阶段十一已完成最终精调，best pipeline 达到 `0.055883`，复跑达到 `0.052947`。当前 KuaiRec 模型实验不再继续加轮次。

3. **补齐双卡 MLU 工程能力**
   - 已完成：单卡 `723,335 samples/s`，双卡 `908,159 samples/s`。
   - 可作为简历中的 MLU 多卡训练适配与性能记录。

## 6. 最终简历表达方向

可以写成：

> 基于 X/Twitter 推荐系统多阶段架构，构建公开数据集推荐训练与评测闭环，完成 MovieLens、MIND、
> KuaiRec 三类场景实验。实现 Popularity、ItemCF、MF、Two-Tower、DNNRanker 和 TwoTower+DNN-Rerank
> pipeline，使用 `Recall@K`、`NDCG@K`、`AUC`、`LogLoss` 等指标评估召回与排序效果。
> 在寒武纪 MLU 服务器上完成推荐模型训练适配，记录 batch size、训练吞吐、显存占用和模型效果，
> 并通过标签阈值、候选集大小、训练样本规模等消融分析优化推荐效果。

## 7. 阶段性收尾产物

当前不建议立刻切换第四批数据集。优先将现有三批实验沉淀为可讲清楚、可追溯、可写进简历的材料：

| 文档 | 用途 |
|---|---|
| `docs/project_summary_report.md` | 汇总 MovieLens、MIND、KuaiRec 三批数据集的目标、模型、指标和结论 |
| `docs/project_architecture.md` | 记录项目总架构图、离线推荐 pipeline、KuaiRec 优化链路和 MLU 训练架构 |
| `docs/experiment_reading_guide.md` | 说明从哪里开始看实验、重点看哪些报告和代码 |
| `docs/resume_project_writeup.md` | 提供简历标题、项目描述、可直接使用的简历要点和面试开场讲法 |
| `docs/interview_qa.md` | 梳理面试中高频问题，例如 AUC 与 TopK 差异、hard negative、ItemCF 为什么强 |
| `experiments/kuairec_short_video/reports/big_matrix_improvement_plan.md` | 设计 KuaiRec big 后续补强方案，默认优先 ItemCF 蒸馏 Two-Tower |
| `experiments/kuairec_short_video/reports/upgrade_experiments_status.md` | 记录 KuaiRec big 四个升级实验的分批调度状态、正式指标和结论 |
| `experiments/kuairec_short_video/reports/stage8_recall_boost_report.md` | 记录 2M 蒸馏、蒸馏 pipeline、LightGCN 调参和序列修正结果 |
| `experiments/kuairec_short_video/reports/stage9_pipeline_tuning_plan.md` | 记录下一轮蒸馏 pipeline 精调的三组实验和成功标准 |
| `experiments/kuairec_short_video/reports/stage9_pipeline_tuning_report.md` | 记录阶段九三组 pipeline 精调指标和结论 |
| `experiments/kuairec_short_video/reports/stage10_soft_label_tuning_report.md` | 记录 soft label 蒸馏精调的四组最终消融 |
| `experiments/kuairec_short_video/reports/stage11_final_kuairec_report.md` | 记录最终复跑和 KuaiRec 收尾结论 |

KuaiRec 数据集当前已完成模型实验收尾。后续如果继续训练，建议切换到 Tenrec 或 KuaiRand；如果目标是简历完整度，则先完善总 README、架构图和面试材料。
