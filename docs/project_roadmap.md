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
| 4 | KuaiRec `big_matrix` | 大规模短视频推荐 | 已完成采样实验 | 观察大候选池和训练规模带来的工程问题 |
| 5 | Tenrec / KuaiRand | 更大规模信息流 | 待定 | 用于最终规模化证明 |

## 4. 已完成关键结论

| 阶段 | 最重要结论 |
|---|---|
| MovieLens | 两阶段 pipeline 已超过 Popularity、ItemCF、MF 和单独 Two-Tower，说明召回 + 排序闭环有效。 |
| MIND | 内容感知 Two-Tower 在 `NDCG@10` 上强于 DNNRanker，说明新闻文本和类别特征对内容推荐有效。 |
| KuaiRec 第一轮 | `Two-Tower` 在 `small_matrix.csv` 上达到当前最佳 `NDCG@20=0.143577`，短视频双塔召回有效。 |
| KuaiRec 阶段二 | `watch_ratio >= 0.8` 和全量 `small_matrix` 训练都提升了双塔，但 Ranker 重排仍未稳定超过 Two-Tower。 |
| KuaiRec 阶段三 | Ranker hard negative 训练生效，`TwoTower+DNN-Rerank@200 NDCG@20=0.203215`，已超过单独 Two-Tower。 |
| KuaiRec `big_matrix` | ItemCF 当前 TopK 最强，神经模型 AUC 高但 TopK 弱，说明大候选池下需要更强负采样和召回训练。 |

## 5. KuaiRec 后续路线

当前优先级不是继续无目的加模型，而是解决三个明确问题：

1. **让 Ranker 重排超过 Two-Tower**
   - 已完成：通过 Two-Tower hard negative 训练，`TwoTower+DNN-Rerank@200` 的 `NDCG@20`
     达到 `0.203215`，超过单独 `Two-Tower` 的 `0.159630`。

2. **提升 `big_matrix.csv` 上的神经 TopK**
   - 当前 `big_matrix` 采样实验中 ItemCF 的 `NDCG@20=0.058148` 最好。
   - 神经模型 AUC 高但 TopK 弱，后续应做 in-batch negative、hard negative 或全量候选召回训练。

3. **补齐双卡 MLU 工程能力**
   - 当前训练是单进程 MLU，主要使用容器逻辑 MLU0。
   - 后续可把神经训练改造成 DDP，使用 `torchrun --nproc_per_node=2`。
   - 成功标准：记录单卡/双卡吞吐、显存和指标对比。

## 6. 最终简历表达方向

可以写成：

> 基于 X/Twitter 推荐系统多阶段架构，构建公开数据集推荐训练与评测闭环，完成 MovieLens、MIND、
> KuaiRec 三类场景实验。实现 Popularity、ItemCF、MF、Two-Tower、DNNRanker 和 TwoTower+DNN-Rerank
> pipeline，使用 `Recall@K`、`NDCG@K`、`AUC`、`LogLoss` 等指标评估召回与排序效果。
> 在寒武纪 MLU 服务器上完成推荐模型训练适配，记录 batch size、训练吞吐、显存占用和模型效果，
> 并通过标签阈值、候选集大小、训练样本规模等消融分析优化推荐效果。
