# 推荐系统项目简历写法

## 1. 推荐简历标题

**开源推荐系统训练与评测闭环项目 | X/Twitter 推荐架构学习复现**

## 2. 一句话版本

基于 X/Twitter 推荐系统多阶段架构，完成 MovieLens、MIND、KuaiRec 三类公开数据集上的推荐训练与离线评测闭环，实现召回、排序、重排、指标分析和寒武纪 MLU 训练适配。

## 3. 简历项目描述

基于 X/Twitter 推荐系统多阶段架构，构建公开数据集推荐训练与评测闭环，覆盖数据清洗、样本构造、召回、排序、重排、离线评测和 MLU 训练适配。使用 MovieLens 1M、MIND-small、KuaiRec 等数据集复现电影推荐、新闻推荐和短视频推荐场景，实现 Popularity、ItemCF、MF、Two-Tower、DNNRanker 和 TwoTower+DNN-Rerank pipeline，并基于 `Recall@K`、`NDCG@K`、`AUC`、`LogLoss`、训练吞吐等指标进行系统评估。

## 4. 可直接放进简历的要点

- 构建多数据集推荐实验 pipeline，完成 MovieLens、MIND、KuaiRec 三类场景的数据处理、训练样本构造、召回、排序、重排和离线评测。
- 实现 Popularity、ItemCF、MF、Two-Tower、DNNRanker 和 TwoTower+DNN-Rerank 等模型，对比 `Recall@K`、`NDCG@K`、`AUC`、`LogLoss` 等指标。
- 在 MovieLens 1M 上完成入门推荐闭环，TwoTower+DNN-Rerank 的 `NDCG@20=0.042451`，超过 Popularity、ItemCF、MF 和单独 Two-Tower。
- 在 MIND-small 新闻推荐中引入标题/摘要哈希文本 embedding，ContentTwoTower 在 MLU 放大实验中达到 `NDCG@10=0.374560`，高于 DNNRanker 的 `0.353507`。
- 在 KuaiRec 短视频推荐中基于 `watch_ratio` 构造完播/接近完播标签，验证 `watch_ratio >= 0.8` 比严格完播更适合当前 TopK 推荐任务。
- 通过 Two-Tower hard negative 优化 Ranker，使 KuaiRec small 上 `DNNRanker NDCG@20=0.240050`，TwoTower+DNN-Rerank@200 达到 `NDCG@20=0.203215`，超过单独 Two-Tower。
- 在 KuaiRec big 上完成 hard negative、in-batch negative、ItemCF 蒸馏、LightGCN、序列兴趣模型和轻量文本 encoder 对比，并通过蒸馏召回 + Ranker pipeline 将神经 `NDCG@20` 提升到 `0.044560`。
- 在公司寒武纪 MLU 服务器上完成推荐模型训练适配和双卡 DDP benchmark，Two-Tower in-batch 训练单卡吞吐 `723,335 samples/s`，双卡吞吐 `908,159 samples/s`。

## 5. 面试开场讲法

这个项目不是只跑一个推荐模型，而是我按工业推荐系统的多阶段流程做了一个离线训练和评测闭环。先用 MovieLens 跑通用户-物品推荐，再用 MIND 引入新闻文本和曝光候选，最后用 KuaiRec 迁移到短视频信息流场景，并在 MLU 服务器上完成训练适配和吞吐测试。项目中我重点比较了召回模型、排序模型和两阶段 pipeline 的差异，也分析了 `AUC` 高但 TopK 弱、hard negative 为什么有效、big 数据下为什么 ItemCF 仍然强于当前神经召回这些问题。

## 6. 面试中应避免的表述

| 不建议这样说 | 推荐说法 |
|---|---|
| 我复现了 X/Twitter 的完整推荐系统 | 我参考 X/Twitter 多阶段推荐思想，在公开数据集上复现了训练评测闭环 |
| 我训练了 Grok 类基座模型 | 当前底座是推荐系统表征模型，包括 MF、Two-Tower 和 DNNRanker |
| 神经模型一定比传统模型强 | KuaiRec big 上 ItemCF 更强，说明召回训练和协同信号仍需补强 |
| 只强调 AUC 很高 | 同时解释 `AUC` 与 `NDCG@K` 的差异，并说明最终推荐更看重 TopK |

## 7. 可以继续升级的方向

1. 精调 ItemCF 蒸馏 Two-Tower + DNNRanker pipeline，优化 teacher 权重、hard negative 配比和 `candidate_k`。
2. 深调 LightGCN 的层数、采样策略和 BPR 训练轮数，判断图召回是否能追上 ItemCF。
3. 修正和增强用户序列兴趣模型，加入最近观看类别、时间窗口和更强的负采样。
4. 接入更强的轻量文本 encoder，替换当前哈希 caption embedding。
5. 如果需要继续扩大数据规模，再迁移到 Tenrec 或 KuaiRand。
