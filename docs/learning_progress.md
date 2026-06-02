# 推荐系统学习进度

## 学习目标

基于 X/Twitter 开源推荐系统架构，搭建一个企业级推荐系统学习项目，覆盖数据处理、
召回、排序、离线评测、训练环境适配和实验报告沉淀。最终目标是形成可复现项目成果，
并能在简历和面试中清楚说明技术价值。

## 当前进度

| 日期 | 阶段 | 状态 | 产出 |
|---|---|---|---|
| 2026-05-27 | MovieLens 1M 最小闭环 | 已完成 | 数据处理、统计报告、Popularity baseline、`Recall@20` |
| 2026-06-01 | MovieLens 1M ItemCF baseline | 已完成 | ItemCF 召回、统一结果表、`Recall@20` 对比 |
| 2026-06-01 | MovieLens 1M MF baseline | 已完成 | PyTorch MF 训练脚本、MLU 容器训练验证、`Recall@20` 对比 |
| 2026-06-01 | MovieLens 1M Two-Tower 召回 | 已完成 | 双塔召回训练脚本、MLU 训练结果、与 MF/ItemCF 对比 |
| 2026-06-01 | MovieLens 1M DNN Ranker 排序 | 已完成 | 排序模型训练脚本、AUC/LogLoss、TopK 对比 |
| 2026-06-01 | Two-Tower + DNN Ranker 两阶段 pipeline | 已完成 | Top 200 召回候选、Ranker 重排 Top 20、pipeline 指标 |
| 2026-06-01 | Candidate K 消融实验 | 已完成 | `candidate_k=50/100/200/500` 对比，分析召回上限和重排效果 |
| 2026-06-01 | MIND-small 数据准备 | 已完成 | RecZoo 镜像下载、新闻元数据、样本 CSV、数据报告 |
| 2026-06-02 | MIND Popularity baseline | 已完成 | 新闻热度排序、AUC/MRR/NDCG@K、MIND 基线报告 |
| 2026-06-02 | MIND 本地完整实验闭环 | 已完成 | Category、DNNRanker、ContentTwoTower、TwoTower+DNN-Rerank、统一结果表 |
| 2026-06-02 | MIND MLU 训练验证 | 已完成 | srv4 容器训练、MLU 结果表、训练记录报告 |
| 2026-06-02 | MIND MLU 放大与 Candidate K 消融 | 已完成 | 1M/500k 训练评估、文本哈希 encoder、`candidate_k=10/20/50/100` 对比 |
| 2026-06-02 | KuaiRec 短视频推荐实验启动 | 已完成 | 第三批数据集方案、数据下载、数据盘点、全套 baseline 与 MLU 训练 |

## 已完成内容

1. 新建 `experiments/movielens_recall/` 作为第一个推荐实验目录。
2. 编写 MovieLens 1M 下载脚本。
3. 编写数据准备脚本，将 `rating >= 4` 定义为正反馈。
4. 按用户时间序列切分训练集、验证集和测试集。
5. 生成数据统计报告。
6. 实现 Popularity baseline。
7. 输出 `Recall@20`、`HitRate@20`、`Precision@20`、`NDCG@20` 和 `Coverage@20`。
8. 实现 ItemCF baseline，并输出模型对比结果。
9. 启动 `xalgorithm-mlu` 后台容器，验证 `torch_mlu` 可见 2 张 MLU。
10. 实现 Matrix Factorization baseline，并在 MLU 上完成训练和评估。
11. 实现 Two-Tower 召回模型，并在 MLU 上完成训练和评估。
12. 实现 DNN Ranker 排序模型，并在 MLU 上完成训练和评估。
13. 实现 Two-Tower 召回候选集 + DNN Ranker 重排的两阶段 pipeline。
14. 完成候选集大小消融实验，验证不同 `candidate_k` 对最终 Top20 效果的影响。
15. 新建 `experiments/mind_news/`，完成 MIND-small 新闻推荐数据准备和数据报告。
16. 实现 MIND Popularity baseline，在验证集曝光候选列表内按新闻点击热度排序并完成评估。
17. 实现 MIND Category baseline、DNN Ranker、内容感知 Two-Tower 和 Two-Tower + DNN Ranker pipeline。
18. 将 MIND 神经模型同步到 srv4 的 `xalgorithm-mlu` 容器，在 `MLU_VISIBLE_DEVICES=2,3` 下完成训练验证。
19. 在 srv4 MLU 上完成 MIND 1M 训练样本、500k 验证样本放大实验，加入标题/摘要哈希文本 embedding，并完成 `candidate_k=10/20/50/100` 消融。
20. 新建 `experiments/kuairec_short_video/`，开始第三批 KuaiRec 短视频推荐实验。
21. 编写 KuaiRec 下载脚本、数据盘点脚本和面向小白的试验方案。
22. 在 srv4 的 `xalgorithm-mlu` 容器内下载并解压 KuaiRec 数据，生成第一版数据盘点报告。
23. 完成 KuaiRec `small_matrix.csv` 第一轮完整训练：Popularity、Category、ItemCF、MF、Two-Tower、DNN Ranker 和 Two-Tower + DNN Ranker pipeline。

## 当前实验结果

| 指标 | 数值 |
|---|---:|
| 原始评分数 | 1,000,209 |
| 用户数 | 6,040 |
| 电影元数据数 | 3,883 |
| 正反馈交互数 | 575,281 |
| 训练集正反馈数 | 563,209 |
| 验证集正反馈数 | 6,035 |
| 测试集正反馈数 | 6,037 |
| Popularity `Recall@20` | 0.067252 |
| Popularity `HitRate@20` | 0.067252 |
| Popularity `NDCG@20` | 0.026149 |
| ItemCF `Recall@20` | 0.082160 |
| ItemCF `HitRate@20` | 0.082160 |
| ItemCF `NDCG@20` | 0.032793 |
| ItemCF `Coverage@20` | 0.132372 |
| MF `Recall@20` | 0.104025 |
| MF `HitRate@20` | 0.104025 |
| MF `NDCG@20` | 0.039807 |
| MF `Coverage@20` | 0.481844 |
| Two-Tower `Recall@20` | 0.101872 |
| Two-Tower `HitRate@20` | 0.101872 |
| Two-Tower `NDCG@20` | 0.040244 |
| Two-Tower `Coverage@20` | 0.340973 |
| DNN-Ranker `Recall@20` | 0.106841 |
| DNN-Ranker `HitRate@20` | 0.106841 |
| DNN-Ranker `NDCG@20` | 0.041302 |
| DNN-Ranker `Coverage@20` | 0.525882 |
| DNN-Ranker `AUC` | 0.899737 |
| DNN-Ranker `LogLoss` | 0.138917 |
| TwoTower+DNN-Rerank `Candidate Recall@200` | 0.471757 |
| TwoTower+DNN-Rerank `Recall@20` | 0.108829 |
| TwoTower+DNN-Rerank `HitRate@20` | 0.108829 |
| TwoTower+DNN-Rerank `NDCG@20` | 0.041877 |
| TwoTower+DNN-Rerank `Coverage@20` | 0.435230 |
| Best Rerank Ablation `candidate_k` | 50 |
| Best Rerank Ablation `Recall@20` | 0.109988 |
| Best Rerank Ablation `NDCG@20` | 0.042451 |
| MIND-small 新闻数 | 65,238 |
| MIND-small 训练样本行数 | 5,843,444 |
| MIND-small 验证样本行数 | 2,740,998 |
| MIND-small 训练 CTR | 4.0446% |
| MIND-small 验证 CTR | 4.0636% |
| MIND Popularity `AUC` | 0.522252 |
| MIND Popularity `MRR` | 0.266074 |
| MIND Popularity `NDCG@5` | 0.247261 |
| MIND Popularity `NDCG@10` | 0.308465 |
| MIND Popularity `HitRate@5` | 0.423433 |
| MIND Popularity `HitRate@10` | 0.611986 |
| MIND Popularity `Coverage@5` | 0.017061 |
| MIND Popularity `Coverage@10` | 0.026380 |
| MIND Category `AUC` full | 0.588720 |
| MIND Category `NDCG@10` full | 0.338507 |
| MIND DNNRanker `AUC` 1M/500k, MLU | 0.592749 |
| MIND DNNRanker `NDCG@10` 1M/500k, MLU | 0.353507 |
| MIND ContentTwoTower `AUC` 1M/500k, MLU | 0.616641 |
| MIND ContentTwoTower `NDCG@10` 1M/500k, MLU | 0.374560 |
| MIND TwoTower+DNN-Rerank@10 `AUC` 1M/500k, MLU | 0.602720 |
| MIND TwoTower+DNN-Rerank@10 `NDCG@10` 1M/500k, MLU | 0.362443 |
| MIND best `candidate_k` | 10 |
| MIND DNNRanker 训练吞吐 | 755,916 samples/s |
| MIND ContentTwoTower 训练吞吐 | 541,957 samples/s |
| MIND Card 2 采样峰值显存 | 224 MiB |
| MIND Card 3 采样峰值显存 | 0 MiB |
| KuaiRec `big_matrix` 交互数 | 12,530,806 |
| KuaiRec `big_matrix` 用户数 | 7,176 |
| KuaiRec `big_matrix` 视频数 | 10,728 |
| KuaiRec `big_matrix` 平均 `watch_ratio` | 0.944506 |
| KuaiRec `big_matrix` `watch_ratio >= 1.0` 比例 | 33.82% |
| KuaiRec `small_matrix` 交互数 | 4,676,570 |
| KuaiRec `small_matrix` 用户数 | 1,411 |
| KuaiRec `small_matrix` 视频数 | 3,327 |
| KuaiRec `small_matrix` 平均 `watch_ratio` | 0.907069 |
| KuaiRec `small_matrix` `watch_ratio >= 1.0` 比例 | 32.40% |
| KuaiRec Popularity `NDCG@20` | 0.055724 |
| KuaiRec Category `NDCG@20` | 0.083583 |
| KuaiRec ItemCF `NDCG@20` | 0.097525 |
| KuaiRec MF `Recall@20` | 0.025962 |
| KuaiRec MF `NDCG@20` | 0.127836 |
| KuaiRec Two-Tower `Recall@20` | 0.026051 |
| KuaiRec Two-Tower `NDCG@20` | 0.143577 |
| KuaiRec DNNRanker `AUC` | 0.656796 |
| KuaiRec DNNRanker `NDCG@20` | 0.113215 |
| KuaiRec best `candidate_k` | 50 |
| KuaiRec TwoTower+DNN-Rerank@50 `NDCG@20` | 0.113354 |

## 当前理解沉淀

- 推荐系统不是单个模型，而是数据、召回、排序、过滤、评测组成的系统工程。
- 第一阶段的价值是建立可复现的离线评测闭环，而不是追求复杂模型。
- Popularity baseline 指标不高是正常现象，它用于给后续 ItemCF、MF 和 Two-Tower 提供对比基准。
- ItemCF 已经体现个性化召回价值，`Recall@20` 和 `NDCG@20` 均高于 Popularity baseline。
- MF 在 MLU 上完成训练，`Recall@20` 继续高于 ItemCF，说明可训练 embedding 模型已经带来更强召回能力。
- Two-Tower 使用用户塔和物品塔分别生成向量，更接近企业推荐系统中的向量召回范式；当前 `Recall@20` 接近 MF，`NDCG@20` 略高于 MF。
- 当前 Two-Tower 不是 Transformer 架构，而是 embedding + MLP 的双塔结构；每个塔包含 2 个线性层，其中 1 个隐藏层和 1 个输出投影层。
- DNN Ranker 将项目从召回推进到排序阶段，开始补充 `AUC` 和 `LogLoss` 这类 CTR 排序指标。
- Two-Tower + DNN Ranker 两阶段 pipeline 已跑通：候选召回决定重排上限，Ranker 在 Top 200 候选中重排 Top 20 后继续提升 `Recall@20` 和 `NDCG@20`。
- 候选集大小消融显示：`Candidate Recall` 随 `candidate_k` 增大而上升，但最终 Top20 指标不线性上升；当前 `candidate_k=50` 的 `Recall@20` 和 `NDCG@20` 最好。
- MIND-small 阶段已经完成数据入口；相比 MovieLens，它额外提供新闻标题、摘要、类别和用户点击历史，更适合做内容推荐。
- 官方 MIND Azure Blob 当前不可直接公开访问，本阶段使用 RecZoo `MIND_small_x1` 镜像继续推进。
- MIND Popularity baseline 已完成；它只按新闻全局点击热度排序，`AUC=0.522252`，说明热门程度只有弱收益，后续需要引入类别、用户历史和文本内容。
- MIND 当前评估方式和 MovieLens 不同：MovieLens 是从全量物品推荐 TopK，MIND 是在一次真实曝光候选列表内排序。
- MIND Category baseline 已在全量数据上超过 Popularity，说明类别 CTR 和用户历史类别偏好是有效特征。
- MIND DNNRanker、ContentTwoTower 和 TwoTower+DNN-Rerank 已在 srv4 MLU 容器完成 1M/500k 放大实验，并加入标题/摘要哈希文本 embedding。
- 当前 MLU 放大口径下 ContentTwoTower 的 `AUC=0.616641`、`NDCG@10=0.374560`，高于 DNNRanker 的 `AUC=0.592749`、`NDCG@10=0.353507`。
- Candidate K 消融显示当前 `candidate_k=10` 最好，`NDCG@10=0.362443`；继续增大候选数会引入更多噪声，重排效果反而下降。
- 当前训练脚本是单进程 MLU，主要使用容器逻辑 MLU0，也就是宿主 Card 2；Card 3 尚未参与 DDP。
- 按时间顺序切分比随机切分更接近真实推荐场景，因为模型只能利用用户过去行为预测未来偏好。
- 第三批数据集选择 KuaiRec，是因为它比 MovieLens/MIND 更接近短视频信息流推荐，可学习观看时长、完播率、视频类别和内容描述等行为信号。
- KuaiRec 核心交互表包含 `watch_ratio`，第一轮可以将 `watch_ratio >= 1.0` 定义为正反馈，表示完播或重复观看。
- KuaiRec 第一轮完整训练已完成，当前 `NDCG@20` 最好的是 Two-Tower，说明用户兴趣塔和视频内容塔在短视频 TopK 推荐上已经超过统计 baseline 和 ItemCF。
- KuaiRec DNNRanker 的 `AUC=0.656796`，但 `NDCG@20` 未超过 Two-Tower，说明点击/完播概率排序和最终 TopK 推荐效果并不完全等价。

## 下一步计划

1. 做 KuaiRec 标签阈值消融：`watch_ratio >= 1.0` 对比 `watch_ratio >= 0.8`。
2. 将神经模型训练样本从 120 万扩展到 `small_matrix.csv` 全量训练。
3. 将 KuaiRec 从 `small_matrix.csv` 扩展到 `big_matrix.csv`。
4. 优化 DNN Ranker 特征和损失，使重排 TopK 指标超过单独 Two-Tower。
5. 如果要体现双卡能力，再做 DDP 或 `torchrun --nproc_per_node=2`。

## 后续总体规划

MovieLens 阶段已经完成初步验证：数据处理、召回、排序、两阶段 pipeline、MLU 训练、
指标报告和消融实验都已经跑通。下一阶段不建议继续在 MovieLens 上堆复杂模型，
而应该迁移到更接近真实内容推荐的数据集。

建议路线：

1. `MIND-small`：做新闻推荐，重点引入标题、摘要、类别和用户点击历史。
2. 内容感知 Two-Tower：用户塔编码点击历史，新闻塔编码新闻文本和类别。
3. 内容感知 Ranker：融合用户行为、新闻文本 embedding、类别、热度和时序特征。
4. 两阶段 pipeline：新闻召回 TopN，再用 Ranker 重排 TopK。
5. `KuaiRec`：迁移到短视频推荐，重点建模观看反馈、完播率、类别和视频文本。
6. 如需要继续扩展，再考虑 `MIND-large`、Tenrec 或 KuaiRand，体现更大规模训练能力。

如果要结合“Grok / X”方向，当前更合理的目标不是从零训练大语言模型基座，
而是构建一个小型内容推荐基座表征模型：用文本编码器理解内容，用用户行为序列建模兴趣，
再通过召回和排序任务联合评估。这样更贴近 X/Twitter 推荐系统，也更适合当前两张 80GB MLU 的学习目标。
