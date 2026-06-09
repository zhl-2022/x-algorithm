# 推荐系统项目学习路线

## 1. 学习目标

这份路线用于指导你从零开始学习当前推荐系统项目。目标不是先看上游 X/Twitter 源码，也不是直接读 KuaiRec 最终大脚本，而是按“结论 -> 架构 -> 最小闭环 -> 内容推荐 -> 短视频大规模实验 -> 简历面试”的顺序逐步掌握。

完成后你应该能讲清楚：

1. 推荐系统为什么要拆成召回、排序、重排和评测。
2. 三批数据集分别解决了什么学习问题。
3. 每个模型输入是什么、输出是什么、指标怎么计算。
4. 为什么 KuaiRec big 上 ItemCF 强，ItemCF 蒸馏为什么有效。
5. MLU 单卡/双卡训练在项目里体现了什么工程能力。

## 2. 推荐学习顺序

| 顺序 | 阶段 | 先读文档 | 重点代码 | 阶段目标 |
|---:|---|---|---|---|
| 1 | 全局地图 | `README.md`、`docs/project_summary_report.md`、`docs/project_architecture.md` | 暂不读代码 | 用 3 分钟讲清楚项目做了什么 |
| 2 | MovieLens | `experiments/movielens_recall/README.md`、`reports/指标解析.md` | `prepare_movielens.py`、`itemcf_baseline.py`、`mf_train.py`、`two_tower_train.py`、`ranker_train.py`、`rerank_pipeline.py` | 掌握最小推荐闭环 |
| 3 | MIND | `experiments/mind_news/README.md`、`reports/试验解析.md` | `prepare_mind.py`、`run_all_experiments.py` | 掌握内容推荐和曝光候选 |
| 4 | KuaiRec | `experiments/kuairec_short_video/README.md`、`reports/试验方案.md`、最终阶段报告 | `prepare_kuairec.py`、`run_all_experiments.py`、`run_upgrade_experiments.py`、`benchmark_mlu_ddp.py` | 掌握短视频推荐、蒸馏和 MLU 工程 |
| 5 | 简历面试 | `docs/resume_project_writeup.md`、`docs/interview_qa.md` | 回看关键代码 | 把实验转成简历表达和面试回答 |

## 3. 阶段学习清单

### 3.1 全局地图

- [ ] 阅读 `README.md`，区分上游源码和本项目实验层。
- [ ] 阅读 `docs/project_summary_report.md`，记住三批数据集的关键结论。
- [ ] 阅读 `docs/project_architecture.md`，理解数据、召回、排序、重排、评测和 MLU 链路。
- [ ] 能讲清楚：当前项目不是训练 Grok/LLM，而是推荐系统离线训练评测闭环。

### 3.2 MovieLens：最小推荐闭环

- [ ] 阅读 `experiments/movielens_recall/README.md`。
- [ ] 阅读 `experiments/movielens_recall/reports/指标解析.md`。
- [ ] 看懂 `prepare_movielens.py`：评分如何转成 `rating >= 4.0` 的隐式正反馈。
- [ ] 看懂 `itemcf_baseline.py`：共现相似度如何生成 TopK 推荐。
- [ ] 看懂 `mf_train.py` 和 `two_tower_train.py`：embedding 与双塔召回如何训练。
- [ ] 看懂 `ranker_train.py` 和 `rerank_pipeline.py`：候选集如何接入排序模型。
- [ ] 能解释 MovieLens 最佳结果：两阶段 pipeline `NDCG@20=0.042451`。

### 3.3 MIND：内容推荐

- [ ] 阅读 `experiments/mind_news/README.md`。
- [ ] 阅读 `experiments/mind_news/reports/试验解析.md`。
- [ ] 看懂 `prepare_mind.py`：新闻元数据、用户历史和曝光样本如何解析。
- [ ] 看懂 `run_all_experiments.py`：Popularity、Category、DNNRanker、ContentTwoTower 和 pipeline 如何统一评测。
- [ ] 能解释 MIND 结论：标题/摘要哈希文本 embedding 有效，`ContentTwoTower NDCG@10=0.374560`。

### 3.4 KuaiRec：短视频与大规模实验

- [ ] 阅读 `experiments/kuairec_short_video/README.md`。
- [ ] 阅读 `experiments/kuairec_short_video/reports/试验方案.md`。
- [ ] 阅读 `stage10_soft_label_tuning_report.md` 和 `stage11_final_kuairec_report.md`。
- [ ] 看懂 `prepare_kuairec.py`：`watch_ratio` 标签如何构造。
- [ ] 看懂 `run_all_experiments.py`：baseline、Two-Tower、Ranker、hard negative、in-batch negative 如何实现。
- [ ] 看懂 `run_upgrade_experiments.py`：ItemCF 蒸馏、LightGCN、GRU、TextCNN、soft label 如何组织。
- [ ] 看懂 `benchmark_mlu_ddp.py`：MLU 单卡/双卡 DDP benchmark 如何跑。
- [ ] 能解释 KuaiRec 结论：big 神经 pipeline 从 `NDCG@20=0.005245` 提升到 `0.055883`，复跑 `0.052947`。

### 3.5 简历与面试表达

- [ ] 阅读 `docs/resume_project_writeup.md`。
- [ ] 阅读 `docs/interview_qa.md`。
- [ ] 能用 3 分钟讲完整项目背景、技术路线、指标结果和工程价值。
- [ ] 能回答 `AUC` 与 `NDCG@K` 的区别。
- [ ] 能回答 ItemCF 为什么在 KuaiRec big 上强。
- [ ] 能回答 ItemCF 蒸馏、hard negative、soft label 分别解决了什么问题。
- [ ] 能回答 MLU 双卡 DDP 是怎么验证的。

## 4. 学习验收问题

每完成一个阶段，都用下面问题自测：

| 问题 | 你需要能回答到什么程度 |
|---|---|
| 数据集原始字段是什么？ | 能说出关键字段和业务含义 |
| 正反馈标签怎么定义？ | 能说明阈值和原因，例如 `rating >= 4.0`、`click=1`、`watch_ratio >= 0.8` |
| 训练、验证、测试怎么切？ | 能说明是否按用户、时间、曝光样本切分 |
| 使用了哪些模型？ | 能区分 baseline、召回、排序、重排 |
| 核心指标是什么？ | 能解释 `Recall@K`、`NDCG@K`、`AUC`、`LogLoss` |
| 最好结果是多少？ | 能说出每阶段关键数字和对比对象 |
| 为什么有效或无效？ | 能结合数据、标签、负采样、候选质量解释 |
| 当前实验有什么局限？ | 能说明离线评测、文本哈希、分布式 TopK 未完全服务化等限制 |

## 5. 默认学习假设

1. 当前目标是把项目学懂并写进简历，不是继续无止境加模型。
2. 优先学习 `experiments/` 和 `docs/`，上游 X 源码只作为架构参考。
3. 当前最值得深读的是 KuaiRec，因为它包含短视频推荐、ItemCF 蒸馏、hard negative、soft label 和 MLU 双卡。
4. 如果后续继续扩展实验，再考虑 Tenrec 或 KuaiRand；当前学习阶段先把三批数据集讲清楚。
