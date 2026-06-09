# 推荐实验阅读指南

## 1. 先看什么

如果你是第一次回看这个项目，建议按下面顺序读，不要直接从上游 X 源码或 KuaiRec 最终脚本开始。

如果你希望按阶段打卡学习，先使用 `docs/study_plan.md`；本文件更偏“读哪些报告和代码”。

| 顺序 | 文件 | 读它的目的 |
|---:|---|---|
| 1 | `README.md` | 先理解这个仓库里“上游源码”和“个人实验”两层关系 |
| 2 | `docs/project_summary_report.md` | 看三批数据集最终结论和关键指标 |
| 3 | `docs/project_architecture.md` | 建立数据、模型、pipeline、MLU 的整体架构图 |
| 4 | `docs/study_plan.md` | 按阶段完成学习清单和验收问题 |
| 5 | `docs/resume_project_writeup.md` | 看如何把实验转成简历项目 |
| 6 | `docs/interview_qa.md` | 看面试中最容易被追问的问题 |

## 2. 实验代码阅读顺序

### 2.1 MovieLens：先学最小闭环

先看 MovieLens，因为它的数据最简单，能把推荐系统流程讲清楚。

| 文件 | 重点看什么 |
|---|---|
| `experiments/movielens_recall/README.md` | MovieLens 阶段的实验设定、命令和指标 |
| `experiments/movielens_recall/reports/指标解析.md` | 面向初学者的指标解释 |
| `experiments/movielens_recall/scripts/prepare_movielens.py` | 如何把评分转成 `rating >= 4.0` 的隐式正反馈 |
| `experiments/movielens_recall/scripts/itemcf_baseline.py` | ItemCF 如何用共现相似度做推荐 |
| `experiments/movielens_recall/scripts/mf_train.py` | MF 如何用 embedding 学用户-电影协同信号 |
| `experiments/movielens_recall/scripts/two_tower_train.py` | Two-Tower 召回模型的最小实现 |
| `experiments/movielens_recall/scripts/ranker_train.py` | DNNRanker 如何做排序 |
| `experiments/movielens_recall/scripts/rerank_pipeline.py` | 召回候选如何接入 Ranker 重排 |

MovieLens 阶段最重要的结论是：两阶段 pipeline 的 `NDCG@20=0.042451`，超过 Popularity、ItemCF、MF 和单独 Two-Tower。

### 2.2 MIND：再学内容推荐

MIND 比 MovieLens 更接近内容流推荐，因为它有新闻标题、摘要、类别、曝光候选和点击标签。

| 文件 | 重点看什么 |
|---|---|
| `experiments/mind_news/README.md` | MIND 数据、实验命令和最终结果 |
| `experiments/mind_news/reports/试验解析.md` | 小白视角的新闻推荐实验解释 |
| `experiments/mind_news/scripts/prepare_mind.py` | 如何解析新闻元数据和曝光样本 |
| `experiments/mind_news/scripts/run_all_experiments.py` | Popularity、Category、DNNRanker、ContentTwoTower 和 pipeline 的统一实现 |

MIND 阶段最重要的结论是：加入标题/摘要哈希文本 embedding 后，`ContentTwoTower NDCG@10=0.374560`，高于 `DNNRanker NDCG@10=0.353507`。

### 2.3 KuaiRec：最后看短视频和工程化

KuaiRec 是当前项目最重要、最接近简历亮点的阶段。它包含 `small_matrix.csv` 的快速验证、`big_matrix.csv` 的大候选池挑战，以及 MLU 训练验证。

| 文件 | 重点看什么 |
|---|---|
| `experiments/kuairec_short_video/README.md` | KuaiRec 全部阶段和最终结果 |
| `experiments/kuairec_short_video/reports/试验方案.md` | 数据、标签、模型、指标和每轮实验目的 |
| `experiments/kuairec_short_video/scripts/prepare_kuairec.py` | 如何读取 KuaiRec 字段、构造 `watch_ratio` 标签 |
| `experiments/kuairec_short_video/scripts/run_all_experiments.py` | 基础模型、标签消融、Ranker hard negative、in-batch negative |
| `experiments/kuairec_short_video/scripts/run_upgrade_experiments.py` | ItemCF 蒸馏、LightGCN、GRU 序列、TextCNN、soft label 精调 |
| `experiments/kuairec_short_video/scripts/benchmark_mlu_ddp.py` | MLU 单卡/双卡 DDP benchmark |
| `experiments/kuairec_short_video/scripts/start_stage10_11_kuairec_final.sh` | 最终 Stage10/Stage11 收尾实验启动方式 |

KuaiRec 阶段最重要的结论是：big 神经 pipeline 从 Stage5 的 `NDCG@20=0.005245`，通过 ItemCF 蒸馏、hard negative、teacher/negative 配比和 soft label 精调，提升到 Stage10 最佳 `0.055883`，换 seed 复跑 `0.052947`。

## 3. 重点代码怎么读

### 3.1 数据处理代码

优先掌握三件事：

1. 原始数据有哪些字段。
2. 正反馈标签怎么定义。
3. 训练、验证、测试怎么切分。

对应代码：

```text
experiments/movielens_recall/scripts/prepare_movielens.py
experiments/mind_news/scripts/prepare_mind.py
experiments/kuairec_short_video/scripts/prepare_kuairec.py
```

### 3.2 评测代码

推荐系统面试一定会追问指标。重点看 `Recall@K`、`NDCG@K`、`Coverage@K`、`AUC`、`LogLoss` 是怎么计算的。

对应代码：

```text
experiments/movielens_recall/scripts/*_train.py
experiments/mind_news/scripts/run_all_experiments.py
experiments/kuairec_short_video/scripts/run_all_experiments.py
experiments/kuairec_short_video/scripts/run_upgrade_experiments.py
```

读代码时重点找这些函数或逻辑：

- 用户维度 TopK 推荐列表如何生成。
- 训练集中已看过内容如何过滤。
- 测试集真实正样本如何构造。
- `NDCG@K` 为什么越靠前命中权重越高。
- `AUC` 和 TopK 指标为什么可能不一致。

### 3.3 召回模型代码

召回模型是项目主线。建议按复杂度读：

1. `itemcf_baseline.py`：传统协同过滤，理解共现统计。
2. `mf_train.py`：embedding 点积，理解可训练协同信号。
3. `two_tower_train.py`：双塔召回，理解用户塔和物品塔。
4. `run_upgrade_experiments.py` 里的 ItemCF 蒸馏：理解为什么用强 ItemCF teacher 补神经召回。

### 3.4 排序和重排代码

重点看 Ranker 如何接入召回候选：

```text
experiments/movielens_recall/scripts/ranker_train.py
experiments/movielens_recall/scripts/rerank_pipeline.py
experiments/kuairec_short_video/scripts/run_all_experiments.py
experiments/kuairec_short_video/scripts/run_upgrade_experiments.py
```

需要理解：

- Ranker 是二分类模型，但最终要服务 TopK。
- hard negative 为什么比随机负样本更贴近重排场景。
- `candidate_k` 变大为什么可能提升候选召回，但降低最终 Top20 质量。
- blend 为什么有时优于纯 rerank。

### 3.5 MLU 工程代码

MLU 相关重点看：

```text
scripts/mlu/start_xalgorithm_mlu.sh
scripts/mlu/check_srv4_mlu.ps1
experiments/kuairec_short_video/scripts/benchmark_mlu_ddp.py
```

需要掌握：

- `torch_mlu` 如何检查设备可用。
- `MLU_VISIBLE_DEVICES=2,3` 如何限制训练卡。
- DDP 如何用 `torchrun --nproc_per_node=2` 启动。
- 寒武纪多卡训练中使用 `cncl` 后端。

## 4. 面试时怎么讲项目

推荐用三段式讲法：

1. **为什么做**：参考 X/Twitter 多阶段推荐思想，做一个可复现的离线推荐训练评测闭环。
2. **怎么做**：用 MovieLens 跑通最小闭环，用 MIND 加入内容特征，用 KuaiRec 做短视频 big 场景和 MLU 工程验证。
3. **结果是什么**：MovieLens pipeline 最优，MIND 内容双塔有效，KuaiRec big 通过 ItemCF 蒸馏和 soft label 将神经 pipeline 提升到 `NDCG@20=0.055883`，MLU 双卡吞吐达到 `908,159 samples/s`。

## 5. 最应该讲清楚的 6 个问题

1. 为什么推荐系统要分召回、排序和重排？
2. 为什么 `AUC` 高不代表 `NDCG@K` 高？
3. ItemCF 的共现相似度是怎么计算的？
4. Two-Tower 为什么适合召回？
5. hard negative 为什么能提升 Ranker？
6. KuaiRec big 上为什么 ItemCF 仍强于神经模型，ItemCF 蒸馏解决了什么问题？

这些问题的详细回答见 `docs/interview_qa.md`。
