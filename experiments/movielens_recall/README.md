# MovieLens 召回与排序实验

这个实验是当前项目的第一个可复现推荐系统 baseline，用于从数据处理、
召回、排序评测和训练闭环的角度学习 X algorithm 项目。

## 目标

跑通一个最小但完整的推荐系统流程：

1. 下载 MovieLens 1M。
2. 将评分数据转换为隐式正反馈交互。
3. 按用户时间序列切分正反馈样本。
4. 生成数据统计报告。
5. 使用 `Recall@20` 评估 Popularity baseline。
6. 逐步实现 ItemCF、MF、Two-Tower 召回模型并对比离线指标。
7. 实现 DNN Ranker 排序模型，补充 AUC 和 LogLoss 评估。
8. 串联 Two-Tower 召回和 DNN Ranker 重排，形成两阶段推荐 pipeline。

## 实验设定

| 项目 | 设置 |
|---|---|
| 数据集 | MovieLens 1M |
| 正反馈定义 | `rating >= 4.0` |
| 切分方式 | 每个用户内部按时间顺序切分 |
| Baseline | Popularity、ItemCF、MF、Two-Tower、DNN Ranker、TwoTower+DNN-Rerank |
| 核心指标 | `Recall@20`、`NDCG@20`、`AUC`、`LogLoss` |

至少有 3 条正反馈的用户会产生训练集、验证集和测试集样本。有 2 条正反馈的用户
会产生训练集和测试集样本。只有 1 条正反馈的用户只进入训练集。

## 快速开始

在当前目录运行：

```powershell
python scripts/download_movielens.py
python scripts/prepare_movielens.py
python scripts/popularity_baseline.py --k 20
python scripts/itemcf_baseline.py --k 20
python scripts/mf_train.py --k 20
python scripts/two_tower_train.py --k 20
python scripts/ranker_train.py --k 20
python scripts/rerank_pipeline.py --k 20 --candidate-k 200
python scripts/rerank_ablation.py --k 20 --candidate-ks 50,100,200,500
```

也可以在数据缺失时由准备脚本自动下载：

```powershell
python scripts/prepare_movielens.py --download
python scripts/popularity_baseline.py --k 20
python scripts/itemcf_baseline.py --k 20
python scripts/mf_train.py --k 20
python scripts/two_tower_train.py --k 20
python scripts/ranker_train.py --k 20
python scripts/rerank_pipeline.py --k 20 --candidate-k 200
python scripts/rerank_ablation.py --k 20 --candidate-ks 50,100,200,500
```

## 输出文件

| 路径 | 说明 |
|---|---|
| `data/raw/ml-1m/` | 解压后的 MovieLens 1M 原始文件 |
| `data/processed/train.csv` | 训练集正反馈交互 |
| `data/processed/valid.csv` | 验证集正反馈交互 |
| `data/processed/test.csv` | 测试集正反馈交互 |
| `data/processed/item_metadata.csv` | 电影标题和类别元数据 |
| `reports/data_report.md` | 数据统计和切分报告 |
| `reports/baseline_report.md` | Popularity baseline 指标报告 |
| `reports/itemcf_report.md` | ItemCF baseline 指标报告 |
| `reports/mf_report.md` | Matrix Factorization baseline 指标报告 |
| `reports/two_tower_report.md` | Two-Tower 召回模型指标报告 |
| `reports/ranker_report.md` | DNN Ranker 排序模型指标报告 |
| `reports/rerank_report.md` | Two-Tower + DNN Ranker 两阶段重排报告 |
| `reports/rerank_ablation_report.md` | 候选集大小消融实验报告 |
| `outputs/popularity_results.csv` | 机器可读的指标结果 |
| `outputs/itemcf_results.csv` | ItemCF 机器可读指标结果 |
| `outputs/mf_results.csv` | Matrix Factorization 机器可读指标结果 |
| `outputs/two_tower_results.csv` | Two-Tower 机器可读指标结果 |
| `outputs/ranker_results.csv` | DNN Ranker 机器可读指标结果 |
| `outputs/rerank_results.csv` | 两阶段重排机器可读指标结果 |
| `outputs/rerank_ablation_results.csv` | 候选集大小消融实验结果 |
| `outputs/experiment_results.csv` | 多模型统一指标表 |

## 当前结果

| 模型 | Recall@20 | NDCG@20 | Coverage@20 |
|---|---:|---:|---:|
| Popularity | 0.067252 | 0.026149 | 0.045326 |
| ItemCF | 0.082160 | 0.032793 | 0.132372 |
| MF | 0.104025 | 0.039807 | 0.481844 |
| Two-Tower | 0.101872 | 0.040244 | 0.340973 |
| DNN-Ranker | 0.106841 | 0.041302 | 0.525882 |
| TwoTower+DNN-Rerank | 0.108829 | 0.041877 | 0.435230 |

## 当前 Pipeline 指标

| Pipeline | Candidate Recall@200 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|
| Two-Tower -> DNN Ranker | 0.471757 | 0.108829 | 0.041877 |

## Candidate K 消融

| Candidate K | Candidate Recall | Recall@20 | NDCG@20 | Coverage@20 | 重排耗时（秒） |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.198277 | 0.109988 | 0.042451 | 0.366469 | 4.19 |
| 100 | 0.315388 | 0.108332 | 0.041980 | 0.399176 | 4.37 |
| 200 | 0.471757 | 0.108829 | 0.041877 | 0.435230 | 4.63 |
| 500 | 0.733477 | 0.107172 | 0.041467 | 0.477981 | 5.71 |

当前 Top20 指标最好的候选数是 `candidate_k=50`。这说明候选池变大可以提升召回上限，
但会引入更多噪声候选，最终 Top20 效果不一定同步提升。

## 当前排序指标

| 模型 | AUC | LogLoss |
|---|---:|---:|
| DNN-Ranker | 0.899737 | 0.138917 |

## 后续模型

当前 baseline 稳定后，建议按顺序增加以下模型：

1. 整理 MovieLens 阶段报告，形成可复现实验闭环。
2. 迁移到 MIND-small，开始新闻推荐实验。
3. 在 MIND 上加入文本内容编码，升级为内容理解 + 行为推荐 pipeline。
