# Two-Tower + DNN Ranker 重排报告

## Pipeline 指标

| 指标 | 数值 |
|---|---:|
| 模型 | TwoTower+DNN-Rerank |
| 召回候选数 | 200 |
| 最终推荐 K | 20 |
| 评估用户数 | 6,037 |
| Candidate Recall@200 | 0.471757 |
| Candidate HitRate@200 | 0.471757 |
| Recall@20 | 0.108829 |
| HitRate@20 | 0.108829 |
| Precision@20 | 0.005441 |
| NDCG@20 | 0.041877 |
| Coverage@20 | 0.435230 |

## Ranker 排序指标

| 指标 | 数值 |
|---|---:|
| AUC | 0.899737 |
| LogLoss | 0.138917 |
| 排序评估样本数 | 609,737 |
| 排序评估正样本数 | 6,037 |
| 排序评估负样本数 | 603,700 |

## 训练与评估设置

| 项目 | 数值 |
|---|---:|
| 设备 | mlu |
| 用户数 | 6,038 |
| 电影数 | 3,883 |
| Two-Tower Embedding Dim | 128 |
| Two-Tower Hidden Dim | 256 |
| Two-Tower Tower Dim | 128 |
| Two-Tower Epochs | 10 |
| Two-Tower Final Loss | 0.313643 |
| Two-Tower 训练耗时（秒） | 23.38 |
| Ranker Embedding Dim | 128 |
| Ranker Hidden Dims | 512 -> 256 -> 128 |
| Ranker Epochs | 8 |
| Ranker Final Loss | 0.205959 |
| Ranker 训练耗时（秒） | 25.74 |
| 候选生成耗时（秒） | 7.76 |
| 重排耗时（秒） | 4.94 |
| 总评估耗时（秒） | 15.09 |

## 说明

- 这个实验是两阶段推荐 pipeline：Two-Tower 先从全量电影中召回候选，DNN Ranker 再对候选集重排。
- `Candidate Recall` 表示真实测试电影是否进入了 Two-Tower 召回候选集，它决定了重排阶段的效果上限。
- 最终 `Recall@K` 和 `NDCG@K` 只在 Ranker 重排后的 TopK 上计算。
- Ranker 的 AUC 和 LogLoss 仍使用测试正样本加随机负样本评估。
