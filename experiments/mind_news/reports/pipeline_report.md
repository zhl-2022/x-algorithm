# MIND Two-Tower + DNN Ranker Pipeline 报告

Pipeline 先用 Two-Tower 在曝光候选内筛选 TopK，再用 DNN Ranker 重排。

| 模型 | 范围 | 训练样本 | 验证样本 | 曝光组 | AUC | MRR | NDCG@5 | NDCG@10 | HitRate@10 | 训练耗时秒 | 评估耗时秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoTower+DNN-Rerank | sample | 100,000 | 100,000 | 2,695 | 0.555439 | 0.273330 | 0.243986 | 0.319061 | 0.650093 | 4.21 | 0.19 |

## 说明

- Two-Tower 在每个曝光组内筛选 Top 20 候选，DNN Ranker 再重排。
