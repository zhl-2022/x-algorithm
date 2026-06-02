# KuaiRec Two-Tower + DNN Ranker Pipeline 报告

先用 Two-Tower 召回候选，再用 DNN Ranker 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoTower+DNN-Rerank@50 | small_matrix | 1,200,000 | 1,410 | 0.025497 | 0.888652 | 0.113354 | 0.149684 | 0.656796 | 0.759420 | 6.08 | 1.69 |
| TwoTower+DNN-Rerank@100 | small_matrix | 1,200,000 | 1,410 | 0.025479 | 0.891489 | 0.113166 | 0.149384 | 0.656796 | 0.759420 | 6.08 | 1.69 |
| TwoTower+DNN-Rerank@200 | small_matrix | 1,200,000 | 1,410 | 0.025458 | 0.891489 | 0.112958 | 0.148182 | 0.656796 | 0.759420 | 6.08 | 1.69 |

## 说明

- TwoTower+DNN-Rerank@50：Two-Tower 先取 Top50 候选，再由 DNN Ranker 重排 Top20。
- TwoTower+DNN-Rerank@100：Two-Tower 先取 Top100 候选，再由 DNN Ranker 重排 Top20。
- TwoTower+DNN-Rerank@200：Two-Tower 先取 Top200 候选，再由 DNN Ranker 重排 Top20。
