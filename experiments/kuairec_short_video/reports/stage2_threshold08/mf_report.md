# KuaiRec Matrix Factorization 报告

使用用户和视频 embedding 的点积预测完播正反馈。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MF | small_matrix_sample | 1,200,000 | 1,411 | 0.018616 | 0.915663 | 0.136451 | 0.152690 | 0.678856 | 0.635658 | 1.67 | 0.03 |

## 说明

- MF：用户和视频 embedding 点积 + bias，使用 BCE 学习完播正反馈。
