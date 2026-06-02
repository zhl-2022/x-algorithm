# KuaiRec Matrix Factorization 报告

使用用户和视频 embedding 的点积预测完播正反馈。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MF | big_matrix_sample | 2,000,000 | 7,173 | 0.001905 | 0.083368 | 0.004222 | 0.096290 | 0.776112 | 0.524590 | 3.10 | 0.23 |

## 说明

- MF：用户和视频 embedding 点积 + bias，使用 BCE 学习完播正反馈。
