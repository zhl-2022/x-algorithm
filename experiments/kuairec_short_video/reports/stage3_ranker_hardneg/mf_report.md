# KuaiRec Matrix Factorization 报告

使用用户和视频 embedding 的点积预测完播正反馈。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MF | small_matrix_sample | 1,200,000 | 1,411 | 0.019001 | 0.922041 | 0.143591 | 0.183950 | 0.679292 | 0.637416 | 3.72 | 0.06 |

## 说明

- MF：用户和视频 embedding 点积 + bias，使用 BCE 学习完播正反馈。
