# KuaiRec big LightGCN 图召回报告

将用户-视频正反馈建成二部图，训练 LightGCN 召回 embedding。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGCN | big_matrix | 800,000 | 7,174 | 0.002571 | 0.166992 | 0.011096 | 0.323919 | 0.600340 | 0.693026 | 5.01 | 0.50 |

## 说明

- LightGCN：基于用户-视频正反馈二部图训练 LightGCN 图召回。
