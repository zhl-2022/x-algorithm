# KuaiRec big LightGCN 图召回报告

将用户-视频正反馈建成二部图，训练 LightGCN 召回 embedding。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGCN | big_matrix | 800,000 | 7,174 | 0.002102 | 0.134095 | 0.007371 | 0.969426 | 0.500204 | 0.693154 | 1.34 | 0.45 |

## 说明

- LightGCN：基于用户-视频正反馈二部图训练 LightGCN 图召回。
