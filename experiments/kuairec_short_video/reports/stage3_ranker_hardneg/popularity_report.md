# KuaiRec Popularity Baseline 报告

按视频全局正反馈热度排序。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity | small_matrix | 3,595,097 | 1,411 | 0.007644 | 0.627923 | 0.053773 | 0.324316 | 0.479285 | 0.710028 | 0.00 | 0.00 |

## 说明

- Popularity：视频全局完播正反馈热度。
