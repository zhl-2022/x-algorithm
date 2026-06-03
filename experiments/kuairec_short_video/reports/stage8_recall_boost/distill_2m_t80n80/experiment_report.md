# KuaiRec big ItemCF 蒸馏 Two-Tower 报告

用 ItemCF teacher 的强协同 TopK 信号补强 Two-Tower 召回。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ItemCF-Distill-TwoTower | big_matrix | 2,000,000 | 7,174 | 0.006202 | 0.314608 | 0.027320 | 0.046048 | 0.608339 | 0.677686 | 8.65 | 0.39 |

## 说明

- ItemCF-Distill-TwoTower：用 ItemCF TopK teacher 样本和真实行为样本蒸馏训练 Two-Tower。
