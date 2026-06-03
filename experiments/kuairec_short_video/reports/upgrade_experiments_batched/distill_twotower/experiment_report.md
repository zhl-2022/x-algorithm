# KuaiRec big ItemCF 蒸馏 Two-Tower 报告

用 ItemCF teacher 的强协同 TopK 信号补强 Two-Tower 召回。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ItemCF-Distill-TwoTower | big_matrix | 800,000 | 7,174 | 0.006874 | 0.363953 | 0.033562 | 0.024981 | 0.604001 | 0.679956 | 3.36 | 0.32 |

## 说明

- ItemCF-Distill-TwoTower：用 ItemCF TopK teacher 样本和真实行为样本蒸馏训练 Two-Tower。
