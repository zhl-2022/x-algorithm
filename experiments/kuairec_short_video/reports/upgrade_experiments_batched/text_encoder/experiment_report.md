# KuaiRec big 轻量文本 Encoder 报告

用 TextCNN 对 caption 哈希 token 做轻量编码。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TextCNN-TwoTower | big_matrix | 800,000 | 7,174 | 0.003039 | 0.152635 | 0.007997 | 0.129754 | 0.554473 | 0.691228 | 3.24 | 0.29 |

## 说明

- TextCNN-TwoTower：用 TextCNN 编码 caption 哈希 token，替代简单均值文本池化。
