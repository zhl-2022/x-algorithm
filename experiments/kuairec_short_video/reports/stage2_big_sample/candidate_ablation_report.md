# KuaiRec Candidate K 消融报告

比较不同 `candidate_k` 对最终 Top20 结果的影响。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoTower+DNN-Blend@50a0 | big_matrix | 2,000,000 | 7,173 | 0.000682 | 0.030392 | 0.001448 | 0.026752 | 0.500755 | N/A | 9.88 | 19.17 |
| TwoTower+DNN-Blend@50a0.5 | big_matrix | 2,000,000 | 7,173 | 0.001027 | 0.041824 | 0.001977 | 0.023210 | 0.500756 | N/A | 9.88 | 19.17 |
| TwoTower+DNN-Rerank@50 | big_matrix | 2,000,000 | 7,173 | 0.000948 | 0.039035 | 0.001949 | 0.024609 | 0.500756 | N/A | 9.88 | 19.17 |
| TwoTower+DNN-Blend@100a0 | big_matrix | 2,000,000 | 7,173 | 0.000682 | 0.030392 | 0.001448 | 0.026752 | 0.501908 | N/A | 9.88 | 19.17 |
| TwoTower+DNN-Blend@100a0.5 | big_matrix | 2,000,000 | 7,173 | 0.001741 | 0.069009 | 0.003278 | 0.019482 | 0.501908 | N/A | 9.88 | 19.17 |
| TwoTower+DNN-Rerank@100 | big_matrix | 2,000,000 | 7,173 | 0.001727 | 0.069148 | 0.003415 | 0.021532 | 0.501908 | N/A | 9.88 | 19.17 |

## 说明

- TwoTower+DNN-Blend@50a0：Two-Tower 先取 Top50 候选，再按 0*Ranker + 1*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@50a0.5：Two-Tower 先取 Top50 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- TwoTower+DNN-Rerank@50：Two-Tower 先取 Top50 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@100a0：Two-Tower 先取 Top100 候选，再按 0*Ranker + 1*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@100a0.5：Two-Tower 先取 Top100 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- TwoTower+DNN-Rerank@100：Two-Tower 先取 Top100 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
