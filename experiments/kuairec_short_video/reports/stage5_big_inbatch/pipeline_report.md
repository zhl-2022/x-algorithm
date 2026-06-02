# KuaiRec Two-Tower + DNN Ranker Pipeline 报告

先用 Two-Tower 召回候选，再用 DNN Ranker 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoTower+DNN-Blend@100a0 | big_matrix | 2,000,000 | 7,174 | 0.001720 | 0.088235 | 0.004706 | 0.268550 | 0.542569 | 1.379468 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@100a0.25 | big_matrix | 2,000,000 | 7,174 | 0.001715 | 0.096599 | 0.004806 | 0.231357 | 0.683764 | 1.044925 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@100a0.5 | big_matrix | 2,000,000 | 7,174 | 0.001791 | 0.103150 | 0.004936 | 0.233221 | 0.736746 | 0.790469 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@100a0.75 | big_matrix | 2,000,000 | 7,174 | 0.001843 | 0.108029 | 0.005144 | 0.237136 | 0.748750 | 0.639209 | 28.06 | 20.19 |
| TwoTower+DNN-Rerank@100 | big_matrix | 2,000,000 | 7,174 | 0.001857 | 0.110677 | 0.005245 | 0.243102 | 0.745916 | 0.582823 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@200a0 | big_matrix | 2,000,000 | 7,174 | 0.001720 | 0.088235 | 0.004706 | 0.268550 | 0.564147 | 1.305441 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@200a0.25 | big_matrix | 2,000,000 | 7,174 | 0.001474 | 0.078060 | 0.003892 | 0.204884 | 0.683133 | 1.004625 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@200a0.5 | big_matrix | 2,000,000 | 7,174 | 0.001371 | 0.075551 | 0.003468 | 0.193885 | 0.732871 | 0.776533 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@200a0.75 | big_matrix | 2,000,000 | 7,174 | 0.001335 | 0.074993 | 0.003423 | 0.198639 | 0.744970 | 0.641124 | 28.06 | 20.19 |
| TwoTower+DNN-Rerank@200 | big_matrix | 2,000,000 | 7,174 | 0.001251 | 0.073878 | 0.003359 | 0.206003 | 0.739266 | 0.593660 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@500a0 | big_matrix | 2,000,000 | 7,174 | 0.001720 | 0.088235 | 0.004706 | 0.268550 | 0.570534 | 1.199486 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@500a0.25 | big_matrix | 2,000,000 | 7,174 | 0.001164 | 0.059799 | 0.003019 | 0.181674 | 0.669351 | 0.943237 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@500a0.5 | big_matrix | 2,000,000 | 7,174 | 0.000749 | 0.037636 | 0.001716 | 0.139075 | 0.716693 | 0.750783 | 28.06 | 20.19 |
| TwoTower+DNN-Blend@500a0.75 | big_matrix | 2,000,000 | 7,174 | 0.000712 | 0.033733 | 0.001528 | 0.135720 | 0.731336 | 0.637347 | 28.06 | 20.19 |
| TwoTower+DNN-Rerank@500 | big_matrix | 2,000,000 | 7,174 | 0.000769 | 0.036242 | 0.001672 | 0.145507 | 0.727796 | 0.600681 | 28.06 | 20.19 |

## 说明

- TwoTower+DNN-Blend@100a0：Two-Tower 先取 Top100 候选，再按 0*Ranker + 1*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@100a0.25：Two-Tower 先取 Top100 候选，再按 0.25*Ranker + 0.75*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@100a0.5：Two-Tower 先取 Top100 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@100a0.75：Two-Tower 先取 Top100 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- TwoTower+DNN-Rerank@100：Two-Tower 先取 Top100 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@200a0：Two-Tower 先取 Top200 候选，再按 0*Ranker + 1*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@200a0.25：Two-Tower 先取 Top200 候选，再按 0.25*Ranker + 0.75*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@200a0.5：Two-Tower 先取 Top200 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@200a0.75：Two-Tower 先取 Top200 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- TwoTower+DNN-Rerank@200：Two-Tower 先取 Top200 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@500a0：Two-Tower 先取 Top500 候选，再按 0*Ranker + 1*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@500a0.25：Two-Tower 先取 Top500 候选，再按 0.25*Ranker + 0.75*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@500a0.5：Two-Tower 先取 Top500 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@500a0.75：Two-Tower 先取 Top500 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- TwoTower+DNN-Rerank@500：Two-Tower 先取 Top500 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
