# KuaiRec Two-Tower + DNN Ranker Pipeline 报告

先用 Two-Tower 召回候选，再用 DNN Ranker 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoTower+DNN-Blend@100a0 | big_matrix | 2,000,000 | 7,174 | 0.000261 | 0.019236 | 0.000937 | 0.027871 | 0.658928 | 0.438130 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@100a0.25 | big_matrix | 2,000,000 | 7,174 | 0.000528 | 0.038333 | 0.001763 | 0.029642 | 0.724026 | 0.428608 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@100a0.5 | big_matrix | 2,000,000 | 7,174 | 0.000604 | 0.047115 | 0.002680 | 0.033837 | 0.725827 | 0.426156 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@100a0.75 | big_matrix | 2,000,000 | 7,174 | 0.000614 | 0.047115 | 0.003065 | 0.036820 | 0.723702 | 0.431543 | 16.29 | 19.69 |
| TwoTower+DNN-Rerank@100 | big_matrix | 2,000,000 | 7,174 | 0.000614 | 0.046557 | 0.003153 | 0.039150 | 0.721696 | 0.445246 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@200a0 | big_matrix | 2,000,000 | 7,174 | 0.000261 | 0.019236 | 0.000937 | 0.027871 | 0.642556 | 0.452808 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@200a0.25 | big_matrix | 2,000,000 | 7,174 | 0.000542 | 0.040284 | 0.001823 | 0.030667 | 0.719014 | 0.445318 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@200a0.5 | big_matrix | 2,000,000 | 7,174 | 0.000913 | 0.073599 | 0.003762 | 0.036074 | 0.726803 | 0.444880 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@200a0.75 | big_matrix | 2,000,000 | 7,174 | 0.000975 | 0.078338 | 0.004466 | 0.040828 | 0.725897 | 0.452257 | 16.29 | 19.69 |
| TwoTower+DNN-Rerank@200 | big_matrix | 2,000,000 | 7,174 | 0.000994 | 0.080011 | 0.004714 | 0.042226 | 0.724385 | 0.467923 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@500a0 | big_matrix | 2,000,000 | 7,174 | 0.000261 | 0.019236 | 0.000937 | 0.027871 | 0.643985 | 0.507486 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@500a0.25 | big_matrix | 2,000,000 | 7,174 | 0.000542 | 0.040284 | 0.001821 | 0.032252 | 0.692907 | 0.500508 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@500a0.5 | big_matrix | 2,000,000 | 7,174 | 0.000963 | 0.076805 | 0.003865 | 0.039057 | 0.698819 | 0.501162 | 16.29 | 19.69 |
| TwoTower+DNN-Blend@500a0.75 | big_matrix | 2,000,000 | 7,174 | 0.001106 | 0.087120 | 0.004707 | 0.042599 | 0.697121 | 0.510206 | 16.29 | 19.69 |
| TwoTower+DNN-Rerank@500 | big_matrix | 2,000,000 | 7,174 | 0.001192 | 0.093253 | 0.004991 | 0.046421 | 0.694432 | 0.528086 | 16.29 | 19.69 |

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
