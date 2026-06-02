# KuaiRec Two-Tower + DNN Ranker Pipeline 报告

先用 Two-Tower 召回候选，再用 DNN Ranker 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoTower+DNN-Blend@50a0 | small_matrix | 1,200,000 | 1,411 | 0.017476 | 0.909993 | 0.153744 | 0.195672 | 0.515512 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@50a0.25 | small_matrix | 1,200,000 | 1,411 | 0.017393 | 0.912119 | 0.147511 | 0.175834 | 0.515514 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@50a0.5 | small_matrix | 1,200,000 | 1,411 | 0.017898 | 0.910702 | 0.148415 | 0.166516 | 0.515512 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@50a0.75 | small_matrix | 1,200,000 | 1,411 | 0.018176 | 0.915663 | 0.146756 | 0.166817 | 0.515508 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Rerank@50 | small_matrix | 1,200,000 | 1,411 | 0.018461 | 0.920624 | 0.145019 | 0.168019 | 0.515504 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@100a0 | small_matrix | 1,200,000 | 1,411 | 0.017476 | 0.909993 | 0.153744 | 0.195672 | 0.528941 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@100a0.25 | small_matrix | 1,200,000 | 1,411 | 0.017403 | 0.912828 | 0.147611 | 0.175534 | 0.528952 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@100a0.5 | small_matrix | 1,200,000 | 1,411 | 0.017921 | 0.911410 | 0.148611 | 0.160505 | 0.528954 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@100a0.75 | small_matrix | 1,200,000 | 1,411 | 0.018061 | 0.915663 | 0.145915 | 0.155095 | 0.528947 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Rerank@100 | small_matrix | 1,200,000 | 1,411 | 0.018148 | 0.913536 | 0.142382 | 0.155696 | 0.528936 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@200a0 | small_matrix | 1,200,000 | 1,411 | 0.017476 | 0.909993 | 0.153744 | 0.195672 | 0.538034 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@200a0.25 | small_matrix | 1,200,000 | 1,411 | 0.017408 | 0.912828 | 0.147634 | 0.175534 | 0.538085 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@200a0.5 | small_matrix | 1,200,000 | 1,411 | 0.017951 | 0.912119 | 0.148884 | 0.160204 | 0.538114 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Blend@200a0.75 | small_matrix | 1,200,000 | 1,411 | 0.018094 | 0.917789 | 0.146209 | 0.152390 | 0.538119 | N/A | 7.04 | 1.69 |
| TwoTower+DNN-Rerank@200 | small_matrix | 1,200,000 | 1,411 | 0.018162 | 0.917080 | 0.142500 | 0.152690 | 0.538096 | N/A | 7.04 | 1.69 |

## 说明

- TwoTower+DNN-Blend@50a0：Two-Tower 先取 Top50 候选，再按 0*Ranker + 1*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@50a0.25：Two-Tower 先取 Top50 候选，再按 0.25*Ranker + 0.75*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@50a0.5：Two-Tower 先取 Top50 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- TwoTower+DNN-Blend@50a0.75：Two-Tower 先取 Top50 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- TwoTower+DNN-Rerank@50：Two-Tower 先取 Top50 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
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
