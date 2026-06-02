# KuaiRec Two-Tower + DNN Ranker Pipeline 报告

先用 Two-Tower 召回候选，再用 DNN Ranker 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoTower+DNN-Blend@50a0 | small_matrix | 1,200,000 | 1,411 | 0.018152 | 0.917080 | 0.159630 | 0.182747 | 0.747227 | 0.440293 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@50a0.25 | small_matrix | 1,200,000 | 1,411 | 0.018439 | 0.927711 | 0.163016 | 0.183348 | 0.750077 | 0.444772 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@50a0.5 | small_matrix | 1,200,000 | 1,411 | 0.019013 | 0.927711 | 0.167467 | 0.189360 | 0.735325 | 0.451804 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@50a0.75 | small_matrix | 1,200,000 | 1,411 | 0.019447 | 0.934798 | 0.169771 | 0.200180 | 0.713082 | 0.461561 | 7.55 | 1.15 |
| TwoTower+DNN-Rerank@50 | small_matrix | 1,200,000 | 1,411 | 0.019543 | 0.936924 | 0.170111 | 0.208596 | 0.690666 | 0.474199 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@100a0 | small_matrix | 1,200,000 | 1,411 | 0.018152 | 0.917080 | 0.159630 | 0.182747 | 0.684276 | 0.493401 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@100a0.25 | small_matrix | 1,200,000 | 1,411 | 0.018461 | 0.929128 | 0.163269 | 0.183348 | 0.699203 | 0.496040 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@100a0.5 | small_matrix | 1,200,000 | 1,411 | 0.020732 | 0.948264 | 0.178336 | 0.184250 | 0.694845 | 0.500695 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@100a0.75 | small_matrix | 1,200,000 | 1,411 | 0.021951 | 0.963855 | 0.190630 | 0.199880 | 0.677161 | 0.507458 | 7.55 | 1.15 |
| TwoTower+DNN-Rerank@100 | small_matrix | 1,200,000 | 1,411 | 0.022158 | 0.963147 | 0.197553 | 0.212203 | 0.654515 | 0.516412 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@200a0 | small_matrix | 1,200,000 | 1,411 | 0.018152 | 0.917080 | 0.159630 | 0.182747 | 0.704481 | 0.585691 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@200a0.25 | small_matrix | 1,200,000 | 1,411 | 0.018461 | 0.929128 | 0.163269 | 0.182447 | 0.711277 | 0.585723 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@200a0.5 | small_matrix | 1,200,000 | 1,411 | 0.021151 | 0.950390 | 0.180832 | 0.182447 | 0.712292 | 0.587364 | 7.55 | 1.15 |
| TwoTower+DNN-Blend@200a0.75 | small_matrix | 1,200,000 | 1,411 | 0.022699 | 0.970943 | 0.196753 | 0.190562 | 0.705119 | 0.590651 | 7.55 | 1.15 |
| TwoTower+DNN-Rerank@200 | small_matrix | 1,200,000 | 1,411 | 0.022686 | 0.974486 | 0.203215 | 0.201383 | 0.687100 | 0.595605 | 7.55 | 1.15 |

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
