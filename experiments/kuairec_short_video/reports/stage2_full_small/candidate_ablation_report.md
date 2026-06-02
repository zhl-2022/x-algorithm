# KuaiRec Candidate K 消融报告

比较不同 `candidate_k` 对最终 Top20 结果的影响。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TwoTower+DNN-Blend@50a0 | small_matrix | 3,595,097 | 1,410 | 0.027291 | 0.908511 | 0.149288 | 0.138563 | 0.520547 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@50a0.25 | small_matrix | 3,595,097 | 1,410 | 0.027135 | 0.904255 | 0.147315 | 0.134055 | 0.520544 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@50a0.5 | small_matrix | 3,595,097 | 1,410 | 0.027289 | 0.904965 | 0.147579 | 0.133754 | 0.520539 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@50a0.75 | small_matrix | 3,595,097 | 1,410 | 0.027415 | 0.909929 | 0.146681 | 0.134656 | 0.520532 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Rerank@50 | small_matrix | 3,595,097 | 1,410 | 0.028026 | 0.917021 | 0.145412 | 0.139766 | 0.520524 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@100a0 | small_matrix | 3,595,097 | 1,410 | 0.027291 | 0.908511 | 0.149288 | 0.138563 | 0.533662 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@100a0.25 | small_matrix | 3,595,097 | 1,410 | 0.027135 | 0.904255 | 0.147315 | 0.133754 | 0.533660 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@100a0.5 | small_matrix | 3,595,097 | 1,410 | 0.027298 | 0.904965 | 0.147600 | 0.132852 | 0.533651 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@100a0.75 | small_matrix | 3,595,097 | 1,410 | 0.027547 | 0.912057 | 0.147167 | 0.133153 | 0.533635 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Rerank@100 | small_matrix | 3,595,097 | 1,410 | 0.028425 | 0.921986 | 0.147254 | 0.137060 | 0.533610 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@200a0 | small_matrix | 3,595,097 | 1,410 | 0.027291 | 0.908511 | 0.149288 | 0.138563 | 0.535822 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@200a0.25 | small_matrix | 3,595,097 | 1,410 | 0.027135 | 0.904255 | 0.147315 | 0.133754 | 0.535876 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@200a0.5 | small_matrix | 3,595,097 | 1,410 | 0.027298 | 0.904965 | 0.147600 | 0.132852 | 0.535915 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Blend@200a0.75 | small_matrix | 3,595,097 | 1,410 | 0.027559 | 0.912057 | 0.147212 | 0.133153 | 0.535933 | N/A | 16.93 | 1.50 |
| TwoTower+DNN-Rerank@200 | small_matrix | 3,595,097 | 1,410 | 0.028359 | 0.921277 | 0.146904 | 0.135257 | 0.535913 | N/A | 16.93 | 1.50 |

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
