# KuaiRec big 蒸馏双塔两阶段 Pipeline 报告

用 ItemCF 蒸馏 Two-Tower 作为召回底座，再接 DNNRanker hard negative 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ItemCF-Distill-TwoTower | big_matrix | 2,000,000 | 7,174 | 0.006250 | 0.321578 | 0.028065 | 0.036353 | 0.598066 | 0.680752 | 7.27 | 0.27 |
| DNNRanker | big_matrix | 2,559,393 | 7,174 | 0.001413 | 0.069278 | 0.003670 | 0.013609 | 0.764700 | 0.629068 | 6.98 | 20.08 |
| DistillTwoTower+DNN-Blend@100a0.5 | big_matrix | 2,000,000 | 7,174 | 0.008474 | 0.377753 | 0.039845 | 0.064877 | 0.699481 | 0.671878 | 14.25 | 20.35 |
| DistillTwoTower+DNN-Blend@100a0.75 | big_matrix | 2,000,000 | 7,174 | 0.008507 | 0.381935 | 0.039473 | 0.069724 | 0.699426 | 0.630601 | 14.25 | 20.35 |
| DistillTwoTower+DNN-Rerank@100 | big_matrix | 2,000,000 | 7,174 | 0.008492 | 0.382771 | 0.039138 | 0.071682 | 0.698508 | 0.669534 | 14.25 | 20.35 |
| DistillTwoTower+DNN-Blend@200a0.5 | big_matrix | 2,000,000 | 7,174 | 0.008501 | 0.394062 | 0.038678 | 0.070377 | 0.685548 | 0.688418 | 14.25 | 20.35 |
| DistillTwoTower+DNN-Blend@200a0.75 | big_matrix | 2,000,000 | 7,174 | 0.008596 | 0.398662 | 0.037903 | 0.074012 | 0.684043 | 0.640627 | 14.25 | 20.35 |
| DistillTwoTower+DNN-Rerank@200 | big_matrix | 2,000,000 | 7,174 | 0.008650 | 0.400474 | 0.037265 | 0.076249 | 0.682372 | 0.663466 | 14.25 | 20.35 |

## 说明

- ItemCF-Distill-TwoTower：用 ItemCF TopK teacher 样本和真实行为样本蒸馏训练 Two-Tower。
- DNNRanker：融合用户、视频、类别、caption 哈希文本和统计特征的 MLP 排序模型，并追加 559,393 条蒸馏 Two-Tower 高分难负样本训练。
- DistillTwoTower+DNN-Blend@100a0.5：Two-Tower 先取 Top100 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@100a0.75：Two-Tower 先取 Top100 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@100：Two-Tower 先取 Top100 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@200a0.5：Two-Tower 先取 Top200 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@200a0.75：Two-Tower 先取 Top200 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@200：Two-Tower 先取 Top200 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
