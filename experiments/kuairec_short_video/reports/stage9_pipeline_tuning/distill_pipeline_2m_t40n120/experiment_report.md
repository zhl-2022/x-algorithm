# KuaiRec big 蒸馏双塔两阶段 Pipeline 报告

用 ItemCF 蒸馏 Two-Tower 作为召回底座，再接 DNNRanker hard negative 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ItemCF-Distill-TwoTower | big_matrix | 2,000,000 | 7,174 | 0.009351 | 0.413856 | 0.041418 | 0.041760 | 0.614358 | 0.675834 | 8.47 | 0.39 |
| DNNRanker | big_matrix | 2,559,393 | 7,174 | 0.000217 | 0.014776 | 0.000657 | 0.015474 | 0.761383 | 0.619076 | 7.47 | 21.17 |
| DistillTwoTower+DNN-Blend@100a0.5 | big_matrix | 2,000,000 | 7,174 | 0.011512 | 0.479788 | 0.047521 | 0.054810 | 0.684191 | 0.659482 | 15.94 | 21.57 |
| DistillTwoTower+DNN-Blend@100a0.75 | big_matrix | 2,000,000 | 7,174 | 0.011545 | 0.484249 | 0.047939 | 0.057420 | 0.685393 | 0.625609 | 15.94 | 21.57 |
| DistillTwoTower+DNN-Rerank@100 | big_matrix | 2,000,000 | 7,174 | 0.011560 | 0.486200 | 0.048158 | 0.059471 | 0.685021 | 0.672621 | 15.94 | 21.57 |
| DistillTwoTower+DNN-Blend@200a0.5 | big_matrix | 2,000,000 | 7,174 | 0.011853 | 0.486340 | 0.045455 | 0.053225 | 0.677591 | 0.659147 | 15.94 | 21.57 |
| DistillTwoTower+DNN-Blend@200a0.75 | big_matrix | 2,000,000 | 7,174 | 0.012109 | 0.491915 | 0.045782 | 0.058259 | 0.678502 | 0.626430 | 15.94 | 21.57 |
| DistillTwoTower+DNN-Rerank@200 | big_matrix | 2,000,000 | 7,174 | 0.012324 | 0.497073 | 0.045962 | 0.061242 | 0.677883 | 0.664003 | 15.94 | 21.57 |

## 说明

- ItemCF-Distill-TwoTower：用 ItemCF TopK teacher 样本和真实行为样本蒸馏训练 Two-Tower。
- DNNRanker：融合用户、视频、类别、caption 哈希文本和统计特征的 MLP 排序模型，并追加 559,393 条蒸馏 Two-Tower 高分难负样本训练。
- DistillTwoTower+DNN-Blend@100a0.5：Two-Tower 先取 Top100 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@100a0.75：Two-Tower 先取 Top100 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@100：Two-Tower 先取 Top100 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@200a0.5：Two-Tower 先取 Top200 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@200a0.75：Two-Tower 先取 Top200 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@200：Two-Tower 先取 Top200 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
