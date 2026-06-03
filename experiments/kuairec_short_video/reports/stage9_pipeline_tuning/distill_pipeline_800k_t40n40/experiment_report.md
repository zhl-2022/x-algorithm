# KuaiRec big 蒸馏双塔两阶段 Pipeline 报告

用 ItemCF 蒸馏 Two-Tower 作为召回底座，再接 DNNRanker hard negative 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ItemCF-Distill-TwoTower | big_matrix | 800,000 | 7,174 | 0.006514 | 0.338584 | 0.031221 | 0.027405 | 0.604214 | 0.678463 | 4.04 | 0.38 |
| DNNRanker | big_matrix | 1,359,393 | 7,174 | 0.000885 | 0.072763 | 0.003175 | 0.017431 | 0.726550 | 0.638297 | 3.96 | 20.18 |
| DistillTwoTower+DNN-Blend@100a0.5 | big_matrix | 800,000 | 7,174 | 0.006599 | 0.325481 | 0.029738 | 0.045861 | 0.671385 | 0.651121 | 7.99 | 20.56 |
| DistillTwoTower+DNN-Blend@100a0.75 | big_matrix | 800,000 | 7,174 | 0.006579 | 0.327432 | 0.029630 | 0.048192 | 0.670703 | 0.656902 | 7.99 | 20.56 |
| DistillTwoTower+DNN-Rerank@100 | big_matrix | 800,000 | 7,174 | 0.006614 | 0.331057 | 0.029748 | 0.049683 | 0.669991 | 0.769732 | 7.99 | 20.56 |
| DistillTwoTower+DNN-Blend@200a0.5 | big_matrix | 800,000 | 7,174 | 0.008522 | 0.407025 | 0.038479 | 0.056301 | 0.665905 | 0.657427 | 7.99 | 20.56 |
| DistillTwoTower+DNN-Blend@200a0.75 | big_matrix | 800,000 | 7,174 | 0.008458 | 0.407722 | 0.038914 | 0.060030 | 0.664600 | 0.654231 | 7.99 | 20.56 |
| DistillTwoTower+DNN-Rerank@200 | big_matrix | 800,000 | 7,174 | 0.008508 | 0.410231 | 0.039138 | 0.061521 | 0.663471 | 0.745776 | 7.99 | 20.56 |

## 说明

- ItemCF-Distill-TwoTower：用 ItemCF TopK teacher 样本和真实行为样本蒸馏训练 Two-Tower。
- DNNRanker：融合用户、视频、类别、caption 哈希文本和统计特征的 MLP 排序模型，并追加 559,393 条蒸馏 Two-Tower 高分难负样本训练。
- DistillTwoTower+DNN-Blend@100a0.5：Two-Tower 先取 Top100 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@100a0.75：Two-Tower 先取 Top100 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@100：Two-Tower 先取 Top100 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@200a0.5：Two-Tower 先取 Top200 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@200a0.75：Two-Tower 先取 Top200 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@200：Two-Tower 先取 Top200 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
