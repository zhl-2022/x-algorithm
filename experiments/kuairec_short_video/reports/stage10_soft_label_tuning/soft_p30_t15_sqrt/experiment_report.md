# KuaiRec big 蒸馏双塔两阶段 Pipeline 报告

用 ItemCF 蒸馏 Two-Tower 作为召回底座，再接 DNNRanker hard negative 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ItemCF-Distill-TwoTower | big_matrix | 2,000,000 | 7,174 | 0.012373 | 0.467103 | 0.051765 | 0.040362 | 0.619820 | 0.671715 | 7.98 | 0.36 |
| DNNRanker | big_matrix | 2,559,393 | 7,174 | 0.000105 | 0.007945 | 0.000384 | 0.015753 | 0.760781 | 0.621132 | 7.33 | 19.83 |
| DistillTwoTower+DNN-Blend@100a0.5 | big_matrix | 2,000,000 | 7,174 | 0.014698 | 0.535963 | 0.055883 | 0.048937 | 0.676812 | 0.655660 | 15.31 | 20.19 |
| DistillTwoTower+DNN-Blend@100a0.75 | big_matrix | 2,000,000 | 7,174 | 0.014902 | 0.547672 | 0.055372 | 0.053039 | 0.680917 | 0.618710 | 15.31 | 20.19 |
| DistillTwoTower+DNN-Rerank@100 | big_matrix | 2,000,000 | 7,174 | 0.015095 | 0.552411 | 0.055224 | 0.055276 | 0.681210 | 0.667802 | 15.31 | 20.19 |

## 说明

- ItemCF-Distill-TwoTower：用 ItemCF TopK teacher 样本和真实行为样本蒸馏训练 Two-Tower。
- DNNRanker：融合用户、视频、类别、caption 哈希文本和统计特征的 MLP 排序模型，并追加 559,393 条蒸馏 Two-Tower 高分难负样本训练。
- DistillTwoTower+DNN-Blend@100a0.5：Two-Tower 先取 Top100 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@100a0.75：Two-Tower 先取 Top100 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@100：Two-Tower 先取 Top100 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
