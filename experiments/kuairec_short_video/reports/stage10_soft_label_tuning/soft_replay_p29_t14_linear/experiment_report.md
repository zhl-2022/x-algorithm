# KuaiRec big 蒸馏双塔两阶段 Pipeline 报告

用 ItemCF 蒸馏 Two-Tower 作为召回底座，再接 DNNRanker hard negative 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ItemCF-Distill-TwoTower | big_matrix | 2,000,000 | 7,174 | 0.010326 | 0.426261 | 0.045149 | 0.042972 | 0.613541 | 0.673973 | 9.01 | 0.27 |
| DNNRanker | big_matrix | 2,559,393 | 7,174 | 0.000214 | 0.015194 | 0.000636 | 0.016219 | 0.761580 | 0.623020 | 10.97 | 20.75 |
| DistillTwoTower+DNN-Blend@100a0.5 | big_matrix | 2,000,000 | 7,174 | 0.012032 | 0.485364 | 0.051210 | 0.052386 | 0.692200 | 0.656050 | 19.99 | 21.02 |
| DistillTwoTower+DNN-Blend@100a0.75 | big_matrix | 2,000,000 | 7,174 | 0.012197 | 0.492612 | 0.051595 | 0.055742 | 0.694056 | 0.620415 | 19.99 | 21.02 |
| DistillTwoTower+DNN-Rerank@100 | big_matrix | 2,000,000 | 7,174 | 0.012189 | 0.495958 | 0.051452 | 0.058072 | 0.693939 | 0.666262 | 19.99 | 21.02 |

## 说明

- ItemCF-Distill-TwoTower：用 ItemCF TopK teacher 样本和真实行为样本蒸馏训练 Two-Tower。
- DNNRanker：融合用户、视频、类别、caption 哈希文本和统计特征的 MLP 排序模型，并追加 559,393 条蒸馏 Two-Tower 高分难负样本训练。
- DistillTwoTower+DNN-Blend@100a0.5：Two-Tower 先取 Top100 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@100a0.75：Two-Tower 先取 Top100 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@100：Two-Tower 先取 Top100 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
