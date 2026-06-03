# KuaiRec big 蒸馏双塔两阶段 Pipeline 报告

用 ItemCF 蒸馏 Two-Tower 作为召回底座，再接 DNNRanker hard negative 重排。

| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ItemCF-Distill-TwoTower | big_matrix | 2,000,000 | 7,174 | 0.006202 | 0.314608 | 0.027320 | 0.046048 | 0.608339 | 0.677686 | 9.11 | 0.40 |
| DNNRanker | big_matrix | 2,559,393 | 7,174 | 0.000615 | 0.042654 | 0.002183 | 0.016499 | 0.763745 | 0.622476 | 10.16 | 20.94 |
| DistillTwoTower+DNN-Blend@100a0 | big_matrix | 2,000,000 | 7,174 | 0.006202 | 0.314608 | 0.027320 | 0.046048 | 0.516586 | 0.943843 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@100a0.25 | big_matrix | 2,000,000 | 7,174 | 0.008419 | 0.393365 | 0.035481 | 0.059004 | 0.699007 | 0.769285 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@100a0.5 | big_matrix | 2,000,000 | 7,174 | 0.009077 | 0.411904 | 0.038299 | 0.066555 | 0.705238 | 0.657763 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@100a0.75 | big_matrix | 2,000,000 | 7,174 | 0.009197 | 0.421662 | 0.039178 | 0.070843 | 0.704443 | 0.623176 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Rerank@100 | big_matrix | 2,000,000 | 7,174 | 0.009365 | 0.429189 | 0.039596 | 0.072334 | 0.703364 | 0.666228 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@200a0 | big_matrix | 2,000,000 | 7,174 | 0.006202 | 0.314608 | 0.027320 | 0.046048 | 0.496338 | 0.937507 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@200a0.25 | big_matrix | 2,000,000 | 7,174 | 0.009726 | 0.435322 | 0.038138 | 0.060309 | 0.678170 | 0.779353 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@200a0.5 | big_matrix | 2,000,000 | 7,174 | 0.010792 | 0.468776 | 0.042824 | 0.069444 | 0.684367 | 0.674544 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@200a0.75 | big_matrix | 2,000,000 | 7,174 | 0.010940 | 0.471425 | 0.044004 | 0.073453 | 0.683124 | 0.635127 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Rerank@200 | big_matrix | 2,000,000 | 7,174 | 0.011031 | 0.473237 | 0.044560 | 0.075783 | 0.681781 | 0.663163 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@500a0 | big_matrix | 2,000,000 | 7,174 | 0.006202 | 0.314608 | 0.027320 | 0.046048 | 0.477020 | 0.891534 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@500a0.25 | big_matrix | 2,000,000 | 7,174 | 0.009943 | 0.436716 | 0.035756 | 0.055276 | 0.664053 | 0.765169 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@500a0.5 | big_matrix | 2,000,000 | 7,174 | 0.010950 | 0.459019 | 0.036815 | 0.055835 | 0.672928 | 0.678187 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Blend@500a0.75 | big_matrix | 2,000,000 | 7,174 | 0.010820 | 0.459297 | 0.036387 | 0.059657 | 0.671384 | 0.639068 | 19.26 | 21.34 |
| DistillTwoTower+DNN-Rerank@500 | big_matrix | 2,000,000 | 7,174 | 0.010749 | 0.453304 | 0.035964 | 0.061801 | 0.669798 | 0.650206 | 19.26 | 21.34 |

## 说明

- ItemCF-Distill-TwoTower：用 ItemCF TopK teacher 样本和真实行为样本蒸馏训练 Two-Tower。
- DNNRanker：融合用户、视频、类别、caption 哈希文本和统计特征的 MLP 排序模型，并追加 559,393 条蒸馏 Two-Tower 高分难负样本训练。
- DistillTwoTower+DNN-Blend@100a0：Two-Tower 先取 Top100 候选，再按 0*Ranker + 1*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@100a0.25：Two-Tower 先取 Top100 候选，再按 0.25*Ranker + 0.75*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@100a0.5：Two-Tower 先取 Top100 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@100a0.75：Two-Tower 先取 Top100 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@100：Two-Tower 先取 Top100 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@200a0：Two-Tower 先取 Top200 候选，再按 0*Ranker + 1*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@200a0.25：Two-Tower 先取 Top200 候选，再按 0.25*Ranker + 0.75*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@200a0.5：Two-Tower 先取 Top200 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@200a0.75：Two-Tower 先取 Top200 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@200：Two-Tower 先取 Top200 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@500a0：Two-Tower 先取 Top500 候选，再按 0*Ranker + 1*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@500a0.25：Two-Tower 先取 Top500 候选，再按 0.25*Ranker + 0.75*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@500a0.5：Two-Tower 先取 Top500 候选，再按 0.5*Ranker + 0.5*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Blend@500a0.75：Two-Tower 先取 Top500 候选，再按 0.75*Ranker + 0.25*TwoTower 融合重排 Top20。
- DistillTwoTower+DNN-Rerank@500：Two-Tower 先取 Top500 候选，再按 1*Ranker + 0*TwoTower 融合重排 Top20。
