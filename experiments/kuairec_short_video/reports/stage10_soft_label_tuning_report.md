# KuaiRec 阶段十：Soft Label 蒸馏精调报告

## 1. 实验目标

阶段十围绕阶段九最佳 `2m_t40n120` 继续精调 teacher soft label、正样本比例、teacher 比例和随机负样本比例。

## 2. 结果汇总

| 实验 | 最佳模型 | Recall@20 | NDCG@20 | AUC | 判断 |
|---|---|---:|---:|---:|---|
| `soft_p30_t15_sqrt` | `DistillTwoTower+DNN-Blend@100a0.5` | 0.014698 | 0.055883 | 0.676812 | 超过 Stage9 |
| `soft_p30_t15_linear` | `DistillTwoTower+DNN-Rerank@100` | 0.013347 | 0.053271 | 0.692285 | 超过 Stage9 |
| `soft_p30_t10_linear` | `DistillTwoTower+DNN-Blend@100a0.5` | 0.012395 | 0.052608 | 0.683012 | 超过 Stage9 |
| `soft_replay_p29_t14_linear` | `DistillTwoTower+DNN-Blend@100a0.75` | 0.012197 | 0.051595 | 0.694056 | 超过 Stage9 |

## 3. 阶段结论

阶段十最佳为 `soft_p30_t15_sqrt` 的 `DistillTwoTower+DNN-Blend@100a0.5`，`NDCG@20=0.055883`。

该结果达到或超过阶段九最佳，说明 soft label/比例修正带来收益。
