# KuaiRec 阶段五：big_matrix Two-Tower in-batch negative 实验

## 1. 实验目标

阶段四显示 big 场景下 Two-Tower BCE 召回很弱，`NDCG@20=0.000937`。
本阶段改用 in-batch negative 训练 Two-Tower，让模型在一个 batch 内区分真实正样本和其他视频，
更接近召回任务。

## 2. 实验设置

| 项目 | 数值 |
|---|---:|
| 数据源 | `big_matrix.csv` |
| 标签 | `watch_ratio >= 0.8` |
| Two-Tower loss | `inbatch` |
| Two-Tower 正样本训练行 | 2,000,000 |
| Ranker 训练样本 | 2,559,393 |
| Epochs | 3 |
| Two-Tower batch size | 4,096 |
| Ranker hard negatives | 559,393 |
| Candidate K | 100/200/500 |

## 3. 核心结果

| 模型 | Stage 4 NDCG@20 | Stage 5 NDCG@20 | 结论 |
|---|---:|---:|---|
| Two-Tower | 0.000937 | 0.004706 | 召回底座明显改善 |
| DNNRanker | 0.006151 | 0.002005 | Ranker 全量 TopK 下降 |
| Best pipeline | 0.004991 | 0.005245 | pipeline 小幅提升 |
| ItemCF | 0.065921 | 0.065921 | 仍是 big TopK 最强 |

## 4. 结论

in-batch negative 对 Two-Tower 召回是有效的，`NDCG@20` 从 `0.000937` 提升到 `0.004706`。
但是它仍没有解决 big 场景的整体 TopK 问题，最佳 pipeline 只有 `NDCG@20=0.005245`，
远低于 ItemCF。

当前判断：

- `small_matrix.csv` 上 Ranker hard negative 已经成功。
- `big_matrix.csv` 上神经模型还需要更强的召回学习方式。
- 后续如果继续深挖，应尝试更大训练样本、用户历史序列建模、LightGCN 或 item-to-item 蒸馏。

