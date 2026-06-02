# KuaiRec 阶段三：Ranker 优化实验总结

## 1. 本阶段目标

阶段二的主要问题是：`DNNRanker` 的 `AUC` 不低，但 `TwoTower+DNN-Rerank`
没有稳定超过单独 `Two-Tower`。这说明第一版 Ranker 虽然能做二分类判断，
但还没有真正把 Two-Tower 召回候选重新排好。

阶段三的目标是：

> 让 Ranker 重排真正超过 Two-Tower，使两阶段 pipeline 产生明确收益。

## 2. 本轮怎么优化 Ranker

本轮新增了 hard negative 训练能力：

1. 先训练 `Two-Tower`。
2. 用训练好的 `Two-Tower` 给训练集里的负样本打分。
3. 每个用户最多选出 100 条 Two-Tower 打分很高、但标签为负的样本。
4. 把这些样本追加到 Ranker 训练集中。
5. 提高 Ranker 正样本权重到 4，让模型更重视完播/接近完播样本。

小白可以这样理解：

> Two-Tower 容易把某些“不该推荐的视频”召回到前面，hard negative 就是把这些难错题挑出来，
> 专门拿给 Ranker 学，让 Ranker 学会在重排时把它们压下去。

## 3. 实验设置

| 项目 | 数值 |
|---|---:|
| 数据源 | `small_matrix.csv` |
| 标签 | `watch_ratio >= 0.8` |
| 神经模型基础训练样本 | 1,200,000 |
| Ranker hard negative pool | 1,500,000 |
| 每用户 hard negatives | 100 |
| 实际追加 hard negatives | 141,100 |
| Ranker 最终训练样本 | 1,341,100 |
| Epochs | 3 |
| Batch size | 8,192 |
| Ranker hidden dims | `256,128,64` |
| Ranker 正样本权重 | 4 |
| 候选集 | `candidate_k=50/100/200` |
| 融合权重 | `alpha=0/0.25/0.5/0.75/1` |

## 4. 核心结果

| 模型 | Recall@20 | HitRate@20 | Precision@20 | NDCG@20 | AUC | 结论 |
|---|---:|---:|---:|---:|---:|---|
| Two-Tower | 0.018152 | 0.917080 | 0.132282 | 0.159630 | 0.508734 | 当前召回底座 |
| DNNRanker | 0.028238 | 0.978738 | 0.213855 | 0.240050 | 0.653649 | 全量打分 TopK 最强 |
| TwoTower+DNN-Rerank@50 | 0.019543 | 0.936924 | 0.142275 | 0.170111 | 0.690666 | 重排超过 Two-Tower |
| TwoTower+DNN-Rerank@100 | 0.022158 | 0.963147 | 0.162544 | 0.197553 | 0.654515 | 候选更大，效果继续提升 |
| TwoTower+DNN-Rerank@200 | 0.022686 | 0.974486 | 0.167080 | 0.203215 | 0.687100 | 本轮最佳两阶段 pipeline |

## 5. 和阶段二相比提升多少

| 对比项 | 阶段二 | 阶段三 | 提升 |
|---|---:|---:|---:|
| Two-Tower `NDCG@20` | 0.153744 | 0.159630 | +3.83% |
| DNNRanker `NDCG@20` | 0.142891 | 0.240050 | +68.00% |
| 最佳两阶段 pipeline `NDCG@20` | 0.153744 | 0.203215 | +32.18% |

说明：阶段二最佳两阶段 pipeline 未超过 Two-Tower，因此这里用阶段二最佳 Two-Tower
作为 pipeline 需要超过的目标。

## 6. 本轮结论

本轮已经解决阶段二留下的核心问题：

- Ranker 全量打分已经明显超过 Two-Tower。
- `TwoTower+DNN-Rerank@200` 也明显超过单独 Two-Tower。
- hard negative 训练对 Ranker TopK 指标有明显帮助。
- `candidate_k=200` 比 50 和 100 更好，说明本轮 Ranker 有能力从更大候选集中挑出更靠前的正样本。

## 7. 下一步

阶段三已经证明 Ranker 优化方向有效。下一步不建议继续只在 `small_matrix.csv` 上调参，
而应该把这个方案迁移到 `big_matrix.csv`：

1. 在 `big_matrix.csv` 上做 hard negative Ranker。
2. 优化 Two-Tower 召回 loss，例如 in-batch negative。
3. 对比 `ItemCF`、`Two-Tower`、`DNNRanker` 和两阶段 pipeline 的大规模 TopK 指标。
4. 如果要体现工程能力，再做 MLU 双卡 DDP。

