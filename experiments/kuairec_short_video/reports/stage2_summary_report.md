# KuaiRec 阶段二实验总结

## 1. 本阶段目标

第一轮 KuaiRec `small_matrix.csv` 已经跑通 `Popularity`、`Category`、`ItemCF`、`MF`、
`Two-Tower`、`DNNRanker` 和 `TwoTower+DNN-Rerank`。第一轮最佳 `NDCG@20`
是 `Two-Tower=0.143577`。

阶段二不再只是加模型，而是验证三个问题：

1. 正反馈阈值从 `watch_ratio >= 1.0` 放宽到 `watch_ratio >= 0.8` 是否更适合 TopK 推荐。
2. 神经模型训练样本从 120 万扩展到 `small_matrix.csv` 全量训练交互后，指标是否提升。
3. 切换到 `big_matrix.csv` 后，现有模型是否还能保持稳定 TopK 效果。

## 2. 当前底座模型架构

当前底座不是 Transformer，也不是 Grok/LLM，而是推荐系统里的轻量表征模型：

| 模型 | 结构 | 当前用途 |
|---|---|---|
| `MF` | 用户 embedding + 视频 embedding 点积 + bias | 学习协同过滤信号 |
| `Two-Tower` | 用户塔 MLP + 视频塔 MLP，输出 64 维归一化向量 | 向量召回 |
| `DNNRanker` | ID、类别、caption 哈希文本、统计特征拼接后进入 MLP | 完播概率排序 |
| `TwoTower+DNN-Blend/Rerank` | 双塔先取候选，再按 Ranker 与 Two-Tower 分数融合重排 | 两阶段推荐 pipeline |

## 3. 实验设置

| 实验 | 数据 | 标签 | 神经训练样本 | 候选集 | 融合权重 |
|---|---|---|---:|---|---|
| 标签阈值消融 | `small_matrix.csv` | `watch_ratio >= 0.8` | 1,200,000 | 50/100/200 | 0/0.25/0.5/0.75/1 |
| 全量 small 训练 | `small_matrix.csv` | `watch_ratio >= 1.0` | 3,595,097 | 50/100/200 | 0/0.25/0.5/0.75/1 |
| big 采样放大 | `big_matrix.csv` | `watch_ratio >= 1.0` | 2,000,000 | 50/100 | 0/0.5/1 |

说明：`alpha=0` 表示只使用 Two-Tower 分数，`alpha=1` 表示只使用 Ranker 分数，
中间值表示 `alpha * Ranker + (1 - alpha) * TwoTower`。

## 4. 标签阈值消融结论

`watch_ratio >= 0.8` 的核心结果：

| 模型 | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---:|---:|---:|---|
| Popularity | 0.007644 | 0.053773 | 0.479285 | 只看热门，最低基线 |
| Category | 0.012311 | 0.085090 | 0.539413 | 类别偏好有效 |
| ItemCF | 0.013332 | 0.104011 | 0.476898 | 个性化共现继续提升 |
| MF | 0.018616 | 0.136451 | 0.678856 | embedding 协同信号有效 |
| Two-Tower | 0.017476 | 0.153744 | 0.513206 | 本实验最佳 NDCG |
| DNNRanker | 0.018240 | 0.142891 | 0.647959 | AUC 较高，但 TopK 不如 Two-Tower |

结论：把正反馈放宽到 `watch_ratio >= 0.8` 后，Two-Tower 的 `NDCG@20`
从第一轮 `0.143577` 提升到 `0.153744`。这说明“接近完播”比“严格完播”
更适合当前 TopK 推荐任务，因为它增加了有效正样本，减少了标签过严带来的稀疏问题。

## 5. 全量 small 训练结论

`small_matrix.csv` 全量训练交互的核心结果：

| 模型 | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---:|---:|---:|---|
| MF | 0.027138 | 0.139644 | 0.691779 | 比 120 万采样训练更强 |
| Two-Tower | 0.027291 | 0.149288 | 0.535261 | 本实验最佳 NDCG |
| DNNRanker | 0.028359 | 0.146904 | 0.646642 | Recall 更高，NDCG 略低 |

结论：把训练样本扩展到全量 `small_matrix` 后，Two-Tower 的 `NDCG@20`
从第一轮 `0.143577` 提升到 `0.149288`。DNNRanker 的 `Recall@20=0.028359`
高于 Two-Tower，但 `NDCG@20` 仍略低，说明 Ranker 能找回更多正样本，
但还没有把命中样本稳定排到更靠前的位置。

## 6. big_matrix 采样放大结论

`big_matrix.csv` 使用 200 万神经训练样本的核心结果：

| 模型 | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---:|---:|---:|---|
| Popularity | 0.002464 | 0.008112 | 0.597722 | 大候选池下热门基线很弱 |
| Category | 0.002795 | 0.008871 | 0.606614 | 类别略有提升 |
| ItemCF | 0.022084 | 0.058148 | 0.605107 | 当前 big TopK 最强 |
| MF | 0.001905 | 0.004222 | 0.776112 | AUC 高但 TopK 弱 |
| Two-Tower | 0.000682 | 0.001448 | 0.736317 | 召回 TopK 明显不足 |
| DNNRanker | 0.002511 | 0.004502 | 0.764070 | 排序 AUC 高，但推荐列表不强 |

结论：`big_matrix.csv` 的候选池和交互规模更大，现有神经模型出现“AUC 高但 TopK 弱”的问题。
这通常说明模型能区分部分正负样本，但在全量视频召回时没有学到足够稳定的近邻结构。
后续应优先做 in-batch negative、hard negative、全量训练或更强召回 loss。

## 7. 为什么重排还没超过 Two-Tower

当前 Ranker 重排未稳定超过 Two-Tower，主要原因可能是：

1. Ranker 的训练目标是逐样本二分类，优化的是 `AUC` 和 `LogLoss`，不直接优化 TopK 排名。
2. Two-Tower 的候选集限制了 Ranker 能看到的视频，候选召回质量决定重排上限。
3. Ranker 特征还偏轻量，caption 只是哈希 embedding，没有真正语义编码。
4. 当前负样本分布较简单，模型没有充分学习难负样本。

## 8. 已完成项

- [x] 增加 `--positive-threshold` 标签消融实验。
- [x] 增加全量 `small_matrix.csv` 神经训练实验。
- [x] 增加 `big_matrix.csv` 采样放大实验。
- [x] 增加 `--rerank-blend-alphas`，支持 Two-Tower 与 Ranker 分数融合。
- [x] 增加 `--ranker-positive-weight`，支持 Ranker 正样本加权。
- [x] 修正 pipeline 候选外分数导致的 `LogLoss` 不适用记录。

## 9. 下一步

1. 优先优化 Ranker，让 `TwoTower+DNN-Rerank` 的 `NDCG@20` 超过单独 Two-Tower。
2. 在 `big_matrix.csv` 上做更强负采样和召回训练，解决神经模型 TopK 偏弱问题。
3. 改造 MLU 双卡 DDP，记录单卡/双卡吞吐、显存和指标对比。

