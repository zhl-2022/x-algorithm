# KuaiRec big_matrix 补强方案设计

## 1. 当前问题

KuaiRec `big_matrix.csv` 已完成 hard negative、in-batch negative 和两阶段重排实验，但当前神经 TopK 仍明显弱于 ItemCF。

| 模型/实验 | NDCG@20 | 结论 |
|---|---:|---|
| ItemCF | 0.065921 | 当前 big TopK 最强 |
| Stage 4 DNNRanker hard negative | 0.006151 | Ranker 有局部收益 |
| Stage 4 TwoTower+DNN-Rerank@500 | 0.004991 | hard negative pipeline 最好 |
| Stage 5 Two-Tower in-batch | 0.004706 | 比 BCE Two-Tower 明显提升 |
| Stage 5 TwoTower+DNN-Rerank@100 | 0.005245 | in-batch pipeline 最好，但仍弱于 ItemCF |

核心判断：当前瓶颈不只是 Ranker，而是 Two-Tower 在 big 场景下的候选召回质量不足。Ranker 只能重排 Two-Tower 召回出的候选，如果候选本身质量低，重排上限也会低。

## 2. 补强方向对比

| 方案 | 思路 | 优点 | 风险 | 优先级 |
|---|---|---|---|---|
| ItemCF 蒸馏 Two-Tower | 用 ItemCF 的高质量 TopK 结果指导 Two-Tower 学习 | 直接针对当前 ItemCF 强、Two-Tower 弱的问题 | 需要设计蒸馏样本和 teacher 分数 | 最高 |
| LightGCN / 图召回 | 将用户-视频交互建成二部图，学习图 embedding | 能显式利用协同结构 | 新增图训练脚本和采样复杂度 | 中 |
| 序列兴趣模型 | 用用户最近观看序列建模短期兴趣 | 更接近短视频业务 | 当前数据字段和时序处理需要再确认 | 中 |
| 轻量文本 encoder | 替换 caption 哈希 embedding | 提升内容语义理解 | 对 big TopK 的协同瓶颈不一定直接有效 | 低 |

默认建议先做 **ItemCF 蒸馏 Two-Tower**，因为它最贴合当前实验证据。

## 2.1 已完成的中等规模验证

2026-06-03 已按“预处理缓存 + 分批启动”的方式完成四个补强方向的第一轮中等规模验证。
统一口径为 `big_matrix.csv`、`watch_ratio >= 0.8`、`800,000` 训练样本和 `7,174` 个评估用户。

| 方向 | Recall@20 | NDCG@20 | AUC | 当前判断 |
|---|---:|---:|---:|---|
| ItemCF 蒸馏 Two-Tower | 0.006874 | 0.033562 | 0.604001 | 本轮最有效，明显高于 stage5 best pipeline |
| LightGCN / 图召回 | 0.002192 | 0.008166 | 0.509714 | 第一版覆盖率高，但排序质量不足 |
| GRU 序列兴趣模型 | 0.000490 | 0.000868 | 0.704649 | AUC 高但 TopK 弱，需要重做序列候选训练 |
| TextCNN 双塔 | 0.003039 | 0.007997 | 0.554473 | 可运行，但单独文本增强不足 |

阶段结论：ItemCF 蒸馏 Two-Tower 已将 big 场景神经 TopK 从 stage5 best pipeline
`NDCG@20=0.005245` 提升到 `0.033562`，方向有效；但仍低于 big ItemCF `0.065921`。
后续若继续投入训练，应优先扩大蒸馏样本并优化 teacher 分数，而不是平均推进四个方向。

## 2.2 阶段八执行结果

2026-06-03 继续执行阶段八，验证 2M 蒸馏放大、蒸馏 pipeline、LightGCN 调参和序列模型修复。

| 实验 | NDCG@20 | 判断 |
|---|---:|---|
| 800k ItemCF 蒸馏 Two-Tower | 0.033562 | 阶段七基线 |
| 2M ItemCF 蒸馏 Two-Tower | 0.027320 | 放大样本后 TopK 下降 |
| DistillTwoTower+DNN-Rerank@200 | 0.044560 | 当前最佳神经 pipeline |
| LightGCN best `l1_e10` | 0.011906 | 调参后提升，但仍不是主线 |
| GRU sequence fixed | 0.000914 | 修复 padding 后仍弱 |

新结论：不要直接盲目扩到 5M。当前更有效的方向是精调蒸馏召回 + Ranker pipeline，
包括 teacher 分数形状、蒸馏正负样本比例、hard negative 配比、`candidate_k=100/200` 和融合权重。

## 3. ItemCF 蒸馏 Two-Tower 方案

### 3.1 训练目标

让 Two-Tower 学习 ItemCF 的 TopK 排序能力，使用户向量和视频向量的点积分数更接近 ItemCF teacher 的推荐偏好。

### 3.2 样本构造

对每个训练用户构造三类样本：

| 样本类型 | 来源 | 标签/目标 |
|---|---|---|
| 行为正样本 | 用户真实 `watch_ratio >= 0.8` 的视频 | 强正样本 |
| 蒸馏正样本 | ItemCF 为用户召回的 TopK 视频 | teacher 正样本或软标签 |
| 负样本 | 用户未接近完播、且不在 ItemCF TopK 的视频 | 负样本 |

第一版不引入复杂 listwise loss，优先使用可控的二分类蒸馏：

```text
target = 1.0                      for 真实正样本
target = itemcf_score_normalized  for ItemCF teacher 样本
target = 0.0                      for 负样本
```

### 3.3 模型保持不变

第一版不改 Two-Tower 架构，只改训练数据和 loss 目标，避免同时引入过多变量。

保持当前结构：

- 用户塔：`user_id embedding + user dense features -> MLP -> user vector`
- 视频塔：`video_id embedding + category embedding + caption hash embedding + video dense features -> MLP -> video vector`
- 打分：向量归一化后点积。

### 3.4 成功标准

| 指标 | 成功标准 |
|---|---|
| Two-Tower `NDCG@20` | 高于 stage5 in-batch 的 `0.004706` |
| Pipeline `NDCG@20` | 高于 stage5 best pipeline 的 `0.005245` |
| 与 ItemCF 差距 | 明显缩小，优先目标达到 `NDCG@20 >= 0.02` |
| 训练稳定性 | 不出现 `nan/NaN`，AUC 和 TopK 指标可复现 |

## 4. 实验执行顺序

1. Smoke test：在 `big_matrix.csv --max-rows 50000` 上验证蒸馏样本构造和训练脚本。
2. 小规模正式实验：使用 50 万到 100 万训练样本，确认蒸馏方向是否提升 Two-Tower TopK。
3. 中规模实验：扩展到 200 万训练样本，对齐 stage4/stage5 口径。
4. Pipeline 验证：用蒸馏 Two-Tower 作为召回底座，再接 DNNRanker hard negative 重排。
5. MLU 记录：记录训练时间、吞吐、显存和输出指标。

## 5. 暂不优先做的事项

- 暂不切 Tenrec 或 KuaiRand，避免在 KuaiRec big 问题尚未解释清楚时引入新变量。
- 暂不直接加深 MLP 层数，因为当前问题更像召回训练信号不足，而不是模型容量不够。
- 暂不做完整在线服务化，因为当前项目目标是离线训练评测闭环和简历沉淀。

## 6. 阶段结论

KuaiRec big 的下一步应该从“继续堆同类神经模型”转为“让神经召回学习强协同信号”。ItemCF 蒸馏 Two-Tower 是最直接的补强方向：它既利用了当前最强的 ItemCF 结果，又保留了 Two-Tower 适合向量召回和后续工程化的优势。
