# KuaiRec 阶段九：蒸馏 Pipeline 精调报告

## 1. 实验目标

阶段九继续聚焦 `big_matrix.csv`，不切换新数据集，也不直接把训练样本扩到 5M。本轮目标是围绕阶段八最佳结果
`DistillTwoTower+DNN-Rerank@200 NDCG@20=0.044560`，精调 ItemCF 蒸馏 Two-Tower 与 DNNRanker 的两阶段 pipeline。

核心问题是：

> teacher 样本比例、随机负样本比例和候选集大小，是否能继续提升 KuaiRec big 场景的神经 TopK 推荐质量？

## 2. 固定设置

| 项目 | 设置 |
|---|---|
| 数据集 | `big_matrix.csv` |
| 标签 | `watch_ratio >= 0.8` |
| 评估用户数 | 7,174 |
| 训练设备 | srv4 `xalgorithm-mlu` 容器，`MLU_VISIBLE_DEVICES=2` |
| Epochs | 3 |
| Batch size | 8,192 |
| Embedding dim | 64 |
| Tower dim | 64 |
| Hidden dim | 128 |
| Ranker hidden dims | `256,128,64` |
| Ranker positive weight | 4 |
| Ranker hard negative pool | 3,000,000 |
| 每用户 Ranker hard negatives | 80 |
| candidate_k | 100, 200 |
| rerank alpha | 0.5, 0.75, 1 |

## 3. 三组实验设计

| 实验名 | 训练样本 | teacher/user | negative/user | 设计目的 |
|---|---:|---:|---:|---|
| `distill_pipeline_800k_t40n40` | 800,000 | 40 | 40 | 验证较小蒸馏样本接 Ranker 后是否优于阶段八 |
| `distill_pipeline_2m_t40n120` | 2,000,000 | 40 | 120 | 降低 teacher 占比，提高随机负样本比例 |
| `distill_pipeline_2m_t120n40` | 2,000,000 | 120 | 40 | 提高 teacher 占比，增强 ItemCF 排序信号 |

执行脚本：

```bash
cd /root/zhl/x-algorithm
bash experiments/kuairec_short_video/scripts/start_stage9_pipeline_tuning.sh
```

服务器日志显示三组实验均已完成，训练结束后 Card 2/3 显存均为 `0 MiB/81920 MiB`。

## 4. 实验结果

### 4.1 单独蒸馏双塔结果

| 实验 | Recall@20 | NDCG@20 | AUC | 判断 |
|---|---:|---:|---:|---|
| `800k_t40n40` | 0.006514 | 0.031221 | 0.604214 | 低于阶段七 800k 蒸馏基线 |
| `2m_t40n120` | 0.009351 | 0.041418 | 0.614358 | 单独双塔本轮最好 |
| `2m_t120n40` | 0.006250 | 0.028065 | 0.598066 | teacher 占比过高后 TopK 下降 |

### 4.2 Pipeline 最佳结果

| 实验 | 最佳 pipeline | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---|---:|---:|---:|---|
| `800k_t40n40` | `DistillTwoTower+DNN-Rerank@200` | 0.008508 | 0.039138 | 0.663471 | 没超过阶段八 |
| `2m_t40n120` | `DistillTwoTower+DNN-Rerank@100` | 0.011560 | 0.048158 | 0.685021 | 本轮最佳，超过阶段八 |
| `2m_t120n40` | `DistillTwoTower+DNN-Blend@100a0.5` | 0.008474 | 0.039845 | 0.699481 | AUC 高，但 TopK 不足 |

## 5. 与关键基线对比

| 模型/阶段 | NDCG@20 | 说明 |
|---|---:|---|
| Stage 5 best pipeline | 0.005245 | 未蒸馏前 big 神经 pipeline 基线 |
| Stage 7 ItemCF-Distill-TwoTower | 0.033562 | 第一版蒸馏双塔 |
| Stage 8 best pipeline | 0.044560 | 蒸馏双塔接 Ranker 后的阶段八最佳 |
| Stage 9 best pipeline | 0.048158 | 本轮最佳，`2m_t40n120 + Rerank@100` |
| Big ItemCF | 0.065921 | 当前统计协同过滤上限参考 |

阶段九最佳结果相对阶段八提升约 `8.1%`：

$$
\frac{0.048158 - 0.044560}{0.044560} \approx 8.1\%
$$

但距离 big ItemCF 仍有约 `26.9%` 差距：

$$
\frac{0.065921 - 0.048158}{0.065921} \approx 26.9\%
$$

## 6. 关键结论

1. `2m_t40n120` 是本轮最优配置，说明降低 teacher 样本占比、增加随机负样本，有助于校准双塔召回边界。
2. `2m_t120n40` 没有提升，说明直接增加 ItemCF teacher 样本不一定更好；teacher 信号可能会带来噪声或过拟合 ItemCF 局部排序。
3. 最佳候选规模是 `candidate_k=100`，而不是 `200`。当前 Ranker 在过大的候选集中会引入更多噪声，Top20 排序质量下降。
4. Stage9 已把 KuaiRec big 神经 pipeline 从 `0.044560` 提升到 `0.048158`，方向有效，但仍未追上 ItemCF。

## 7. 下一步建议

下一轮不建议直接扩到 5M，也不建议继续平均推进 LightGCN、GRU 和 TextCNN。更合理的方向是：

1. 继续围绕 `2m_t40n120` 做小范围精调，例如 `teacher_items_per_user=20/40/60`、`negative_items_per_user=120/160/200`。
2. 设计 teacher soft label，将 ItemCF 分数归一化为连续目标，而不是简单把 teacher TopK 当正样本。
3. 保持 `candidate_k=100` 作为默认候选规模，只有在召回底座进一步增强后再扩大到 200/500。
4. 如需体现工程能力，再把当前最佳蒸馏 pipeline 改造成 DDP 训练，记录单卡/双卡吞吐和最终指标。
