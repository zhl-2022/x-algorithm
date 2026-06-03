# KuaiRec 阶段八：big 神经召回补强实验报告

## 1. 实验目标

阶段八继续围绕 `big_matrix.csv` 的核心瓶颈做补强：ItemCF 仍是 big TopK 最强基线，
但神经召回需要进一步提升，才能支撑后续向量召回和两阶段 pipeline。

本轮重点验证四件事：

1. 将 ItemCF 蒸馏 Two-Tower 从 `800,000` 样本放大到 `2,000,000`。
2. 将蒸馏 Two-Tower 接回 `DNNRanker` hard negative 两阶段 pipeline。
3. 调参 LightGCN 的层数和训练轮数。
4. 修正 GRU 序列模型的 padding 表示后重新验证。

## 2. 运行方式

服务器：srv4 `xalgorithm-mlu` 容器。

执行脚本：

```bash
docker exec -d xalgorithm-mlu bash -lc \
  "cd /root/zhl/x-algorithm && bash experiments/kuairec_short_video/scripts/start_stage8_recall_boost.sh > experiments/kuairec_short_video/logs/stage8_recall_boost/supervisor.log 2>&1"
```

调度方式：

| 阶段 | 任务 | 设备 |
|---|---|---|
| Batch 1 | `distill_2m_t80n80` | Card 2 |
| Batch 1 | `sequence_fixed_800k` | Card 3 |
| Batch 1 | `lightgcn_l{1,2,3}_e{2,5,10}` | CPU 顺序执行 |
| Batch 2 | `distill_pipeline_2m` | Card 2 |

本轮复用缓存 `experiments/kuairec_short_video/data/cache/big_matrix_threshold08_prepared.pkl`。

## 3. 关键结果

| 实验 | 训练样本 | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---:|---:|---:|---:|---|
| ItemCF-Distill-TwoTower 800k | 800,000 | 0.006874 | 0.033562 | 0.604001 | 阶段七蒸馏基线 |
| ItemCF-Distill-TwoTower 2M | 2,000,000 | 0.006202 | 0.027320 | 0.608339 | 放大后 AUC 略升，但 TopK 下降 |
| LightGCN best `l1_e10` | 800,000 | 0.002902 | 0.011906 | 0.593868 | LightGCN 调参后提升，但仍弱于蒸馏 |
| GRU-Sequence fixed | 800,000 | 0.000488 | 0.000914 | 0.683278 | padding 修复后仍未形成有效 TopK |
| DistillTwoTower+DNN-Rerank@200 | 2,000,000 | 0.011031 | 0.044560 | 0.681781 | 本轮最佳神经 pipeline |
| ItemCF big baseline | - | - | 0.065921 | - | big TopK 仍然最强 |

## 4. Pipeline 消融

| Pipeline | Recall@20 | NDCG@20 | AUC | 说明 |
|---|---:|---:|---:|---|
| DistillTwoTower+DNN-Rerank@100 | 0.009365 | 0.039596 | 0.703364 | Top100 候选内纯 Ranker 重排 |
| DistillTwoTower+DNN-Rerank@200 | 0.011031 | 0.044560 | 0.681781 | 本轮最佳 |
| DistillTwoTower+DNN-Rerank@500 | 0.010749 | 0.035964 | 0.669798 | 候选扩大到 500 后噪声增加 |
| DistillTwoTower+DNN-Blend@200a0.75 | 0.010940 | 0.044004 | 0.683124 | 融合分数接近纯 Ranker |

结论：蒸馏双塔接回 Ranker 后明显优于单独蒸馏双塔，说明“蒸馏召回 + hard negative 排序”
是当前 big 场景最有效的神经 pipeline 路线。

## 5. LightGCN 调参

| 配置 | Recall@20 | NDCG@20 | AUC |
|---|---:|---:|---:|
| `l1_e2` | 0.002102 | 0.007371 | 0.500204 |
| `l1_e5` | 0.002557 | 0.010301 | 0.522183 |
| `l1_e10` | 0.002902 | 0.011906 | 0.593868 |
| `l2_e2` | 0.002192 | 0.008166 | 0.509272 |
| `l2_e5` | 0.002740 | 0.011823 | 0.575342 |
| `l2_e10` | 0.002412 | 0.010261 | 0.609910 |
| `l3_e2` | 0.002520 | 0.010294 | 0.527576 |
| `l3_e5` | 0.002571 | 0.011096 | 0.600340 |
| `l3_e10` | 0.001820 | 0.007719 | 0.613213 |

结论：当前设置下浅层 LightGCN 更稳，`l1_e10` 最好；但 TopK 仍明显低于蒸馏 pipeline。

## 6. 阶段结论

1. 2M 蒸馏单模型没有超过 800k 蒸馏，说明简单增加样本并不必然提升 TopK；teacher 样本质量、
   软标签权重和正负样本配比比样本量更关键。
2. 蒸馏双塔接 DNNRanker 后达到 `NDCG@20=0.044560`，高于阶段七蒸馏双塔 `0.033562`，
   也高于 stage5 best pipeline `0.005245`。
3. LightGCN 调参能提升到 `NDCG@20=0.011906`，但暂时不是主线。
4. 序列模型修复后仍弱，后续不建议继续放大，除非重构序列样本和负采样。
5. 当前 big 场景下一步应围绕蒸馏 pipeline 做精调，而不是盲目扩到 5M。
