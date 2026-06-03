# KuaiRec 阶段九：蒸馏 Pipeline 精调计划

## 1. 目标

阶段九不切换新数据集，也不直接把蒸馏样本扩到 5M。当前目标是围绕阶段八最佳结果
`DistillTwoTower+DNN-Rerank@200 NDCG@20=0.044560` 做蒸馏召回 + Ranker pipeline 精调，
判断 teacher 样本比例、随机负样本比例和候选集大小是否还能继续提升 big 场景 TopK。

成功标准：

| 标准 | 数值 |
|---|---:|
| 最低成功线 | 任一 pipeline `NDCG@20 > 0.044560` |
| 强成功线 | 任一 pipeline `NDCG@20 > 0.050000` |
| 参考上限 | big ItemCF `NDCG@20=0.065921` |

## 2. 固定设置

| 参数 | 数值 |
|---|---|
| 数据集 | `big_matrix.csv` |
| 标签 | `watch_ratio >= 0.8` |
| 缓存 | `data/cache/big_matrix_threshold08_prepared.pkl` |
| 设备 | srv4 `xalgorithm-mlu` 容器，优先 Card 2 |
| Epochs | `3` |
| Batch size | `8192` |
| Embedding dim | `64` |
| Tower dim | `64` |
| Hidden dim | `128` |
| Ranker hidden dims | `256,128,64` |
| Ranker positive weight | `4` |
| candidate_k | `100,200` |
| rerank alpha | `0.5,0.75,1` |

## 3. 三组实验

| 实验名 | 训练样本 | teacher/user | negative/user | 目的 |
|---|---:|---:|---:|---|
| `distill_pipeline_800k_t40n40` | 800,000 | 40 | 40 | 验证 800k 蒸馏底座接 Ranker 是否优于 2M |
| `distill_pipeline_2m_t40n120` | 2,000,000 | 40 | 120 | 降低 teacher 占比、提高随机负样本 |
| `distill_pipeline_2m_t120n40` | 2,000,000 | 120 | 40 | 提高 teacher 占比，增强 ItemCF 排序信号 |

## 4. 建议执行命令

在 srv4 容器内执行，三组建议分批跑，避免重复打分任务同时占用 Card 2。当前已固化为脚本：

```bash
cd /root/zhl/x-algorithm
bash experiments/kuairec_short_video/scripts/start_stage9_pipeline_tuning.sh
```

如果需要手动跑单组，可以使用下面的命令模板。

```bash
cd /root/zhl/x-algorithm
source /torch/venv3/pytorch/bin/activate

MLU_VISIBLE_DEVICES=2 python experiments/kuairec_short_video/scripts/run_upgrade_experiments.py \
  --experiment distill_pipeline \
  --run-name distill_pipeline_800k_t40n40 \
  --matrix big_matrix.csv \
  --positive-threshold 0.8 \
  --prepared-cache experiments/kuairec_short_video/data/cache/big_matrix_threshold08_prepared.pkl \
  --train-rows 800000 \
  --ranker-train-rows 800000 \
  --teacher-items-per-user 40 \
  --negative-items-per-user 40 \
  --ranker-hard-negatives-per-user 80 \
  --ranker-hard-negative-pool-rows 3000000 \
  --candidate-ks 100,200 \
  --rerank-blend-alphas 0.5,0.75,1 \
  --auc-rows 500000 \
  --epochs 3 \
  --batch-size 8192 \
  --embedding-dim 64 \
  --tower-dim 64 \
  --hidden-dim 128 \
  --ranker-hidden-dims 256,128,64 \
  --ranker-positive-weight 4 \
  --outputs-dir experiments/kuairec_short_video/outputs/stage9_pipeline_tuning \
  --reports-dir experiments/kuairec_short_video/reports/stage9_pipeline_tuning \
  --device auto
```

第二、三组只替换：

| 实验名 | 参数替换 |
|---|---|
| `distill_pipeline_2m_t40n120` | `--train-rows 2000000 --ranker-train-rows 2000000 --teacher-items-per-user 40 --negative-items-per-user 120` |
| `distill_pipeline_2m_t120n40` | `--train-rows 2000000 --ranker-train-rows 2000000 --teacher-items-per-user 120 --negative-items-per-user 40` |

## 5. 判断规则

1. 如果 `800k_t40n40` 优于 stage8 best，说明阶段八 2M 单模型下降可能来自样本配比或 teacher 噪声。
2. 如果 `2m_t40n120` 优于 stage8 best，说明当前需要更多随机负样本来校准召回边界。
3. 如果 `2m_t120n40` 优于 stage8 best，说明增强 ItemCF teacher 信号更关键。
4. 如果三组都不超过 `0.044560`，下一步不继续改样本量，转向 teacher soft label 设计。
