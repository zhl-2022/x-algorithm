# KuaiRec 阶段十一：最终复跑与收尾报告

## 1. 目标

阶段十一用阶段十最佳配置换 `seed=2027` 复跑一次，验证最终 KuaiRec big 神经 pipeline 的稳定性。

## 2. 关键基线

| 基线 | NDCG@20 | 说明 |
|---|---:|---|
| Stage9 best | 0.048158 | 现有最佳神经 pipeline |
| Stage10 best | 0.055883 | soft label 精调最佳 |
| Stage11 replay | 0.052947 | 最终复跑结果 |
| ItemCF | 0.065921 | big TopK 统计协同过滤参考上限 |

## 3. 最终判断

最终最佳为 Stage10 `soft_p30_t15_sqrt` 的 `DistillTwoTower+DNN-Blend@100a0.5`，`NDCG@20=0.055883`。

KuaiRec big 神经 pipeline 已超过阶段九基线，但仍需承认 ItemCF 仍是当前 TopK 上限参考。
