# Two-Tower 召回模型报告

## 指标

| 指标 | 数值 |
|---|---:|
| 模型 | Two-Tower |
| K | 20 |
| 评估用户数 | 6,037 |
| Recall@20 | 0.101872 |
| HitRate@20 | 0.101872 |
| Precision@20 | 0.005094 |
| NDCG@20 | 0.040244 |
| Coverage@20 | 0.340973 |

## 训练设置

| 项目 | 数值 |
|---|---:|
| 训练目标 | bce |
| 设备 | mlu |
| 用户数 | 6,038 |
| 电影数 | 3,883 |
| Embedding Dim | 128 |
| Tower Dim | 128 |
| Hidden Dim | 256 |
| 最大历史电影数 | 100 |
| Epochs | 10 |
| Batch Size | 4,096 |
| 每个正样本负采样数 | 1 |
| Learning Rate | 0.005 |
| Temperature | 0.1 |
| Final Loss | 0.313643 |
| 训练耗时（秒） | 22.82 |
| 评估耗时（秒） | 7.45 |

## 说明

- 用户塔输入为 `user_id embedding` 与用户训练集历史正反馈电影 embedding 均值的拼接。
- 物品塔输入为 `item_id embedding`。
- 两个塔分别经过 MLP 后做 L2 归一化，使用向量点积作为召回分数。
- 当前正式配置使用随机负采样 BCE 训练。
- 也可以通过 `--loss in-batch` 使用 batch 内其他正样本物品作为隐式负样本。
- 正样本来自训练集 `rating >= 4` 的交互。
- 评估时会计算用户向量与全量电影向量的相似度，过滤训练集中已看过电影，再取 TopK。
