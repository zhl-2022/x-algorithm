# KuaiRec 阶段四：big_matrix hard negative 迁移实验

## 1. 实验目标

阶段三在 `small_matrix.csv` 上证明 Ranker hard negative 有效。本阶段把同样策略迁移到
`big_matrix.csv`，验证它在更大候选池和更多用户下是否仍能提升 TopK 推荐。

## 2. 实验设置

| 项目 | 数值 |
|---|---:|
| 数据源 | `big_matrix.csv` |
| 标签 | `watch_ratio >= 0.8` |
| 训练交互 | 10,021,757 |
| 测试交互 | 1,256,352 |
| 评估用户 | 7,174 |
| 神经基础训练样本 | 2,000,000 |
| Ranker hard negative pool | 3,000,000 |
| 每用户 hard negatives | 80 |
| 实际追加 hard negatives | 559,393 |
| Ranker 最终训练样本 | 2,559,393 |
| Epochs | 3 |
| Batch size | 8,192 |
| Candidate K | 100/200/500 |

## 3. 核心结果

| 模型 | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---:|---:|---:|---|
| ItemCF | 0.019777 | 0.065921 | 0.607427 | 当前 big TopK 最强 |
| Two-Tower BCE | 0.000261 | 0.000937 | 0.752882 | AUC 高但 TopK 很弱 |
| DNNRanker hard negative | 0.001951 | 0.006151 | 0.762044 | 比上一轮 big DNN 有提升 |
| TwoTower+DNN-Rerank@500 | 0.001192 | 0.004991 | 0.694432 | pipeline 有提升但仍弱 |

## 4. 结论

hard negative 迁移到 `big_matrix.csv` 后，Ranker 和 pipeline 相比上一轮 big 神经模型有提升，
但仍远低于 ItemCF 的 `NDCG@20=0.065921`。

这说明问题不只在 Ranker，核心瓶颈仍然是 big 场景下 Two-Tower 候选召回质量不足。

## 5. 下一步

继续执行阶段五：把 Two-Tower 从 BCE 二分类训练改成 in-batch negative 召回训练，
观察召回底座是否改善。

