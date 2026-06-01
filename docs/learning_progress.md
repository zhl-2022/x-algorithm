# 推荐系统学习进度

## 学习目标

基于 X/Twitter 开源推荐系统架构，搭建一个企业级推荐系统学习项目，覆盖数据处理、
召回、排序、离线评测、训练环境适配和实验报告沉淀。最终目标是形成可复现项目成果，
并能在简历和面试中清楚说明技术价值。

## 当前进度

| 日期 | 阶段 | 状态 | 产出 |
|---|---|---|---|
| 2026-05-27 | MovieLens 1M 最小闭环 | 已完成 | 数据处理、统计报告、Popularity baseline、`Recall@20` |
| 2026-06-01 | MovieLens 1M ItemCF baseline | 已完成 | ItemCF 召回、统一结果表、`Recall@20` 对比 |

## 已完成内容

1. 新建 `experiments/movielens_recall/` 作为第一个推荐实验目录。
2. 编写 MovieLens 1M 下载脚本。
3. 编写数据准备脚本，将 `rating >= 4` 定义为正反馈。
4. 按用户时间序列切分训练集、验证集和测试集。
5. 生成数据统计报告。
6. 实现 Popularity baseline。
7. 输出 `Recall@20`、`HitRate@20`、`Precision@20`、`NDCG@20` 和 `Coverage@20`。
8. 实现 ItemCF baseline，并输出模型对比结果。

## 当前实验结果

| 指标 | 数值 |
|---|---:|
| 原始评分数 | 1,000,209 |
| 用户数 | 6,040 |
| 电影元数据数 | 3,883 |
| 正反馈交互数 | 575,281 |
| 训练集正反馈数 | 563,209 |
| 验证集正反馈数 | 6,035 |
| 测试集正反馈数 | 6,037 |
| Popularity `Recall@20` | 0.067252 |
| Popularity `HitRate@20` | 0.067252 |
| Popularity `NDCG@20` | 0.026149 |
| ItemCF `Recall@20` | 0.082160 |
| ItemCF `HitRate@20` | 0.082160 |
| ItemCF `NDCG@20` | 0.032793 |
| ItemCF `Coverage@20` | 0.132372 |

## 当前理解沉淀

- 推荐系统不是单个模型，而是数据、召回、排序、过滤、评测组成的系统工程。
- 第一阶段的价值是建立可复现的离线评测闭环，而不是追求复杂模型。
- Popularity baseline 指标不高是正常现象，它用于给后续 ItemCF、MF 和 Two-Tower 提供对比基准。
- ItemCF 已经体现个性化召回价值，`Recall@20` 和 `NDCG@20` 均高于 Popularity baseline。
- 按时间顺序切分比随机切分更接近真实推荐场景，因为模型只能利用用户过去行为预测未来偏好。

## 下一步计划

1. 准备 Matrix Factorization 的训练脚本，为后续 MLU 训练适配做铺垫。
2. 在 `xalgorithm-mlu` 容器中验证 PyTorch/torch_mlu 训练最小样例。
3. 将 MF 指标加入 `outputs/experiment_results.csv`。
4. 开始设计 Two-Tower 召回模型。
