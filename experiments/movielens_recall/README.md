# MovieLens 召回实验

这个实验是当前项目的第一个可复现推荐系统 baseline，用于从数据处理、
召回评测和训练闭环的角度学习 X algorithm 项目。

## 目标

跑通一个最小但完整的推荐系统流程：

1. 下载 MovieLens 1M。
2. 将评分数据转换为隐式正反馈交互。
3. 按用户时间序列切分正反馈样本。
4. 生成数据统计报告。
5. 使用 `Recall@20` 评估 Popularity baseline。

## 实验设定

| 项目 | 设置 |
|---|---|
| 数据集 | MovieLens 1M |
| 正反馈定义 | `rating >= 4.0` |
| 切分方式 | 每个用户内部按时间顺序切分 |
| Baseline | 基于训练集全局物品热度 |
| 核心指标 | `Recall@20` |

至少有 3 条正反馈的用户会产生训练集、验证集和测试集样本。有 2 条正反馈的用户
会产生训练集和测试集样本。只有 1 条正反馈的用户只进入训练集。

## 快速开始

在当前目录运行：

```powershell
python scripts/download_movielens.py
python scripts/prepare_movielens.py
python scripts/popularity_baseline.py --k 20
```

也可以在数据缺失时由准备脚本自动下载：

```powershell
python scripts/prepare_movielens.py --download
python scripts/popularity_baseline.py --k 20
```

## 输出文件

| 路径 | 说明 |
|---|---|
| `data/raw/ml-1m/` | 解压后的 MovieLens 1M 原始文件 |
| `data/processed/train.csv` | 训练集正反馈交互 |
| `data/processed/valid.csv` | 验证集正反馈交互 |
| `data/processed/test.csv` | 测试集正反馈交互 |
| `data/processed/item_metadata.csv` | 电影标题和类别元数据 |
| `reports/data_report.md` | 数据统计和切分报告 |
| `reports/baseline_report.md` | Popularity baseline 指标报告 |
| `outputs/popularity_results.csv` | 机器可读的指标结果 |

## 后续模型

当前 baseline 稳定后，建议按顺序增加以下模型：

1. ItemCF
2. Matrix Factorization 矩阵分解
3. Two-Tower 召回模型
4. DNN Ranker 排序模型
