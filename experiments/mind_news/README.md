# MIND 新闻推荐实验

这个实验是 MovieLens 阶段之后的第二个推荐系统实验，用于从电影 ID 推荐升级到
新闻内容推荐。MIND-small 包含新闻标题、摘要、类别、用户点击历史和曝光点击标签，
更接近 X/Twitter 类信息流推荐场景。

## 目标

1. 下载并解析 MIND-small。
2. 生成新闻元数据、曝光样本和用户历史。
3. 输出数据统计报告。
4. 后续实现内容感知 Two-Tower 召回模型。
5. 后续实现内容感知 DNN Ranker 排序模型。
6. 最终复用 MovieLens 阶段的两阶段 pipeline：召回候选，再排序重排。

## 数据来源

MIND 是 Microsoft News Recommendation Dataset。当前实验使用 MIND-small：

| 文件 | 说明 |
|---|---|
| `MINDsmall_train.zip` | 训练集新闻和用户行为 |
| `MINDsmall_dev.zip` | 验证集新闻和用户行为 |

优先下载地址来自 Microsoft Azure Open Datasets 文档中的 MIND 数据集说明。
当前官方 Azure Blob 返回 `Public access is not permitted`，因此本阶段使用可访问的
RecZoo `MIND_small_x1` 镜像作为替代数据源。该镜像已将原始曝光展开为逐条
`user_id-news_id-click` 样本。

## 快速开始

在当前目录运行：

```powershell
python scripts/download_mind.py
python scripts/prepare_mind.py
```

也可以让准备脚本在数据缺失时自动下载：

```powershell
python scripts/prepare_mind.py --download
```

如果官方源不可用，直接使用 RecZoo 镜像：

```powershell
python scripts/download_mind.py --source reczoo
python scripts/prepare_mind.py --source reczoo --sample-rows 100000
```

运行 MIND Popularity baseline：

```powershell
python scripts/popularity_baseline.py
```

运行当前 MIND 阶段全部实验：

```powershell
python scripts/run_all_experiments.py
```

## 输出文件

| 路径 | 说明 |
|---|---|
| `data/raw/MINDsmall_train/` | 解压后的训练集原始文件 |
| `data/raw/MINDsmall_dev/` | 解压后的验证集原始文件 |
| `data/raw/MIND_small_x1/` | RecZoo 镜像解压目录 |
| `data/processed/news_metadata.csv` | 合并去重后的新闻元数据 |
| `data/processed/train_sample.csv` | 训练样本前 100,000 行，用于快速调试 |
| `data/processed/valid_sample.csv` | 验证样本前 100,000 行，用于快速调试 |
| `reports/data_report.md` | MIND-small 数据统计报告 |
| `reports/baseline_report.md` | MIND Popularity baseline 报告 |
| `reports/all_experiments_report.md` | MIND 阶段全部实验汇总报告 |
| `reports/category_report.md` | Category baseline 报告 |
| `reports/ranker_report.md` | DNN Ranker 报告 |
| `reports/two_tower_report.md` | 内容感知 Two-Tower 报告 |
| `reports/pipeline_report.md` | Two-Tower + DNN Ranker pipeline 报告 |
| `reports/mlu_training_report.md` | srv4 MLU 训练记录 |
| `reports/试验解析.md` | 面向初学者的 MIND 阶段完整学习说明和实验清单 |
| `outputs/popularity_results.csv` | Popularity baseline 机器可读结果 |
| `outputs/experiment_results.csv` | 当前 MIND 阶段统一实验结果表 |

## 当前数据统计

| 指标 | 数值 |
|---|---:|
| 新闻数 | 65,238 |
| Category 数 | 18 |
| Subcategory 数 | 270 |
| 训练样本行数 | 5,843,444 |
| 验证样本行数 | 2,740,998 |
| 训练用户数 | 50,000 |
| 验证用户数 | 50,000 |
| 训练 CTR | 4.0446% |
| 验证 CTR | 4.0636% |

## 当前实验结果

统计类 baseline 使用全量 `train.csv` 和 `valid.csv`；神经网络模型默认使用
`train_sample.csv` 和 `valid_sample.csv` 各 100,000 行。当前神经模型结果已在
srv4 的 `xalgorithm-mlu` 容器中使用 `MLU_VISIBLE_DEVICES=2,3` 跑通。

| 模型 | 范围 | AUC | MRR | NDCG@5 | NDCG@10 | HitRate@10 |
|---|---|---:|---:|---:|---:|---:|
| Popularity | full | 0.522252 | 0.266074 | 0.247261 | 0.308465 | 0.611986 |
| Category | full | 0.588720 | 0.291866 | 0.274974 | 0.338507 | 0.657152 |
| DNNRanker | sample, MLU | 0.547961 | 0.267107 | 0.241286 | 0.311570 | 0.635250 |
| ContentTwoTower | sample, MLU | 0.562748 | 0.281838 | 0.256427 | 0.324701 | 0.650835 |
| TwoTower+DNN-Rerank | sample, MLU | 0.550229 | 0.271336 | 0.244329 | 0.316776 | 0.646011 |

## 后续模型

1. 将 DNN Ranker 和 ContentTwoTower 从 100k 样本训练扩展到百万级或全量训练。
2. 在 MLU 容器中补充 `candidate_k=10/20/50/100` pipeline 消融。
3. 引入更强文本 encoder，把标题和摘要从统计特征升级为语义向量。
4. 后续切换 MIND-large、KuaiRec 或 Tenrec，做更大规模训练性能记录。

详细学习清单见 `reports/试验解析.md`。
