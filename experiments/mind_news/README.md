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
| `reports/试验解析.md` | 面向初学者的 MIND 阶段完整学习说明和实验清单 |

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

## 后续模型

1. 内容感知 Two-Tower：用户塔编码点击历史，新闻塔编码标题、摘要和类别。
2. 内容感知 DNN Ranker：融合用户、新闻、类别、热度和文本 embedding。
3. 两阶段 pipeline：Two-Tower 召回 TopN，Ranker 重排 TopK。

详细学习清单见 `reports/试验解析.md`。
