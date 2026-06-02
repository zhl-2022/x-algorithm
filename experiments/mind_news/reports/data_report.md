# MIND-small 数据报告

## 数据源说明

官方 Azure Blob 下载当前返回 `Public access is not permitted`，本报告使用可访问的 RecZoo `MIND_small_x1` 镜像生成。
该镜像已将原始曝光展开为逐条 `user_id-news_id-click` 样本，适合后续直接训练排序模型或构造召回样本。

## 新闻内容概览

| 指标 | 数值 |
|---|---:|
| 新闻数 | 65,238 |
| Category 数 | 18 |
| Subcategory 数 | 270 |
| 平均标题词数 | 10.74 |
| 平均摘要词数 | 34.86 |

## 行为样本统计

| Split | 样本行数 | 曝光 ID 数 | 用户数 | 有历史用户数 | 平均历史长度 | 最大历史长度 | 正样本 | 负样本 | CTR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5,843,444 | 156,965 | 50,000 | 49,108 | 25.23 | 50 | 236,344 | 5,607,100 | 4.0446% |
| valid | 2,740,998 | 73,152 | 50,000 | 48,593 | 25.21 | 50 | 111,383 | 2,629,615 | 4.0636% |

## Top Category

| Category | 新闻数 | 占比 |
|---|---:|---:|
| news | 20,039 | 30.7168% |
| sports | 19,368 | 29.6882% |
| finance | 3,786 | 5.8034% |
| foodanddrink | 3,123 | 4.7871% |
| travel | 3,013 | 4.6185% |
| lifestyle | 2,991 | 4.5848% |
| video | 2,712 | 4.1571% |
| weather | 2,601 | 3.9869% |
| health | 2,207 | 3.3830% |
| autos | 2,076 | 3.1822% |

## Top Subcategory

| Subcategory | 新闻数 | 占比 |
|---|---:|---:|
| newsus | 8,840 | 13.5504% |
| football_nfl | 7,464 | 11.4412% |
| newspolitics | 3,478 | 5.3312% |
| newscrime | 2,665 | 4.0850% |
| weathertopstories | 2,599 | 3.9839% |
| baseball_mlb | 2,166 | 3.3202% |
| football_ncaa | 2,107 | 3.2297% |
| newsworld | 2,022 | 3.0994% |
| basketball_nba | 1,997 | 3.0611% |
| news | 1,711 | 2.6227% |

## 输出文件

| 文件 | 说明 |
|---|---|
| `data/processed/news_metadata.csv` | 新闻内容元数据，后续新闻塔文本/类别编码使用 |
| `data/processed/train_sample.csv` | 训练样本前 `100,000` 行，用于快速调试模型 |
| `data/processed/valid_sample.csv` | 验证样本前 `100,000` 行，用于快速调试模型 |

## 说明

- `click=1` 表示用户点击该新闻，`click=0` 表示曝光但未点击。
- `news_his` 是用户历史点击新闻 ID 序列，是后续用户塔建模兴趣的核心输入。
- `cat`、`sub_cat`、`title`、`abstract` 是后续新闻塔和 Ranker 的内容特征。
- RecZoo 原始 `train.csv` 和 `valid.csv` 体积较大，当前不复制完整处理文件，只输出样本 CSV 和全量统计报告。
