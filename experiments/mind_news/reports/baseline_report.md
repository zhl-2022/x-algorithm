# MIND Popularity Baseline 报告

## 实验目的

这个实验用于建立 MIND 新闻推荐阶段的最低基线。模型不理解用户个性，也不理解新闻文本，
只根据训练集中新闻被点击的次数判断新闻热度，然后在验证集每一次曝光的候选新闻列表内部排序。

## 输入与输出

| 项目 | 内容 |
|---|---|
| 训练输入 | `data/raw/MIND_small_x1/train.csv` |
| 验证输入 | `data/raw/MIND_small_x1/valid.csv` |
| 训练信号 | `click=1` 的新闻点击次数 |
| 排序对象 | 同一个 `imp_id + user_id` 下的候选新闻 |
| 输出报告 | `reports/baseline_report.md` |
| 输出结果表 | `outputs/popularity_results.csv`、`outputs/experiment_results.csv` |

## 热度打分方式

每篇新闻的主分数是训练集点击次数：

$$
score(news) = click\_count(news)
$$

如果两篇新闻点击次数相同，脚本会用一个极小的曝光次数项打破并列：

$$
score(news) = click\_count(news) + \lambda \cdot log(1 + exposure\_count(news))
$$

本次实验的 $\lambda$ 为 `1e-06`。这个项很小，主要用于减少完全相同分数导致的排序并列。

## 数据统计

| 指标 | 数值 |
|---|---:|
| 训练样本行数 | 5,843,444 |
| 训练正样本数 | 236,344 |
| 训练负样本数 | 5,607,100 |
| 训练 CTR | 4.0446% |
| 训练集中出现新闻数 | 20,288 |
| 训练集中被点击新闻数 | 7,713 |
| 验证样本行数 | 2,740,998 |
| 验证正样本数 | 111,383 |
| 验证负样本数 | 2,629,615 |
| 评估曝光组数 | 73,152 |
| 平均每组候选新闻数 | 37.47 |

## 评估指标

| 指标 | 数值 | 小白解释 |
|---|---:|---|
| AUC | 0.522252 | 随机拿一条点击新闻和一条未点击新闻，热门模型把点击新闻排得更高的概率 |
| MRR | 0.266074 | 每个曝光列表里第一条被点击新闻出现得越靠前，数值越高 |
| NDCG@5 | 0.247261 | Top5 内命中点击新闻且位置越靠前，数值越高 |
| NDCG@10 | 0.308465 | Top10 内命中点击新闻且位置越靠前，数值越高 |
| HitRate@5 | 0.423433 | Top5 里至少有一条点击新闻的曝光组比例 |
| HitRate@10 | 0.611986 | Top10 里至少有一条点击新闻的曝光组比例 |
| Coverage@5 | 0.017061 | 所有 Top5 推荐覆盖到的不同新闻占总新闻数比例 |
| Coverage@10 | 0.026380 | 所有 Top10 推荐覆盖到的不同新闻占总新闻数比例 |

## 训练集中最热门的新闻

| 排名 | 新闻 ID | 点击数 | 曝光数 | CTR | Category | Subcategory | 标题 |
|---:|---|---:|---:|---:|---|---|---|
| 1 | N55689 | 4,316 | 18,315 | 23.5654% | sports | football_nfl | Charles Rogers, former Michigan State football, Detroit Lions star, dead at 38 |
| 2 | N35729 | 3,346 | 15,418 | 21.7019% | news | newsus | Porsche launches into second story of New Jersey building, killing 2 |
| 3 | N33619 | 3,246 | 15,062 | 21.5509% | news | newsus | College gymnast dies following training accident in Connecticut |
| 4 | N53585 | 2,835 | 9,908 | 28.6132% | tv | tvnews | Rip Taylor's Cause of Death Revealed, Memorial Service Scheduled for Later This Month |
| 5 | N63970 | 2,578 | 14,276 | 18.0583% | finance | finance-companies | Dean Foods files for bankruptcy |
| 6 | N49685 | 2,294 | 7,229 | 31.7333% | music | music-celebrity | Broadway Star Laurel Griggs Suffered Asthma Attack Before She Died at Age 13 |
| 7 | N49279 | 2,270 | 6,229 | 36.4424% | music | musicnews | Broadway Actress Laurel Griggs Dies at Age 13 |
| 8 | N287 | 2,128 | 10,019 | 21.2396% | news | newscrime | Three school workers charged in death of special needs student |
| 9 | N23446 | 1,930 | 15,500 | 12.4516% | lifestyle | lifestyleroyals | Prince Harry and Meghan Markle just shared a never-before-seen photo of baby Archie with his 'Grandpa' Prince Charles to celebrate his birthday |
| 10 | N51048 | 1,875 | 19,242 | 9.7443% | news | elections-2020-us | Rep. Tim Ryan endorses Biden in Democratic primary |
| 11 | N58363 | 1,603 | 13,565 | 11.8172% | finance | finance-companies | Supreme Court refuses to block lawsuit against gun manufacturer brought by Sandy Hook families |
| 12 | N62360 | 1,594 | 16,869 | 9.4493% | news | newsworld | The son of a Chinese billionaire has been banned from flying first class, playing golf, buying property, or going clubbing |
| 13 | N38779 | 1,490 | 18,101 | 8.2316% | news | newsus | 'One in a million' deer captured on camera in Michigan woods |
| 14 | N41881 | 1,478 | 8,638 | 17.1104% | tv | tv-celebrity | Marlboro Man Bob Norris dies at 90, having reportedly never been a smoker |
| 15 | N42977 | 1,406 | 9,727 | 14.4546% | news | newsus | 'It's not over': Sarah Palin says she is fighting to repair her marriage |
| 16 | N40839 | 1,296 | 11,929 | 10.8643% | sports | sports_news | Stephen Curry calls out Michael Jordan for being a 'hater' |
| 17 | N4642 | 1,189 | 12,176 | 9.7651% | music | music-celebrity | Kodak Black Sentenced to Over 3 Years in Prison in Weapons Case |
| 18 | N56214 | 1,168 | 10,334 | 11.3025% | weather | weathertopstories | Deadly Arctic blast breaks records set more than 100 years ago |
| 19 | N26262 | 1,139 | 19,106 | 5.9615% | entertainment | entertainment-celebrity | Celebrity plastic surgery transformations |
| 20 | N41020 | 1,139 | 15,551 | 7.3243% | entertainment | celebrity | Lamar Odom Is Engaged to Sabrina Parr: See Her Ring! |

## 怎么理解这个结果

- Popularity baseline 是非个性化模型：同一组候选新闻中，它只偏向训练集中更热门的新闻。
- 这个模型不能利用 `user_id`、`news_his`、标题、摘要或类别，因此它只是后续模型的最低对照线。
- 如果后续 Category baseline、DNN Ranker 或 Two-Tower 没有超过这个基线，说明个性化或内容特征没有真正发挥作用。
- MIND 的验证方式和 MovieLens 不同：MovieLens 是从全量电影里推荐 TopK；MIND 当前是在一次真实曝光的候选列表内部排序。
