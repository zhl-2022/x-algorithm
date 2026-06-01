# ItemCF Baseline 报告

## 指标

| 指标 | 数值 |
|---|---:|
| 模型 | ItemCF |
| K | 20 |
| 评估用户数 | 6,037 |
| Recall@20 | 0.082160 |
| HitRate@20 | 0.082160 |
| Precision@20 | 0.004108 |
| NDCG@20 | 0.032793 |
| Coverage@20 | 0.132372 |

## ItemCF 设置

| 项目 | 数值 |
|---|---:|
| 训练用户数 | 6,038 |
| 训练物品数 | 3,525 |
| 共现物品对数 | 3,887,159 |
| 每个物品保留相似物品数 | 100 |
| 相似度构建耗时（秒） | 63.81 |
| 评估耗时（秒） | 20.66 |

## 示例相似物品

| 源电影 ID | 源电影 | 相似电影 ID | 相似电影 | 相似度 |
|---:|---|---:|---|---:|
| 1 | Toy Story (1995) | 3114 | Toy Story 2 (1999) | 0.598948 |
| 2 | Jumanji (1995) | 3489 | Hook (1991) | 0.337757 |
| 3 | Grumpier Old Men (1995) | 3450 | Grumpy Old Men (1993) | 0.375349 |
| 4 | Waiting to Exhale (1995) | 1621 | Soul Food (1997) | 0.305556 |
| 5 | Father of the Bride Part II (1995) | 500 | Mrs. Doubtfire (1993) | 0.251365 |
| 6 | Heat (1995) | 47 | Seven (Se7en) (1995) | 0.437179 |
| 7 | Sabrina (1995) | 539 | Sleepless in Seattle (1993) | 0.344453 |
| 8 | Tom and Huck (1995) | 484 | Lassie (1994) | 0.338062 |
| 9 | Sudden Death (1995) | 1497 | Double Team (1997) | 0.385758 |
| 10 | GoldenEye (1995) | 1722 | Tomorrow Never Dies (1997) | 0.570312 |
| 11 | American President, The (1995) | 440 | Dave (1993) | 0.484595 |
| 12 | Dracula: Dead and Loving It (1995) | 2207 | Jamaica Inn (1939) | 0.272166 |
| 13 | Balto (1995) | 2089 | Rescuers Down Under, The (1990) | 0.221249 |
| 14 | Nixon (1995) | 1120 | People vs. Larry Flynt, The (1996) | 0.267530 |
| 15 | Cutthroat Island (1995) | 533 | Shadow, The (1994) | 0.219034 |
| 16 | Casino (1995) | 1213 | GoodFellas (1990) | 0.407475 |
| 17 | Sense and Sensibility (1995) | 838 | Emma (1996) | 0.422392 |
| 18 | Four Rooms (1995) | 1089 | Reservoir Dogs (1992) | 0.220579 |
| 19 | Ace Ventura: When Nature Calls (1995) | 344 | Ace Ventura: Pet Detective (1994) | 0.368945 |
| 20 | Money Train (1995) | 227 | Drop Zone (1994) | 0.367065 |

## 说明

- ItemCF 使用训练集正反馈构建物品共现相似度。
- 相似度公式为 `co_count(i, j) / sqrt(count(i) * count(j))`。
- 对每个用户，使用其训练集历史电影的相似电影累加得分并推荐 TopK。
- 评估前会过滤用户在训练集中已经看过的电影。
