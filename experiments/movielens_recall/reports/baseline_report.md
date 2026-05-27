# Popularity Baseline 报告

## 指标

| 指标 | 数值 |
|---|---:|
| 模型 | Popularity |
| K | 20 |
| 评估用户数 | 6,037 |
| Recall@20 | 0.067252 |
| HitRate@20 | 0.067252 |
| Precision@20 | 0.003363 |
| NDCG@20 | 0.026149 |
| Coverage@20 | 0.045326 |

## 训练集中最热门的电影

| 排名 | 电影 ID | 训练集正反馈数 | 标题 | 类别 |
|---:|---:|---:|---|---|
| 1 | 2858 | 2,800 | American Beauty (1999) | Comedy, Drama |
| 2 | 260 | 2,599 | Star Wars: Episode IV - A New Hope (1977) | Action, Adventure, Fantasy, Sci-Fi |
| 3 | 1196 | 2,495 | Star Wars: Episode V - The Empire Strikes Back (1980) | Action, Adventure, Drama, Sci-Fi, War |
| 4 | 1198 | 2,241 | Raiders of the Lost Ark (1981) | Action, Adventure |
| 5 | 2028 | 2,226 | Saving Private Ryan (1998) | Action, Drama, War |
| 6 | 593 | 2,222 | Silence of the Lambs, The (1991) | Drama, Thriller |
| 7 | 2571 | 2,134 | Matrix, The (1999) | Action, Sci-Fi, Thriller |
| 8 | 1210 | 2,107 | Star Wars: Episode VI - Return of the Jedi (1983) | Action, Adventure, Romance, Sci-Fi, War |
| 9 | 2762 | 2,085 | Sixth Sense, The (1999) | Thriller |
| 10 | 527 | 2,051 | Schindler's List (1993) | Drama, War |
| 11 | 608 | 2,043 | Fargo (1996) | Crime, Drama, Thriller |
| 12 | 318 | 2,019 | Shawshank Redemption, The (1994) | Drama |
| 13 | 589 | 2,018 | Terminator 2: Judgment Day (1991) | Action, Sci-Fi, Thriller |
| 14 | 858 | 1,974 | Godfather, The (1972) | Action, Crime, Drama |
| 15 | 110 | 1,959 | Braveheart (1995) | Action, Drama, War |
| 16 | 1197 | 1,902 | Princess Bride, The (1987) | Action, Adventure, Comedy, Romance |
| 17 | 1270 | 1,895 | Back to the Future (1985) | Comedy, Sci-Fi |
| 18 | 1617 | 1,828 | L.A. Confidential (1997) | Crime, Film-Noir, Mystery, Thriller |
| 19 | 2396 | 1,828 | Shakespeare in Love (1998) | Comedy, Romance |
| 20 | 296 | 1,744 | Pulp Fiction (1994) | Crime, Drama |

## 说明

- Popularity baseline 只使用训练集正反馈统计物品热度。
- 评估前会过滤用户在训练集中已经看过的物品。
- 当前 leave-last 切分方式下，每个测试用户通常只有 1 个相关物品。
