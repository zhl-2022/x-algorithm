# KuaiRec 数据盘点报告

## 生成说明

- 原始数据目录：`/root/zhl/x-algorithm/experiments/kuairec_short_video/data/raw`
- 处理数据目录：`/root/zhl/x-algorithm/experiments/kuairec_short_video/data/processed`
- CSV 文件数量：10

## 文件清单

| 文件 | 相对路径 | 大小 | 行数 | 样例文件 |
|---|---|---:|---:|---|
| `big_matrix.csv` | `KuaiRec/KuaiRec 2.0/data/big_matrix.csv` | 1.01 GiB | 12530806 | `data/processed/samples/KuaiRec_KuaiRec_2_0_data_big_matrix_csv_sample.csv` |
| `item_categories.csv` | `KuaiRec/KuaiRec 2.0/data/item_categories.csv` | 110.48 KiB | 10728 | `data/processed/samples/KuaiRec_KuaiRec_2_0_data_item_categories_csv_sample.csv` |
| `item_daily_features.csv` | `KuaiRec/KuaiRec 2.0/data/item_daily_features.csv` | 81.88 MiB | 343341 | `data/processed/samples/KuaiRec_KuaiRec_2_0_data_item_daily_features_csv_sample.csv` |
| `kuairec_caption_category.csv` | `KuaiRec/KuaiRec 2.0/data/kuairec_caption_category.csv` | 1.87 MiB | 10732 | `data/processed/samples/KuaiRec_KuaiRec_2_0_data_kuairec_caption_category_csv_sample.csv` |
| `small_matrix.csv` | `KuaiRec/KuaiRec 2.0/data/small_matrix.csv` | 387.34 MiB | 4676570 | `data/processed/samples/KuaiRec_KuaiRec_2_0_data_small_matrix_csv_sample.csv` |
| `social_network.csv` | `KuaiRec/KuaiRec 2.0/data/social_network.csv` | 6.75 KiB | 472 | `data/processed/samples/KuaiRec_KuaiRec_2_0_data_social_network_csv_sample.csv` |
| `user_features.csv` | `KuaiRec/KuaiRec 2.0/data/user_features.csv` | 726.73 KiB | 7176 | `data/processed/samples/KuaiRec_KuaiRec_2_0_data_user_features_csv_sample.csv` |
| `kuairec_caption_category.csv` | `kuairec_caption_category.csv` | 1.87 MiB | 10732 | `data/processed/samples/kuairec_caption_category_csv_sample.csv` |
| `user_features_raw.csv` | `user_features_raw.csv` | 1.47 MiB | 7176 | `data/processed/samples/user_features_raw_csv_sample.csv` |
| `video_raw_categories_multi.csv` | `video_raw_categories_multi.csv` | 1.64 MiB | 26826 | `data/processed/samples/video_raw_categories_multi_csv_sample.csv` |

## 关键交互表候选

- `KuaiRec/KuaiRec 2.0/data/big_matrix.csv`：字段 `user_id|video_id|play_duration|video_duration|time|date|timestamp|watch_ratio`
- `KuaiRec/KuaiRec 2.0/data/small_matrix.csv`：字段 `user_id|video_id|play_duration|video_duration|time|date|timestamp|watch_ratio`

## 交互表统计

| 文件 | 行数 | 用户数 | 视频数 | 平均 `watch_ratio` | `watch_ratio >= 1.0` | 比例 | `watch_ratio >= 0.8` | 比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `KuaiRec/KuaiRec 2.0/data/big_matrix.csv` | 12,530,806 | 7,176 | 10,728 | 0.944506 | 4,238,228 | 33.82% | 5,683,062 | 45.35% |
| `KuaiRec/KuaiRec 2.0/data/small_matrix.csv` | 4,676,570 | 1,411 | 3,327 | 0.907069 | 1,515,050 | 32.40% | 2,218,724 | 47.44% |

## 当前标签建议

- 首选二分类标签：`positive = watch_ratio >= 1.0`，表示完播或重复观看。
- 可做消融：`positive = watch_ratio >= 0.8`，观察更宽松正反馈对 Recall/NDCG 的影响。

## 下一步

1. 确认核心交互表和时间字段。
2. 根据实际字段确定正反馈标签。
3. 按用户时间序列切分训练集、验证集和测试集。
4. 先实现 Popularity baseline，再迁移 Category、ItemCF、MF、Two-Tower 和 Ranker。

## 样例行

### `KuaiRec/KuaiRec 2.0/data/big_matrix.csv`

```json
[
  {
    "user_id": "0",
    "video_id": "3649",
    "play_duration": "13838",
    "video_duration": "10867",
    "time": "2020-07-05 00:08:23.438",
    "date": "20200705",
    "timestamp": "1593878903.438",
    "watch_ratio": "1.2733965215790926"
  },
  {
    "user_id": "0",
    "video_id": "9598",
    "play_duration": "13665",
    "video_duration": "10984",
    "time": "2020-07-05 00:13:41.297",
    "date": "20200705",
    "timestamp": "1593879221.297",
    "watch_ratio": "1.2440823015294975"
  }
]
```

### `KuaiRec/KuaiRec 2.0/data/item_categories.csv`

```json
[
  {
    "video_id": "0",
    "feat": "[8]"
  },
  {
    "video_id": "1",
    "feat": "[27, 9]"
  }
]
```

### `KuaiRec/KuaiRec 2.0/data/item_daily_features.csv`

```json
[
  {
    "video_id": "0",
    "date": "20200705",
    "author_id": "3309",
    "video_type": "NORMAL",
    "upload_dt": "2020-03-30",
    "upload_type": "ShortImport",
    "visible_status": "public",
    "video_duration": "5966.0",
    "video_width": "720",
    "video_height": "1280",
    "music_id": "3350323409",
    "video_tag_id": "841",
    "video_tag_name": "建筑",
    "show_cnt": "14665",
    "show_user_num": "11372",
    "play_cnt": "10141",
    "play_user_num": "7485",
    "play_duration": "88729488",
    "complete_play_cnt": "5657",
    "complete_play_user_num": "4834",
    "valid_play_cnt": "5503",
    "valid_play_user_num": "4775",
    "long_time_play_cnt": "5503",
    "long_time_play_user_num": "4775",
    "short_time_play_cnt": "1939",
    "short_time_play_user_num": "1481",
    "play_progress": "0.7998600086607257",
    "comment_stay_duration": "6629173",
    "like_cnt": "573",
    "like_user_num": "569",
    "click_like_cnt": "315",
    "double_click_cnt": "257",
    "cancel_like_cnt": "87",
    "cancel_like_user_num": "85",
    "comment_cnt": "11",
    "comment_user_num": "11",
    "direct_comment_cnt": "8",
    "reply_comment_cnt": "3",
    "delete_comment_cnt": "0",
    "delete_comment_user_num": "0",
    "comment_like_cnt": "112",
    "comment_like_user_num": "61",
    "follow_cnt": "284",
    "follow_user_num": "284",
    "cancel_follow_cnt": "0",
    "cancel_follow_user_num": "0",
    "share_cnt": "2",
    "share_user_num": "2",
    "download_cnt": "8",
    "download_user_num": "8",
    "report_cnt": "0",
    "report_user_num": "0",
    "reduce_similar_cnt": "3",
    "reduce_similar_user_num": "3",
    "collect_cnt": "",
    "collect_user_num": "",
    "cancel_collect_cnt": "",
    "cancel_collect_user_num": ""
  },
  {
    "video_id": "0",
    "date": "20200706",
    "author_id": "3309",
    "video_type": "NORMAL",
    "upload_dt": "2020-03-30",
    "upload_type": "ShortImport",
    "visible_status": "public",
    "video_duration": "5966.0",
    "video_width": "720",
    "video_height": "1280",
    "music_id": "3350323409",
    "video_tag_id": "841",
    "video_tag_name": "建筑",
    "show_cnt": "10883",
    "show_user_num": "8513",
    "play_cnt": "7321",
    "play_user_num": "5490",
    "play_duration": "64264607",
    "complete_play_cnt": "4162",
    "complete_play_user_num": "3522",
    "valid_play_cnt": "4039",
    "valid_play_user_num": "3468",
    "long_time_play_cnt": "4039",
    "long_time_play_user_num": "3468",
    "short_time_play_cnt": "1340",
    "short_time_play_user_num": "1040",
    "play_progress": "0.8052530649367682",
    "comment_stay_duration": "3997498",
    "like_cnt": "302",
    "like_user_num": "301",
    "click_like_cnt": "159",
    "double_click_cnt": "142",
    "cancel_like_cnt": "47",
    "cancel_like_user_num": "47",
    "comment_cnt": "7",
    "comment_user_num": "7",
    "direct_comment_cnt": "6",
    "reply_comment_cnt": "1",
    "delete_comment_cnt": "0",
    "delete_comment_user_num": "0",
    "comment_like_cnt": "60",
    "comment_like_user_num": "32",
    "follow_cnt": "201",
    "follow_user_num": "200",
    "cancel_follow_cnt": "0",
    "cancel_follow_user_num": "0",
    "share_cnt": "1",
    "share_user_num": "1",
    "download_cnt": "2",
    "download_user_num": "2",
    "report_cnt": "0",
    "report_user_num": "0",
    "reduce_similar_cnt": "5",
    "reduce_similar_user_num": "5",
    "collect_cnt": "",
    "collect_user_num": "",
    "cancel_collect_cnt": "",
    "cancel_collect_user_num": ""
  }
]
```

### `KuaiRec/KuaiRec 2.0/data/kuairec_caption_category.csv`

```json
[
  {
    "video_id": "0",
    "manual_cover_text": "UNKNOWN",
    "caption": "精神小伙路难走 程哥你狗粮慢点撒",
    "topic_tag": "[]",
    "first_level_category_id": "8",
    "first_level_category_name": "颜值",
    "second_level_category_id": "673",
    "second_level_category_name": "颜值随拍",
    "third_level_category_id": "-124",
    "third_level_category_name": "UNKNOWN"
  },
  {
    "video_id": "1",
    "manual_cover_text": "UNKNOWN",
    "caption": "",
    "topic_tag": "[]",
    "first_level_category_id": "27",
    "first_level_category_name": "高新数码",
    "second_level_category_id": "-124",
    "second_level_category_name": "UNKNOWN",
    "third_level_category_id": "-124",
    "third_level_category_name": "UNKNOWN"
  }
]
```

### `KuaiRec/KuaiRec 2.0/data/small_matrix.csv`

```json
[
  {
    "user_id": "14",
    "video_id": "148",
    "play_duration": "4381",
    "video_duration": "6067",
    "time": "2020-07-05 05:27:48.378",
    "date": "20200705.0",
    "timestamp": "1593898068.378",
    "watch_ratio": "0.7221031811438932"
  },
  {
    "user_id": "14",
    "video_id": "183",
    "play_duration": "11635",
    "video_duration": "6100",
    "time": "2020-07-05 05:28:00.057",
    "date": "20200705.0",
    "timestamp": "1593898080.057",
    "watch_ratio": "1.907377049180328"
  }
]
```

### `KuaiRec/KuaiRec 2.0/data/social_network.csv`

```json
[
  {
    "user_id": "3371",
    "friend_list": "[2975]"
  },
  {
    "user_id": "24",
    "friend_list": "[2665]"
  }
]
```

### `KuaiRec/KuaiRec 2.0/data/user_features.csv`

```json
[
  {
    "user_id": "0",
    "user_active_degree": "high_active",
    "is_lowactive_period": "0",
    "is_live_streamer": "0",
    "is_video_author": "0",
    "follow_user_num": "5",
    "follow_user_num_range": "(0,10]",
    "fans_user_num": "0",
    "fans_user_num_range": "0",
    "friend_user_num": "0",
    "friend_user_num_range": "0",
    "register_days": "107",
    "register_days_range": "61-90",
    "onehot_feat0": "0",
    "onehot_feat1": "1",
    "onehot_feat2": "17",
    "onehot_feat3": "638",
    "onehot_feat4": "2",
    "onehot_feat5": "0",
    "onehot_feat6": "1",
    "onehot_feat7": "6",
    "onehot_feat8": "184",
    "onehot_feat9": "6",
    "onehot_feat10": "3",
    "onehot_feat11": "0",
    "onehot_feat12": "0",
    "onehot_feat13": "0",
    "onehot_feat14": "0",
    "onehot_feat15": "0",
    "onehot_feat16": "0",
    "onehot_feat17": "0"
  },
  {
    "user_id": "1",
    "user_active_degree": "full_active",
    "is_lowactive_period": "0",
    "is_live_streamer": "0",
    "is_video_author": "0",
    "follow_user_num": "386",
    "follow_user_num_range": "(250,500]",
    "fans_user_num": "4",
    "fans_user_num_range": "[1,10)",
    "friend_user_num": "2",
    "friend_user_num_range": "[1,5)",
    "register_days": "327",
    "register_days_range": "181-365",
    "onehot_feat0": "0",
    "onehot_feat1": "3",
    "onehot_feat2": "25",
    "onehot_feat3": "1021",
    "onehot_feat4": "0",
    "onehot_feat5": "0",
    "onehot_feat6": "1",
    "onehot_feat7": "6",
    "onehot_feat8": "186",
    "onehot_feat9": "6",
    "onehot_feat10": "2",
    "onehot_feat11": "0",
    "onehot_feat12": "0",
    "onehot_feat13": "0",
    "onehot_feat14": "0",
    "onehot_feat15": "0",
    "onehot_feat16": "0",
    "onehot_feat17": "0"
  }
]
```

### `kuairec_caption_category.csv`

```json
[
  {
    "video_id": "0",
    "manual_cover_text": "UNKNOWN",
    "caption": "精神小伙路难走 程哥你狗粮慢点撒",
    "topic_tag": "[]",
    "first_level_category_id": "8",
    "first_level_category_name": "颜值",
    "second_level_category_id": "673",
    "second_level_category_name": "颜值随拍",
    "third_level_category_id": "-124",
    "third_level_category_name": "UNKNOWN"
  },
  {
    "video_id": "1",
    "manual_cover_text": "UNKNOWN",
    "caption": "",
    "topic_tag": "[]",
    "first_level_category_id": "27",
    "first_level_category_name": "高新数码",
    "second_level_category_id": "-124",
    "second_level_category_name": "UNKNOWN",
    "third_level_category_id": "-124",
    "third_level_category_name": "UNKNOWN"
  }
]
```
