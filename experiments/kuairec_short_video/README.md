# KuaiRec 短视频推荐实验

## 实验目标

这是本项目的第三批公开数据集实验。前两批数据集已经完成：

1. MovieLens 1M：电影推荐，重点跑通推荐系统离线闭环。
2. MIND-small：新闻推荐，重点引入内容特征、曝光候选列表和 MLU 训练。

KuaiRec 阶段的目标是迁移到更接近真实信息流业务的短视频推荐场景，重点学习：

- 用户-短视频曝光行为如何构造训练样本。
- 观看时长、完播率、类别、文本描述等特征如何进入推荐模型。
- Popularity、Category、ItemCF、Two-Tower、Ranker 和两阶段 pipeline 如何复用到新数据集。
- 如何在 srv4 的 MLU 容器中做更大规模训练、吞吐和显存记录。

## 数据来源

当前使用 Zenodo 上的新版 KuaiRec 数据：

- 数据记录：<https://zenodo.org/records/18164998>
- 项目主页：<https://kuairec.com/>

Zenodo 记录名为 `KuaiRec Dataset with Raw Features`，包含：

| 文件 | 用途 |
|---|---|
| `KuaiRec.zip` | 主数据包，包含交互矩阵、用户特征、视频特征等核心文件 |
| `kuairec_caption_category.csv` | 视频 caption 和类别文本信息 |
| `video_raw_categories_multi.csv` | 视频原始多类别标签 |
| `user_features_raw.csv` | 原始用户特征 |

## 当前推荐执行顺序

1. 下载并解压 KuaiRec 数据。
2. 运行数据盘点脚本，生成 `reports/data_report.md`。
3. 根据实际字段确认正反馈定义，例如 `watch_ratio >= 1.0` 或 `is_click = 1`。
4. 构造训练集、验证集和测试集。
5. 实现短视频 Popularity baseline。
6. 实现 Category baseline 和 ItemCF。
7. 迁移 MovieLens/MIND 中已有的 MF、Two-Tower、DNN Ranker 和两阶段 pipeline。
8. 在 MLU 上做放大训练，记录 batch size、训练耗时、吞吐、显存和指标。

## 本地命令

```powershell
python experiments\kuairec_short_video\scripts\download_kuairec.py
python experiments\kuairec_short_video\scripts\prepare_kuairec.py --sample-rows 1000
python experiments\kuairec_short_video\scripts\run_all_experiments.py --device cpu --max-rows 50000 --neural-train-rows 20000 --epochs 1
```

## 服务器命令

在 srv4 上建议放到 `/root/zhl/x-algorithm`：

```bash
cd /root/zhl/x-algorithm
python experiments/kuairec_short_video/scripts/download_kuairec.py
python experiments/kuairec_short_video/scripts/prepare_kuairec.py --sample-rows 1000
```

如果后续进入 MLU 容器训练：

```bash
docker exec -it xalgorithm-mlu bash
source /torch/venv3/pytorch/bin/activate
cd /root/zhl/x-algorithm
python experiments/kuairec_short_video/scripts/run_all_experiments.py \
  --matrix small_matrix.csv \
  --neural-train-rows 1200000 \
  --auc-rows 300000 \
  --epochs 2 \
  --batch-size 8192 \
  --embedding-dim 64 \
  --tower-dim 64 \
  --hidden-dim 128 \
  --ranker-hidden-dims 128,64 \
  --candidate-ks 50,100,200 \
  --rerank-blend-alphas 0,0.25,0.5,0.75,1 \
  --ranker-positive-weight 2 \
  --device auto
```

## 第一轮结果

当前已在 srv4 的 `xalgorithm-mlu` 容器内完成 `small_matrix.csv` 第一轮完整实验。

| 模型 | Recall@20 | NDCG@20 | AUC | 设备 |
|---|---:|---:|---:|---|
| Popularity | 0.011655 | 0.055724 | 0.480085 | CPU |
| Category | 0.017828 | 0.083583 | 0.534036 | CPU |
| ItemCF | 0.019690 | 0.097525 | 0.478955 | CPU |
| MF | 0.025962 | 0.127836 | 0.683895 | MLU |
| Two-Tower | 0.026051 | 0.143577 | 0.549570 | MLU |
| DNNRanker | 0.025540 | 0.113215 | 0.656796 | MLU |
| TwoTower+DNN-Rerank@50 | 0.025497 | 0.113354 | 0.656796 | MLU |

当前 `NDCG@20` 最好的是 `Two-Tower`，说明短视频场景下用户塔和视频塔的向量召回已经明显超过热度和协同过滤 baseline。

## 第二轮结果

第二轮围绕“标签阈值、训练规模、候选重排、big_matrix 放大”做系统实验。

| 实验 | 数据 | 标签 | 神经训练样本 | 当前最佳模型 | 最佳 NDCG@20 | 结论 |
|---|---|---|---:|---|---:|---|
| 标签阈值消融 | `small_matrix.csv` | `watch_ratio >= 0.8` | 1,200,000 | Two-Tower | 0.153744 | 放宽到接近完播后，双塔效果提升 |
| 全量 small 训练 | `small_matrix.csv` | `watch_ratio >= 1.0` | 3,595,097 | Two-Tower | 0.149288 | 扩大训练样本后双塔继续提升 |
| big 采样放大 | `big_matrix.csv` | `watch_ratio >= 1.0` | 2,000,000 | ItemCF | 0.058148 | 大候选池下神经模型 TopK 还不稳定 |

阶段二详细解释见：

- `reports/stage2_summary_report.md`
- `reports/stage2_threshold08/all_experiments_report.md`
- `reports/stage2_full_small/all_experiments_report.md`
- `reports/stage2_big_sample/all_experiments_report.md`

当前最重要的结论是：`watch_ratio >= 0.8` 更适合 `small_matrix.csv` 的 TopK 推荐，
但 Ranker 重排还没有稳定超过单独 Two-Tower；`big_matrix.csv` 上神经模型 AUC 较高但 TopK 偏弱，
后续应优先做 hard negative、in-batch negative 或更强召回训练。

## 第三轮结果

第三轮专门优化 Ranker，新增 Two-Tower hard negative 训练：

| 模型 | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---:|---:|---:|---|
| Two-Tower | 0.018152 | 0.159630 | 0.508734 | 召回底座 |
| DNNRanker | 0.028238 | 0.240050 | 0.653649 | 全量打分 TopK 最强 |
| TwoTower+DNN-Rerank@50 | 0.019543 | 0.170111 | 0.690666 | 重排超过 Two-Tower |
| TwoTower+DNN-Rerank@100 | 0.022158 | 0.197553 | 0.654515 | 候选扩大后继续提升 |
| TwoTower+DNN-Rerank@200 | 0.022686 | 0.203215 | 0.687100 | 本轮最佳两阶段 pipeline |

本轮已解决“Ranker 重排没有超过 Two-Tower”的问题。详细说明见
`reports/stage3_ranker_optimization_report.md`。

## 当前状态

- [x] 新建 KuaiRec 实验目录。
- [x] 固化官方数据下载脚本。
- [x] 新建面向小白的试验方案文档。
- [x] 在 srv4 下载并解压数据。
- [x] 生成数据盘点报告。
- [x] 确认第一版标签口径：`watch_ratio >= 1.0`。
- [x] 根据 `watch_ratio` 构造训练样本。
- [x] 实现 KuaiRec Popularity、Category、ItemCF baseline。
- [x] 完成 KuaiRec MF、Two-Tower、DNN Ranker 和两阶段 pipeline 训练。
- [x] 在 srv4 MLU 容器完成第一轮训练验证。
- [x] 完成标签阈值消融：`watch_ratio >= 1.0` 对比 `watch_ratio >= 0.8`。
- [x] 完成 `small_matrix.csv` 全量神经训练实验。
- [x] 完成 `big_matrix.csv` 采样放大实验。
- [x] 完成 Two-Tower 与 DNNRanker 融合重排实验。
- [x] 完成 Ranker hard negative 优化实验，使两阶段 pipeline 超过单独 Two-Tower。
