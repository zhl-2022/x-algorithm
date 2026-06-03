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

第二轮最重要的结论是：`watch_ratio >= 0.8` 更适合 `small_matrix.csv` 的 TopK 推荐；
当时 Ranker 重排尚未稳定超过单独 Two-Tower，因此后续进入 hard negative 优化。

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

## 第四到第六轮结果

后三轮迁移到 `big_matrix.csv` 并补充 MLU 工程实验。

| 轮次 | 实验 | 最好结果 | 结论 |
|---|---|---:|---|
| 第 4 轮 | big hard negative Ranker | Pipeline `NDCG@20=0.004991` | Ranker 有提升，但未超过 ItemCF |
| 第 5 轮 | big in-batch Two-Tower | Pipeline `NDCG@20=0.005245` | Two-Tower 有改善，但仍弱于 ItemCF |
| 第 6 轮 | MLU 单卡/双卡 benchmark | 双卡 `908,159 samples/s` | 双卡 DDP 跑通，吞吐比单卡高约 25.6% |

当前 `big_matrix.csv` 的关键对比如下：

| 模型/配置 | NDCG@20 | 说明 |
|---|---:|---|
| ItemCF | 0.065921 | 目前 big TopK 最强 |
| Stage 4 DNNRanker hard negative | 0.006151 | 比旧 big 神经模型有提升 |
| Stage 4 TwoTower+DNN-Rerank@500 | 0.004991 | hard negative pipeline 最好 |
| Stage 5 Two-Tower in-batch | 0.004706 | 召回底座相对 BCE 明显提升 |
| Stage 5 TwoTower+DNN-Rerank@100 | 0.005245 | in-batch pipeline 最好 |

详细说明见：

- `reports/stage4_big_hardneg_report.md`
- `reports/stage5_big_inbatch_report.md`
- `reports/stage6_mlu_ddp_report.md`

阶段结论：KuaiRec `small_matrix.csv` 已经完成并取得明确收益；`big_matrix.csv` 上统计协同过滤
仍明显强于当前神经召回，说明大规模短视频推荐还需要更强的图召回、序列建模或蒸馏方案。

## 第七轮升级实验结果

第七轮按“预处理缓存 + 分批启动”的方式完成四个 `big_matrix.csv` 补强方向。统一设置为
`watch_ratio >= 0.8`、`800,000` 训练样本、`7,174` 个评估用户。

| 方向 | 设备 | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---|---:|---:|---:|---|
| ItemCF 蒸馏 Two-Tower | MLU | 0.006874 | 0.033562 | 0.604001 | 本轮最好，明显高于 stage5 big pipeline |
| LightGCN / 图召回 | CPU | 0.002192 | 0.008166 | 0.509714 | 覆盖率高，但第一版排序质量不足 |
| 序列兴趣模型 | MLU | 0.000490 | 0.000868 | 0.704649 | AUC 高但 TopK 弱，短期兴趣候选质量不足 |
| 轻量文本 encoder | MLU | 0.003039 | 0.007997 | 0.554473 | 可运行，但单独文本增强收益有限 |

本轮最重要结论是：ItemCF 蒸馏 Two-Tower 将 big 场景神经 TopK 从 stage5 最佳
`NDCG@20=0.005245` 提升到 `0.033562`，方向有效；但仍低于 big ItemCF `NDCG@20=0.065921`，
后续应优先扩大蒸馏规模，而不是继续盲目加深 MLP。

详细设计和状态见：

- `reports/big_matrix_improvement_plan.md`
- `reports/upgrade_experiments_status.md`
- `reports/upgrade_experiments_batched/*/experiment_report.md`

## 第八轮召回补强结果

第八轮继续围绕 `big_matrix.csv` 做神经召回补强：放大 ItemCF 蒸馏 Two-Tower、
接入 DNNRanker pipeline、调参 LightGCN，并修正 GRU 序列模型 padding。

| 实验 | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---:|---:|---:|---|
| ItemCF-Distill-TwoTower 2M | 0.006202 | 0.027320 | 0.608339 | 样本放大后 AUC 略升，但 TopK 低于 800k 蒸馏 |
| LightGCN best `l1_e10` | 0.002902 | 0.011906 | 0.593868 | 调参后有提升，但仍弱于蒸馏路线 |
| GRU-Sequence fixed | 0.000488 | 0.000914 | 0.683278 | padding 修复后 TopK 仍弱 |
| DistillTwoTower+DNN-Rerank@200 | 0.011031 | 0.044560 | 0.681781 | 本轮最佳神经 pipeline |

阶段结论：蒸馏双塔单模型放大到 2M 没有超过 800k 版本，但蒸馏双塔接 DNNRanker 后达到
`NDCG@20=0.044560`，明显高于阶段七蒸馏双塔 `0.033562`，说明当前应继续优化
“蒸馏召回 + hard negative 排序”pipeline，而不是盲目扩到 5M。

详细报告见 `reports/stage8_recall_boost_report.md`。

## 第九轮 Pipeline 精调结果

第九轮围绕阶段八最佳 `DistillTwoTower+DNN-Rerank@200 NDCG@20=0.044560` 做三组蒸馏样本配比实验，
目标是判断 teacher 样本比例、随机负样本比例和 `candidate_k=100/200` 是否还能继续提升 big 场景 TopK。

| 实验 | 最佳模型 | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---|---:|---:|---:|---|
| `distill_pipeline_800k_t40n40` | `DistillTwoTower+DNN-Rerank@200` | 0.008508 | 0.039138 | 0.663471 | 未超过阶段八 |
| `distill_pipeline_2m_t40n120` | `DistillTwoTower+DNN-Rerank@100` | 0.011560 | 0.048158 | 0.685021 | 本轮最佳 |
| `distill_pipeline_2m_t120n40` | `DistillTwoTower+DNN-Blend@100a0.5` | 0.008474 | 0.039845 | 0.699481 | AUC 高但 TopK 不足 |

阶段结论：`2m_t40n120` 将当前最佳神经 pipeline 从 `NDCG@20=0.044560` 提升到
`0.048158`，相对提升约 `8.1%`。这说明降低 teacher 占比、提高随机负样本比例比单纯增加
teacher 样本更有效；但该结果仍低于 big ItemCF `NDCG@20=0.065921`。

详细计划和报告见：

- `reports/stage9_pipeline_tuning_plan.md`
- `reports/stage9_pipeline_tuning_report.md`

## 第十到第十一轮最终收尾结果

第十轮修正蒸馏采样逻辑，让 `negative_items_per_user` 真正参与随机负样本采样，并加入 teacher
soft label 变换。随后围绕阶段九最佳继续做四组比例消融，第十一轮用最佳配置换 `seed=2027` 复跑。

| 阶段 | 实验 | 最佳模型 | Recall@20 | NDCG@20 | AUC | 结论 |
|---|---|---|---:|---:|---:|---|
| Stage10 | `soft_replay_p29_t14_linear` | `Blend@100a0.75` | 0.012197 | 0.051595 | 0.694056 | 超过阶段九 |
| Stage10 | `soft_p30_t10_linear` | `Blend@100a0.5` | 0.012395 | 0.052608 | 0.683012 | 达到强成功线 |
| Stage10 | `soft_p30_t15_linear` | `Rerank@100` | 0.013347 | 0.053271 | 0.692285 | 继续提升 |
| Stage10 | `soft_p30_t15_sqrt` | `Blend@100a0.5` | 0.014698 | 0.055883 | 0.676812 | 当前最佳神经 pipeline |
| Stage11 | `final_replay_soft_p30_t15_sqrt_seed2027` | `Blend@100a0.5` | 0.013915 | 0.052947 | 0.669883 | 换 seed 后仍超过阶段九 |

最终结论：KuaiRec big 神经 pipeline 从 Stage5 的 `NDCG@20=0.005245`，经过 ItemCF 蒸馏、hard negative
排序和 soft label 精调，提升到 Stage10 最佳 `NDCG@20=0.055883`；换 seed 复跑仍有
`NDCG@20=0.052947`。这已经完成本数据集的模型实验收尾，但仍低于 big ItemCF `NDCG@20=0.065921`，
说明统计协同过滤仍是当前 TopK 上限参考。

详细报告见：

- `reports/stage10_soft_label_tuning_report.md`
- `reports/stage11_final_kuairec_report.md`

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
- [x] 完成 `big_matrix.csv` hard negative 迁移实验。
- [x] 完成 `big_matrix.csv` Two-Tower in-batch negative 实验。
- [x] 完成 MLU 单卡/双卡 DDP 吞吐实验。
- [x] 完成 KuaiRec big ItemCF 蒸馏 Two-Tower、LightGCN、序列兴趣模型和轻量文本 encoder 中等规模验证。
- [x] 完成 KuaiRec 阶段八：2M 蒸馏放大、蒸馏 pipeline、LightGCN 调参和序列 padding 修复验证。
- [x] 完成 KuaiRec 阶段九：蒸馏 pipeline 精调，最佳 `DistillTwoTower+DNN-Rerank@100 NDCG@20=0.048158`。
- [x] 完成 KuaiRec 阶段十和阶段十一：soft label 蒸馏精调与最终复跑，最佳 `NDCG@20=0.055883`。
