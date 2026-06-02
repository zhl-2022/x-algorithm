# MIND MLU 训练记录

## 运行环境

| 项目 | 信息 |
|---|---|
| 服务器 | `srv4` / `node2` |
| 容器 | `xalgorithm-mlu` |
| 镜像 | `cambricon-base/pytorch:v25.12.0-torch2.9.1-torchmlu1.30.2-ubuntu22.04-py310` |
| 可见设备 | `MLU_VISIBLE_DEVICES=2,3` |
| 设备 | 2 张 `MLU590-H8`，单卡 80GB |
| PyTorch | `2.9.1+cpu` |
| torch_mlu | `1.30.2+torch2.9.1` |
| MLU 可用性 | `torch.mlu.is_available() = True` |
| MLU 数量 | `torch.mlu.device_count() = 2` |

训练前后 `cnmon` 均确认 Card `2`、`3` 为 `0 MiB/81920 MiB`，未发现残留训练进程。

## 数据准备

服务器容器无法直接从 HuggingFace 下载 RecZoo 镜像，错误为：

```text
Cannot assign requested address
```

因此本次采用本地上传数据压缩包的方式：

```text
本地 MIND_small_x1.zip
  -> srv4:/root/zhl/x-algorithm/experiments/mind_news/data/raw/MIND_small_x1.zip
  -> 容器内解压和 prepare_mind.py 生成处理文件
```

## 运行命令

在服务器容器内执行：

```bash
docker exec -w /root/zhl/x-algorithm/experiments/mind_news \
  xalgorithm-mlu \
  python scripts/run_all_experiments.py
```

脚本会自动选择 `torch.mlu` 作为训练设备。统计类模型使用全量数据；神经模型使用
`train_sample.csv` 和 `valid_sample.csv` 各 100,000 行。

## 实验结果

### 100k 首次 MLU 验证

| 模型 | 范围 | AUC | MRR | NDCG@5 | NDCG@10 | HitRate@10 | 训练耗时秒 | 评估耗时秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | full | 0.522252 | 0.266074 | 0.247261 | 0.308465 | 0.611986 | 61.25 | 69.16 |
| Category | full | 0.588720 | 0.291866 | 0.274974 | 0.338507 | 0.657152 | 61.25 | 69.16 |
| DNNRanker | sample, MLU | 0.547961 | 0.267107 | 0.241286 | 0.311570 | 0.635250 | 0.73 | 0.07 |
| ContentTwoTower | sample, MLU | 0.562748 | 0.281838 | 0.256427 | 0.324701 | 0.650835 | 0.82 | 0.18 |
| TwoTower+DNN-Rerank | sample, MLU | 0.550229 | 0.271336 | 0.244329 | 0.316776 | 0.646011 | 1.55 | 0.25 |

### 500k 放大实验

| 模型 | 范围 | AUC | MRR | NDCG@5 | NDCG@10 | HitRate@10 | 训练耗时秒 | 评估耗时秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity-Sample | 500k/200k, MLU | 0.519972 | 0.266030 | 0.249040 | 0.308639 | 0.607023 | 60.37 | 3.26 |
| Category-Sample | 500k/200k, MLU | 0.590639 | 0.292102 | 0.276646 | 0.338938 | 0.656332 | 60.37 | 3.26 |
| DNNRanker | 500k/200k, MLU | 0.596585 | 0.307982 | 0.292274 | 0.353536 | 0.681360 | 1.72 | 0.10 |
| ContentTwoTower | 500k/200k, MLU | 0.608497 | 0.316013 | 0.299637 | 0.361835 | 0.689018 | 2.11 | 0.17 |

### 1M 文本哈希最终放大实验

| 模型 | 范围 | AUC | MRR | NDCG@5 | NDCG@10 | HitRate@10 | 训练耗时秒 | 评估耗时秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DNNRanker | 1M/500k, MLU | 0.592749 | 0.312269 | 0.289978 | 0.353507 | 0.681180 | 3.97 | 0.33 |
| ContentTwoTower | 1M/500k, MLU | 0.616641 | 0.331905 | 0.313456 | 0.374560 | 0.701867 | 5.54 | 0.39 |

| 模型 | 训练样本遍数 | 训练吞吐 样本/秒 | 评估吞吐 样本/秒 |
|---|---:|---:|---:|
| DNNRanker | 3,000,000 | 755,916 | 1,524,855 |
| ContentTwoTower | 3,000,000 | 541,957 | 1,293,661 |

`cnmon` 1 秒采样显示：Card 2 峰值显存 `224 MiB`，Card 3 峰值显存 `0 MiB`。
这说明当前脚本是单进程 MLU 训练，主要使用容器逻辑 MLU0，也就是宿主 Card 2；
Card 3 尚未参与 DDP。

### Candidate K 消融

| Candidate K | AUC | MRR | NDCG@5 | NDCG@10 | HitRate@10 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.602720 | 0.315350 | 0.294595 | 0.362443 | 0.701867 |
| 20 | 0.597145 | 0.313291 | 0.290818 | 0.355927 | 0.687901 |
| 50 | 0.594094 | 0.312461 | 0.290026 | 0.353712 | 0.682151 |
| 100 | 0.593528 | 0.312284 | 0.289967 | 0.353502 | 0.681180 |

## 结论

- MIND 阶段已经从本地可复现实验推进到公司 MLU 训练环境。
- 全量 `Category` baseline 明显超过全量 `Popularity`，说明类别和用户历史类别偏好有效。
- 1M 文本哈希最终放大实验中，`ContentTwoTower` 的 `AUC=0.616641`、`NDCG@10=0.374560`，
  已超过全量 `Category` baseline 和 DNNRanker。
- Candidate K 消融显示 `candidate_k=10` 当前最好，候选规模继续增大会引入更多噪声。
- 下一步如需体现双卡能力，应将当前单进程训练改造成 DDP；如需提升模型效果，应将哈希词向量升级为预训练文本 encoder。
