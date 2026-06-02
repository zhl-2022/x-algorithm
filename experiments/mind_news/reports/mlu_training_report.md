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

| 模型 | 范围 | AUC | MRR | NDCG@5 | NDCG@10 | HitRate@10 | 训练耗时秒 | 评估耗时秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Popularity | full | 0.522252 | 0.266074 | 0.247261 | 0.308465 | 0.611986 | 61.25 | 69.16 |
| Category | full | 0.588720 | 0.291866 | 0.274974 | 0.338507 | 0.657152 | 61.25 | 69.16 |
| DNNRanker | sample, MLU | 0.547961 | 0.267107 | 0.241286 | 0.311570 | 0.635250 | 0.73 | 0.07 |
| ContentTwoTower | sample, MLU | 0.562748 | 0.281838 | 0.256427 | 0.324701 | 0.650835 | 0.82 | 0.18 |
| TwoTower+DNN-Rerank | sample, MLU | 0.550229 | 0.271336 | 0.244329 | 0.316776 | 0.646011 | 1.55 | 0.25 |

## 结论

- MIND 阶段已经从本地可复现实验推进到公司 MLU 训练环境。
- 全量 `Category` baseline 明显超过全量 `Popularity`，说明类别和用户历史类别偏好有效。
- 神经模型已在 MLU 上跑通训练与评估链路，当前 100k 样本口径下 `ContentTwoTower` 的
  `NDCG@10=0.324701`，高于 `DNNRanker` 的 `0.311570`。
- 下一步应在 MLU 上扩大样本规模，并对 `candidate_k` 做消融实验。
