# KuaiRec 升级实验后台运行状态

## 1. 启动时间

2026-06-03 先在 srv4 的 `xalgorithm-mlu` 容器内启动过四个 KuaiRec big 并行升级实验。
由于四个进程会同时重复读取 `big_matrix.csv`，前期 CPU/IO 压力过大，已按稳定训练策略停止原并行任务，
改为“预处理缓存 + 分批启动”。

## 2. 原并行任务状态

| 任务 | PID | 设备设置 | 目标 |
|---|---:|---|---|
| `distill_twotower` | 7046 | `MLU_VISIBLE_DEVICES=2`，`device=auto` | 用 ItemCF teacher 信号蒸馏 Two-Tower |
| `lightgcn` | 7048 | `device=cpu` | 用用户-视频二部图训练 LightGCN 召回 |
| `sequence_model` | 7050 | `MLU_VISIBLE_DEVICES=3`，`device=auto` | 用 GRU 编码用户最近完播序列 |
| `text_encoder` | 7052 | `MLU_VISIBLE_DEVICES=2,3`，`device=auto` | 用 TextCNN 替代 caption 哈希均值池化 |

以上四个旧任务已执行 `kill` 停止，不再占用 CPU 或 MLU。

## 3. 当前分批调度状态

| 阶段 | 状态 | 结果 |
|---|---|---|
| 全量 `PreparedData` 缓存构建 | 已完成 | `train=10,021,757`、`test=1,256,352`、`eval_users=7,174`，耗时 `1229.42s` |
| 第一批：`distill_twotower` | 已完成 | 使用 Card 2，结果已写入 CSV 和报告 |
| 第一批：`sequence_model` | 已完成 | 使用 Card 3，结果已写入 CSV 和报告 |
| 第一批：`lightgcn` | 已完成 | 使用 CPU，结果已写入 CSV 和报告 |
| 第二批：`text_encoder` | 已完成 | 使用 Card 2，结果已写入 CSV 和报告 |

截至 2026-06-03 10:24，srv4 的 Card 2/3 显存均为 `0 MiB/81920 MiB`，说明本轮训练已经结束并释放 MLU。

## 4. 分批执行顺序

1. 构建一次全量缓存：`experiments/kuairec_short_video/data/cache/big_matrix_threshold08_prepared.pkl`。
2. 第一批并行：
   - `distill_twotower`：Card 2。
   - `sequence_model`：Card 3。
   - `lightgcn`：CPU。
3. 第一批全部完成后，第二批：
   - `text_encoder`：Card 2。

实际执行日志：

| 时间 | 事件 |
|---|---|
| 2026-06-03 09:52:38 | 开始构建 `big_matrix_threshold08_prepared.pkl` |
| 2026-06-03 10:13:16 | 缓存完成，启动第一批 `distill_twotower`、`sequence_model`、`lightgcn` |
| 2026-06-03 10:16:31 | 第一批三个实验完成，启动第二批 `text_encoder` |
| 2026-06-03 10:18:42 | 第二批完成，全部升级实验结束 |

## 5. 统一参数

| 参数 | 数值 |
|---|---:|
| 数据集 | `big_matrix.csv` |
| 标签 | `watch_ratio >= 0.8` |
| 训练样本 | 800,000 |
| AUC 样本 | 300,000 |
| Epochs | 2 |
| Batch size | 4,096 |
| Embedding dim | 64 |
| Tower dim | 64 |
| Hidden dim | 128 |

## 6. 查看命令

```bash
docker exec xalgorithm-mlu bash -lc "ps -ef | grep -E 'run_upgrade_experiments|start_upgrade|torchrun|python' | grep -v grep || true"
docker exec xalgorithm-mlu bash -lc "tail -n 40 experiments/kuairec_short_video/logs/upgrade_experiments_batched/supervisor.log"
docker exec xalgorithm-mlu bash -lc "ls -lh experiments/kuairec_short_video/data/cache"
docker exec xalgorithm-mlu bash -lc "ls -lh experiments/kuairec_short_video/logs/upgrade_experiments_batched"
cnmon
```

## 7. 输出目录

| 类型 | 路径 |
|---|---|
| 缓存 | `experiments/kuairec_short_video/data/cache/` |
| 日志 | `experiments/kuairec_short_video/logs/upgrade_experiments_batched/` |
| PID | `experiments/kuairec_short_video/outputs/upgrade_experiments_batched/pids/` |
| CSV 结果 | `experiments/kuairec_short_video/outputs/upgrade_experiments_batched/<experiment>/experiment_results.csv` |
| Markdown 报告 | `experiments/kuairec_short_video/reports/upgrade_experiments_batched/<experiment>/experiment_report.md` |

## 8. 正式实验结果

本轮使用 `big_matrix.csv`、`watch_ratio >= 0.8`、`800,000` 训练样本和 `7,174` 个评估用户。

| 任务 | 设备 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ItemCF-Distill-TwoTower | MLU | 0.006874 | 0.363953 | 0.033562 | 0.024981 | 0.604001 | 0.679956 | 本轮 TopK 最好，说明蒸馏 ItemCF 协同信号有效 |
| LightGCN | CPU | 0.002192 | 0.144550 | 0.008166 | 0.956283 | 0.509714 | 0.693142 | 覆盖率很高，但排序质量弱，第一版图召回还需要调参 |
| GRU-Sequence-Interest | MLU | 0.000490 | 0.022163 | 0.000868 | 0.005500 | 0.704649 | 0.650996 | AUC 高但 TopK 很弱，序列模型当前没有形成有效候选召回 |
| TextCNN-TwoTower | MLU | 0.003039 | 0.152635 | 0.007997 | 0.129754 | 0.554473 | 0.691228 | 轻量文本 encoder 可运行，但单独文本增强不足以解决 big TopK |

对比阶段五 big 最佳 pipeline `NDCG@20=0.005245`，ItemCF 蒸馏 Two-Tower 提升到 `0.033562`；
但它仍低于 big ItemCF `NDCG@20=0.065921`。这说明“让神经召回学习 ItemCF”是正确方向，
但还需要扩大蒸馏样本、改进 teacher 样本权重和候选评估策略。

## 9. 已完成的 smoke test

四个方向都已在 `big_matrix.csv --max-rows 50000` 上通过 1 epoch CPU smoke test：

| 任务 | Smoke NDCG@20 | 说明 |
|---|---:|---|
| `distill_twotower` | 0.007806 | 蒸馏样本构造、训练、评测可运行 |
| `lightgcn` | 0.022075 | 图构建、BPR 训练、TopK 评测可运行 |
| `sequence_model` | 0.008030 | GRU 序列模型训练、评测可运行 |
| `text_encoder` | 0.002708 | TextCNN 双塔训练、评测可运行 |

## 10. 注意事项

- 本轮任务使用 `nohup` 后台运行，断开本地会话不会中断容器内调度；当前已经全部完成。
- 新方案只读取并解析一次 `big_matrix.csv`，后续实验直接加载缓存。
- 缓存文件位于 `data/cache/`，属于大文件，不提交 Git。
- 本轮是 80 万样本中等规模试验，不是最终最大规模；后续优先把 ItemCF 蒸馏扩大到 200 万或更多样本。
