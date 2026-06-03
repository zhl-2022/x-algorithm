# 推荐系统项目阶段性总实验报告

## 1. 项目目标

本项目围绕 X/Twitter 推荐系统的多阶段推荐思想，复现一个可解释、可评测、可迁移到公司 MLU
服务器的推荐系统训练闭环。当前目标不是完整复刻 X 的生产系统，也不是训练 Grok/LLM 基座，而是把推荐系统中最核心的工程链路跑通：

1. 将公开数据集转成用户行为样本。
2. 实现召回、排序、重排和离线评测。
3. 在 MovieLens、MIND、KuaiRec 三类场景中复用同一套实验思路。
4. 在寒武纪 MLU 服务器上验证训练适配、吞吐和双卡能力。
5. 沉淀可写入简历、可在面试中讲清楚的指标和结论。

## 2. 三批数据集完成情况

| 阶段 | 数据集 | 场景 | 完成内容 | 当前结论 |
|---|---|---|---|---|
| 1 | MovieLens 1M | 电影推荐 | Popularity、ItemCF、MF、Two-Tower、DNNRanker、TwoTower+DNN-Rerank | 最小推荐闭环跑通，两阶段 pipeline 最优 |
| 2 | MIND-small | 新闻推荐 | 热度、类别、DNNRanker、ContentTwoTower、候选重排、MLU 放大 | 文本和类别特征对内容推荐有效 |
| 3 | KuaiRec | 短视频推荐 | 标签消融、small/big 矩阵、hard negative、in-batch、蒸馏/图/序列/文本补强、MLU DDP | small 上 Ranker 有明确收益，big 上 ItemCF 蒸馏最有效但仍弱于 ItemCF |

## 3. 关键指标总览

| 数据集 | 模型/实验 | 关键指标 | 说明 |
|---|---|---:|---|
| MovieLens 1M | Popularity | `NDCG@20=0.026149` | 只按热门电影推荐，作为最低基线 |
| MovieLens 1M | ItemCF | `NDCG@20=0.032793` | 协同过滤开始体现个性化 |
| MovieLens 1M | MF | `NDCG@20=0.039807` | 可训练 embedding 带来提升 |
| MovieLens 1M | Two-Tower | `NDCG@20=0.040244` | 更接近工业召回结构 |
| MovieLens 1M | TwoTower+DNN-Rerank | `NDCG@20=0.042451` | 两阶段 pipeline 在 MovieLens 上最优 |
| MIND-small | DNNRanker, MLU | `NDCG@10=0.353507` | 排序模型具备候选内排序能力 |
| MIND-small | ContentTwoTower, MLU | `NDCG@10=0.374560` | 标题/摘要哈希文本 embedding 有效 |
| MIND-small | TwoTower+DNN-Rerank@10 | `NDCG@10=0.362443` | 候选扩大后会引入噪声，`candidate_k=10` 最好 |
| KuaiRec small | Two-Tower | `NDCG@20=0.143577` | 第一轮短视频召回最强 |
| KuaiRec small | `watch_ratio >= 0.8` Two-Tower | `NDCG@20=0.153744` | 接近完播标签比严格完播更适合 TopK |
| KuaiRec small | DNNRanker hard negative | `NDCG@20=0.240050` | hard negative 明显增强 Ranker |
| KuaiRec small | TwoTower+DNN-Rerank@200 | `NDCG@20=0.203215` | Ranker 重排超过单独 Two-Tower |
| KuaiRec big | ItemCF | `NDCG@20=0.065921` | big 场景当前 TopK 最强 |
| KuaiRec big | Stage 5 Two-Tower in-batch | `NDCG@20=0.004706` | 相比 BCE 有提升，但仍不足 |
| KuaiRec big | Stage 5 best pipeline | `NDCG@20=0.005245` | big 上神经 pipeline 仍弱于 ItemCF |
| KuaiRec big | ItemCF-Distill-TwoTower | `NDCG@20=0.033562` | 蒸馏 ItemCF 协同信号后明显提升神经 TopK |
| KuaiRec big | LightGCN | `NDCG@20=0.008166` | 第一版图召回覆盖率高，但排序质量不足 |
| KuaiRec big | TextCNN-TwoTower | `NDCG@20=0.007997` | 轻量文本 encoder 可运行，但单独文本增强收益有限 |
| KuaiRec MLU | 单卡 benchmark | `723,335 samples/s` | MLU 单卡训练链路跑通 |
| KuaiRec MLU | 双卡 DDP benchmark | `908,159 samples/s` | 双卡吞吐提升约 25.6% |

## 4. 主要技术结论

1. 推荐系统不是单模型问题，而是数据、召回、排序、重排和评测的系统工程。
2. `AUC` 高不等于 `NDCG@K` 高。`AUC` 衡量抽样正负样本排序能力，`NDCG@K` 衡量最终 TopK 推荐列表质量。
3. MovieLens 上，两阶段 pipeline 说明“召回候选 + Ranker 重排”能带来稳定收益。
4. MIND 上，内容特征有效，标题/摘要哈希 embedding 已经能提升新闻推荐效果。
5. KuaiRec small 上，hard negative 是 Ranker 的关键优化点，因为它让 Ranker 学会压低“两塔高分但真实不喜欢”的候选。
6. KuaiRec big 上，ItemCF 明显强于早期神经召回；第七轮验证 ItemCF 蒸馏 Two-Tower 能把神经 TopK 从 `0.005245` 提升到 `0.033562`，说明蒸馏强协同信号是有效补强方向。
7. MLU 工程链路已经打通，当前项目具备单卡训练、双卡 DDP benchmark 和吞吐记录能力。

## 5. 当前限制

| 限制 | 影响 | 后续方向 |
|---|---|---|
| KuaiRec big 神经召回仍弱于 ItemCF | Two-Tower 候选质量不足，蒸馏后仍未完全学到 ItemCF 排序信号 | 扩大 ItemCF 蒸馏样本，优化 teacher 权重和 hard negative 配比 |
| 文本特征仍是哈希 embedding | 只能利用弱文本信号，缺少语义理解 | 后续可接入轻量文本 encoder |
| DDP 只做吞吐 benchmark | 还没有做完整多卡 TopK 评测 | 后续如需工程深化，再做分布式评测 |
| 当前主要是离线评测 | 未包含在线 A/B、延迟和服务化 | 简历中应明确这是离线训练评测闭环 |

## 6. 阶段性结论

当前三批数据集已经足够支撑一个完整的简历项目：从入门数据集到内容推荐，再到短视频大规模推荐和 MLU 工程验证。下一步优先级不是马上切换 Tenrec，而是把现有实验写透；如果继续做模型实验，应优先扩大 KuaiRec big 的 ItemCF 蒸馏 Two-Tower。
