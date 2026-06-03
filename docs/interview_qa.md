# 推荐系统项目面试问答

## 1. 这个项目的核心价值是什么？

核心价值不是“跑了几个模型”，而是完成了推荐系统从数据到指标的训练评测闭环：

1. 数据处理：把评分、点击、观看行为转成训练样本。
2. 召回：用 ItemCF、MF、Two-Tower 找候选内容。
3. 排序：用 DNNRanker 预测用户对候选的兴趣。
4. 重排：用 TwoTower+DNN-Rerank 模拟多阶段推荐。
5. 评测：用 `Recall@K`、`NDCG@K`、`AUC`、`LogLoss` 判断效果。
6. 工程：把训练迁移到寒武纪 MLU，并记录单卡/双卡吞吐。

## 2. 为什么先做 MovieLens？

MovieLens 结构简单，适合先验证推荐系统最小闭环。它能帮助确认：

- 数据切分是否正确。
- 正负样本是否能构造。
- TopK 评测是否能跑通。
- Popularity、ItemCF、MF、Two-Tower、Ranker 是否能形成可对比结果。

MovieLens 阶段的关键结果是 TwoTower+DNN-Rerank `NDCG@20=0.042451`，高于单独模型，说明两阶段 pipeline 是有效的。

## 3. MIND 阶段相比 MovieLens 多了什么？

MIND 是新闻推荐，不再只是用户和物品 ID。它多了新闻标题、摘要、类别、曝光候选列表和用户点击历史。因此它更接近内容推荐业务。

当前 MLU 放大实验中，ContentTwoTower 的 `NDCG@10=0.374560`，高于 DNNRanker 的 `0.353507`，说明新闻文本和类别特征对内容推荐是有效的。

## 4. KuaiRec 阶段为什么重要？

KuaiRec 更接近短视频信息流推荐，因为它有用户-视频曝光矩阵、观看时长、完播率、类别和 caption。它让项目从电影、新闻进一步迁移到更接近真实内容流的场景。

项目中使用 `watch_ratio` 构造正反馈：

- `watch_ratio >= 1.0`：严格完播或重复观看。
- `watch_ratio >= 0.8`：接近完播。

实验显示 `watch_ratio >= 0.8` 的 Two-Tower `NDCG@20=0.153744`，高于严格完播的 `0.143577`，说明接近完播标签更密集、更适合当前 TopK 推荐。

## 5. 为什么 AUC 高但 NDCG@K 低？

`AUC` 和 `NDCG@K` 衡量的问题不同。

| 指标 | 关注点 | 本项目中的意义 |
|---|---|---|
| `AUC` | 随机抽一个正样本和负样本，模型能不能把正样本排在前面 | 衡量二分类排序能力 |
| `NDCG@K` | 真正推荐 TopK 时，命中的内容是否排在靠前位置 | 衡量最终推荐列表质量 |

KuaiRec big 上 MF 和 DNNRanker 的 `AUC` 不低，但 `NDCG@20` 很弱，说明模型在抽样正负样本里能做判断，但从一万多个视频里选 Top20 时还不能稳定找出用户未来喜欢的视频。

## 6. 为什么 KuaiRec big 上 ItemCF 反而最强？

ItemCF 直接利用用户共同观看或共同完播行为，能够捕捉强协同信号。KuaiRec big 的用户-视频交互密度更高，ItemCF 的共现统计更可靠，因此 TopK 表现强。

当前神经 Two-Tower 主要通过 ID embedding、类别、caption 哈希和统计特征学习表征，但训练目标和负采样还没有充分学到 ItemCF 的强协同结构，所以 big 上仍弱于 ItemCF。

## 7. hard negative 为什么有效？

普通负样本通常很容易区分，模型学不到细粒度排序。hard negative 是 Two-Tower 给了高分、但真实标签是负样本的视频。

这类样本对 Ranker 很有价值，因为它们模拟了真实重排场景：

1. Two-Tower 先召回一批看起来相关的候选。
2. Ranker 要在这些候选里把真实不适合的内容压下去。
3. hard negative 让 Ranker 专门学习这种困难边界。

KuaiRec small 上加入 141,100 条 hard negatives 后，`DNNRanker NDCG@20=0.240050`，TwoTower+DNN-Rerank@200 达到 `NDCG@20=0.203215`，说明 hard negative 对 Ranker 有明确收益。

## 8. 当前 Two-Tower 是 Transformer 吗？

不是。当前 Two-Tower 是推荐系统中常见的轻量双塔结构，不是 Transformer，也不是 LLM。

当前结构可以概括为：

- 用户塔：用户 ID embedding + 用户统计特征，通过 `Linear -> ReLU -> Dropout -> Linear` 输出用户向量。
- 视频塔：视频 ID embedding + 类别 embedding + caption 哈希 embedding + 视频统计特征，通过 `Linear -> ReLU -> Dropout -> Linear` 输出视频向量。
- 打分方式：用户向量和视频向量做 L2 normalize 后点积，再除以 temperature 得到召回分数。

## 9. MLU 实验体现了什么能力？

MLU 实验体现的是训练环境适配和工程验证能力：

- 推荐模型能在寒武纪 `torch_mlu` 环境中训练。
- 能限制训练只使用 Card 2/3。
- 能记录吞吐、训练时间和设备数。
- 已完成单卡/双卡 DDP benchmark。

当前 KuaiRec Two-Tower in-batch benchmark 单卡 `723,335 samples/s`，双卡 `908,159 samples/s`，双卡吞吐提升约 25.6%。

## 10. ItemCF 蒸馏做完后怎么看结果？

因为当前瓶颈很明确：KuaiRec big 上 ItemCF 的 `NDCG@20=0.065921`，而 stage5 神经 pipeline 最好只有 `0.005245`。这说明 ItemCF 有神经模型没有学到的协同排序信号。

第七轮实验中，ItemCF 蒸馏 Two-Tower 达到 `NDCG@20=0.033562`，明显高于 stage5 best pipeline `0.005245`。这说明蒸馏方向有效，Two-Tower 确实学到了一部分 ItemCF 的协同排序信号。

阶段八继续验证后，单独 2M 蒸馏 Two-Tower 的 `NDCG@20=0.027320`，反而低于 800k 蒸馏，说明不是样本越多越好；teacher 样本质量、软标签和正负样本比例更关键。

但把蒸馏 Two-Tower 接回 DNNRanker 后，阶段八 `DistillTwoTower+DNN-Rerank@200 NDCG@20=0.044560`，阶段九继续把最佳结果提升到 `DistillTwoTower+DNN-Rerank@100 NDCG@20=0.048158`。最后修正负样本采样并加入 `sqrt` teacher soft label 后，阶段十达到 `NDCG@20=0.055883`，换 seed 复跑仍有 `0.052947`。结论是：蒸馏方向有效，但不是 teacher 样本越多越好；当前更有效的是控制 teacher/negative 配比，并让 Ranker 在更干净的 `candidate_k=100` 候选集上融合重排。

## 11. 阶段九为什么 `2m_t40n120` 最好？

阶段九对比了三种蒸馏样本配比：

| 配置 | NDCG@20 | 解释 |
|---|---:|---|
| `800k_t40n40` | 0.039138 | 训练量偏小，pipeline 没超过阶段八 |
| `2m_t40n120` | 0.048158 | teacher 适中、随机负样本更多，召回边界更稳 |
| `2m_t120n40` | 0.039845 | teacher 占比过高，可能放大了 ItemCF 噪声 |

所以面试里可以这样解释：ItemCF 是强 teacher，但不能把它的 TopK 简单全部当成强正样本。适量 teacher 能提供协同过滤信号，更多随机负样本能帮助 Two-Tower 学会“哪些内容不该推”，二者平衡后 TopK 才提升。

## 12. 最终为什么选择 `soft_p30_t15_sqrt` 作为 KuaiRec 收尾结果？

阶段十修正了一个重要问题：之前 `negative_items_per_user` 只被解析，没有真正限制随机负样本采样。修正后重新做 soft label 消融，结果如下：

| 配置 | NDCG@20 | 解释 |
|---|---:|---|
| `soft_replay_p29_t14_linear` | 0.051595 | 复刻阶段九附近配置，修正采样后已超过阶段九 |
| `soft_p30_t10_linear` | 0.052608 | 降低 teacher 比例，提高负样本比例，有收益 |
| `soft_p30_t15_linear` | 0.053271 | teacher 比例提高到 15%，继续提升 |
| `soft_p30_t15_sqrt` | 0.055883 | 使用 `sqrt` soft label 强化高分 teacher，当前最优 |
| Stage11 replay | 0.052947 | 换 seed 后仍超过阶段九，说明结果不是单次偶然 |

可以这样回答：最终方案不是换更复杂模型，而是把训练信号做干净。ItemCF teacher 提供协同过滤排序信号，随机负样本校准召回边界，`sqrt` soft label 放大高分 teacher 的区分度，最后再由 Ranker 做融合重排。
