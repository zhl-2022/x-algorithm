# X Algorithm 推荐系统学习项目

本仓库当前有两层含义：

1. 上游代码层：保留 X/Twitter For You Feed Algorithm 的源码与说明，用于学习生产级推荐系统的模块拆分。
2. 本项目实验层：在 `experiments/` 和 `docs/` 下，基于公开数据集复现一个可解释、可评测、可写入简历的推荐系统训练闭环。

当前实验目标不是完整复刻 X 的生产系统，也不是训练 Grok/LLM 基座模型，而是参考其“候选召回、排序、过滤、重排、评测”的多阶段推荐思想，完成从数据处理到 MLU 训练验证的离线工程闭环。

## 项目入口

建议先从这些文档看起：

| 文档 | 作用 |
|---|---|
| [`docs/project_summary_report.md`](docs/project_summary_report.md) | 三批数据集的总实验结论和关键指标 |
| [`docs/project_architecture.md`](docs/project_architecture.md) | 项目架构图、模型链路和 MLU 训练链路 |
| [`docs/experiment_reading_guide.md`](docs/experiment_reading_guide.md) | 从哪里开始看实验、重点看哪些代码 |
| [`docs/resume_project_writeup.md`](docs/resume_project_writeup.md) | 可直接整理到简历里的项目描述 |
| [`docs/interview_qa.md`](docs/interview_qa.md) | 面试问答：AUC/TopK、hard negative、ItemCF 蒸馏、MLU 多卡 |
| [`docs/project_roadmap.md`](docs/project_roadmap.md) | 项目路线、阶段状态和后续扩展方向 |

## 已完成实验

| 阶段 | 数据集 | 场景 | 已完成内容 | 关键结论 |
|---|---|---|---|---|
| 1 | MovieLens 1M | 电影推荐 | Popularity、ItemCF、MF、Two-Tower、DNNRanker、两阶段重排 | 最小推荐闭环跑通，`TwoTower+DNN-Rerank` 最优 `NDCG@20=0.042451` |
| 2 | MIND-small | 新闻推荐 | 内容特征、DNNRanker、ContentTwoTower、candidate_k 消融、MLU 放大 | 标题/摘要哈希文本 embedding 有效，`ContentTwoTower NDCG@10=0.374560` |
| 3 | KuaiRec | 短视频推荐 | 标签消融、small/big 矩阵、hard negative、in-batch、ItemCF 蒸馏、soft label、MLU DDP | big 神经 pipeline 从 `0.005245` 提升到 `0.055883`，复跑 `0.052947` |

## 总体架构

```mermaid
flowchart LR
    A[公开数据集<br/>MovieLens / MIND / KuaiRec] --> B[数据处理<br/>清洗 / 切分 / 标签构造]
    B --> C[Baseline<br/>Popularity / Category / ItemCF]
    B --> D[召回模型<br/>MF / Two-Tower / ContentTwoTower / DistillTwoTower]
    B --> E[排序模型<br/>DNNRanker / hard negative]
    D --> F[候选集<br/>candidate_k]
    F --> G[重排与融合<br/>Rerank / Blend]
    E --> G
    G --> H[离线评测<br/>Recall@K / NDCG@K / AUC / LogLoss]
    H --> I[实验报告<br/>CSV / Markdown / 简历材料]
    D --> J[MLU 训练验证<br/>torch_mlu / DDP / 吞吐]
    E --> J
```

更完整的架构解释见 [`docs/project_architecture.md`](docs/project_architecture.md)。

## 推荐阅读顺序

如果你要快速理解当前项目，不建议从上游 X 源码逐行看起。推荐顺序是：

1. 先看 [`docs/project_summary_report.md`](docs/project_summary_report.md)，了解三批数据集得出了什么结论。
2. 再看 [`docs/project_architecture.md`](docs/project_architecture.md)，建立数据、模型、评测和 MLU 的整体地图。
3. 然后按 [`docs/experiment_reading_guide.md`](docs/experiment_reading_guide.md) 从 MovieLens 读到 MIND，再读 KuaiRec。
4. 最后看 [`docs/resume_project_writeup.md`](docs/resume_project_writeup.md) 和 [`docs/interview_qa.md`](docs/interview_qa.md)，把实验转成简历和面试表达。

## 关键代码入口

| 场景 | 重点脚本 | 需要掌握的内容 |
|---|---|---|
| MovieLens | `experiments/movielens_recall/scripts/prepare_movielens.py` | 隐式正反馈、时间切分、数据统计 |
| MovieLens | `experiments/movielens_recall/scripts/itemcf_baseline.py` | ItemCF 共现相似度和 TopK 推荐 |
| MovieLens | `experiments/movielens_recall/scripts/two_tower_train.py`、`ranker_train.py`、`rerank_pipeline.py` | 召回、排序和两阶段 pipeline |
| MIND | `experiments/mind_news/scripts/run_all_experiments.py` | 新闻内容特征、曝光候选、ContentTwoTower |
| KuaiRec | `experiments/kuairec_short_video/scripts/run_all_experiments.py` | 短视频标签、baseline、Two-Tower、Ranker、pipeline |
| KuaiRec | `experiments/kuairec_short_video/scripts/run_upgrade_experiments.py` | ItemCF 蒸馏、LightGCN、序列模型、TextCNN、soft label |
| MLU | `experiments/kuairec_short_video/scripts/benchmark_mlu_ddp.py` | 寒武纪 MLU 单卡/双卡 DDP 吞吐 benchmark |

## 当前项目结论

1. 推荐系统效果来自数据、召回、排序、重排和评测闭环，不是单个模型。
2. `AUC` 高不等于最终 TopK 推荐好，项目中始终以 `NDCG@K`、`Recall@K` 作为核心推荐指标。
3. KuaiRec big 上 ItemCF 仍是当前 TopK 参考上限，说明强协同信号非常重要。
4. ItemCF 蒸馏、hard negative 和 soft label 能显著改善神经召回与重排，最终将 KuaiRec big 神经 pipeline 提升到 `NDCG@20=0.055883`。
5. 寒武纪 MLU 训练链路已跑通，双卡 DDP benchmark 达到 `908,159 samples/s`，比单卡 `723,335 samples/s` 提升约 25.6%。

---

## 上游 X For You Feed Algorithm 原始说明

下面保留上游项目原始 README 内容，供学习 X/Twitter 推荐系统代码结构时参考。

This repository contains the core recommendation system powering the "For You" feed on X. It combines in-network content (from accounts you follow) with out-of-network content (discovered through ML-based retrieval) and ranks everything using a Grok-based transformer model.

> **Note:** The transformer implementation is ported from the [Grok-1 open source release](https://github.com/xai-org/grok-1) by xAI, adapted for recommendation system use cases.

## Table of Contents

- [Updates — May 15th, 2026](#updates--may-15th-2026)
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Components](#components)
  - [Home Mixer](#home-mixer)
  - [Thunder](#thunder)
  - [Phoenix](#phoenix)
  - [Candidate Pipeline](#candidate-pipeline)
- [How It Works](#how-it-works)
  - [Pipeline Stages](#pipeline-stages)
  - [Scoring and Ranking](#scoring-and-ranking)
  - [Filtering](#filtering)
- [Key Design Decisions](#key-design-decisions)
- [License](#license)

---

## Updates — May 15th, 2026

This release updates the For You algorithm code, including a runnable end-to-end inference pipeline alongside new components for content understanding, ads, and candidate sourcing.

1. **End-to-end inference pipeline:** A new [`phoenix/run_pipeline.py`](phoenix/run_pipeline.py) replaces the separate `run_ranker.py` and `run_retrieval.py` scripts with a single entry point that runs **retrieval → ranking** from exported checkpoints, mirroring how the two stages are composed in production.

2. **Pre-trained model artifacts:** A pre-trained mini Phoenix model (256-dim embeddings, 4 attention heads, 2 transformer layers) is now packaged as a ~3 GB archive distributed via Git LFS, enabling out-of-the-box inference without training your own model first.

3. **Grox content-understanding pipeline:** A new [`grox/`](grox/) service is included, providing classifiers, embedders, and a task-execution engine for content understanding workloads such as spam detection, post-category classification, and PTOS policy enforcement.

4. **Ads blending system:** Includes a new [`home-mixer/ads/`](home-mixer/ads/) module that handles ad injection and positioning within the feed, including brand-safety tracking that respects sensitive content boundaries.

5. **Query hydrators:** Home mixer now hydrates user context including followed topics, starter packs, impression bloom filters, IP, mutual follow graphs, and served history.

6. **Candidate hydrators:** Additional hydrators for engagement counts, brand safety signals, language codes, media detection, quote post expansion, mutual follow scores, and more.

7. **Candidate sources:** Adds sources for ads, who to follow, Phoenix MoE, Phoenix topics, prompts, and updates Thunder/Phoenix ones.

---

## Overview

The For You feed algorithm retrieves, ranks, and filters posts from two sources:

1. **In-Network (Thunder)**: Posts from accounts you follow
2. **Out-of-Network (Phoenix Retrieval)**: Posts discovered from a global corpus

Both sources are combined and ranked together using **Phoenix**, a Grok-based transformer model that predicts engagement probabilities for each post. The final score is a weighted combination of these predicted engagements.

We have eliminated every single hand-engineered feature and most heuristics from the system. The Grok-based transformer does all the heavy lifting by understanding your engagement history (what you liked, replied to, shared, etc.) and using that to determine what content is relevant to you.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    FOR YOU FEED REQUEST                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         HOME MIXER                                          │
│                                    (Orchestration Layer)                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                   QUERY HYDRATION                                   │   │
│   │  ┌──────────────────────────┐    ┌──────────────────────────────────────────────┐   │   │
│   │  │ User Action Sequence     │    │ User Features                                │   │   │
│   │  │ (engagement history)     │    │ (following list, preferences, etc.)          │   │   │
│   │  └──────────────────────────┘    └──────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                  CANDIDATE SOURCES                                  │   │
│   │         ┌─────────────────────────────┐    ┌────────────────────────────────┐       │   │
│   │         │        THUNDER              │    │     PHOENIX RETRIEVAL          │       │   │
│   │         │    (In-Network Posts)       │    │   (Out-of-Network Posts)       │       │   │
│   │         │                             │    │                                │       │   │
│   │         │  Posts from accounts        │    │  ML-based similarity search    │       │   │
│   │         │  you follow                 │    │  across global corpus          │       │   │
│   │         └─────────────────────────────┘    └────────────────────────────────┘       │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                      HYDRATION                                      │   │
│   │  Fetch additional data: core post metadata, author info, media entities, etc.       │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                      FILTERING                                      │   │
│   │  Remove: duplicates, old posts, self-posts, blocked authors, muted keywords, etc.   │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                       SCORING                                       │   │
│   │  ┌──────────────────────────┐                                                       │   │
│   │  │  Phoenix Scorer          │    Grok-based Transformer predicts:                   │   │
│   │  │  (ML Predictions)        │    P(like), P(reply), P(repost), P(click)...          │   │
│   │  └──────────────────────────┘                                                       │   │
│   │               │                                                                     │   │
│   │               ▼                                                                     │   │
│   │  ┌──────────────────────────┐                                                       │   │
│   │  │  Weighted Scorer         │    Weighted Score = Σ (weight × P(action))            │   │
│   │  │  (Combine predictions)   │                                                       │   │
│   │  └──────────────────────────┘                                                       │   │
│   │               │                                                                     │   │
│   │               ▼                                                                     │   │
│   │  ┌──────────────────────────┐                                                       │   │
│   │  │  Author Diversity        │    Attenuate repeated author scores                   │   │
│   │  │  Scorer                  │    to ensure feed diversity                           │   │
│   │  └──────────────────────────┘                                                       │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                      SELECTION                                      │   │
│   │                    Sort by final score, select top K candidates                     │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                              │                                              │
│                                              ▼                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              FILTERING (Post-Selection)                             │   │
│   │                 Visibility filtering (deleted/spam/violence/gore etc)               │   │
│   └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     RANKED FEED RESPONSE                                    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Home Mixer

**Location:** [`home-mixer/`](home-mixer/)

The orchestration layer that assembles the For You feed. It leverages the `CandidatePipeline` framework with the following stages:

| Stage | Description |
|-------|-------------|
| Query Hydrators | Fetch user context (engagement history, following list) |
| Sources | Retrieve candidates from Thunder and Phoenix |
| Hydrators | Enrich candidates with additional data |
| Filters | Remove ineligible candidates |
| Scorers | Predict engagement and compute final scores |
| Selector | Sort by score and select top K |
| Post-Selection Filters | Final visibility and dedup checks |
| Side Effects | Cache request info for future use |

The server exposes a gRPC endpoint (`ScoredPostsService`) that returns ranked posts for a given user.

---

### Thunder

**Location:** [`thunder/`](thunder/)

An in-memory post store and realtime ingestion pipeline that tracks recent posts from all users. It:

- Consumes post create/delete events from Kafka
- Maintains per-user stores for original posts, replies/reposts, and video posts
- Serves "in-network" post candidates from accounts the requesting user follows
- Automatically trims posts older than the retention period

Thunder enables sub-millisecond lookups for in-network content without hitting an external database.

---

### Phoenix

**Location:** [`phoenix/`](phoenix/)

The ML component with two main functions:

#### 1. Retrieval (Two-Tower Model)
Finds relevant out-of-network posts:
- **User Tower**: Encodes user features and engagement history into an embedding
- **Candidate Tower**: Encodes all posts into embeddings
- **Similarity Search**: Retrieves top-K posts via dot product similarity

#### 2. Ranking (Transformer with Candidate Isolation)
Predicts engagement probabilities for each candidate:
- Takes user context (engagement history) and candidate posts as input
- Uses special attention masking so candidates cannot attend to each other
- Outputs probabilities for each action type (like, reply, repost, click, etc.)

See [`phoenix/README.md`](phoenix/README.md) for detailed architecture documentation.

---

### Candidate Pipeline

**Location:** [`candidate-pipeline/`](candidate-pipeline/)

A reusable framework for building recommendation pipelines. Defines traits for:

| Trait | Purpose |
|-------|---------|
| `Source` | Fetch candidates from a data source |
| `Hydrator` | Enrich candidates with additional features |
| `Filter` | Remove candidates that shouldn't be shown |
| `Scorer` | Compute scores for ranking |
| `Selector` | Sort and select top candidates |
| `SideEffect` | Run async side effects (caching, logging) |

The framework runs sources and hydrators in parallel where possible, with configurable error handling and logging.

---

## How It Works

### Pipeline Stages

1. **Query Hydration**: Fetch the user's recent engagements history and metadata (eg. following list)

2. **Candidate Sourcing**: Retrieve candidates from:
   - **Thunder**: Recent posts from followed accounts (in-network)
   - **Phoenix Retrieval**: ML-discovered posts from the global corpus (out-of-network)

3. **Candidate Hydration**: Enrich candidates with:
   - Core post data (text, media, etc.)
   - Author information (username, verification status)
   - Video duration (for video posts)
   - Subscription status

4. **Pre-Scoring Filters**: Remove posts that are:
   - Duplicates
   - Too old
   - From the viewer themselves
   - From blocked/muted accounts
   - Containing muted keywords
   - Previously seen or recently served
   - Ineligible subscription content

5. **Scoring**: Apply multiple scorers sequentially:
   - **Phoenix Scorer**: Get ML predictions from the Phoenix transformer model
   - **Weighted Scorer**: Combine predictions into a final relevance score
   - **Author Diversity Scorer**: Attenuate repeated author scores for diversity
   - **OON Scorer**: Adjust scores for out-of-network content

6. **Selection**: Sort by score and select the top K candidates

7. **Post-Selection Processing**: Final validation of post candidates to be served

---

### Scoring and Ranking

The Phoenix Grok-based transformer model predicts probabilities for multiple engagement types:

```
Predictions:
├── P(favorite)
├── P(reply)
├── P(repost)
├── P(quote)
├── P(click)
├── P(profile_click)
├── P(video_view)
├── P(photo_expand)
├── P(share)
├── P(dwell)
├── P(follow_author)
├── P(not_interested)
├── P(block_author)
├── P(mute_author)
└── P(report)
```

The **Weighted Scorer** combines these into a final score:

```
Final Score = Σ (weight_i × P(action_i))
```

Positive actions (like, repost, share) have positive weights. Negative actions (block, mute, report) have negative weights, pushing down content the user would likely dislike.

---

### Filtering

Filters run at two stages:

**Pre-Scoring Filters:**
| Filter | Purpose |
|--------|---------|
| `DropDuplicatesFilter` | Remove duplicate post IDs |
| `CoreDataHydrationFilter` | Remove posts that failed to hydrate core metadata |
| `AgeFilter` | Remove posts older than threshold |
| `SelfpostFilter` | Remove user's own posts |
| `RepostDeduplicationFilter` | Dedupe reposts of same content |
| `IneligibleSubscriptionFilter` | Remove paywalled content user can't access |
| `PreviouslySeenPostsFilter` | Remove posts user has already seen |
| `PreviouslyServedPostsFilter` | Remove posts already served in session |
| `MutedKeywordFilter` | Remove posts with user's muted keywords |
| `AuthorSocialgraphFilter` | Remove posts from blocked/muted authors |

**Post-Selection Filters:**
| Filter | Purpose |
|--------|---------|
| `VFFilter` | Remove posts that are deleted/spam/violence/gore etc. |
| `DedupConversationFilter` | Deduplicate multiple branches of the same conversation thread |

---

## Key Design Decisions

### 1. No Hand-Engineered Features
The system relies entirely on the Grok-based transformer to learn relevance from user engagement sequences. No manual feature engineering for content relevance. This significantly reduces the complexity in our data pipelines and serving infrastructure.

### 2. Candidate Isolation in Ranking
During transformer inference, candidates cannot attend to each other—only to the user context. This ensures the score for a post doesn't depend on which other posts are in the batch, making scores consistent and cacheable.

### 3. Hash-Based Embeddings
Both retrieval and ranking use multiple hash functions for embedding lookup

### 4. Multi-Action Prediction
Rather than predicting a single "relevance" score, the model predicts probabilities for many actions.

### 5. Composable Pipeline Architecture
The `candidate-pipeline` crate provides a flexible framework for building recommendation pipelines with:
- Separation of pipeline execution and monitoring from business logic
- Parallel execution of independent stages and graceful error handling
- Easy addition of new sources, hydrations, filters, and scorers

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
