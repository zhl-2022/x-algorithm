"""Train and evaluate a DNN ranking model on MovieLens."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = EXPERIMENT_DIR / "data" / "processed"
DEFAULT_REPORTS_DIR = EXPERIMENT_DIR / "reports"
DEFAULT_OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
    rating: float
    timestamp: int


@dataclass(frozen=True)
class RetrievalMetrics:
    model: str
    k: int
    evaluated_users: int
    recall: float
    hit_rate: float
    precision: float
    ndcg: float
    coverage: float


@dataclass(frozen=True)
class RankingMetrics:
    auc: float
    logloss: float
    eval_samples: int
    eval_positive_samples: int
    eval_negative_samples: int


@dataclass(frozen=True)
class TrainingStats:
    device: str
    embedding_dim: int
    hidden_dims: tuple[int, ...]
    num_genres: int
    epochs: int
    batch_size: int
    negative_samples: int
    eval_negative_samples: int
    learning_rate: float
    dropout: float
    train_seconds: float
    eval_seconds: float
    final_loss: float
    num_users: int
    num_items: int


class DNNRanker(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        user_dense_features: torch.Tensor,
        item_dense_features: torch.Tensor,
        item_genre_features: torch.Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.register_buffer("user_dense_features", user_dense_features)
        self.register_buffer("item_dense_features", item_dense_features)
        self.register_buffer("item_genre_features", item_genre_features)

        input_dim = (
            embedding_dim * 3
            + user_dense_features.shape[1]
            + item_dense_features.shape[1]
            + item_genre_features.shape[1]
        )
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embedding(user_indices)
        item_vectors = self.item_embedding(item_indices)
        cross_vectors = user_vectors * item_vectors
        features = torch.cat(
            [
                user_vectors,
                item_vectors,
                cross_vectors,
                self.user_dense_features[user_indices],
                self.item_dense_features[item_indices],
                self.item_genre_features[item_indices],
            ],
            dim=1,
        )
        return self.mlp(features).squeeze(1)


def read_interactions(path: Path) -> list[Interaction]:
    interactions: list[Interaction] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            interactions.append(
                Interaction(
                    user_id=int(row["user_id"]),
                    item_id=int(row["item_id"]),
                    rating=float(row["rating"]),
                    timestamp=int(row["timestamp"]),
                )
            )
    return interactions


def read_item_metadata(path: Path) -> dict[int, tuple[str, str]]:
    items: dict[int, tuple[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            items[int(row["item_id"])] = (row["title"], row["genres"])
    return items


def group_items_by_user(rows: list[Interaction]) -> dict[int, set[int]]:
    grouped: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        grouped[row.user_id].add(row.item_id)
    return grouped


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    try:
        import torch_mlu  # noqa: F401

        if hasattr(torch, "mlu") and torch.mlu.is_available():
            return torch.device("mlu")
    except Exception:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not dims:
        raise argparse.ArgumentTypeError("hidden dims cannot be empty")
    return dims


def build_mappings(
    train: list[Interaction],
    test: list[Interaction],
    all_item_ids: set[int],
) -> tuple[dict[int, int], dict[int, int], list[int]]:
    user_ids = sorted({row.user_id for row in train} | {row.user_id for row in test})
    item_ids = sorted(all_item_ids)
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    return user_to_index, item_to_index, item_ids


def build_feature_tensors(
    train: list[Interaction],
    item_metadata: dict[int, tuple[str, str]],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    user_counts = Counter(row.user_id for row in train)
    item_counts = Counter(row.item_id for row in train)

    max_user_log = max((math.log1p(count) for count in user_counts.values()), default=1.0)
    max_item_log = max((math.log1p(count) for count in item_counts.values()), default=1.0)
    max_genre_count = max(
        (
            len(genres.split("|"))
            for _, genres in item_metadata.values()
            if genres and genres != "(no genres listed)"
        ),
        default=1,
    )

    user_dense = torch.zeros((len(user_to_index), 1), dtype=torch.float32)
    for user_id, user_index in user_to_index.items():
        user_dense[user_index, 0] = math.log1p(user_counts[user_id]) / max_user_log

    genres = sorted(
        {
            genre
            for _, genre_text in item_metadata.values()
            for genre in genre_text.split("|")
            if genre and genre != "(no genres listed)"
        }
    )
    genre_to_index = {genre: index for index, genre in enumerate(genres)}

    item_dense = torch.zeros((len(item_to_index), 2), dtype=torch.float32)
    item_genres = torch.zeros((len(item_to_index), len(genres)), dtype=torch.float32)
    for item_id, item_index in item_to_index.items():
        _, genre_text = item_metadata.get(item_id, ("", ""))
        item_genre_list = [
            genre
            for genre in genre_text.split("|")
            if genre and genre != "(no genres listed)"
        ]
        item_dense[item_index, 0] = math.log1p(item_counts[item_id]) / max_item_log
        item_dense[item_index, 1] = len(item_genre_list) / max_genre_count
        for genre in item_genre_list:
            item_genres[item_index, genre_to_index[genre]] = 1.0

    return user_dense, item_dense, item_genres, genres


def sample_negative_item(
    rng: random.Random,
    item_indices: list[int],
    excluded_items: set[int],
) -> int:
    while True:
        candidate = rng.choice(item_indices)
        if candidate not in excluded_items:
            return candidate


def build_train_pairs(
    train: list[Interaction],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
) -> tuple[list[tuple[int, int]], dict[int, set[int]]]:
    train_pairs: list[tuple[int, int]] = []
    user_positive_indices: dict[int, set[int]] = defaultdict(set)
    for row in train:
        if row.user_id in user_to_index and row.item_id in item_to_index:
            pair = (user_to_index[row.user_id], item_to_index[row.item_id])
            train_pairs.append(pair)
            user_positive_indices[pair[0]].add(pair[1])
    return train_pairs, user_positive_indices


def make_training_batch(
    batch_pairs: list[tuple[int, int]],
    user_positive_indices: dict[int, set[int]],
    item_indices: list[int],
    negative_samples: int,
    rng: random.Random,
) -> tuple[list[int], list[int], list[float]]:
    users: list[int] = []
    items: list[int] = []
    labels: list[float] = []

    for user_index, item_index in batch_pairs:
        users.append(user_index)
        items.append(item_index)
        labels.append(1.0)

        positives = user_positive_indices[user_index]
        for _ in range(negative_samples):
            negative_item = sample_negative_item(rng, item_indices, positives)
            users.append(user_index)
            items.append(negative_item)
            labels.append(0.0)

    return users, items, labels


def train_model(
    train: list[Interaction],
    item_metadata: dict[int, tuple[str, str]],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DNNRanker, float, float, int]:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    train_pairs, user_positive_indices = build_train_pairs(
        train=train,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
    )
    user_dense, item_dense, item_genres, genres = build_feature_tensors(
        train=train,
        item_metadata=item_metadata,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
    )

    model = DNNRanker(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        user_dense_features=user_dense,
        item_dense_features=item_dense,
        item_genre_features=item_genres,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    item_indices = list(range(len(item_to_index)))

    train_started = time.perf_counter()
    final_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_pairs)
        epoch_loss = 0.0
        batch_count = 0

        model.train()
        for start in range(0, len(train_pairs), args.batch_size):
            batch_pairs = train_pairs[start : start + args.batch_size]
            users, items, labels = make_training_batch(
                batch_pairs=batch_pairs,
                user_positive_indices=user_positive_indices,
                item_indices=item_indices,
                negative_samples=args.negative_samples,
                rng=rng,
            )
            user_tensor = torch.tensor(users, dtype=torch.long, device=device)
            item_tensor = torch.tensor(items, dtype=torch.long, device=device)
            label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(user_tensor, item_tensor)
            loss = F.binary_cross_entropy_with_logits(logits, label_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            batch_count += 1

        final_loss = epoch_loss / batch_count if batch_count else 0.0
        print(f"epoch={epoch} loss={final_loss:.6f}", flush=True)

    return model, final_loss, time.perf_counter() - train_started, len(genres)


def binary_auc(labels: list[int], scores: list[float]) -> float:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.0

    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2
        for index in range(position, end):
            ranks[order[index]] = average_rank
        position = end

    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def build_ranking_eval_pairs(
    test_by_user: dict[int, set[int]],
    train_by_user: dict[int, set[int]],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    eval_negative_samples: int,
    seed: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    rng = random.Random(seed)
    item_indices = list(range(len(item_to_index)))
    pairs: list[tuple[int, int]] = []
    labels: list[int] = []

    for user_id in sorted(test_by_user):
        if user_id not in user_to_index:
            continue
        user_index = user_to_index[user_id]
        relevant_items = {
            item_to_index[item_id]
            for item_id in test_by_user[user_id]
            if item_id in item_to_index
        }
        train_items = {
            item_to_index[item_id]
            for item_id in train_by_user.get(user_id, set())
            if item_id in item_to_index
        }
        excluded_items = train_items | relevant_items

        for item_index in sorted(relevant_items):
            pairs.append((user_index, item_index))
            labels.append(1)

        for _ in range(eval_negative_samples):
            negative_item = sample_negative_item(rng, item_indices, excluded_items)
            pairs.append((user_index, negative_item))
            labels.append(0)

    return pairs, labels


def score_pairs(
    model: DNNRanker,
    pairs: list[tuple[int, int]],
    batch_size: int,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            user_tensor = torch.tensor(
                [user_index for user_index, _ in batch_pairs],
                dtype=torch.long,
                device=device,
            )
            item_tensor = torch.tensor(
                [item_index for _, item_index in batch_pairs],
                dtype=torch.long,
                device=device,
            )
            scores.extend(model(user_tensor, item_tensor).detach().cpu().tolist())
    return scores


def logloss_from_logits(labels: list[int], logits: list[float]) -> float:
    losses = []
    for label, logit in zip(labels, logits):
        if logit >= 0:
            losses.append((1 - label) * logit + math.log1p(math.exp(-logit)))
        else:
            losses.append(-label * logit + math.log1p(math.exp(logit)))
    return sum(losses) / len(losses) if losses else 0.0


def dcg_for_hits(recommendations: list[int], relevant_items: set[int]) -> float:
    dcg = 0.0
    for index, item_id in enumerate(recommendations, start=1):
        if item_id in relevant_items:
            dcg += 1.0 / math.log2(index + 1)
    return dcg


def ideal_dcg(num_relevant: int, k: int) -> float:
    return sum(1.0 / math.log2(index + 1) for index in range(1, min(num_relevant, k) + 1))


def evaluate_topk(
    model: DNNRanker,
    train_by_user: dict[int, set[int]],
    test_by_user: dict[int, set[int]],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    index_to_item: list[int],
    k: int,
    score_batch_size: int,
    device: torch.device,
) -> RetrievalMetrics:
    model.eval()
    total_recall = 0.0
    total_hit_rate = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    recommended_items: set[int] = set()
    all_item_indices = list(range(len(index_to_item)))

    with torch.no_grad():
        for user_id in sorted(test_by_user):
            if user_id not in user_to_index:
                continue
            user_index = user_to_index[user_id]
            scores: list[float] = []
            for start in range(0, len(all_item_indices), score_batch_size):
                batch_items = all_item_indices[start : start + score_batch_size]
                user_tensor = torch.full(
                    (len(batch_items),),
                    user_index,
                    dtype=torch.long,
                    device=device,
                )
                item_tensor = torch.tensor(batch_items, dtype=torch.long, device=device)
                scores.extend(model(user_tensor, item_tensor).detach().cpu().tolist())

            score_tensor = torch.tensor(scores, dtype=torch.float32)
            for seen_item in train_by_user.get(user_id, set()):
                seen_index = item_to_index.get(seen_item)
                if seen_index is not None:
                    score_tensor[seen_index] = -float("inf")

            top_indices = torch.topk(score_tensor, k=min(k, len(score_tensor))).indices.tolist()
            recommendations = [index_to_item[index] for index in top_indices]
            relevant_items = test_by_user[user_id]
            recommended_items.update(recommendations)

            hits = len(set(recommendations) & relevant_items)
            total_recall += hits / len(relevant_items)
            total_hit_rate += 1.0 if hits > 0 else 0.0
            total_precision += hits / k
            user_dcg = dcg_for_hits(recommendations, relevant_items)
            user_idcg = ideal_dcg(len(relevant_items), k)
            total_ndcg += user_dcg / user_idcg if user_idcg else 0.0

    evaluated_users = len(test_by_user)
    if evaluated_users == 0:
        raise ValueError("No test users found. Run prepare_movielens.py first.")

    return RetrievalMetrics(
        model="DNN-Ranker",
        k=k,
        evaluated_users=evaluated_users,
        recall=total_recall / evaluated_users,
        hit_rate=total_hit_rate / evaluated_users,
        precision=total_precision / evaluated_users,
        ndcg=total_ndcg / evaluated_users,
        coverage=len(recommended_items) / len(index_to_item),
    )


def evaluate_model(
    model: DNNRanker,
    train_by_user: dict[int, set[int]],
    test_by_user: dict[int, set[int]],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    index_to_item: list[int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[RetrievalMetrics, RankingMetrics, float]:
    eval_started = time.perf_counter()
    eval_pairs, eval_labels = build_ranking_eval_pairs(
        test_by_user=test_by_user,
        train_by_user=train_by_user,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        eval_negative_samples=args.eval_negative_samples,
        seed=args.seed + 17,
    )
    eval_logits = score_pairs(
        model=model,
        pairs=eval_pairs,
        batch_size=args.score_batch_size,
        device=device,
    )
    ranking_metrics = RankingMetrics(
        auc=binary_auc(eval_labels, eval_logits),
        logloss=logloss_from_logits(eval_labels, eval_logits),
        eval_samples=len(eval_labels),
        eval_positive_samples=sum(eval_labels),
        eval_negative_samples=len(eval_labels) - sum(eval_labels),
    )
    retrieval_metrics = evaluate_topk(
        model=model,
        train_by_user=train_by_user,
        test_by_user=test_by_user,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_item=index_to_item,
        k=args.k,
        score_batch_size=args.score_batch_size,
        device=device,
    )
    return retrieval_metrics, ranking_metrics, time.perf_counter() - eval_started


def write_ranker_csv(
    path: Path,
    retrieval_metrics: RetrievalMetrics,
    ranking_metrics: RankingMetrics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "k",
        "evaluated_users",
        "recall",
        "hit_rate",
        "precision",
        "ndcg",
        "coverage",
        "auc",
        "logloss",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model": retrieval_metrics.model,
                "k": retrieval_metrics.k,
                "evaluated_users": retrieval_metrics.evaluated_users,
                "recall": f"{retrieval_metrics.recall:.8f}",
                "hit_rate": f"{retrieval_metrics.hit_rate:.8f}",
                "precision": f"{retrieval_metrics.precision:.8f}",
                "ndcg": f"{retrieval_metrics.ndcg:.8f}",
                "coverage": f"{retrieval_metrics.coverage:.8f}",
                "auc": f"{ranking_metrics.auc:.8f}",
                "logloss": f"{ranking_metrics.logloss:.8f}",
            }
        )


def write_experiment_results(outputs_dir: Path, result_paths: list[Path]) -> None:
    rows: list[dict[str, str]] = []
    fieldnames = [
        "model",
        "k",
        "evaluated_users",
        "recall",
        "hit_rate",
        "precision",
        "ndcg",
        "coverage",
        "auc",
        "logloss",
    ]

    for path in result_paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))

    output_path = outputs_dir / "experiment_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def format_float(value: float) -> str:
    return f"{value:.6f}"


def generate_report(
    path: Path,
    retrieval_metrics: RetrievalMetrics,
    ranking_metrics: RankingMetrics,
    stats: TrainingStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden_dims = " -> ".join(str(dim) for dim in stats.hidden_dims)
    lines = [
        "# DNN Ranker 排序模型报告",
        "",
        "## TopK 指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 模型 | {retrieval_metrics.model} |",
        f"| K | {retrieval_metrics.k} |",
        f"| 评估用户数 | {retrieval_metrics.evaluated_users:,} |",
        f"| Recall@{retrieval_metrics.k} | {format_float(retrieval_metrics.recall)} |",
        f"| HitRate@{retrieval_metrics.k} | {format_float(retrieval_metrics.hit_rate)} |",
        f"| Precision@{retrieval_metrics.k} | {format_float(retrieval_metrics.precision)} |",
        f"| NDCG@{retrieval_metrics.k} | {format_float(retrieval_metrics.ndcg)} |",
        f"| Coverage@{retrieval_metrics.k} | {format_float(retrieval_metrics.coverage)} |",
        "",
        "## 排序指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| AUC | {format_float(ranking_metrics.auc)} |",
        f"| LogLoss | {format_float(ranking_metrics.logloss)} |",
        f"| 排序评估样本数 | {ranking_metrics.eval_samples:,} |",
        f"| 排序评估正样本数 | {ranking_metrics.eval_positive_samples:,} |",
        f"| 排序评估负样本数 | {ranking_metrics.eval_negative_samples:,} |",
        "",
        "## 训练设置",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 设备 | {stats.device} |",
        f"| 用户数 | {stats.num_users:,} |",
        f"| 电影数 | {stats.num_items:,} |",
        f"| Genre 特征数 | {stats.num_genres} |",
        f"| Embedding Dim | {stats.embedding_dim} |",
        f"| MLP Hidden Dims | {hidden_dims} |",
        f"| Epochs | {stats.epochs} |",
        f"| Batch Size | {stats.batch_size:,} |",
        f"| 每个正样本训练负采样数 | {stats.negative_samples} |",
        f"| 每个测试用户排序评估负采样数 | {stats.eval_negative_samples} |",
        f"| Learning Rate | {stats.learning_rate} |",
        f"| Dropout | {stats.dropout} |",
        f"| Final Loss | {stats.final_loss:.6f} |",
        f"| 训练耗时（秒） | {stats.train_seconds:.2f} |",
        f"| 评估耗时（秒） | {stats.eval_seconds:.2f} |",
        "",
        "## 说明",
        "",
        "- DNN Ranker 是排序模型，不是召回模型；它学习用户-电影候选对的点击/喜欢概率。",
        "- 输入特征包括 `user_id embedding`、`item_id embedding`、二者逐元素乘积、用户活跃度、电影热度、电影类别数量和电影类别 multi-hot 特征。",
        "- 训练正样本来自训练集 `rating >= 4` 的交互，负样本从用户未出现过正反馈的电影中随机采样。",
        "- AUC 和 LogLoss 使用测试正样本加随机负样本评估。",
        "- TopK 指标仍然使用全量电影打分，过滤用户训练集中已看过电影，再取 TopK。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dims", type=parse_hidden_dims, default=(512, 256, 128))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--negative-samples", type=int, default=3)
    parser.add_argument("--eval-negative-samples", type=int, default=100)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mlu.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    train_path = args.processed_dir / "train.csv"
    test_path = args.processed_dir / "test.csv"
    metadata_path = args.processed_dir / "item_metadata.csv"

    for path in [train_path, test_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}. Run prepare_movielens.py first."
            )

    train = read_interactions(train_path)
    test = read_interactions(test_path)
    item_metadata = read_item_metadata(metadata_path)
    train_by_user = group_items_by_user(train)
    test_by_user = group_items_by_user(test)
    user_to_index, item_to_index, index_to_item = build_mappings(
        train=train,
        test=test,
        all_item_ids=set(item_metadata),
    )
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    model, final_loss, train_seconds, num_genres = train_model(
        train=train,
        item_metadata=item_metadata,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        args=args,
        device=device,
    )
    retrieval_metrics, ranking_metrics, eval_seconds = evaluate_model(
        model=model,
        train_by_user=train_by_user,
        test_by_user=test_by_user,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_item=index_to_item,
        args=args,
        device=device,
    )
    stats = TrainingStats(
        device=str(device),
        embedding_dim=args.embedding_dim,
        hidden_dims=args.hidden_dims,
        num_genres=num_genres,
        epochs=args.epochs,
        batch_size=args.batch_size,
        negative_samples=args.negative_samples,
        eval_negative_samples=args.eval_negative_samples,
        learning_rate=args.lr,
        dropout=args.dropout,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        final_loss=final_loss,
        num_users=len(user_to_index),
        num_items=len(index_to_item),
    )

    ranker_result_path = args.outputs_dir / "ranker_results.csv"
    write_ranker_csv(ranker_result_path, retrieval_metrics, ranking_metrics)
    write_experiment_results(
        outputs_dir=args.outputs_dir,
        result_paths=[
            args.outputs_dir / "popularity_results.csv",
            args.outputs_dir / "itemcf_results.csv",
            args.outputs_dir / "mf_results.csv",
            args.outputs_dir / "two_tower_results.csv",
            ranker_result_path,
        ],
    )
    generate_report(
        path=args.reports_dir / "ranker_report.md",
        retrieval_metrics=retrieval_metrics,
        ranking_metrics=ranking_metrics,
        stats=stats,
    )

    print(f"Recall@{args.k}: {retrieval_metrics.recall:.6f}")
    print(f"NDCG@{args.k}: {retrieval_metrics.ndcg:.6f}")
    print(f"Coverage@{args.k}: {retrieval_metrics.coverage:.6f}")
    print(f"AUC: {ranking_metrics.auc:.6f}")
    print(f"LogLoss: {ranking_metrics.logloss:.6f}")
    print(f"Wrote DNN Ranker report: {args.reports_dir / 'ranker_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
