"""Train and evaluate a Two-Tower recall model on MovieLens."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from collections import defaultdict
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
class Metrics:
    model: str
    k: int
    evaluated_users: int
    recall: float
    hit_rate: float
    precision: float
    ndcg: float
    coverage: float


@dataclass(frozen=True)
class TrainingStats:
    training_objective: str
    device: str
    embedding_dim: int
    tower_dim: int
    hidden_dim: int
    max_history_items: int
    epochs: int
    batch_size: int
    negative_samples: int
    learning_rate: float
    temperature: float
    train_seconds: float
    eval_seconds: float
    final_loss: float
    num_users: int
    num_items: int


class TwoTowerRecall(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: int,
        tower_dim: int,
        user_history_items: torch.Tensor,
        user_history_lengths: torch.Tensor,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.history_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tower_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tower_dim),
        )
        self.register_buffer("user_history_items", user_history_items)
        self.register_buffer("user_history_lengths", user_history_lengths)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.history_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        with torch.no_grad():
            self.history_embedding.weight[0].zero_()

    def encode_users(self, user_indices: torch.Tensor) -> torch.Tensor:
        history_items = self.user_history_items[user_indices]
        history_lengths = self.user_history_lengths[user_indices].clamp_min(1).unsqueeze(1)
        history_vectors = self.history_embedding(history_items).sum(dim=1) / history_lengths
        user_vectors = self.user_embedding(user_indices)
        tower_input = torch.cat([user_vectors, history_vectors], dim=1)
        return F.normalize(self.user_tower(tower_input), dim=1)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        item_vectors = self.item_embedding(item_indices)
        return F.normalize(self.item_tower(item_vectors), dim=1)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_vectors = self.encode_users(user_indices)
        item_vectors = self.encode_items(item_indices)
        return (user_vectors * item_vectors).sum(dim=1) / self.temperature


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


def build_user_history_tensors(
    train: list[Interaction],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    max_history_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    history_by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in train:
        if row.user_id in user_to_index and row.item_id in item_to_index:
            user_index = user_to_index[row.user_id]
            item_index = item_to_index[row.item_id] + 1
            history_by_user[user_index].append((row.timestamp, item_index))

    num_users = len(user_to_index)
    history_items = torch.zeros((num_users, max_history_items), dtype=torch.long)
    history_lengths = torch.zeros(num_users, dtype=torch.float32)

    for user_index, rows in history_by_user.items():
        rows.sort(key=lambda row: row[0])
        item_indices = [item_index for _, item_index in rows[-max_history_items:]]
        history_lengths[user_index] = len(item_indices)
        if item_indices:
            history_items[user_index, : len(item_indices)] = torch.tensor(
                item_indices,
                dtype=torch.long,
            )

    return history_items, history_lengths


def sample_negative_item(
    rng: random.Random,
    item_indices: list[int],
    user_positive_items: set[int],
) -> int:
    while True:
        candidate = rng.choice(item_indices)
        if candidate not in user_positive_items:
            return candidate


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
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TwoTowerRecall, float, float]:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    train_pairs = [
        (user_to_index[row.user_id], item_to_index[row.item_id])
        for row in train
        if row.user_id in user_to_index and row.item_id in item_to_index
    ]
    user_positive_indices: dict[int, set[int]] = defaultdict(set)
    for user_index, item_index in train_pairs:
        user_positive_indices[user_index].add(item_index)

    history_items, history_lengths = build_user_history_tensors(
        train=train,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        max_history_items=args.max_history_items,
    )
    item_indices = list(range(len(item_to_index)))
    model = TwoTowerRecall(
        num_users=len(user_to_index),
        num_items=len(item_to_index),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        tower_dim=args.tower_dim,
        user_history_items=history_items,
        user_history_lengths=history_lengths,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_started = time.perf_counter()
    final_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_pairs)
        epoch_loss = 0.0
        batch_count = 0

        model.train()
        for start in range(0, len(train_pairs), args.batch_size):
            batch_pairs = train_pairs[start : start + args.batch_size]

            optimizer.zero_grad(set_to_none=True)
            if args.loss == "in-batch":
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
                user_vectors = model.encode_users(user_tensor)
                item_vectors = model.encode_items(item_tensor)
                logits = (user_vectors @ item_vectors.T) / args.temperature
                labels = torch.arange(len(batch_pairs), dtype=torch.long, device=device)
                loss = (
                    F.cross_entropy(logits, labels)
                    + F.cross_entropy(logits.T, labels)
                ) / 2
            else:
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
                logits = model(user_tensor, item_tensor)
                loss = F.binary_cross_entropy_with_logits(logits, label_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            batch_count += 1

        final_loss = epoch_loss / batch_count if batch_count else 0.0
        print(f"epoch={epoch} loss={final_loss:.6f}", flush=True)

    train_seconds = time.perf_counter() - train_started
    return model, final_loss, train_seconds


def dcg_for_hits(recommendations: list[int], relevant_items: set[int]) -> float:
    dcg = 0.0
    for index, item_id in enumerate(recommendations, start=1):
        if item_id in relevant_items:
            dcg += 1.0 / math.log2(index + 1)
    return dcg


def ideal_dcg(num_relevant: int, k: int) -> float:
    return sum(1.0 / math.log2(index + 1) for index in range(1, min(num_relevant, k) + 1))


def evaluate_model(
    model: TwoTowerRecall,
    train_by_user: dict[int, set[int]],
    test_by_user: dict[int, set[int]],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    index_to_item: list[int],
    k: int,
    device: torch.device,
) -> tuple[Metrics, float]:
    eval_started = time.perf_counter()
    model.eval()

    total_recall = 0.0
    total_hit_rate = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    recommended_items: set[int] = set()

    all_item_indices = torch.arange(len(index_to_item), dtype=torch.long, device=device)

    with torch.no_grad():
        item_vectors = model.encode_items(all_item_indices)
        for user_id in sorted(test_by_user):
            if user_id not in user_to_index:
                continue

            user_index = torch.tensor([user_to_index[user_id]], dtype=torch.long, device=device)
            user_vector = model.encode_users(user_index)
            scores = (user_vector @ item_vectors.T).squeeze(0).detach().cpu()

            for seen_item in train_by_user.get(user_id, set()):
                seen_index = item_to_index.get(seen_item)
                if seen_index is not None:
                    scores[seen_index] = -float("inf")

            top_indices = torch.topk(scores, k=min(k, len(scores))).indices.tolist()
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

    metrics = Metrics(
        model="Two-Tower",
        k=k,
        evaluated_users=evaluated_users,
        recall=total_recall / evaluated_users,
        hit_rate=total_hit_rate / evaluated_users,
        precision=total_precision / evaluated_users,
        ndcg=total_ndcg / evaluated_users,
        coverage=len(recommended_items) / len(index_to_item),
    )
    return metrics, time.perf_counter() - eval_started


def write_metrics_csv(path: Path, metrics: Metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "k",
                "evaluated_users",
                "recall",
                "hit_rate",
                "precision",
                "ndcg",
                "coverage",
            ]
        )
        writer.writerow(
            [
                metrics.model,
                metrics.k,
                metrics.evaluated_users,
                f"{metrics.recall:.8f}",
                f"{metrics.hit_rate:.8f}",
                f"{metrics.precision:.8f}",
                f"{metrics.ndcg:.8f}",
                f"{metrics.coverage:.8f}",
            ]
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
            writer.writerow({field: row[field] for field in fieldnames})


def format_float(value: float) -> str:
    return f"{value:.6f}"


def generate_report(path: Path, metrics: Metrics, stats: TrainingStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Two-Tower 召回模型报告",
        "",
        "## 指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 模型 | {metrics.model} |",
        f"| K | {metrics.k} |",
        f"| 评估用户数 | {metrics.evaluated_users:,} |",
        f"| Recall@{metrics.k} | {format_float(metrics.recall)} |",
        f"| HitRate@{metrics.k} | {format_float(metrics.hit_rate)} |",
        f"| Precision@{metrics.k} | {format_float(metrics.precision)} |",
        f"| NDCG@{metrics.k} | {format_float(metrics.ndcg)} |",
        f"| Coverage@{metrics.k} | {format_float(metrics.coverage)} |",
        "",
        "## 训练设置",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 训练目标 | {stats.training_objective} |",
        f"| 设备 | {stats.device} |",
        f"| 用户数 | {stats.num_users:,} |",
        f"| 电影数 | {stats.num_items:,} |",
        f"| Embedding Dim | {stats.embedding_dim} |",
        f"| Tower Dim | {stats.tower_dim} |",
        f"| Hidden Dim | {stats.hidden_dim} |",
        f"| 最大历史电影数 | {stats.max_history_items} |",
        f"| Epochs | {stats.epochs} |",
        f"| Batch Size | {stats.batch_size:,} |",
        f"| 每个正样本负采样数 | {stats.negative_samples} |",
        f"| Learning Rate | {stats.learning_rate} |",
        f"| Temperature | {stats.temperature} |",
        f"| Final Loss | {stats.final_loss:.6f} |",
        f"| 训练耗时（秒） | {stats.train_seconds:.2f} |",
        f"| 评估耗时（秒） | {stats.eval_seconds:.2f} |",
        "",
        "## 说明",
        "",
        "- 用户塔输入为 `user_id embedding` 与用户训练集历史正反馈电影 embedding 均值的拼接。",
        "- 物品塔输入为 `item_id embedding`。",
        "- 两个塔分别经过 MLP 后做 L2 归一化，使用向量点积作为召回分数。",
        "- 当前正式配置使用随机负采样 BCE 训练。",
        "- 也可以通过 `--loss in-batch` 使用 batch 内其他正样本物品作为隐式负样本。",
        "- 正样本来自训练集 `rating >= 4` 的交互。",
        "- 评估时会计算用户向量与全量电影向量的相似度，过滤训练集中已看过电影，再取 TopK。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--tower-dim", type=int, default=128)
    parser.add_argument("--max-history-items", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--negative-samples", type=int, default=1)
    parser.add_argument("--loss", choices=["bce", "in-batch"], default="bce")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.1)
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

    model, final_loss, train_seconds = train_model(
        train=train,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        args=args,
        device=device,
    )
    metrics, eval_seconds = evaluate_model(
        model=model,
        train_by_user=train_by_user,
        test_by_user=test_by_user,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_item=index_to_item,
        k=args.k,
        device=device,
    )
    stats = TrainingStats(
        training_objective=args.loss,
        device=str(device),
        embedding_dim=args.embedding_dim,
        tower_dim=args.tower_dim,
        hidden_dim=args.hidden_dim,
        max_history_items=args.max_history_items,
        epochs=args.epochs,
        batch_size=args.batch_size,
        negative_samples=args.negative_samples,
        learning_rate=args.lr,
        temperature=args.temperature,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        final_loss=final_loss,
        num_users=len(user_to_index),
        num_items=len(index_to_item),
    )

    two_tower_result_path = args.outputs_dir / "two_tower_results.csv"
    write_metrics_csv(two_tower_result_path, metrics)
    write_experiment_results(
        outputs_dir=args.outputs_dir,
        result_paths=[
            args.outputs_dir / "popularity_results.csv",
            args.outputs_dir / "itemcf_results.csv",
            args.outputs_dir / "mf_results.csv",
            two_tower_result_path,
        ],
    )
    generate_report(
        path=args.reports_dir / "two_tower_report.md",
        metrics=metrics,
        stats=stats,
    )

    print(f"Recall@{args.k}: {metrics.recall:.6f}")
    print(f"HitRate@{args.k}: {metrics.hit_rate:.6f}")
    print(f"NDCG@{args.k}: {metrics.ndcg:.6f}")
    print(f"Coverage@{args.k}: {metrics.coverage:.6f}")
    print(f"Wrote Two-Tower report: {args.reports_dir / 'two_tower_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
