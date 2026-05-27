"""Evaluate a global Popularity baseline on MovieLens implicit splits."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


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


def popularity_rank(train: list[Interaction]) -> list[tuple[int, int]]:
    item_counts = Counter(row.item_id for row in train)
    return sorted(item_counts.items(), key=lambda item: (-item[1], item[0]))


def recommend_for_user(
    ranked_items: list[tuple[int, int]],
    seen_items: set[int],
    k: int,
) -> list[int]:
    recommendations: list[int] = []
    for item_id, _ in ranked_items:
        if item_id in seen_items:
            continue
        recommendations.append(item_id)
        if len(recommendations) == k:
            break
    return recommendations


def dcg_for_hits(recommendations: list[int], relevant_items: set[int]) -> float:
    dcg = 0.0
    for index, item_id in enumerate(recommendations, start=1):
        if item_id in relevant_items:
            dcg += 1.0 / math.log2(index + 1)
    return dcg


def ideal_dcg(num_relevant: int, k: int) -> float:
    return sum(1.0 / math.log2(index + 1) for index in range(1, min(num_relevant, k) + 1))


def evaluate_popularity(
    train: list[Interaction],
    test: list[Interaction],
    all_item_ids: set[int],
    k: int,
) -> tuple[Metrics, list[tuple[int, int]], dict[int, list[int]]]:
    ranked_items = popularity_rank(train)
    train_by_user = group_items_by_user(train)
    test_by_user = group_items_by_user(test)

    total_recall = 0.0
    total_hit_rate = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    recommended_items: set[int] = set()
    recommendations_by_user: dict[int, list[int]] = {}

    for user_id in sorted(test_by_user):
        relevant_items = test_by_user[user_id]
        recommendations = recommend_for_user(
            ranked_items=ranked_items,
            seen_items=train_by_user.get(user_id, set()),
            k=k,
        )
        recommendations_by_user[user_id] = recommendations
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
        model="Popularity",
        k=k,
        evaluated_users=evaluated_users,
        recall=total_recall / evaluated_users,
        hit_rate=total_hit_rate / evaluated_users,
        precision=total_precision / evaluated_users,
        ndcg=total_ndcg / evaluated_users,
        coverage=len(recommended_items) / len(all_item_ids),
    )
    return metrics, ranked_items, recommendations_by_user


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


def format_float(value: float) -> str:
    return f"{value:.6f}"


def generate_report(
    path: Path,
    metrics: Metrics,
    ranked_items: list[tuple[int, int]],
    item_metadata: dict[int, tuple[str, str]],
    top_n: int = 20,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Popularity Baseline 报告",
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
        "## 训练集中最热门的电影",
        "",
        "| 排名 | 电影 ID | 训练集正反馈数 | 标题 | 类别 |",
        "|---:|---:|---:|---|---|",
    ]

    for rank, (item_id, count) in enumerate(ranked_items[:top_n], start=1):
        title, genres = item_metadata.get(item_id, ("", ""))
        safe_title = title.replace("|", "\\|")
        safe_genres = genres.replace("|", ", ")
        lines.append(f"| {rank} | {item_id} | {count:,} | {safe_title} | {safe_genres} |")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- Popularity baseline 只使用训练集正反馈统计物品热度。",
            "- 评估前会过滤用户在训练集中已经看过的物品。",
            "- 当前 leave-last 切分方式下，每个测试用户通常只有 1 个相关物品。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing train.csv, test.csv, and item_metadata.csv.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory for Markdown reports.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=DEFAULT_OUTPUTS_DIR,
        help="Directory for machine-readable outputs.",
    )
    parser.add_argument("--k", type=int, default=20, help="Recommendation cutoff.")
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
    metrics, ranked_items, _ = evaluate_popularity(
        train=train,
        test=test,
        all_item_ids=set(item_metadata),
        k=args.k,
    )

    write_metrics_csv(args.outputs_dir / "popularity_results.csv", metrics)
    generate_report(
        path=args.reports_dir / "baseline_report.md",
        metrics=metrics,
        ranked_items=ranked_items,
        item_metadata=item_metadata,
    )

    print(f"Recall@{args.k}: {metrics.recall:.6f}")
    print(f"HitRate@{args.k}: {metrics.hit_rate:.6f}")
    print(f"NDCG@{args.k}: {metrics.ndcg:.6f}")
    print(f"Wrote baseline report: {args.reports_dir / 'baseline_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
