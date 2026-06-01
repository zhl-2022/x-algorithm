"""Evaluate an ItemCF baseline on MovieLens implicit-feedback splits."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
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


@dataclass(frozen=True)
class RunStats:
    train_users: int
    train_items: int
    cooccurrence_pairs: int
    top_similar_per_item: int
    build_seconds: float
    eval_seconds: float


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


def build_itemcf_similarity(
    train_by_user: dict[int, set[int]],
    top_similar_per_item: int,
) -> tuple[dict[int, list[tuple[int, float]]], Counter[int], int]:
    item_counts: Counter[int] = Counter()
    co_counts: dict[int, Counter[int]] = defaultdict(Counter)

    for user_items in train_by_user.values():
        sorted_items = sorted(user_items)
        item_counts.update(sorted_items)
        for index, item_i in enumerate(sorted_items):
            for item_j in sorted_items[index + 1 :]:
                co_counts[item_i][item_j] += 1
                co_counts[item_j][item_i] += 1

    similarities: dict[int, list[tuple[int, float]]] = {}
    for item_i, neighbors in co_counts.items():
        scored_neighbors: list[tuple[int, float]] = []
        count_i = item_counts[item_i]
        for item_j, co_count in neighbors.items():
            denominator = math.sqrt(count_i * item_counts[item_j])
            score = co_count / denominator if denominator else 0.0
            scored_neighbors.append((item_j, score))

        scored_neighbors.sort(key=lambda row: (-row[1], row[0]))
        similarities[item_i] = scored_neighbors[:top_similar_per_item]

    cooccurrence_pairs = sum(len(neighbors) for neighbors in co_counts.values()) // 2
    return similarities, item_counts, cooccurrence_pairs


def recommend_for_user(
    history_items: set[int],
    similarities: dict[int, list[tuple[int, float]]],
    item_counts: Counter[int],
    k: int,
) -> list[int]:
    candidate_scores: dict[int, float] = defaultdict(float)
    for history_item in history_items:
        for candidate_item, similarity in similarities.get(history_item, []):
            if candidate_item in history_items:
                continue
            candidate_scores[candidate_item] += similarity

    ranked_candidates = sorted(
        candidate_scores.items(),
        key=lambda row: (-row[1], -item_counts[row[0]], row[0]),
    )
    return [item_id for item_id, _ in ranked_candidates[:k]]


def dcg_for_hits(recommendations: list[int], relevant_items: set[int]) -> float:
    dcg = 0.0
    for index, item_id in enumerate(recommendations, start=1):
        if item_id in relevant_items:
            dcg += 1.0 / math.log2(index + 1)
    return dcg


def ideal_dcg(num_relevant: int, k: int) -> float:
    return sum(1.0 / math.log2(index + 1) for index in range(1, min(num_relevant, k) + 1))


def evaluate_itemcf(
    train: list[Interaction],
    test: list[Interaction],
    all_item_ids: set[int],
    k: int,
    top_similar_per_item: int,
) -> tuple[Metrics, RunStats, dict[int, list[tuple[int, float]]]]:
    train_by_user = group_items_by_user(train)
    test_by_user = group_items_by_user(test)

    build_started = time.perf_counter()
    similarities, item_counts, cooccurrence_pairs = build_itemcf_similarity(
        train_by_user=train_by_user,
        top_similar_per_item=top_similar_per_item,
    )
    build_seconds = time.perf_counter() - build_started

    eval_started = time.perf_counter()
    total_recall = 0.0
    total_hit_rate = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    recommended_items: set[int] = set()

    for user_id in sorted(test_by_user):
        relevant_items = test_by_user[user_id]
        recommendations = recommend_for_user(
            history_items=train_by_user.get(user_id, set()),
            similarities=similarities,
            item_counts=item_counts,
            k=k,
        )
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
        model="ItemCF",
        k=k,
        evaluated_users=evaluated_users,
        recall=total_recall / evaluated_users,
        hit_rate=total_hit_rate / evaluated_users,
        precision=total_precision / evaluated_users,
        ndcg=total_ndcg / evaluated_users,
        coverage=len(recommended_items) / len(all_item_ids),
    )
    stats = RunStats(
        train_users=len(train_by_user),
        train_items=len(item_counts),
        cooccurrence_pairs=cooccurrence_pairs,
        top_similar_per_item=top_similar_per_item,
        build_seconds=build_seconds,
        eval_seconds=time.perf_counter() - eval_started,
    )
    return metrics, stats, similarities


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
            reader = csv.DictReader(file)
            rows.extend(reader)

    output_path = outputs_dir / "experiment_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def format_float(value: float) -> str:
    return f"{value:.6f}"


def generate_report(
    path: Path,
    metrics: Metrics,
    stats: RunStats,
    similarities: dict[int, list[tuple[int, float]]],
    item_metadata: dict[int, tuple[str, str]],
    top_n: int = 20,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# ItemCF Baseline 报告",
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
        "## ItemCF 设置",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 训练用户数 | {stats.train_users:,} |",
        f"| 训练物品数 | {stats.train_items:,} |",
        f"| 共现物品对数 | {stats.cooccurrence_pairs:,} |",
        f"| 每个物品保留相似物品数 | {stats.top_similar_per_item:,} |",
        f"| 相似度构建耗时（秒） | {stats.build_seconds:.2f} |",
        f"| 评估耗时（秒） | {stats.eval_seconds:.2f} |",
        "",
        "## 示例相似物品",
        "",
        "| 源电影 ID | 源电影 | 相似电影 ID | 相似电影 | 相似度 |",
        "|---:|---|---:|---|---:|",
    ]

    for source_item in sorted(similarities)[:top_n]:
        source_title = item_metadata.get(source_item, ("", ""))[0].replace("|", "\\|")
        if not similarities[source_item]:
            continue
        target_item, score = similarities[source_item][0]
        target_title = item_metadata.get(target_item, ("", ""))[0].replace("|", "\\|")
        lines.append(
            f"| {source_item} | {source_title} | {target_item} | {target_title} | {score:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- ItemCF 使用训练集正反馈构建物品共现相似度。",
            "- 相似度公式为 `co_count(i, j) / sqrt(count(i) * count(j))`。",
            "- 对每个用户，使用其训练集历史电影的相似电影累加得分并推荐 TopK。",
            "- 评估前会过滤用户在训练集中已经看过的电影。",
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
    parser.add_argument(
        "--top-similar-per-item",
        type=int,
        default=100,
        help="Number of similar items retained for each item.",
    )
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
    metrics, stats, similarities = evaluate_itemcf(
        train=train,
        test=test,
        all_item_ids=set(item_metadata),
        k=args.k,
        top_similar_per_item=args.top_similar_per_item,
    )

    itemcf_result_path = args.outputs_dir / "itemcf_results.csv"
    write_metrics_csv(itemcf_result_path, metrics)
    write_experiment_results(
        outputs_dir=args.outputs_dir,
        result_paths=[args.outputs_dir / "popularity_results.csv", itemcf_result_path],
    )
    generate_report(
        path=args.reports_dir / "itemcf_report.md",
        metrics=metrics,
        stats=stats,
        similarities=similarities,
        item_metadata=item_metadata,
    )

    print(f"Recall@{args.k}: {metrics.recall:.6f}")
    print(f"HitRate@{args.k}: {metrics.hit_rate:.6f}")
    print(f"NDCG@{args.k}: {metrics.ndcg:.6f}")
    print(f"Coverage@{args.k}: {metrics.coverage:.6f}")
    print(f"Wrote ItemCF report: {args.reports_dir / 'itemcf_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
