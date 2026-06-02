"""Evaluate a global Popularity baseline on MIND news impressions."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = EXPERIMENT_DIR / "data" / "raw"
DEFAULT_PROCESSED_DIR = EXPERIMENT_DIR / "data" / "processed"
DEFAULT_REPORTS_DIR = EXPERIMENT_DIR / "reports"
DEFAULT_OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"


@dataclass(frozen=True)
class NewsInfo:
    category: str
    subcategory: str
    title: str


@dataclass(frozen=True)
class TrainStats:
    path: Path
    rows: int
    positive_rows: int
    negative_rows: int
    unique_news: int
    clicked_news: int
    ctr: float


@dataclass(frozen=True)
class RankingMetrics:
    model: str
    valid_path: Path
    evaluated_impressions: int
    auc_impressions: int
    valid_rows: int
    positive_rows: int
    negative_rows: int
    avg_candidates: float
    auc: float
    mrr: float
    ndcg5: float
    ndcg10: float
    hitrate5: float
    hitrate10: float
    coverage5: float
    coverage10: float


def resolve_default_train_path(processed_dir: Path, raw_dir: Path) -> Path:
    raw_path = raw_dir / "MIND_small_x1" / "train.csv"
    if raw_path.exists():
        return raw_path
    return processed_dir / "train_sample.csv"


def resolve_default_valid_path(processed_dir: Path, raw_dir: Path) -> Path:
    raw_path = raw_dir / "MIND_small_x1" / "valid.csv"
    if raw_path.exists():
        return raw_path
    return processed_dir / "valid_sample.csv"


def validate_reczoo_schema(path: Path, fieldnames: list[str] | None) -> None:
    required = {"imp_id", "click", "user_id", "news_id"}
    if not fieldnames or not required.issubset(set(fieldnames)):
        raise ValueError(f"Unexpected MIND interaction schema in {path}: {fieldnames}")


def read_news_metadata(path: Path) -> dict[str, NewsInfo]:
    if not path.exists():
        return {}

    news: dict[str, NewsInfo] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"news_id", "category", "subcategory", "title"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Unexpected news metadata schema in {path}: {reader.fieldnames}")
        for row in reader:
            news[row["news_id"]] = NewsInfo(
                category=row["category"],
                subcategory=row["subcategory"],
                title=row["title"],
            )
    return news


def build_popularity(train_path: Path) -> tuple[Counter[str], Counter[str], TrainStats]:
    click_counts: Counter[str] = Counter()
    exposure_counts: Counter[str] = Counter()
    rows = 0
    positive_rows = 0
    negative_rows = 0

    with train_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        validate_reczoo_schema(train_path, reader.fieldnames)
        for row in reader:
            rows += 1
            news_id = row["news_id"]
            click = row["click"]
            exposure_counts[news_id] += 1
            if click == "1":
                click_counts[news_id] += 1
                positive_rows += 1
            elif click == "0":
                negative_rows += 1

    labeled_rows = positive_rows + negative_rows
    stats = TrainStats(
        path=train_path,
        rows=rows,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        unique_news=len(exposure_counts),
        clicked_news=len(click_counts),
        ctr=positive_rows / labeled_rows if labeled_rows else 0.0,
    )
    return click_counts, exposure_counts, stats


def popularity_score(
    news_id: str,
    click_counts: Counter[str],
    exposure_counts: Counter[str],
    tie_breaker_weight: float,
) -> float:
    return float(click_counts[news_id]) + tie_breaker_weight * math.log1p(exposure_counts[news_id])


def auc_score(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(ranked):
        next_index = index + 1
        while next_index < len(ranked) and ranked[next_index][0] == ranked[index][0]:
            next_index += 1
        average_rank = (rank + rank + (next_index - index) - 1) / 2.0
        rank_sum += average_rank * sum(label for _, label in ranked[index:next_index])
        rank += next_index - index
        index = next_index

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def dcg_at_k(labels: list[int], k: int) -> float:
    return sum(label / math.log2(index + 2) for index, label in enumerate(labels[:k]))


def ndcg_at_k(labels: list[int], k: int) -> float:
    ideal = sorted(labels, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg_at_k(labels, k) / ideal_dcg


def mrr_score(labels: list[int]) -> float:
    for index, label in enumerate(labels, start=1):
        if label == 1:
            return 1.0 / index
    return 0.0


def evaluate_group(
    rows: list[dict[str, str]],
    click_counts: Counter[str],
    exposure_counts: Counter[str],
    tie_breaker_weight: float,
) -> tuple[list[int], list[str], float | None]:
    scored_rows = []
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        label = int(row["click"])
        score = popularity_score(
            news_id=row["news_id"],
            click_counts=click_counts,
            exposure_counts=exposure_counts,
            tie_breaker_weight=tie_breaker_weight,
        )
        labels.append(label)
        scores.append(score)
        scored_rows.append((score, row["news_id"], label))

    ranked_rows = sorted(scored_rows, key=lambda row: (-row[0], row[1]))
    ranked_labels = [label for _, _, label in ranked_rows]
    ranked_news_ids = [news_id for _, news_id, _ in ranked_rows]
    return ranked_labels, ranked_news_ids, auc_score(labels, scores)


def evaluate_popularity(
    valid_path: Path,
    click_counts: Counter[str],
    exposure_counts: Counter[str],
    total_news: int,
    tie_breaker_weight: float,
) -> RankingMetrics:
    evaluated_impressions = 0
    auc_impressions = 0
    valid_rows = 0
    positive_rows = 0
    negative_rows = 0
    auc_total = 0.0
    mrr_total = 0.0
    ndcg5_total = 0.0
    ndcg10_total = 0.0
    hitrate5_total = 0.0
    hitrate10_total = 0.0
    recommended5: set[str] = set()
    recommended10: set[str] = set()
    current_key: tuple[str, str] | None = None
    current_rows: list[dict[str, str]] = []

    def consume_group(rows: list[dict[str, str]]) -> None:
        nonlocal evaluated_impressions
        nonlocal auc_impressions
        nonlocal auc_total
        nonlocal mrr_total
        nonlocal ndcg5_total
        nonlocal ndcg10_total
        nonlocal hitrate5_total
        nonlocal hitrate10_total

        if not rows:
            return

        ranked_labels, ranked_news_ids, group_auc = evaluate_group(
            rows=rows,
            click_counts=click_counts,
            exposure_counts=exposure_counts,
            tie_breaker_weight=tie_breaker_weight,
        )
        evaluated_impressions += 1
        if group_auc is not None:
            auc_impressions += 1
            auc_total += group_auc
        mrr_total += mrr_score(ranked_labels)
        ndcg5_total += ndcg_at_k(ranked_labels, 5)
        ndcg10_total += ndcg_at_k(ranked_labels, 10)
        hitrate5_total += 1.0 if any(ranked_labels[:5]) else 0.0
        hitrate10_total += 1.0 if any(ranked_labels[:10]) else 0.0
        recommended5.update(ranked_news_ids[:5])
        recommended10.update(ranked_news_ids[:10])

    with valid_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        validate_reczoo_schema(valid_path, reader.fieldnames)
        for row in reader:
            valid_rows += 1
            if row["click"] == "1":
                positive_rows += 1
            elif row["click"] == "0":
                negative_rows += 1

            key = (row["imp_id"], row["user_id"])
            if current_key is not None and key != current_key:
                consume_group(current_rows)
                current_rows = []
            current_key = key
            current_rows.append(row)

    consume_group(current_rows)

    if evaluated_impressions == 0:
        raise ValueError("No validation impressions found. Run prepare_mind.py first.")

    return RankingMetrics(
        model="Popularity",
        valid_path=valid_path,
        evaluated_impressions=evaluated_impressions,
        auc_impressions=auc_impressions,
        valid_rows=valid_rows,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        avg_candidates=valid_rows / evaluated_impressions,
        auc=auc_total / auc_impressions if auc_impressions else 0.0,
        mrr=mrr_total / evaluated_impressions,
        ndcg5=ndcg5_total / evaluated_impressions,
        ndcg10=ndcg10_total / evaluated_impressions,
        hitrate5=hitrate5_total / evaluated_impressions,
        hitrate10=hitrate10_total / evaluated_impressions,
        coverage5=len(recommended5) / total_news if total_news else 0.0,
        coverage10=len(recommended10) / total_news if total_news else 0.0,
    )


def top_popular_news(
    click_counts: Counter[str],
    exposure_counts: Counter[str],
    top_n: int,
) -> list[tuple[str, int, int, float]]:
    rows = []
    for news_id, exposure_count in exposure_counts.items():
        click_count = click_counts[news_id]
        ctr = click_count / exposure_count if exposure_count else 0.0
        rows.append((news_id, click_count, exposure_count, ctr))
    return sorted(rows, key=lambda row: (-row[1], -row[2], row[0]))[:top_n]


def write_results_csv(path: Path, train_stats: TrainStats, metrics: RankingMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "train_rows",
                "train_positive_rows",
                "train_ctr",
                "valid_rows",
                "valid_positive_rows",
                "evaluated_impressions",
                "avg_candidates",
                "auc",
                "mrr",
                "ndcg5",
                "ndcg10",
                "hitrate5",
                "hitrate10",
                "coverage5",
                "coverage10",
            ]
        )
        writer.writerow(
            [
                metrics.model,
                train_stats.rows,
                train_stats.positive_rows,
                f"{train_stats.ctr:.8f}",
                metrics.valid_rows,
                metrics.positive_rows,
                metrics.evaluated_impressions,
                f"{metrics.avg_candidates:.8f}",
                f"{metrics.auc:.8f}",
                f"{metrics.mrr:.8f}",
                f"{metrics.ndcg5:.8f}",
                f"{metrics.ndcg10:.8f}",
                f"{metrics.hitrate5:.8f}",
                f"{metrics.hitrate10:.8f}",
                f"{metrics.coverage5:.8f}",
                f"{metrics.coverage10:.8f}",
            ]
        )


def write_experiment_results_csv(path: Path, train_stats: TrainStats, metrics: RankingMetrics) -> None:
    write_results_csv(path, train_stats, metrics)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(EXPERIMENT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def generate_report(
    path: Path,
    train_stats: TrainStats,
    metrics: RankingMetrics,
    popular_news: list[tuple[str, int, int, float]],
    news_metadata: dict[str, NewsInfo],
    tie_breaker_weight: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# MIND Popularity Baseline 报告",
        "",
        "## 实验目的",
        "",
        "这个实验用于建立 MIND 新闻推荐阶段的最低基线。模型不理解用户个性，也不理解新闻文本，",
        "只根据训练集中新闻被点击的次数判断新闻热度，然后在验证集每一次曝光的候选新闻列表内部排序。",
        "",
        "## 输入与输出",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 训练输入 | `{display_path(train_stats.path)}` |",
        f"| 验证输入 | `{display_path(metrics.valid_path)}` |",
        "| 训练信号 | `click=1` 的新闻点击次数 |",
        "| 排序对象 | 同一个 `imp_id + user_id` 下的候选新闻 |",
        "| 输出报告 | `reports/baseline_report.md` |",
        "| 输出结果表 | `outputs/popularity_results.csv`、`outputs/experiment_results.csv` |",
        "",
        "## 热度打分方式",
        "",
        "每篇新闻的主分数是训练集点击次数：",
        "",
        "$$",
        "score(news) = click\\_count(news)",
        "$$",
        "",
        "如果两篇新闻点击次数相同，脚本会用一个极小的曝光次数项打破并列：",
        "",
        "$$",
        "score(news) = click\\_count(news) + \\lambda \\cdot log(1 + exposure\\_count(news))",
        "$$",
        "",
        f"本次实验的 $\\lambda$ 为 `{tie_breaker_weight}`。这个项很小，主要用于减少完全相同分数导致的排序并列。",
        "",
        "## 数据统计",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 训练样本行数 | {train_stats.rows:,} |",
        f"| 训练正样本数 | {train_stats.positive_rows:,} |",
        f"| 训练负样本数 | {train_stats.negative_rows:,} |",
        f"| 训练 CTR | {format_percent(train_stats.ctr)} |",
        f"| 训练集中出现新闻数 | {train_stats.unique_news:,} |",
        f"| 训练集中被点击新闻数 | {train_stats.clicked_news:,} |",
        f"| 验证样本行数 | {metrics.valid_rows:,} |",
        f"| 验证正样本数 | {metrics.positive_rows:,} |",
        f"| 验证负样本数 | {metrics.negative_rows:,} |",
        f"| 评估曝光组数 | {metrics.evaluated_impressions:,} |",
        f"| 平均每组候选新闻数 | {metrics.avg_candidates:.2f} |",
        "",
        "## 评估指标",
        "",
        "| 指标 | 数值 | 小白解释 |",
        "|---|---:|---|",
        f"| AUC | {format_float(metrics.auc)} | 随机拿一条点击新闻和一条未点击新闻，热门模型把点击新闻排得更高的概率 |",
        f"| MRR | {format_float(metrics.mrr)} | 每个曝光列表里第一条被点击新闻出现得越靠前，数值越高 |",
        f"| NDCG@5 | {format_float(metrics.ndcg5)} | Top5 内命中点击新闻且位置越靠前，数值越高 |",
        f"| NDCG@10 | {format_float(metrics.ndcg10)} | Top10 内命中点击新闻且位置越靠前，数值越高 |",
        f"| HitRate@5 | {format_float(metrics.hitrate5)} | Top5 里至少有一条点击新闻的曝光组比例 |",
        f"| HitRate@10 | {format_float(metrics.hitrate10)} | Top10 里至少有一条点击新闻的曝光组比例 |",
        f"| Coverage@5 | {format_float(metrics.coverage5)} | 所有 Top5 推荐覆盖到的不同新闻占总新闻数比例 |",
        f"| Coverage@10 | {format_float(metrics.coverage10)} | 所有 Top10 推荐覆盖到的不同新闻占总新闻数比例 |",
        "",
        "## 训练集中最热门的新闻",
        "",
        "| 排名 | 新闻 ID | 点击数 | 曝光数 | CTR | Category | Subcategory | 标题 |",
        "|---:|---|---:|---:|---:|---|---|---|",
    ]

    for rank, (news_id, click_count, exposure_count, ctr) in enumerate(popular_news, start=1):
        info = news_metadata.get(news_id, NewsInfo(category="", subcategory="", title=""))
        lines.append(
            "| "
            f"{rank} | "
            f"{news_id} | "
            f"{click_count:,} | "
            f"{exposure_count:,} | "
            f"{format_percent(ctr)} | "
            f"{escape_markdown(info.category)} | "
            f"{escape_markdown(info.subcategory)} | "
            f"{escape_markdown(info.title)} |"
        )

    lines.extend(
        [
            "",
            "## 怎么理解这个结果",
            "",
            "- Popularity baseline 是非个性化模型：同一组候选新闻中，它只偏向训练集中更热门的新闻。",
            "- 这个模型不能利用 `user_id`、`news_his`、标题、摘要或类别，因此它只是后续模型的最低对照线。",
            "- 如果后续 Category baseline、DNN Ranker 或 Two-Tower 没有超过这个基线，说明个性化或内容特征没有真正发挥作用。",
            "- MIND 的验证方式和 MovieLens 不同：MovieLens 是从全量电影里推荐 TopK；MIND 当前是在一次真实曝光的候选列表内部排序。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing raw MIND files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing processed MIND files.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=None,
        help="Training CSV. Defaults to raw RecZoo train.csv, then train_sample.csv.",
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=None,
        help="Validation CSV. Defaults to raw RecZoo valid.csv, then valid_sample.csv.",
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
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of popular news rows to include in the report.",
    )
    parser.add_argument(
        "--tie-breaker-weight",
        type=float,
        default=1e-6,
        help="Tiny exposure-count weight used to break equal click-count scores.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    train_path = args.train_path or resolve_default_train_path(args.processed_dir, args.raw_dir)
    valid_path = args.valid_path or resolve_default_valid_path(args.processed_dir, args.raw_dir)
    news_metadata_path = args.processed_dir / "news_metadata.csv"

    if not train_path.exists():
        print(f"Missing train CSV: {train_path}", file=sys.stderr)
        return 1
    if not valid_path.exists():
        print(f"Missing validation CSV: {valid_path}", file=sys.stderr)
        return 1

    news_metadata = read_news_metadata(news_metadata_path)
    click_counts, exposure_counts, train_stats = build_popularity(train_path)
    total_news = len(news_metadata) if news_metadata else len(exposure_counts)
    metrics = evaluate_popularity(
        valid_path=valid_path,
        click_counts=click_counts,
        exposure_counts=exposure_counts,
        total_news=total_news,
        tie_breaker_weight=args.tie_breaker_weight,
    )
    popular_news = top_popular_news(
        click_counts=click_counts,
        exposure_counts=exposure_counts,
        top_n=args.top_n,
    )

    report_path = args.reports_dir / "baseline_report.md"
    popularity_results_path = args.outputs_dir / "popularity_results.csv"
    experiment_results_path = args.outputs_dir / "experiment_results.csv"
    generate_report(
        path=report_path,
        train_stats=train_stats,
        metrics=metrics,
        popular_news=popular_news,
        news_metadata=news_metadata,
        tie_breaker_weight=args.tie_breaker_weight,
    )
    write_results_csv(popularity_results_path, train_stats, metrics)
    write_experiment_results_csv(experiment_results_path, train_stats, metrics)

    print(f"Wrote report: {report_path}")
    print(f"Wrote results: {popularity_results_path}")
    print(f"Wrote experiment results: {experiment_results_path}")
    print(
        "Metrics: "
        f"AUC={metrics.auc:.6f}, "
        f"MRR={metrics.mrr:.6f}, "
        f"NDCG@5={metrics.ndcg5:.6f}, "
        f"NDCG@10={metrics.ndcg10:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
