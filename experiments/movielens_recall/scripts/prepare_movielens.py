"""Prepare MovieLens 1M implicit-feedback splits and dataset report."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from download_movielens import DEFAULT_RAW_DIR, download_and_extract


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = EXPERIMENT_DIR / "data" / "processed"
DEFAULT_REPORTS_DIR = EXPERIMENT_DIR / "reports"


@dataclass(frozen=True)
class Rating:
    user_id: int
    item_id: int
    rating: float
    timestamp: int


@dataclass(frozen=True)
class Movie:
    item_id: int
    title: str
    genres: str


def read_ratings(path: Path) -> list[Rating]:
    ratings: list[Rating] = []
    with path.open("r", encoding="latin-1") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != 4:
                raise ValueError(f"Invalid ratings.dat line {line_number}: {line}")
            user_id, item_id, rating, timestamp = parts
            ratings.append(
                Rating(
                    user_id=int(user_id),
                    item_id=int(item_id),
                    rating=float(rating),
                    timestamp=int(timestamp),
                )
            )
    return ratings


def read_movies(path: Path) -> dict[int, Movie]:
    movies: dict[int, Movie] = {}
    with path.open("r", encoding="latin-1") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != 3:
                raise ValueError(f"Invalid movies.dat line {line_number}: {line}")
            item_id, title, genres = parts
            item_id_int = int(item_id)
            movies[item_id_int] = Movie(
                item_id=item_id_int,
                title=title,
                genres=genres,
            )
    return movies


def split_positive_interactions(
    ratings: list[Rating],
    min_rating: float,
) -> tuple[list[Rating], list[Rating], list[Rating]]:
    positives_by_user: dict[int, list[Rating]] = defaultdict(list)
    for rating in ratings:
        if rating.rating >= min_rating:
            positives_by_user[rating.user_id].append(rating)

    train: list[Rating] = []
    valid: list[Rating] = []
    test: list[Rating] = []

    for user_id in sorted(positives_by_user):
        user_ratings = sorted(
            positives_by_user[user_id],
            key=lambda row: (row.timestamp, row.item_id),
        )

        if len(user_ratings) >= 3:
            train.extend(user_ratings[:-2])
            valid.append(user_ratings[-2])
            test.append(user_ratings[-1])
        elif len(user_ratings) == 2:
            train.append(user_ratings[0])
            test.append(user_ratings[1])
        else:
            train.extend(user_ratings)

    return train, valid, test


def write_ratings_csv(path: Path, rows: list[Rating]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "item_id", "rating", "timestamp"])
        for row in rows:
            writer.writerow([row.user_id, row.item_id, row.rating, row.timestamp])


def write_movies_csv(path: Path, movies: dict[int, Movie]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["item_id", "title", "genres"])
        for item_id in sorted(movies):
            movie = movies[item_id]
            writer.writerow([movie.item_id, movie.title, movie.genres])


def format_percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def safe_mean(values: list[int]) -> float:
    return statistics.mean(values) if values else 0.0


def generate_report(
    path: Path,
    ratings: list[Rating],
    movies: dict[int, Movie],
    train: list[Rating],
    valid: list[Rating],
    test: list[Rating],
    min_rating: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rating_users = {row.user_id for row in ratings}
    rating_items = {row.item_id for row in ratings}
    positive_rows = [row for row in ratings if row.rating >= min_rating]
    positive_users = {row.user_id for row in positive_rows}
    positive_items = {row.item_id for row in positive_rows}
    rating_distribution = Counter(int(row.rating) for row in ratings)
    train_users = {row.user_id for row in train}
    valid_users = {row.user_id for row in valid}
    test_users = {row.user_id for row in test}

    ratings_per_user = Counter(row.user_id for row in ratings)
    ratings_per_item = Counter(row.item_id for row in ratings)
    positives_per_user = Counter(row.user_id for row in positive_rows)
    positives_per_item = Counter(row.item_id for row in positive_rows)

    all_density = len(ratings) / (len(rating_users) * len(movies))
    positive_density = len(positive_rows) / (len(positive_users) * len(movies))

    lines = [
        "# MovieLens 1M 数据报告",
        "",
        "## 数据集概览",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 原始评分数 | {len(ratings):,} |",
        f"| 有评分行为的用户数 | {len(rating_users):,} |",
        f"| 元数据中的电影数 | {len(movies):,} |",
        f"| 有评分的电影数 | {len(rating_items):,} |",
        f"| 评分矩阵密度 | {format_percent(all_density)} |",
        f"| 评分矩阵稀疏度 | {format_percent(1 - all_density)} |",
        "",
        "## 正反馈定义",
        "",
        f"本实验将 `rating >= {min_rating:g}` 定义为正反馈。",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 正反馈交互数 | {len(positive_rows):,} |",
        f"| 有正反馈的用户数 | {len(positive_users):,} |",
        f"| 有正反馈的电影数 | {len(positive_items):,} |",
        f"| 正反馈矩阵密度 | {format_percent(positive_density)} |",
        f"| 正反馈矩阵稀疏度 | {format_percent(1 - positive_density)} |",
        "",
        "## 交互分布",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 用户平均评分数 | {safe_mean(list(ratings_per_user.values())):.2f} |",
        f"| 有评分电影的平均评分数 | {safe_mean(list(ratings_per_item.values())):.2f} |",
        f"| 正反馈用户平均正反馈数 | {safe_mean(list(positives_per_user.values())):.2f} |",
        f"| 正反馈电影平均正反馈数 | {safe_mean(list(positives_per_item.values())):.2f} |",
        "",
        "## 评分分布",
        "",
        "| 评分 | 数量 | 占比 |",
        "|---:|---:|---:|",
    ]

    for rating_value in sorted(rating_distribution):
        count = rating_distribution[rating_value]
        lines.append(
            f"| {rating_value} | {count:,} | {format_percent(count / len(ratings))} |"
        )

    lines.extend(
        [
            "",
            "## 按时间顺序切分",
            "",
            "| 数据集 | 交互数 | 用户数 |",
            "|---|---:|---:|",
            f"| 训练集 | {len(train):,} | {len(train_users):,} |",
            f"| 验证集 | {len(valid):,} | {len(valid_users):,} |",
            f"| 测试集 | {len(test):,} | {len(test_users):,} |",
            "",
            "## 说明",
            "",
            "- 数据切分只使用正反馈交互。",
            "- 每个用户的交互序列按时间戳排序。",
            "- 测试集用户数是 baseline 报告中 `Recall@K` 的评估用户分母。",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing the extracted ml-1m dataset.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory for generated CSV files.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Directory for Markdown reports.",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=4.0,
        help="Minimum rating treated as positive feedback.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract MovieLens 1M if the raw files are missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    dataset_dir = args.raw_dir / "ml-1m"

    if args.download and not dataset_dir.exists():
        download_and_extract(raw_dir=args.raw_dir)

    ratings_path = dataset_dir / "ratings.dat"
    movies_path = dataset_dir / "movies.dat"
    if not ratings_path.exists() or not movies_path.exists():
        raise FileNotFoundError(
            "MovieLens 1M files are missing. Run download_movielens.py first "
            "or pass --download."
        )

    ratings = read_ratings(ratings_path)
    movies = read_movies(movies_path)
    train, valid, test = split_positive_interactions(
        ratings=ratings,
        min_rating=args.min_rating,
    )

    write_ratings_csv(args.processed_dir / "train.csv", train)
    write_ratings_csv(args.processed_dir / "valid.csv", valid)
    write_ratings_csv(args.processed_dir / "test.csv", test)
    write_movies_csv(args.processed_dir / "item_metadata.csv", movies)
    generate_report(
        path=args.reports_dir / "data_report.md",
        ratings=ratings,
        movies=movies,
        train=train,
        valid=valid,
        test=test,
        min_rating=args.min_rating,
    )

    print(f"Wrote processed data: {args.processed_dir}")
    print(f"Wrote data report: {args.reports_dir / 'data_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
