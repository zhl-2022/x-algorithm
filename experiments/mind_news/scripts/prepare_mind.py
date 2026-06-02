"""Prepare MIND-small processed CSVs and a dataset report."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from download_mind import DEFAULT_RAW_DIR, download_and_extract


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = EXPERIMENT_DIR / "data" / "processed"
DEFAULT_REPORTS_DIR = EXPERIMENT_DIR / "reports"


@dataclass(frozen=True)
class News:
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    title_entities: str
    abstract_entities: str


@dataclass(frozen=True)
class Impression:
    split: str
    impression_id: str
    user_id: str
    time: str
    history: str
    news_id: str
    label: int | None


@dataclass(frozen=True)
class BehaviorSummary:
    split: str
    behavior_rows: int
    users: int
    users_with_history: int
    avg_history_length: float
    max_history_length: int
    impressions: int
    positive_impressions: int
    negative_impressions: int
    ctr: float


@dataclass(frozen=True)
class ReczooSummary:
    split: str
    rows: int
    impression_ids: int
    users: int
    users_with_history: int
    avg_history_length: float
    max_history_length: int
    positive_rows: int
    negative_rows: int
    ctr: float


def read_news(path: Path) -> dict[str, News]:
    news: dict[str, News] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 8:
                raise ValueError(f"Invalid news.tsv line {line_number}: expected 8 columns")
            row = News(*parts)
            news[row.news_id] = row
    return news


def read_reczoo_news(path: Path) -> dict[str, News]:
    news: dict[str, News] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        required = {
            "news_id",
            "cat",
            "sub_cat",
            "title_entities",
            "abstract_entities",
            "title",
            "abstract",
        }
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Unexpected RecZoo news schema in {path}: {reader.fieldnames}")
        for row in reader:
            news[row["news_id"]] = News(
                news_id=row["news_id"],
                category=row["cat"],
                subcategory=row["sub_cat"],
                title=row["title"],
                abstract=row["abstract"],
                url="",
                title_entities=row["title_entities"],
                abstract_entities=row["abstract_entities"],
            )
    return news


def parse_impression_token(token: str) -> tuple[str, int | None]:
    if "-" not in token:
        return token, None
    news_id, label = token.rsplit("-", 1)
    return news_id, int(label)


def read_impressions(path: Path, split: str) -> tuple[list[Impression], BehaviorSummary]:
    impressions: list[Impression] = []
    users: set[str] = set()
    users_with_history: set[str] = set()
    history_lengths: list[int] = []
    positive_count = 0
    negative_count = 0
    behavior_rows = 0

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                raise ValueError(f"Invalid behaviors.tsv line {line_number}: expected 5 columns")
            impression_id, user_id, time_text, history, impression_text = parts
            behavior_rows += 1
            users.add(user_id)
            history_items = history.split() if history else []
            history_lengths.append(len(history_items))
            if history_items:
                users_with_history.add(user_id)

            for token in impression_text.split():
                news_id, label = parse_impression_token(token)
                if label == 1:
                    positive_count += 1
                elif label == 0:
                    negative_count += 1
                impressions.append(
                    Impression(
                        split=split,
                        impression_id=impression_id,
                        user_id=user_id,
                        time=time_text,
                        history=history,
                        news_id=news_id,
                        label=label,
                    )
                )

    labeled_count = positive_count + negative_count
    summary = BehaviorSummary(
        split=split,
        behavior_rows=behavior_rows,
        users=len(users),
        users_with_history=len(users_with_history),
        avg_history_length=statistics.mean(history_lengths) if history_lengths else 0.0,
        max_history_length=max(history_lengths) if history_lengths else 0,
        impressions=len(impressions),
        positive_impressions=positive_count,
        negative_impressions=negative_count,
        ctr=positive_count / labeled_count if labeled_count else 0.0,
    )
    return impressions, summary


def summarize_reczoo_interactions(
    path: Path,
    split: str,
    sample_path: Path,
    max_sample_rows: int,
) -> ReczooSummary:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    positive_rows = 0
    negative_rows = 0
    impression_ids: set[str] = set()
    users: set[str] = set()
    users_with_history: set[str] = set()
    history_lengths: list[int] = []

    with path.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "imp_id",
            "click",
            "hour",
            "user_id",
            "news_id",
            "cat",
            "sub_cat",
            "title_entities",
            "abstract_entities",
            "news_his",
            "cat_his",
            "subcat_his",
        }
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Unexpected RecZoo interaction schema in {path}: {reader.fieldnames}")

        with sample_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=["split", *reader.fieldnames])
            writer.writeheader()

            for row in reader:
                rows += 1
                impression_ids.add(row["imp_id"])
                users.add(row["user_id"])
                history_items = row["news_his"].split("^") if row["news_his"] else []
                history_lengths.append(len(history_items))
                if history_items:
                    users_with_history.add(row["user_id"])

                if row["click"] == "1":
                    positive_rows += 1
                elif row["click"] == "0":
                    negative_rows += 1

                if rows <= max_sample_rows:
                    writer.writerow({"split": split, **row})

    labeled_count = positive_rows + negative_rows
    return ReczooSummary(
        split=split,
        rows=rows,
        impression_ids=len(impression_ids),
        users=len(users),
        users_with_history=len(users_with_history),
        avg_history_length=statistics.mean(history_lengths) if history_lengths else 0.0,
        max_history_length=max(history_lengths) if history_lengths else 0,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        ctr=positive_rows / labeled_count if labeled_count else 0.0,
    )


def write_news_csv(path: Path, news: dict[str, News]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "news_id",
                "category",
                "subcategory",
                "title",
                "abstract",
                "url",
                "title_entities",
                "abstract_entities",
                "title_word_count",
                "abstract_word_count",
            ]
        )
        for news_id in sorted(news):
            row = news[news_id]
            writer.writerow(
                [
                    row.news_id,
                    row.category,
                    row.subcategory,
                    row.title,
                    row.abstract,
                    row.url,
                    row.title_entities,
                    row.abstract_entities,
                    len(row.title.split()),
                    len(row.abstract.split()),
                ]
            )


def write_impressions_csv(path: Path, impressions: list[Impression]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "split",
                "impression_id",
                "user_id",
                "time",
                "history",
                "news_id",
                "label",
            ]
        )
        for row in impressions:
            writer.writerow(
                [
                    row.split,
                    row.impression_id,
                    row.user_id,
                    row.time,
                    row.history,
                    row.news_id,
                    "" if row.label is None else row.label,
                ]
            )


def write_user_history_csv(path: Path, impressions: list[Impression]) -> None:
    seen: set[tuple[str, str, str]] = set()
    rows: list[tuple[str, str, str, int]] = []
    for row in impressions:
        key = (row.split, row.impression_id, row.user_id)
        if key in seen:
            continue
        seen.add(key)
        history_length = len(row.history.split()) if row.history else 0
        rows.append((row.split, row.user_id, row.history, history_length))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "user_id", "history", "history_length"])
        writer.writerows(rows)


def safe_mean(values: list[int]) -> float:
    return statistics.mean(values) if values else 0.0


def format_percent(value: float) -> str:
    return f"{value * 100:.4f}%"


def generate_report(
    path: Path,
    train_news: dict[str, News],
    dev_news: dict[str, News],
    all_news: dict[str, News],
    train_summary: BehaviorSummary,
    dev_summary: BehaviorSummary,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    categories = Counter(row.category for row in all_news.values())
    subcategories = Counter(row.subcategory for row in all_news.values())
    title_lengths = [len(row.title.split()) for row in all_news.values()]
    abstract_lengths = [len(row.abstract.split()) for row in all_news.values()]
    train_only_news = set(train_news) - set(dev_news)
    dev_only_news = set(dev_news) - set(train_news)
    shared_news = set(train_news) & set(dev_news)

    lines = [
        "# MIND-small 数据报告",
        "",
        "## 数据集概览",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 训练集新闻数 | {len(train_news):,} |",
        f"| 验证集新闻数 | {len(dev_news):,} |",
        f"| 合并去重新闻数 | {len(all_news):,} |",
        f"| 训练/验证共有新闻数 | {len(shared_news):,} |",
        f"| 仅训练集出现新闻数 | {len(train_only_news):,} |",
        f"| 仅验证集出现新闻数 | {len(dev_only_news):,} |",
        f"| Category 数 | {len(categories):,} |",
        f"| Subcategory 数 | {len(subcategories):,} |",
        f"| 平均标题词数 | {safe_mean(title_lengths):.2f} |",
        f"| 平均摘要词数 | {safe_mean(abstract_lengths):.2f} |",
        "",
        "## 行为与曝光统计",
        "",
        "| Split | 行为行数 | 用户数 | 有历史用户数 | 平均历史长度 | 最大历史长度 | 曝光数 | 正样本 | 负样本 | CTR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in [train_summary, dev_summary]:
        lines.append(
            "| "
            f"{summary.split} | "
            f"{summary.behavior_rows:,} | "
            f"{summary.users:,} | "
            f"{summary.users_with_history:,} | "
            f"{summary.avg_history_length:.2f} | "
            f"{summary.max_history_length:,} | "
            f"{summary.impressions:,} | "
            f"{summary.positive_impressions:,} | "
            f"{summary.negative_impressions:,} | "
            f"{format_percent(summary.ctr)} |"
        )

    lines.extend(
        [
            "",
            "## Top Category",
            "",
            "| Category | 新闻数 | 占比 |",
            "|---|---:|---:|",
        ]
    )
    for category, count in categories.most_common(10):
        lines.append(f"| {category} | {count:,} | {format_percent(count / len(all_news))} |")

    lines.extend(
        [
            "",
            "## Top Subcategory",
            "",
            "| Subcategory | 新闻数 | 占比 |",
            "|---|---:|---:|",
        ]
    )
    for subcategory, count in subcategories.most_common(10):
        lines.append(f"| {subcategory} | {count:,} | {format_percent(count / len(all_news))} |")

    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `news.tsv` 提供新闻 ID、类别、标题、摘要、URL 和实体标注，是后续新闻塔内容编码的基础。",
            "- `behaviors.tsv` 提供用户点击历史和一次推荐曝光中的候选新闻及点击标签，是后续召回与排序训练的基础。",
            "- `label=1` 表示用户在该次曝光中点击了新闻，`label=0` 表示曝光但未点击。",
            "- MIND 阶段会从 MovieLens 的 ID 推荐升级为内容推荐：新闻标题、摘要和类别会进入物品塔或 Ranker 特征。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_reczoo_report(
    path: Path,
    news: dict[str, News],
    train_summary: ReczooSummary,
    valid_summary: ReczooSummary,
    max_sample_rows: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    categories = Counter(row.category for row in news.values())
    subcategories = Counter(row.subcategory for row in news.values())
    title_lengths = [len(row.title.split()) for row in news.values()]
    abstract_lengths = [len(row.abstract.split()) for row in news.values()]

    lines = [
        "# MIND-small 数据报告",
        "",
        "## 数据源说明",
        "",
        "官方 Azure Blob 下载当前返回 `Public access is not permitted`，本报告使用可访问的 RecZoo `MIND_small_x1` 镜像生成。",
        "该镜像已将原始曝光展开为逐条 `user_id-news_id-click` 样本，适合后续直接训练排序模型或构造召回样本。",
        "",
        "## 新闻内容概览",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 新闻数 | {len(news):,} |",
        f"| Category 数 | {len(categories):,} |",
        f"| Subcategory 数 | {len(subcategories):,} |",
        f"| 平均标题词数 | {safe_mean(title_lengths):.2f} |",
        f"| 平均摘要词数 | {safe_mean(abstract_lengths):.2f} |",
        "",
        "## 行为样本统计",
        "",
        "| Split | 样本行数 | 曝光 ID 数 | 用户数 | 有历史用户数 | 平均历史长度 | 最大历史长度 | 正样本 | 负样本 | CTR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in [train_summary, valid_summary]:
        lines.append(
            "| "
            f"{summary.split} | "
            f"{summary.rows:,} | "
            f"{summary.impression_ids:,} | "
            f"{summary.users:,} | "
            f"{summary.users_with_history:,} | "
            f"{summary.avg_history_length:.2f} | "
            f"{summary.max_history_length:,} | "
            f"{summary.positive_rows:,} | "
            f"{summary.negative_rows:,} | "
            f"{format_percent(summary.ctr)} |"
        )

    lines.extend(
        [
            "",
            "## Top Category",
            "",
            "| Category | 新闻数 | 占比 |",
            "|---|---:|---:|",
        ]
    )
    for category, count in categories.most_common(10):
        lines.append(f"| {category} | {count:,} | {format_percent(count / len(news))} |")

    lines.extend(
        [
            "",
            "## Top Subcategory",
            "",
            "| Subcategory | 新闻数 | 占比 |",
            "|---|---:|---:|",
        ]
    )
    for subcategory, count in subcategories.most_common(10):
        lines.append(f"| {subcategory} | {count:,} | {format_percent(count / len(news))} |")

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "| 文件 | 说明 |",
            "|---|---|",
            "| `data/processed/news_metadata.csv` | 新闻内容元数据，后续新闻塔文本/类别编码使用 |",
            f"| `data/processed/train_sample.csv` | 训练样本前 `{max_sample_rows:,}` 行，用于快速调试模型 |",
            f"| `data/processed/valid_sample.csv` | 验证样本前 `{max_sample_rows:,}` 行，用于快速调试模型 |",
            "",
            "## 说明",
            "",
            "- `click=1` 表示用户点击该新闻，`click=0` 表示曝光但未点击。",
            "- `news_his` 是用户历史点击新闻 ID 序列，是后续用户塔建模兴趣的核心输入。",
            "- `cat`、`sub_cat`、`title`、`abstract` 是后续新闻塔和 Ranker 的内容特征。",
            "- RecZoo 原始 `train.csv` 和 `valid.csv` 体积较大，当前不复制完整处理文件，只输出样本 CSV 和全量统计报告。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_reczoo(args: argparse.Namespace, reczoo_dir: Path) -> int:
    news = read_reczoo_news(reczoo_dir / "news_corpus.tsv")
    write_news_csv(args.processed_dir / "news_metadata.csv", news)
    train_summary = summarize_reczoo_interactions(
        path=reczoo_dir / "train.csv",
        split="train",
        sample_path=args.processed_dir / "train_sample.csv",
        max_sample_rows=args.sample_rows,
    )
    valid_summary = summarize_reczoo_interactions(
        path=reczoo_dir / "valid.csv",
        split="valid",
        sample_path=args.processed_dir / "valid_sample.csv",
        max_sample_rows=args.sample_rows,
    )
    generate_reczoo_report(
        path=args.reports_dir / "data_report.md",
        news=news,
        train_summary=train_summary,
        valid_summary=valid_summary,
        max_sample_rows=args.sample_rows,
    )
    print(f"Wrote processed data: {args.processed_dir}")
    print(f"Wrote data report: {args.reports_dir / 'data_report.md'}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--source",
        choices=["auto", "official", "reczoo"],
        default="auto",
        help="Dataset source format to prepare.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=100_000,
        help="Rows copied from RecZoo train/valid CSVs for quick model debugging.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and extract MIND-small if files are missing.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    train_dir = args.raw_dir / "MINDsmall_train"
    dev_dir = args.raw_dir / "MINDsmall_dev"
    reczoo_dir = args.raw_dir / "MIND_small_x1"

    if args.download and args.source == "reczoo" and not reczoo_dir.exists():
        download_and_extract(raw_dir=args.raw_dir, source="reczoo")
    elif args.download and args.source in {"auto", "official"} and (
        not train_dir.exists() or not dev_dir.exists()
    ):
        download_and_extract(raw_dir=args.raw_dir, source=args.source)

    if args.source in {"auto", "reczoo"} and (reczoo_dir / "news_corpus.tsv").exists():
        return prepare_reczoo(args=args, reczoo_dir=reczoo_dir)

    required_paths = [
        train_dir / "news.tsv",
        train_dir / "behaviors.tsv",
        dev_dir / "news.tsv",
        dev_dir / "behaviors.tsv",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}. Run download_mind.py first "
                "or pass --download."
            )

    train_news = read_news(train_dir / "news.tsv")
    dev_news = read_news(dev_dir / "news.tsv")
    all_news = {**train_news, **dev_news}
    train_impressions, train_summary = read_impressions(
        train_dir / "behaviors.tsv",
        split="train",
    )
    dev_impressions, dev_summary = read_impressions(
        dev_dir / "behaviors.tsv",
        split="dev",
    )

    write_news_csv(args.processed_dir / "news_metadata.csv", all_news)
    write_impressions_csv(args.processed_dir / "train_impressions.csv", train_impressions)
    write_impressions_csv(args.processed_dir / "dev_impressions.csv", dev_impressions)
    write_user_history_csv(
        args.processed_dir / "user_history.csv",
        train_impressions + dev_impressions,
    )
    generate_report(
        path=args.reports_dir / "data_report.md",
        train_news=train_news,
        dev_news=dev_news,
        all_news=all_news,
        train_summary=train_summary,
        dev_summary=dev_summary,
    )

    print(f"Wrote processed data: {args.processed_dir}")
    print(f"Wrote data report: {args.reports_dir / 'data_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
