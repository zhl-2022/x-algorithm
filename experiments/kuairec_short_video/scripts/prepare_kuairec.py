"""Inspect KuaiRec raw files and generate a first data report.

This script does not train a model. Its job is to make the downloaded dataset
observable before we decide the final label and split strategy.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_REPORT = ROOT / "reports" / "data_report.md"

KNOWN_INTERACTION_NAMES = {
    "big_matrix.csv",
    "small_matrix.csv",
    "user_video_act.csv",
    "interactions.csv",
}


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def read_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            return next(reader, [])
    except UnicodeDecodeError:
        with path.open("r", encoding="gb18030", newline="") as file:
            reader = csv.reader(file)
            return next(reader, [])
    except Exception:
        return []


def read_sample_rows(path: Path, limit: int = 3) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(itertools.islice(csv.DictReader(file), limit))
    except UnicodeDecodeError:
        with path.open("r", encoding="gb18030", newline="") as file:
            return list(itertools.islice(csv.DictReader(file), limit))
    except Exception:
        return []


def count_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            return max(sum(1 for _ in file) - 1, 0)
    except UnicodeDecodeError:
        with path.open("r", encoding="gb18030", newline="") as file:
            return max(sum(1 for _ in file) - 1, 0)


def open_text_csv(path: Path):
    return path.open("r", encoding="utf-8-sig", errors="replace", newline="")


def find_csv_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.rglob("*.csv"))


def sample_name(path: Path, raw_dir: Path) -> str:
    relative = path.relative_to(raw_dir).as_posix()
    safe = "".join(char if char.isalnum() else "_" for char in relative)
    return f"{safe}_sample.csv"


def copy_sample(path: Path, raw_dir: Path, output_dir: Path, sample_rows: int) -> Path | None:
    if sample_rows <= 0:
        return None

    header = read_header(path)
    if not header:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / sample_name(path, raw_dir)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as src, target.open("w", encoding="utf-8", newline="") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            for row in itertools.islice(reader, sample_rows + 1):
                writer.writerow(row)
    except UnicodeDecodeError:
        with path.open("r", encoding="gb18030", newline="") as src, target.open("w", encoding="utf-8", newline="") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            for row in itertools.islice(reader, sample_rows + 1):
                writer.writerow(row)
    return target


def build_inventory(raw_dir: Path, processed_dir: Path, sample_rows: int, should_count_rows: bool) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    sample_dir = processed_dir / "samples"

    for path in find_csv_files(raw_dir):
        header = read_header(path)
        sample_path = copy_sample(path, raw_dir, sample_dir, sample_rows)
        row_count = count_rows(path) if should_count_rows else -1
        inventory.append(
            {
                "file": path.name,
                "relative_path": path.relative_to(raw_dir).as_posix(),
                "size_bytes": str(path.stat().st_size),
                "size": format_size(path.stat().st_size),
                "row_count": "" if row_count < 0 else str(row_count),
                "columns": "|".join(header),
                "sample_file": "" if sample_path is None else sample_path.relative_to(ROOT).as_posix(),
            }
        )
    return inventory


def write_inventory_csv(inventory: list[dict[str, str]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "relative_path", "size_bytes", "size", "row_count", "columns", "sample_file"]
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(inventory)


def infer_label_plan(headers: list[str]) -> list[str]:
    header_set = set(headers)
    plans: list[str] = []
    if "watch_ratio" in header_set:
        plans.append("首选二分类标签：`positive = watch_ratio >= 1.0`，表示完播或重复观看。")
        plans.append("可做消融：`positive = watch_ratio >= 0.8`，观察更宽松正反馈对 Recall/NDCG 的影响。")
    if "is_click" in header_set:
        plans.append("可选点击标签：`positive = is_click == 1`，用于点击预测口径。")
    if {"play_time", "video_duration"}.issubset(header_set):
        plans.append("可计算完播率：`watch_ratio = play_time / video_duration`。")
    if {"watch_time", "duration"}.issubset(header_set):
        plans.append("可计算完播率：`watch_ratio = watch_time / duration`。")
    if not plans:
        plans.append("暂未从表头识别出观看反馈字段，需要查看样例行后再确定标签。")
    return plans


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def profile_interaction_file(path: Path, raw_dir: Path) -> dict[str, str]:
    users: set[str] = set()
    videos: set[str] = set()
    rows = 0
    valid_watch_ratio = 0
    positive_1 = 0
    positive_08 = 0
    watch_ratio_sum = 0.0
    min_timestamp: float | None = None
    max_timestamp: float | None = None

    with open_text_csv(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows += 1
            user_id = row.get("user_id", "")
            video_id = row.get("video_id", "")
            if user_id:
                users.add(user_id)
            if video_id:
                videos.add(video_id)

            ratio = to_float(row.get("watch_ratio", ""))
            if ratio is not None:
                valid_watch_ratio += 1
                watch_ratio_sum += ratio
                if ratio >= 1.0:
                    positive_1 += 1
                if ratio >= 0.8:
                    positive_08 += 1

            timestamp = to_float(row.get("timestamp", ""))
            if timestamp is not None:
                min_timestamp = timestamp if min_timestamp is None else min(min_timestamp, timestamp)
                max_timestamp = timestamp if max_timestamp is None else max(max_timestamp, timestamp)

    avg_ratio = watch_ratio_sum / valid_watch_ratio if valid_watch_ratio else 0.0
    return {
        "file": path.name,
        "relative_path": path.relative_to(raw_dir).as_posix(),
        "rows": str(rows),
        "users": str(len(users)),
        "videos": str(len(videos)),
        "watch_ratio_rows": str(valid_watch_ratio),
        "avg_watch_ratio": f"{avg_ratio:.6f}",
        "positive_ge_1": str(positive_1),
        "positive_ge_1_rate": f"{positive_1 / rows:.6f}" if rows else "0.000000",
        "positive_ge_08": str(positive_08),
        "positive_ge_08_rate": f"{positive_08 / rows:.6f}" if rows else "0.000000",
        "min_timestamp": "" if min_timestamp is None else f"{min_timestamp:.3f}",
        "max_timestamp": "" if max_timestamp is None else f"{max_timestamp:.3f}",
    }


def build_interaction_profiles(raw_dir: Path, inventory: list[dict[str, str]], enabled: bool) -> list[dict[str, str]]:
    if not enabled:
        return []
    profiles: list[dict[str, str]] = []
    for item in inventory:
        if item["file"] in KNOWN_INTERACTION_NAMES:
            profiles.append(profile_interaction_file(raw_dir / item["relative_path"], raw_dir))
    return profiles


def write_profiles_csv(profiles: list[dict[str, str]], target: Path) -> None:
    if not profiles:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "relative_path",
        "rows",
        "users",
        "videos",
        "watch_ratio_rows",
        "avg_watch_ratio",
        "positive_ge_1",
        "positive_ge_1_rate",
        "positive_ge_08",
        "positive_ge_08_rate",
        "min_timestamp",
        "max_timestamp",
    ]
    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(profiles)


def write_report(
    raw_dir: Path,
    processed_dir: Path,
    inventory: list[dict[str, str]],
    interaction_profiles: list[dict[str, str]],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    interaction_files = [item for item in inventory if item["file"] in KNOWN_INTERACTION_NAMES]
    primary = interaction_files[0] if interaction_files else (inventory[0] if inventory else None)
    primary_headers = primary["columns"].split("|") if primary and primary["columns"] else []
    label_plans = infer_label_plan(primary_headers)

    lines: list[str] = [
        "# KuaiRec 数据盘点报告",
        "",
        "## 生成说明",
        "",
        f"- 原始数据目录：`{raw_dir}`",
        f"- 处理数据目录：`{processed_dir}`",
        f"- CSV 文件数量：{len(inventory)}",
        "",
        "## 文件清单",
        "",
        "| 文件 | 相对路径 | 大小 | 行数 | 样例文件 |",
        "|---|---|---:|---:|---|",
    ]

    for item in inventory:
        row_count = item["row_count"] or "未统计"
        sample_file = f"`{item['sample_file']}`" if item["sample_file"] else "-"
        lines.append(f"| `{item['file']}` | `{item['relative_path']}` | {item['size']} | {row_count} | {sample_file} |")

    lines.extend(
        [
            "",
            "## 关键交互表候选",
            "",
        ]
    )
    if interaction_files:
        for item in interaction_files:
            lines.append(f"- `{item['relative_path']}`：字段 `{item['columns']}`")
    else:
        lines.append("- 暂未发现标准命名的交互表，请根据文件清单人工确认。")

    lines.extend(
        [
            "",
            "## 交互表统计",
            "",
        ]
    )
    if interaction_profiles:
        lines.extend(
            [
                "| 文件 | 行数 | 用户数 | 视频数 | 平均 `watch_ratio` | `watch_ratio >= 1.0` | 比例 | `watch_ratio >= 0.8` | 比例 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for profile in interaction_profiles:
            lines.append(
                "| "
                f"`{profile['relative_path']}` | "
                f"{int(profile['rows']):,} | "
                f"{int(profile['users']):,} | "
                f"{int(profile['videos']):,} | "
                f"{profile['avg_watch_ratio']} | "
                f"{int(profile['positive_ge_1']):,} | "
                f"{float(profile['positive_ge_1_rate']) * 100:.2f}% | "
                f"{int(profile['positive_ge_08']):,} | "
                f"{float(profile['positive_ge_08_rate']) * 100:.2f}% |"
            )
    else:
        lines.append("- 本次未启用交互表 profile。可使用 `--profile-interactions` 重新生成。")

    lines.extend(
        [
            "",
            "## 当前标签建议",
            "",
        ]
    )
    for plan in label_plans:
        lines.append(f"- {plan}")

    lines.extend(
        [
            "",
            "## 下一步",
            "",
            "1. 确认核心交互表和时间字段。",
            "2. 根据实际字段确定正反馈标签。",
            "3. 按用户时间序列切分训练集、验证集和测试集。",
            "4. 先实现 Popularity baseline，再迁移 Category、ItemCF、MF、Two-Tower 和 Ranker。",
            "",
            "## 样例行",
            "",
        ]
    )

    for item in inventory[:8]:
        path = raw_dir / item["relative_path"]
        sample_rows = read_sample_rows(path, limit=2)
        lines.append(f"### `{item['relative_path']}`")
        lines.append("")
        if sample_rows:
            lines.append("```json")
            lines.append(json.dumps(sample_rows, ensure_ascii=False, indent=2))
            lines.append("```")
        else:
            lines.append("未读取到样例行。")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect downloaded KuaiRec files.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Raw data directory.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR, help="Processed data directory.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Markdown report path.")
    parser.add_argument("--sample-rows", type=int, default=0, help="Copy N sample rows from each CSV into processed samples.")
    parser.add_argument("--count-rows", action="store_true", help="Count CSV rows. This can be slow for large files.")
    parser.add_argument("--profile-interactions", action="store_true", help="Profile interaction tables such as big_matrix.csv.")
    parser.add_argument("--clean-samples", action="store_true", help="Remove old processed sample files before writing new ones.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    report_path = args.report.resolve()

    if not raw_dir.exists():
        print(f"[error] raw data directory does not exist: {raw_dir}")
        print("Run download_kuairec.py first.")
        return 1

    sample_dir = processed_dir / "samples"
    if args.clean_samples and sample_dir.exists():
        shutil.rmtree(sample_dir)

    inventory = build_inventory(raw_dir, processed_dir, args.sample_rows, args.count_rows)
    interaction_profiles = build_interaction_profiles(raw_dir, inventory, args.profile_interactions)
    write_inventory_csv(inventory, processed_dir / "file_inventory.csv")
    write_profiles_csv(interaction_profiles, processed_dir / "interaction_profile.csv")
    write_report(raw_dir, processed_dir, inventory, interaction_profiles, report_path)

    print(f"[done] inventory: {processed_dir / 'file_inventory.csv'}")
    print(f"[done] report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
