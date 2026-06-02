"""Evaluate reranking sensitivity to Two-Tower candidate set size."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import rerank_pipeline
import two_tower_train


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = EXPERIMENT_DIR / "data" / "processed"
DEFAULT_REPORTS_DIR = EXPERIMENT_DIR / "reports"
DEFAULT_OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"


@dataclass(frozen=True)
class AblationRun:
    candidate_k: int
    candidate_recall: float
    recall: float
    hit_rate: float
    precision: float
    ndcg: float
    coverage: float
    rerank_seconds: float


def parse_candidate_ks(value: str) -> list[int]:
    candidate_ks = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not candidate_ks:
        raise argparse.ArgumentTypeError("candidate ks cannot be empty")
    if any(candidate_k <= 0 for candidate_k in candidate_ks):
        raise argparse.ArgumentTypeError("candidate ks must be positive")
    return candidate_ks


def write_ablation_csv(path: Path, runs: list[AblationRun]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_k",
        "candidate_recall",
        "recall",
        "hit_rate",
        "precision",
        "ndcg",
        "coverage",
        "rerank_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "candidate_k": run.candidate_k,
                    "candidate_recall": f"{run.candidate_recall:.8f}",
                    "recall": f"{run.recall:.8f}",
                    "hit_rate": f"{run.hit_rate:.8f}",
                    "precision": f"{run.precision:.8f}",
                    "ndcg": f"{run.ndcg:.8f}",
                    "coverage": f"{run.coverage:.8f}",
                    "rerank_seconds": f"{run.rerank_seconds:.2f}",
                }
            )


def generate_report(
    path: Path,
    runs: list[AblationRun],
    candidate_generation_seconds: float,
    two_tower_train_seconds: float,
    ranker_train_seconds: float,
    auc: float,
    logloss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best_recall = max(runs, key=lambda run: (run.recall, run.ndcg, -run.rerank_seconds))
    best_ndcg = max(runs, key=lambda run: (run.ndcg, run.recall, -run.rerank_seconds))
    lines = [
        "# Candidate K 消融实验报告",
        "",
        "## 实验目的",
        "",
        "评估 Two-Tower 召回候选数量对 DNN Ranker 重排效果的影响。",
        "本实验只训练一次 Two-Tower 和 DNN Ranker，然后用最大候选数生成候选集，",
        "再分别截断为不同 `candidate_k` 做重排评估，保证不同候选数之间可直接对比。",
        "",
        "## 消融结果",
        "",
        "| Candidate K | Candidate Recall | Recall@20 | NDCG@20 | Coverage@20 | 重排耗时（秒） |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            "| "
            f"{run.candidate_k} | "
            f"{run.candidate_recall:.6f} | "
            f"{run.recall:.6f} | "
            f"{run.ndcg:.6f} | "
            f"{run.coverage:.6f} | "
            f"{run.rerank_seconds:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 排序模型指标",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| AUC | {auc:.6f} |",
            f"| LogLoss | {logloss:.6f} |",
            "",
            "## 运行耗时",
            "",
            "| 阶段 | 耗时（秒） |",
            "|---|---:|",
            f"| Two-Tower 训练 | {two_tower_train_seconds:.2f} |",
            f"| DNN Ranker 训练 | {ranker_train_seconds:.2f} |",
            f"| 最大候选集生成 | {candidate_generation_seconds:.2f} |",
            "",
            "## 结论",
            "",
            f"- 当前 Recall@20 最优候选数是 `{best_recall.candidate_k}`，Recall@20 为 `{best_recall.recall:.6f}`。",
            f"- 当前 NDCG@20 最优候选数是 `{best_ndcg.candidate_k}`，NDCG@20 为 `{best_ndcg.ndcg:.6f}`。",
            "- `Candidate Recall` 会随候选数增大而上升，但最终 Top20 指标不一定线性上升。",
            "- 候选数越大，Ranker 需要重排的候选越多，重排耗时也会增加。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--candidate-ks", type=parse_candidate_ks, default=[50, 100, 200, 500])
    parser.add_argument("--max-history-items", type=int, default=100)
    parser.add_argument("--two-tower-embedding-dim", type=int, default=128)
    parser.add_argument("--two-tower-hidden-dim", type=int, default=256)
    parser.add_argument("--two-tower-tower-dim", type=int, default=128)
    parser.add_argument("--two-tower-epochs", type=int, default=10)
    parser.add_argument("--two-tower-batch-size", type=int, default=4096)
    parser.add_argument("--two-tower-negative-samples", type=int, default=1)
    parser.add_argument("--two-tower-loss", choices=["bce", "in-batch"], default="bce")
    parser.add_argument("--two-tower-lr", type=float, default=0.005)
    parser.add_argument("--two-tower-dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--ranker-embedding-dim", type=int, default=128)
    parser.add_argument(
        "--ranker-hidden-dims",
        type=rerank_pipeline.ranker_train.parse_hidden_dims,
        default=(512, 256, 128),
    )
    parser.add_argument("--ranker-epochs", type=int, default=8)
    parser.add_argument("--ranker-batch-size", type=int, default=4096)
    parser.add_argument("--ranker-negative-samples", type=int, default=3)
    parser.add_argument("--ranker-lr", type=float, default=0.001)
    parser.add_argument("--ranker-dropout", type=float, default=0.1)
    parser.add_argument("--eval-negative-samples", type=int, default=100)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
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

    train = two_tower_train.read_interactions(train_path)
    test = two_tower_train.read_interactions(test_path)
    item_metadata = two_tower_train.read_item_metadata(metadata_path)
    train_by_user = two_tower_train.group_items_by_user(train)
    test_by_user = two_tower_train.group_items_by_user(test)
    user_to_index, item_to_index, index_to_item = two_tower_train.build_mappings(
        train=train,
        test=test,
        all_item_ids=set(item_metadata),
    )
    device = rerank_pipeline.choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    two_tower_args = rerank_pipeline.build_two_tower_args(args)
    two_tower, _, two_tower_seconds = two_tower_train.train_model(
        train=train,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        args=two_tower_args,
        device=device,
    )

    ranker_args = rerank_pipeline.build_ranker_args(args)
    ranker, _, ranker_seconds, _ = rerank_pipeline.ranker_train.train_model(
        train=train,
        item_metadata=item_metadata,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        args=ranker_args,
        device=device,
    )

    max_candidate_k = max(args.candidate_ks)
    candidates_by_user, candidate_seconds = rerank_pipeline.generate_candidates(
        model=two_tower,
        test_by_user=test_by_user,
        train_by_user=train_by_user,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        candidate_k=max_candidate_k,
        device=device,
    )
    ranking_metrics = rerank_pipeline.evaluate_ranker_auc(
        ranker=ranker,
        train_by_user=train_by_user,
        test_by_user=test_by_user,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        eval_negative_samples=args.eval_negative_samples,
        score_batch_size=args.score_batch_size,
        seed=args.seed,
        device=device,
    )

    runs: list[AblationRun] = []
    for candidate_k in args.candidate_ks:
        sliced_candidates = {
            user_id: candidates[:candidate_k]
            for user_id, candidates in candidates_by_user.items()
        }
        metrics, rerank_seconds = rerank_pipeline.evaluate_rerank(
            ranker=ranker,
            candidates_by_user=sliced_candidates,
            test_by_user=test_by_user,
            user_to_index=user_to_index,
            item_to_index=item_to_index,
            index_to_item=index_to_item,
            k=args.k,
            candidate_k=candidate_k,
            score_batch_size=args.score_batch_size,
            device=device,
        )
        runs.append(
            AblationRun(
                candidate_k=candidate_k,
                candidate_recall=metrics.candidate_recall,
                recall=metrics.recall,
                hit_rate=metrics.hit_rate,
                precision=metrics.precision,
                ndcg=metrics.ndcg,
                coverage=metrics.coverage,
                rerank_seconds=rerank_seconds,
            )
        )
        print(
            "candidate_k="
            f"{candidate_k} candidate_recall={metrics.candidate_recall:.6f} "
            f"recall={metrics.recall:.6f} ndcg={metrics.ndcg:.6f} "
            f"coverage={metrics.coverage:.6f} rerank_seconds={rerank_seconds:.2f}",
            flush=True,
        )

    write_ablation_csv(args.outputs_dir / "rerank_ablation_results.csv", runs)
    generate_report(
        path=args.reports_dir / "rerank_ablation_report.md",
        runs=runs,
        candidate_generation_seconds=candidate_seconds,
        two_tower_train_seconds=two_tower_seconds,
        ranker_train_seconds=ranker_seconds,
        auc=ranking_metrics.auc,
        logloss=ranking_metrics.logloss,
    )
    print(f"AUC: {ranking_metrics.auc:.6f}")
    print(f"LogLoss: {ranking_metrics.logloss:.6f}")
    print(f"Wrote ablation report: {args.reports_dir / 'rerank_ablation_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
