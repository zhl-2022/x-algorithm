"""Run a Two-Tower recall plus DNN Ranker reranking pipeline on MovieLens."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

import ranker_train
import two_tower_train


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = EXPERIMENT_DIR / "data" / "processed"
DEFAULT_REPORTS_DIR = EXPERIMENT_DIR / "reports"
DEFAULT_OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"


@dataclass(frozen=True)
class RerankMetrics:
    model: str
    k: int
    candidate_k: int
    evaluated_users: int
    recall: float
    hit_rate: float
    precision: float
    ndcg: float
    coverage: float
    candidate_recall: float
    candidate_hit_rate: float


@dataclass(frozen=True)
class RankingMetrics:
    auc: float
    logloss: float
    eval_samples: int
    eval_positive_samples: int
    eval_negative_samples: int


@dataclass(frozen=True)
class PipelineStats:
    device: str
    two_tower_embedding_dim: int
    two_tower_hidden_dim: int
    two_tower_tower_dim: int
    two_tower_epochs: int
    two_tower_final_loss: float
    two_tower_train_seconds: float
    ranker_embedding_dim: int
    ranker_hidden_dims: tuple[int, ...]
    ranker_epochs: int
    ranker_final_loss: float
    ranker_train_seconds: float
    candidate_seconds: float
    rerank_seconds: float
    eval_seconds: float
    num_users: int
    num_items: int


def choose_device(device_arg: str) -> torch.device:
    return two_tower_train.choose_device(device_arg)


def dcg_for_hits(recommendations: list[int], relevant_items: set[int]) -> float:
    dcg = 0.0
    for index, item_id in enumerate(recommendations, start=1):
        if item_id in relevant_items:
            dcg += 1.0 / math.log2(index + 1)
    return dcg


def ideal_dcg(num_relevant: int, k: int) -> float:
    return sum(1.0 / math.log2(index + 1) for index in range(1, min(num_relevant, k) + 1))


def build_two_tower_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_dim=args.two_tower_embedding_dim,
        hidden_dim=args.two_tower_hidden_dim,
        tower_dim=args.two_tower_tower_dim,
        max_history_items=args.max_history_items,
        epochs=args.two_tower_epochs,
        batch_size=args.two_tower_batch_size,
        negative_samples=args.two_tower_negative_samples,
        loss=args.two_tower_loss,
        lr=args.two_tower_lr,
        weight_decay=args.weight_decay,
        dropout=args.two_tower_dropout,
        temperature=args.temperature,
        seed=args.seed,
    )


def build_ranker_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        embedding_dim=args.ranker_embedding_dim,
        hidden_dims=args.ranker_hidden_dims,
        epochs=args.ranker_epochs,
        batch_size=args.ranker_batch_size,
        negative_samples=args.ranker_negative_samples,
        eval_negative_samples=args.eval_negative_samples,
        score_batch_size=args.score_batch_size,
        lr=args.ranker_lr,
        weight_decay=args.weight_decay,
        dropout=args.ranker_dropout,
        seed=args.seed,
        k=args.k,
    )


def generate_candidates(
    model: two_tower_train.TwoTowerRecall,
    test_by_user: dict[int, set[int]],
    train_by_user: dict[int, set[int]],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    candidate_k: int,
    device: torch.device,
) -> tuple[dict[int, list[int]], float]:
    started = time.perf_counter()
    model.eval()
    all_item_indices = torch.arange(len(item_to_index), dtype=torch.long, device=device)
    candidates_by_user: dict[int, list[int]] = {}

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

            top_indices = torch.topk(
                scores,
                k=min(candidate_k, len(scores)),
            ).indices.tolist()
            candidates_by_user[user_id] = top_indices

    return candidates_by_user, time.perf_counter() - started


def score_candidate_items(
    model: ranker_train.DNNRanker,
    user_index: int,
    candidate_indices: list[int],
    score_batch_size: int,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(candidate_indices), score_batch_size):
            batch_items = candidate_indices[start : start + score_batch_size]
            user_tensor = torch.full(
                (len(batch_items),),
                user_index,
                dtype=torch.long,
                device=device,
            )
            item_tensor = torch.tensor(batch_items, dtype=torch.long, device=device)
            scores.extend(model(user_tensor, item_tensor).detach().cpu().tolist())
    return scores


def evaluate_rerank(
    ranker: ranker_train.DNNRanker,
    candidates_by_user: dict[int, list[int]],
    test_by_user: dict[int, set[int]],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    index_to_item: list[int],
    k: int,
    candidate_k: int,
    score_batch_size: int,
    device: torch.device,
) -> tuple[RerankMetrics, float]:
    started = time.perf_counter()
    ranker.eval()
    total_recall = 0.0
    total_hit_rate = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    total_candidate_recall = 0.0
    total_candidate_hit_rate = 0.0
    recommended_items: set[int] = set()

    for user_id in sorted(test_by_user):
        if user_id not in user_to_index:
            continue

        candidate_indices = candidates_by_user.get(user_id, [])
        relevant_items = test_by_user[user_id]
        relevant_indices = {
            item_to_index[item_id]
            for item_id in relevant_items
            if item_id in item_to_index
        }
        candidate_hits = len(set(candidate_indices) & relevant_indices)
        total_candidate_recall += candidate_hits / len(relevant_items)
        total_candidate_hit_rate += 1.0 if candidate_hits > 0 else 0.0

        ranker_scores = score_candidate_items(
            model=ranker,
            user_index=user_to_index[user_id],
            candidate_indices=candidate_indices,
            score_batch_size=score_batch_size,
            device=device,
        )
        ranked_candidates = sorted(
            zip(candidate_indices, ranker_scores),
            key=lambda row: (-row[1], row[0]),
        )
        recommendations = [
            index_to_item[item_index]
            for item_index, _ in ranked_candidates[:k]
        ]
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

    return (
        RerankMetrics(
            model="TwoTower+DNN-Rerank",
            k=k,
            candidate_k=candidate_k,
            evaluated_users=evaluated_users,
            recall=total_recall / evaluated_users,
            hit_rate=total_hit_rate / evaluated_users,
            precision=total_precision / evaluated_users,
            ndcg=total_ndcg / evaluated_users,
            coverage=len(recommended_items) / len(index_to_item),
            candidate_recall=total_candidate_recall / evaluated_users,
            candidate_hit_rate=total_candidate_hit_rate / evaluated_users,
        ),
        time.perf_counter() - started,
    )


def evaluate_ranker_auc(
    ranker: ranker_train.DNNRanker,
    train_by_user: dict[int, set[int]],
    test_by_user: dict[int, set[int]],
    user_to_index: dict[int, int],
    item_to_index: dict[int, int],
    eval_negative_samples: int,
    score_batch_size: int,
    seed: int,
    device: torch.device,
) -> RankingMetrics:
    eval_pairs, eval_labels = ranker_train.build_ranking_eval_pairs(
        test_by_user=test_by_user,
        train_by_user=train_by_user,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        eval_negative_samples=eval_negative_samples,
        seed=seed + 17,
    )
    eval_logits = ranker_train.score_pairs(
        model=ranker,
        pairs=eval_pairs,
        batch_size=score_batch_size,
        device=device,
    )
    return RankingMetrics(
        auc=ranker_train.binary_auc(eval_labels, eval_logits),
        logloss=ranker_train.logloss_from_logits(eval_labels, eval_logits),
        eval_samples=len(eval_labels),
        eval_positive_samples=sum(eval_labels),
        eval_negative_samples=len(eval_labels) - sum(eval_labels),
    )


def write_rerank_csv(
    path: Path,
    metrics: RerankMetrics,
    ranking_metrics: RankingMetrics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "k",
        "candidate_k",
        "evaluated_users",
        "recall",
        "hit_rate",
        "precision",
        "ndcg",
        "coverage",
        "candidate_recall",
        "candidate_hit_rate",
        "auc",
        "logloss",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "model": metrics.model,
                "k": metrics.k,
                "candidate_k": metrics.candidate_k,
                "evaluated_users": metrics.evaluated_users,
                "recall": f"{metrics.recall:.8f}",
                "hit_rate": f"{metrics.hit_rate:.8f}",
                "precision": f"{metrics.precision:.8f}",
                "ndcg": f"{metrics.ndcg:.8f}",
                "coverage": f"{metrics.coverage:.8f}",
                "candidate_recall": f"{metrics.candidate_recall:.8f}",
                "candidate_hit_rate": f"{metrics.candidate_hit_rate:.8f}",
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

    with (outputs_dir / "experiment_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def format_float(value: float) -> str:
    return f"{value:.6f}"


def generate_report(
    path: Path,
    metrics: RerankMetrics,
    ranking_metrics: RankingMetrics,
    stats: PipelineStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ranker_hidden_dims = " -> ".join(str(dim) for dim in stats.ranker_hidden_dims)
    lines = [
        "# Two-Tower + DNN Ranker 重排报告",
        "",
        "## Pipeline 指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 模型 | {metrics.model} |",
        f"| 召回候选数 | {metrics.candidate_k} |",
        f"| 最终推荐 K | {metrics.k} |",
        f"| 评估用户数 | {metrics.evaluated_users:,} |",
        f"| Candidate Recall@{metrics.candidate_k} | {format_float(metrics.candidate_recall)} |",
        f"| Candidate HitRate@{metrics.candidate_k} | {format_float(metrics.candidate_hit_rate)} |",
        f"| Recall@{metrics.k} | {format_float(metrics.recall)} |",
        f"| HitRate@{metrics.k} | {format_float(metrics.hit_rate)} |",
        f"| Precision@{metrics.k} | {format_float(metrics.precision)} |",
        f"| NDCG@{metrics.k} | {format_float(metrics.ndcg)} |",
        f"| Coverage@{metrics.k} | {format_float(metrics.coverage)} |",
        "",
        "## Ranker 排序指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| AUC | {format_float(ranking_metrics.auc)} |",
        f"| LogLoss | {format_float(ranking_metrics.logloss)} |",
        f"| 排序评估样本数 | {ranking_metrics.eval_samples:,} |",
        f"| 排序评估正样本数 | {ranking_metrics.eval_positive_samples:,} |",
        f"| 排序评估负样本数 | {ranking_metrics.eval_negative_samples:,} |",
        "",
        "## 训练与评估设置",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 设备 | {stats.device} |",
        f"| 用户数 | {stats.num_users:,} |",
        f"| 电影数 | {stats.num_items:,} |",
        f"| Two-Tower Embedding Dim | {stats.two_tower_embedding_dim} |",
        f"| Two-Tower Hidden Dim | {stats.two_tower_hidden_dim} |",
        f"| Two-Tower Tower Dim | {stats.two_tower_tower_dim} |",
        f"| Two-Tower Epochs | {stats.two_tower_epochs} |",
        f"| Two-Tower Final Loss | {stats.two_tower_final_loss:.6f} |",
        f"| Two-Tower 训练耗时（秒） | {stats.two_tower_train_seconds:.2f} |",
        f"| Ranker Embedding Dim | {stats.ranker_embedding_dim} |",
        f"| Ranker Hidden Dims | {ranker_hidden_dims} |",
        f"| Ranker Epochs | {stats.ranker_epochs} |",
        f"| Ranker Final Loss | {stats.ranker_final_loss:.6f} |",
        f"| Ranker 训练耗时（秒） | {stats.ranker_train_seconds:.2f} |",
        f"| 候选生成耗时（秒） | {stats.candidate_seconds:.2f} |",
        f"| 重排耗时（秒） | {stats.rerank_seconds:.2f} |",
        f"| 总评估耗时（秒） | {stats.eval_seconds:.2f} |",
        "",
        "## 说明",
        "",
        "- 这个实验是两阶段推荐 pipeline：Two-Tower 先从全量电影中召回候选，DNN Ranker 再对候选集重排。",
        "- `Candidate Recall` 表示真实测试电影是否进入了 Two-Tower 召回候选集，它决定了重排阶段的效果上限。",
        "- 最终 `Recall@K` 和 `NDCG@K` 只在 Ranker 重排后的 TopK 上计算。",
        "- Ranker 的 AUC 和 LogLoss 仍使用测试正样本加随机负样本评估。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int, default=200)
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
        type=ranker_train.parse_hidden_dims,
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
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    two_tower_args = build_two_tower_args(args)
    two_tower, two_tower_loss, two_tower_seconds = two_tower_train.train_model(
        train=train,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        args=two_tower_args,
        device=device,
    )

    ranker_args = build_ranker_args(args)
    ranker, ranker_loss, ranker_seconds, _ = ranker_train.train_model(
        train=train,
        item_metadata=item_metadata,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        args=ranker_args,
        device=device,
    )

    eval_started = time.perf_counter()
    candidates_by_user, candidate_seconds = generate_candidates(
        model=two_tower,
        test_by_user=test_by_user,
        train_by_user=train_by_user,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        candidate_k=args.candidate_k,
        device=device,
    )
    metrics, rerank_seconds = evaluate_rerank(
        ranker=ranker,
        candidates_by_user=candidates_by_user,
        test_by_user=test_by_user,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_item=index_to_item,
        k=args.k,
        candidate_k=args.candidate_k,
        score_batch_size=args.score_batch_size,
        device=device,
    )
    ranking_metrics = evaluate_ranker_auc(
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
    eval_seconds = time.perf_counter() - eval_started

    stats = PipelineStats(
        device=str(device),
        two_tower_embedding_dim=args.two_tower_embedding_dim,
        two_tower_hidden_dim=args.two_tower_hidden_dim,
        two_tower_tower_dim=args.two_tower_tower_dim,
        two_tower_epochs=args.two_tower_epochs,
        two_tower_final_loss=two_tower_loss,
        two_tower_train_seconds=two_tower_seconds,
        ranker_embedding_dim=args.ranker_embedding_dim,
        ranker_hidden_dims=args.ranker_hidden_dims,
        ranker_epochs=args.ranker_epochs,
        ranker_final_loss=ranker_loss,
        ranker_train_seconds=ranker_seconds,
        candidate_seconds=candidate_seconds,
        rerank_seconds=rerank_seconds,
        eval_seconds=eval_seconds,
        num_users=len(user_to_index),
        num_items=len(index_to_item),
    )

    rerank_result_path = args.outputs_dir / "rerank_results.csv"
    write_rerank_csv(rerank_result_path, metrics, ranking_metrics)
    write_experiment_results(
        outputs_dir=args.outputs_dir,
        result_paths=[
            args.outputs_dir / "popularity_results.csv",
            args.outputs_dir / "itemcf_results.csv",
            args.outputs_dir / "mf_results.csv",
            args.outputs_dir / "two_tower_results.csv",
            args.outputs_dir / "ranker_results.csv",
            rerank_result_path,
        ],
    )
    generate_report(
        path=args.reports_dir / "rerank_report.md",
        metrics=metrics,
        ranking_metrics=ranking_metrics,
        stats=stats,
    )

    print(f"Candidate Recall@{args.candidate_k}: {metrics.candidate_recall:.6f}")
    print(f"Recall@{args.k}: {metrics.recall:.6f}")
    print(f"NDCG@{args.k}: {metrics.ndcg:.6f}")
    print(f"Coverage@{args.k}: {metrics.coverage:.6f}")
    print(f"AUC: {ranking_metrics.auc:.6f}")
    print(f"LogLoss: {ranking_metrics.logloss:.6f}")
    print(f"Wrote rerank report: {args.reports_dir / 'rerank_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
