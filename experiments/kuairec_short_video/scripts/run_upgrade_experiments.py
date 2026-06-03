"""Run KuaiRec big-matrix upgrade experiments.

This script is intentionally separate from run_all_experiments.py.  The base
suite already compares standard baselines; this file focuses on follow-up
experiments that target the current big_matrix bottleneck: neural TopK recall
is much weaker than ItemCF.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from run_all_experiments import (
    DEFAULT_OUTPUTS_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_REPORTS_DIR,
    ExperimentMetric,
    FeatureTensors,
    KuaiRecTwoTower,
    PreparedData,
    RawInteraction,
    add_auc_logloss,
    build_feature_tensors,
    build_prepared_data,
    choose_device,
    dense_for_pairs,
    evaluate_topk,
    move_batch,
    score_two_tower_matrix,
    train_neural_model,
    write_metric_csv,
    write_single_report,
)


@dataclass(frozen=True)
class SequenceFeatureTensors:
    users: torch.Tensor
    items: torch.Tensor
    labels: torch.Tensor
    histories: torch.Tensor
    history_lengths: torch.Tensor


class TextConvTwoTower(KuaiRecTwoTower):
    """Two-Tower variant that replaces mean text pooling with a small TextCNN."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        text_buckets: int,
        embedding_dim: int,
        tower_dim: int,
        hidden_dim: int,
        dense_dim: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__(
            num_users=num_users,
            num_items=num_items,
            num_categories=num_categories,
            text_buckets=text_buckets,
            embedding_dim=embedding_dim,
            tower_dim=tower_dim,
            hidden_dim=hidden_dim,
            dense_dim=dense_dim,
            dropout=dropout,
            temperature=temperature,
        )
        self.text_conv = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=3, padding=1)

    def _text_vector(self, text_indices: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.text_embedding(text_indices).transpose(1, 2)
        encoded = F.relu(self.text_conv(embedded))
        mask = (text_indices > 0).unsqueeze(1)
        encoded = encoded.masked_fill(~mask, -1e4)
        return encoded.max(dim=2).values


class SequenceInterestModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        hidden_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.item_bias = nn.Embedding(num_items + 1, 1, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.item_projection = nn.Linear(embedding_dim, hidden_dim)

    def encode_history(self, histories: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.item_embedding(histories + 1)
        output, hidden = self.gru(embedded)
        fallback = output.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)
        state = hidden[-1]
        return torch.where((lengths > 0).unsqueeze(1), state, fallback)

    def forward(self, batch: SequenceFeatureTensors) -> torch.Tensor:
        state = F.normalize(self.encode_history(batch.histories, batch.history_lengths), dim=1)
        item_vector = F.normalize(self.item_projection(self.item_embedding(batch.items + 1)), dim=1)
        return (state * item_vector).sum(dim=1) / self.temperature + self.item_bias(batch.items + 1).squeeze(1)

    @torch.no_grad()
    def score_matrix(
        self,
        histories: torch.Tensor,
        lengths: torch.Tensor,
        item_count: int,
        device: torch.device,
        batch_users: int,
    ) -> torch.Tensor:
        self.eval()
        item_ids = torch.arange(item_count, dtype=torch.long, device=device)
        item_vectors = F.normalize(self.item_projection(self.item_embedding(item_ids + 1)), dim=1)
        item_bias = self.item_bias(item_ids + 1).squeeze(1)
        rows: list[torch.Tensor] = []
        for start in range(0, len(histories), batch_users):
            user_histories = histories[start : start + batch_users].to(device)
            user_lengths = lengths[start : start + batch_users].to(device)
            state = F.normalize(self.encode_history(user_histories, user_lengths), dim=1)
            rows.append((state @ item_vectors.T / self.temperature + item_bias).detach().cpu())
        return torch.cat(rows, dim=0)


class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, layers: int) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.layers = layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)

    def propagate(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        outputs = [embeddings]
        current = embeddings
        for _ in range(self.layers):
            current = torch.sparse.mm(adj, current)
            outputs.append(current)
        final = torch.stack(outputs, dim=0).mean(dim=0)
        return final[: self.num_users], final[self.num_users :]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["distill_twotower", "lightgcn", "sequence_model", "text_encoder"],
        required=True,
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--prepared-cache", type=Path, default=None)
    parser.add_argument("--matrix", choices=["small_matrix.csv", "big_matrix.csv"], default="big_matrix.csv")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--positive-threshold", type=float, default=0.8)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--train-rows", type=int, default=800_000)
    parser.add_argument("--auc-rows", type=int, default=300_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--score-batch-users", type=int, default=64)
    parser.add_argument("--score-batch-items", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--tower-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--text-buckets", type=int, default=8192)
    parser.add_argument("--max-text-tokens", type=int, default=16)
    parser.add_argument("--itemcf-history", type=int, default=80)
    parser.add_argument("--itemcf-neighbors", type=int, default=100)
    parser.add_argument("--teacher-items-per-user", type=int, default=40)
    parser.add_argument("--negative-items-per-user", type=int, default=40)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--lightgcn-layers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR / "upgrade_experiments")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR / "upgrade_experiments")
    return parser.parse_args(argv)


def load_or_build_prepared_data(args: argparse.Namespace) -> PreparedData:
    if args.prepared_cache and args.prepared_cache.exists():
        started = time.perf_counter()
        with args.prepared_cache.open("rb") as file:
            data = pickle.load(file)
        print(f"Loaded prepared cache: {args.prepared_cache} in {time.perf_counter() - started:.2f}s", flush=True)
        return data

    data = build_prepared_data(args)
    if args.prepared_cache:
        args.prepared_cache.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with args.prepared_cache.open("wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Wrote prepared cache: {args.prepared_cache} in {time.perf_counter() - started:.2f}s", flush=True)
    return data


def build_pair_features(data: PreparedData, users: list[int], items: list[int], labels: list[float]) -> FeatureTensors:
    users_t = torch.tensor(users, dtype=torch.long)
    items_t = torch.tensor(items, dtype=torch.long)
    return FeatureTensors(
        users=users_t,
        items=items_t,
        labels=torch.tensor(labels, dtype=torch.float32),
        categories=data.item_categories[items_t],
        text_indices=data.item_text_indices[items_t],
        text_lengths=data.item_text_lengths[items_t],
        dense=dense_for_pairs(data, users_t, items_t),
    )


def build_itemcf_neighbors(data: PreparedData, max_history: int, max_neighbors: int) -> dict[int, list[tuple[int, float]]]:
    item_positive_counts = Counter()
    co_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for history in data.train_positive_by_user.values():
        unique_history = list(dict.fromkeys(history[-max_history:]))
        for item in unique_history:
            item_positive_counts[item] += 1
        for index, item_i in enumerate(unique_history):
            for item_j in unique_history[index + 1 :]:
                co_counts[item_i][item_j] += 1
                co_counts[item_j][item_i] += 1

    neighbors: dict[int, list[tuple[int, float]]] = {}
    for item, counter in co_counts.items():
        sims: list[tuple[int, float]] = []
        for other, co_count in counter.items():
            denom = math.sqrt(item_positive_counts[item] * item_positive_counts[other])
            if denom:
                sims.append((other, co_count / denom))
        sims.sort(key=lambda pair: (-pair[1], pair[0]))
        neighbors[item] = sims[:max_neighbors]
    return neighbors


def build_distill_features(data: PreparedData, args: argparse.Namespace) -> FeatureTensors:
    rng = random.Random(args.seed)
    started = time.perf_counter()
    neighbors = build_itemcf_neighbors(data, args.itemcf_history, args.itemcf_neighbors)
    print(f"Built ItemCF teacher neighbors in {time.perf_counter() - started:.2f}s", flush=True)
    users: list[int] = []
    items: list[int] = []
    labels: list[float] = []
    target = args.train_rows

    positive_pairs = [
        (user, item)
        for user, history in data.train_positive_by_user.items()
        for item in history[-args.itemcf_history :]
    ]
    rng.shuffle(positive_pairs)
    for user, item in positive_pairs[: max(1, target // 3)]:
        users.append(user)
        items.append(item)
        labels.append(1.0)
    positive_count = len(labels)

    shuffled_users = list(data.train_positive_by_user)
    rng.shuffle(shuffled_users)
    for user in shuffled_users:
        seen = data.train_seen_by_user.get(user, set())
        scored: dict[int, float] = defaultdict(float)
        for hist_item in data.train_positive_by_user[user][-args.itemcf_history :]:
            for neighbor, score in neighbors.get(hist_item, []):
                if neighbor not in seen:
                    scored[neighbor] += score
        if not scored:
            continue
        ranked = sorted(scored.items(), key=lambda pair: (-pair[1], pair[0]))[: args.teacher_items_per_user]
        max_score = ranked[0][1] if ranked else 1.0
        for item, score in ranked:
            users.append(user)
            items.append(item)
            labels.append(0.5 + 0.5 * min(score / max_score, 1.0))
            if len(labels) >= target * 2 // 3:
                break
        if len(labels) >= target * 2 // 3:
            break
    teacher_count = len(labels) - positive_count

    all_items = len(data.index_to_item)
    max_attempts = max(target * 100, 10_000)
    attempts = 0
    while len(labels) < target and attempts < max_attempts:
        attempts += 1
        user = rng.choice(shuffled_users)
        seen = data.train_seen_by_user.get(user, set())
        item = rng.randrange(all_items)
        if item in seen:
            continue
        users.append(user)
        items.append(item)
        labels.append(0.0)
    while len(labels) < target:
        user = rng.choice(shuffled_users)
        item = rng.randrange(all_items)
        users.append(user)
        items.append(item)
        labels.append(0.0)

    print(
        f"Built distill features: positive={positive_count:,} "
        f"teacher={teacher_count:,} negative={target - positive_count - teacher_count:,} "
        f"total={target:,}",
        flush=True,
    )
    return build_pair_features(data, users[:target], items[:target], labels[:target])


def build_sequence_features(data: PreparedData, rows: list[RawInteraction], max_rows: int, args: argparse.Namespace) -> SequenceFeatureTensors:
    rng = random.Random(args.seed)
    if max_rows and len(rows) > max_rows:
        rows = rng.sample(rows, max_rows)
    histories: list[list[int]] = []
    lengths: list[int] = []
    users: list[int] = []
    items: list[int] = []
    labels: list[int] = []
    for row in rows:
        user = data.user_to_index[row.user_id]
        item = data.item_to_index[row.video_id]
        history = [hist_item for hist_item in data.train_positive_by_user.get(user, []) if hist_item != item]
        history = history[-args.sequence_length :]
        padded = history + [0] * (args.sequence_length - len(history))
        users.append(user)
        items.append(item)
        labels.append(row.label)
        histories.append(padded)
        lengths.append(len(history))
    return SequenceFeatureTensors(
        users=torch.tensor(users, dtype=torch.long),
        items=torch.tensor(items, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.float32),
        histories=torch.tensor(histories, dtype=torch.long),
        history_lengths=torch.tensor(lengths, dtype=torch.float32),
    )


def move_sequence_batch(batch: SequenceFeatureTensors, indices: torch.Tensor, device: torch.device) -> SequenceFeatureTensors:
    return SequenceFeatureTensors(
        users=batch.users[indices].to(device),
        items=batch.items[indices].to(device),
        labels=batch.labels[indices].to(device),
        histories=batch.histories[indices].to(device),
        history_lengths=batch.history_lengths[indices].to(device),
    )


def train_sequence_model(
    model: SequenceInterestModel,
    features: SequenceFeatureTensors,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[SequenceInterestModel, float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(args.seed + 77)
    n = len(features.labels)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, generator=rng)
        total_loss = 0.0
        batches = 0
        model.train()
        for start in range(0, n, args.batch_size):
            batch_indices = order[start : start + args.batch_size]
            batch = move_sequence_batch(features, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(batch), batch.labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        print(f"epoch={epoch} sequence_loss={total_loss / max(batches, 1):.6f}", flush=True)
    return model, time.perf_counter() - started


def build_eval_histories(data: PreparedData, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    histories: list[list[int]] = []
    lengths: list[int] = []
    for user in data.eval_user_indices:
        history = data.train_positive_by_user.get(user, [])[-max_len:]
        histories.append(history + [0] * (max_len - len(history)))
        lengths.append(len(history))
    return torch.tensor(histories, dtype=torch.long), torch.tensor(lengths, dtype=torch.float32)


def build_positive_edges(data: PreparedData, max_edges: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [(user, item) for user, history in data.train_positive_by_user.items() for item in history]
    rng = random.Random(seed)
    if max_edges and len(pairs) > max_edges:
        pairs = rng.sample(pairs, max_edges)
    users = torch.tensor([user for user, _ in pairs], dtype=torch.long)
    items = torch.tensor([item for _, item in pairs], dtype=torch.long)
    return users, items


def build_lightgcn_adj(num_users: int, num_items: int, users: torch.Tensor, items: torch.Tensor, device: torch.device) -> torch.Tensor:
    item_nodes = items + num_users
    rows = torch.cat([users, item_nodes])
    cols = torch.cat([item_nodes, users])
    node_count = num_users + num_items
    degree = torch.zeros(node_count, dtype=torch.float32)
    degree.scatter_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
    values = 1.0 / torch.sqrt(degree[rows].clamp_min(1) * degree[cols].clamp_min(1))
    indices = torch.stack([rows, cols], dim=0)
    return torch.sparse_coo_tensor(indices, values, (node_count, node_count)).coalesce().to(device)


def train_lightgcn(
    model: LightGCN,
    users: torch.Tensor,
    items: torch.Tensor,
    data: PreparedData,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[LightGCN, torch.Tensor, torch.Tensor, float]:
    users = users.to(device)
    items = items.to(device)
    adj = build_lightgcn_adj(len(data.index_to_user), len(data.index_to_item), users.cpu(), items.cpu(), device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator(device=device).manual_seed(args.seed + 88)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        user_vectors, item_vectors = model.propagate(adj)
        negative_items = torch.randint(0, len(data.index_to_item), (len(items),), generator=rng, device=device)
        pos_scores = (user_vectors[users] * item_vectors[items]).sum(dim=1)
        neg_scores = (user_vectors[users] * item_vectors[negative_items]).sum(dim=1)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch} lightgcn_bpr_loss={float(loss.detach().cpu()):.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        user_vectors, item_vectors = model.propagate(adj)
    return model, user_vectors.detach().cpu(), item_vectors.detach().cpu(), time.perf_counter() - started


def train_inbatch_tower(
    model: KuaiRecTwoTower,
    features: FeatureTensors,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[KuaiRecTwoTower, float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = torch.Generator().manual_seed(args.seed + 66)
    n = len(features.labels)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, generator=rng)
        total_loss = 0.0
        batches = 0
        model.train()
        for start in range(0, n, args.batch_size):
            batch_indices = order[start : start + args.batch_size]
            if len(batch_indices) < 2:
                continue
            batch = move_batch(features, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            user_vectors = model.encode_user(batch.users, batch.dense)
            item_vectors = model.encode_item(batch.items, batch.categories, batch.text_indices, batch.text_lengths, batch.dense)
            logits = user_vectors @ item_vectors.T / model.temperature
            loss = F.cross_entropy(logits, torch.arange(len(batch_indices), dtype=torch.long, device=device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        print(f"epoch={epoch} text_inbatch_loss={total_loss / max(batches, 1):.6f}", flush=True)
    return model, time.perf_counter() - started


def write_upgrade_outputs(
    metric: ExperimentMetric,
    args: argparse.Namespace,
    title: str,
    description: str,
) -> None:
    output_dir = args.outputs_dir / args.experiment
    report_dir = args.reports_dir / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_metric_csv(output_dir / "experiment_results.csv", [metric])
    write_single_report(report_dir / "experiment_report.md", title, [metric], description)


def run_distill(data: PreparedData, args: argparse.Namespace, device: torch.device) -> ExperimentMetric:
    train_features = build_distill_features(data, args)
    print(f"Distill train features ready: rows={len(train_features.labels):,}", flush=True)
    model = KuaiRecTwoTower(
        len(data.index_to_user),
        len(data.index_to_item),
        int(data.item_categories.max()),
        args.text_buckets,
        args.embedding_dim,
        args.tower_dim,
        args.hidden_dim,
        train_features.dense.shape[1],
        args.dropout,
        args.temperature,
    )
    model, _, train_seconds = train_neural_model(model, train_features, device, args.epochs, args.batch_size, args.lr, args.weight_decay, args.seed + 10, 1.0)
    started = time.perf_counter()
    scores = score_two_tower_matrix(model, data, data.eval_user_indices, device, args.score_batch_users, args.score_batch_items)
    metric = evaluate_topk(scores, data, "ItemCF-Distill-TwoTower", args.matrix.replace(".csv", ""), 20, len(train_features.labels), train_seconds, time.perf_counter() - started, str(device), "用 ItemCF TopK teacher 样本和真实行为样本蒸馏训练 Two-Tower。")
    return add_auc_logloss(metric, data.test_rows[: args.auc_rows], scores, data)


def run_text_encoder(data: PreparedData, args: argparse.Namespace, device: torch.device) -> ExperimentMetric:
    train_features = build_feature_tensors(data, data.train_rows, args.train_rows, args.seed + 11, positive_only=True)
    model = TextConvTwoTower(
        len(data.index_to_user),
        len(data.index_to_item),
        int(data.item_categories.max()),
        args.text_buckets,
        args.embedding_dim,
        args.tower_dim,
        args.hidden_dim,
        train_features.dense.shape[1],
        args.dropout,
        args.temperature,
    )
    model, train_seconds = train_inbatch_tower(model, train_features, device, args)
    started = time.perf_counter()
    scores = score_two_tower_matrix(model, data, data.eval_user_indices, device, args.score_batch_users, args.score_batch_items)
    metric = evaluate_topk(scores, data, "TextCNN-TwoTower", args.matrix.replace(".csv", ""), 20, len(train_features.labels), train_seconds, time.perf_counter() - started, str(device), "用 TextCNN 编码 caption 哈希 token，替代简单均值文本池化。")
    return add_auc_logloss(metric, data.test_rows[: args.auc_rows], scores, data)


def run_sequence(data: PreparedData, args: argparse.Namespace, device: torch.device) -> ExperimentMetric:
    features = build_sequence_features(data, data.train_rows, args.train_rows, args)
    model = SequenceInterestModel(len(data.index_to_item), args.embedding_dim, args.hidden_dim, args.temperature)
    model, train_seconds = train_sequence_model(model, features, device, args)
    histories, lengths = build_eval_histories(data, args.sequence_length)
    started = time.perf_counter()
    scores = model.score_matrix(histories, lengths, len(data.index_to_item), device, args.score_batch_users)
    metric = evaluate_topk(scores, data, "GRU-Sequence-Interest", args.matrix.replace(".csv", ""), 20, len(features.labels), train_seconds, time.perf_counter() - started, str(device), "用用户最近完播视频序列的 GRU 状态表示短期兴趣。")
    return add_auc_logloss(metric, data.test_rows[: args.auc_rows], scores, data)


def run_lightgcn(data: PreparedData, args: argparse.Namespace, device: torch.device) -> ExperimentMetric:
    users, items = build_positive_edges(data, args.train_rows, args.seed + 12)
    model = LightGCN(len(data.index_to_user), len(data.index_to_item), args.embedding_dim, args.lightgcn_layers)
    model, user_vectors, item_vectors, train_seconds = train_lightgcn(model, users, items, data, device, args)
    started = time.perf_counter()
    scores = user_vectors[data.eval_user_indices] @ item_vectors.T
    metric = evaluate_topk(scores, data, "LightGCN", args.matrix.replace(".csv", ""), 20, len(items), train_seconds, time.perf_counter() - started, str(device), "基于用户-视频正反馈二部图训练 LightGCN 图召回。")
    return add_auc_logloss(metric, data.test_rows[: args.auc_rows], scores, data)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)
    data = load_or_build_prepared_data(args)
    print(
        f"Loaded {args.matrix}: train={len(data.train_rows):,} "
        f"test={len(data.test_rows):,} eval_users={len(data.eval_user_indices):,}",
        flush=True,
    )

    if args.experiment == "distill_twotower":
        metric = run_distill(data, args, device)
        title = "KuaiRec big ItemCF 蒸馏 Two-Tower 报告"
        description = "用 ItemCF teacher 的强协同 TopK 信号补强 Two-Tower 召回。"
    elif args.experiment == "lightgcn":
        metric = run_lightgcn(data, args, device)
        title = "KuaiRec big LightGCN 图召回报告"
        description = "将用户-视频正反馈建成二部图，训练 LightGCN 召回 embedding。"
    elif args.experiment == "sequence_model":
        metric = run_sequence(data, args, device)
        title = "KuaiRec big 序列兴趣模型报告"
        description = "用用户最近完播视频序列建模短期兴趣。"
    else:
        metric = run_text_encoder(data, args, device)
        title = "KuaiRec big 轻量文本 Encoder 报告"
        description = "用 TextCNN 对 caption 哈希 token 做轻量编码。"

    write_upgrade_outputs(metric, args, title, description)
    print(
        f"{metric.model}: Recall@20={metric.recall:.6f} "
        f"NDCG@20={metric.ndcg:.6f} AUC={metric.auc:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
