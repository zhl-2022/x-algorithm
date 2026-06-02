"""Run the KuaiRec short-video recommendation experiment suite.

The first production-sized pass uses small_matrix.csv to keep iteration fast
while still covering millions of short-video watch events. Statistical
baselines use the full split. Neural models use a configurable training sample
so the same script can run locally for smoke tests and on srv4 MLU for the
main experiment.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = EXPERIMENT_DIR / "data" / "raw"
DEFAULT_REPORTS_DIR = EXPERIMENT_DIR / "reports"
DEFAULT_OUTPUTS_DIR = EXPERIMENT_DIR / "outputs"


@dataclass(frozen=True, slots=True)
class RawInteraction:
    user_id: str
    video_id: str
    timestamp: float
    watch_ratio: float
    play_duration: float
    video_duration: float
    label: int


@dataclass(frozen=True)
class ItemInfo:
    category: str
    categories: tuple[str, ...]
    caption: str
    category_name: str
    text_tokens: tuple[int, ...]
    text_length: int


@dataclass(frozen=True)
class PreparedData:
    source_path: Path
    train_rows: list[RawInteraction]
    valid_rows: list[RawInteraction]
    test_rows: list[RawInteraction]
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    index_to_user: list[str]
    index_to_item: list[str]
    item_info: dict[str, ItemInfo]
    item_categories: torch.Tensor
    item_text_indices: torch.Tensor
    item_text_lengths: torch.Tensor
    item_duration_norm: torch.Tensor
    item_popularity_norm: torch.Tensor
    item_avg_watch_norm: torch.Tensor
    category_ctr_norm: torch.Tensor
    user_positive_rate: torch.Tensor
    user_avg_watch_norm: torch.Tensor
    user_category_pref: torch.Tensor
    train_seen_by_user: dict[int, set[int]]
    train_positive_by_user: dict[int, list[int]]
    test_positive_by_user: dict[int, set[int]]
    eval_user_indices: list[int]
    positive_threshold: float


@dataclass(frozen=True)
class FeatureTensors:
    users: torch.Tensor
    items: torch.Tensor
    labels: torch.Tensor
    categories: torch.Tensor
    text_indices: torch.Tensor
    text_lengths: torch.Tensor
    dense: torch.Tensor


@dataclass(frozen=True)
class ExperimentMetric:
    model: str
    scope: str
    k: int
    train_rows: int
    eval_users: int
    recall: float
    hit_rate: float
    precision: float
    ndcg: float
    coverage: float
    auc: float
    logloss: float
    train_seconds: float
    eval_seconds: float
    device: str
    notes: str


class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor | FeatureTensors, items: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(users, FeatureTensors):
            items = users.items
            users = users.users
        if items is None:
            raise ValueError("items must be provided when users is not a FeatureTensors batch")
        user_vectors = self.user_embedding(users)
        item_vectors = self.item_embedding(items)
        return (
            (user_vectors * item_vectors).sum(dim=1)
            + self.user_bias(users).squeeze(1)
            + self.item_bias(items).squeeze(1)
            + self.global_bias
        )

    @torch.no_grad()
    def score_matrix(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embedding(user_indices)
        item_vectors = self.item_embedding(item_indices)
        scores = user_vectors @ item_vectors.T
        scores = scores + self.user_bias(user_indices)
        scores = scores + self.item_bias(item_indices).T
        scores = scores + self.global_bias
        return scores


class KuaiRecTwoTower(nn.Module):
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
        super().__init__()
        self.temperature = temperature
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories + 1, embedding_dim // 2, padding_idx=0)
        self.text_embedding = nn.Embedding(text_buckets + 1, embedding_dim, padding_idx=0)
        self.dense_projection = nn.Linear(dense_dim, embedding_dim)
        user_dense_dim = dense_dim - 3
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim + user_dense_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tower_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim * 3 + embedding_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tower_dim),
        )

    def _text_vector(self, text_indices: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        lengths = text_lengths.clamp_min(1).unsqueeze(1)
        return self.text_embedding(text_indices).sum(dim=1) / lengths

    def encode_user(self, users: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        user_vector = self.user_embedding(users)
        user_dense = dense[:, 3:]
        return F.normalize(self.user_tower(torch.cat([user_vector, user_dense], dim=1)), dim=1)

    def encode_item(
        self,
        items: torch.Tensor,
        categories: torch.Tensor,
        text_indices: torch.Tensor,
        text_lengths: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        item_vector = self.item_embedding(items)
        category_vector = self.category_embedding(categories)
        text_vector = self._text_vector(text_indices, text_lengths)
        dense_vector = self.dense_projection(dense)
        return F.normalize(
            self.item_tower(torch.cat([item_vector, category_vector, text_vector, dense_vector], dim=1)),
            dim=1,
        )

    def forward(self, batch: FeatureTensors) -> torch.Tensor:
        user_vectors = self.encode_user(batch.users, batch.dense)
        item_vectors = self.encode_item(
            batch.items,
            batch.categories,
            batch.text_indices,
            batch.text_lengths,
            batch.dense,
        )
        return (user_vectors * item_vectors).sum(dim=1) / self.temperature


class KuaiRecRanker(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_categories: int,
        text_buckets: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dense_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories + 1, embedding_dim // 2, padding_idx=0)
        self.text_embedding = nn.Embedding(text_buckets + 1, embedding_dim, padding_idx=0)
        input_dim = embedding_dim * 3 + embedding_dim // 2 + dense_dim
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: FeatureTensors) -> torch.Tensor:
        text_lengths = batch.text_lengths.clamp_min(1).unsqueeze(1)
        text_vector = self.text_embedding(batch.text_indices).sum(dim=1) / text_lengths
        features = torch.cat(
            [
                self.user_embedding(batch.users),
                self.item_embedding(batch.items),
                self.category_embedding(batch.categories),
                text_vector,
                batch.dense,
            ],
            dim=1,
        )
        return self.mlp(features).squeeze(1)


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_category_list(value: str) -> tuple[str, ...]:
    if not value:
        return ("unknown",)
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return (str(value),)
    if isinstance(parsed, list) and parsed:
        return tuple(str(item) for item in parsed)
    return ("unknown",)


def hash_text(text: str, buckets: int, max_tokens: int) -> tuple[tuple[int, ...], int]:
    text = (text or "").strip()
    tokens: list[str] = []
    for raw in text.replace("_", " ").split():
        raw = raw.strip()
        if raw:
            tokens.append(raw)
    if not tokens and text:
        tokens = [char for char in text if not char.isspace()]
    deduped: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        bucket = zlib.crc32(token.encode("utf-8")) % buckets + 1
        if bucket in seen:
            continue
        seen.add(bucket)
        deduped.append(bucket)
        if len(deduped) == max_tokens:
            break
    length = len(deduped)
    if len(deduped) < max_tokens:
        deduped.extend([0] * (max_tokens - len(deduped)))
    return tuple(deduped), length


def find_source_file(raw_dir: Path, matrix: str) -> Path:
    matches = sorted(raw_dir.rglob(matrix))
    if not matches:
        raise FileNotFoundError(f"Cannot find {matrix} under {raw_dir}")
    return matches[0]


def load_item_info(raw_dir: Path, text_buckets: int, max_text_tokens: int) -> dict[str, ItemInfo]:
    info: dict[str, ItemInfo] = {}
    category_files = sorted(raw_dir.rglob("item_categories.csv"))
    if category_files:
        with category_files[0].open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                categories = parse_category_list(row.get("feat", ""))
                info[row["video_id"]] = ItemInfo(
                    category=categories[0],
                    categories=categories,
                    caption="",
                    category_name=categories[0],
                    text_tokens=tuple([0] * max_text_tokens),
                    text_length=0,
                )

    caption_files = sorted(raw_dir.rglob("kuairec_caption_category.csv"))
    if caption_files:
        with caption_files[0].open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
            for row in csv.DictReader(file):
                video_id = row["video_id"]
                current = info.get(video_id)
                first_category = row.get("first_level_category_id") or (current.category if current else "unknown")
                category_name = row.get("first_level_category_name") or first_category
                caption = " ".join(
                    part
                    for part in [
                        row.get("manual_cover_text", ""),
                        row.get("caption", ""),
                        row.get("first_level_category_name", ""),
                        row.get("second_level_category_name", ""),
                        row.get("third_level_category_name", ""),
                    ]
                    if part and part != "UNKNOWN"
                )
                text_tokens, text_length = hash_text(caption, text_buckets, max_text_tokens)
                categories = current.categories if current else (first_category,)
                info[video_id] = ItemInfo(
                    category=first_category,
                    categories=categories,
                    caption=caption,
                    category_name=category_name,
                    text_tokens=text_tokens,
                    text_length=text_length,
                )
    return info


def load_interactions(path: Path, threshold: float, max_rows: int) -> list[RawInteraction]:
    rows: list[RawInteraction] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            if max_rows and index >= max_rows:
                break
            try:
                watch_ratio = float(row["watch_ratio"])
                timestamp = float(row["timestamp"])
                play_duration = float(row["play_duration"])
                video_duration = float(row["video_duration"])
            except (TypeError, ValueError):
                continue
            rows.append(
                RawInteraction(
                    user_id=row["user_id"],
                    video_id=row["video_id"],
                    timestamp=timestamp,
                    watch_ratio=watch_ratio,
                    play_duration=play_duration,
                    video_duration=video_duration,
                    label=1 if watch_ratio >= threshold else 0,
                )
            )
    return rows


def split_by_user(
    rows: list[RawInteraction],
    train_ratio: float,
    valid_ratio: float,
) -> tuple[list[RawInteraction], list[RawInteraction], list[RawInteraction]]:
    grouped: dict[str, list[RawInteraction]] = defaultdict(list)
    for row in rows:
        grouped[row.user_id].append(row)

    train_rows: list[RawInteraction] = []
    valid_rows: list[RawInteraction] = []
    test_rows: list[RawInteraction] = []
    for user_rows in grouped.values():
        user_rows.sort(key=lambda item: item.timestamp)
        n = len(user_rows)
        if n < 5:
            train_rows.extend(user_rows)
            continue
        train_end = max(1, int(n * train_ratio))
        valid_end = max(train_end + 1, int(n * (train_ratio + valid_ratio)))
        valid_end = min(valid_end, n - 1)
        train_rows.extend(user_rows[:train_end])
        valid_rows.extend(user_rows[train_end:valid_end])
        test_rows.extend(user_rows[valid_end:])
    return train_rows, valid_rows, test_rows


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


def build_prepared_data(args: argparse.Namespace) -> PreparedData:
    source_path = find_source_file(args.raw_dir, args.matrix)
    item_info = load_item_info(args.raw_dir, args.text_buckets, args.max_text_tokens)
    rows = load_interactions(source_path, args.positive_threshold, args.max_rows)
    train_rows, valid_rows, test_rows = split_by_user(rows, args.train_ratio, args.valid_ratio)

    user_ids = sorted({row.user_id for row in rows})
    item_ids = sorted({row.video_id for row in rows})
    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}

    category_values = sorted({item.category for item in item_info.values()} | {"unknown"})
    category_to_index = {category: index + 1 for index, category in enumerate(category_values)}
    num_items = len(item_ids)
    num_users = len(user_ids)
    num_categories = len(category_to_index)

    item_positive_counts = torch.zeros(num_items, dtype=torch.float32)
    item_exposure_counts = torch.zeros(num_items, dtype=torch.float32)
    item_watch_sum = torch.zeros(num_items, dtype=torch.float32)
    item_duration_sum = torch.zeros(num_items, dtype=torch.float32)
    category_positive_counts = torch.zeros(num_categories + 1, dtype=torch.float32)
    category_exposure_counts = torch.zeros(num_categories + 1, dtype=torch.float32)
    user_positive_counts = torch.zeros(num_users, dtype=torch.float32)
    user_exposure_counts = torch.zeros(num_users, dtype=torch.float32)
    user_watch_sum = torch.zeros(num_users, dtype=torch.float32)
    user_category_positive = torch.zeros((num_users, num_categories + 1), dtype=torch.float32)

    item_categories = torch.zeros(num_items, dtype=torch.long)
    item_text_indices = torch.zeros((num_items, args.max_text_tokens), dtype=torch.long)
    item_text_lengths = torch.zeros(num_items, dtype=torch.float32)
    for item_id, item_index in item_to_index.items():
        meta = item_info.get(item_id)
        category = meta.category if meta else "unknown"
        category_index = category_to_index.get(category, category_to_index["unknown"])
        item_categories[item_index] = category_index
        if meta:
            item_text_indices[item_index] = torch.tensor(meta.text_tokens, dtype=torch.long)
            item_text_lengths[item_index] = meta.text_length

    train_seen_by_user: dict[int, set[int]] = defaultdict(set)
    train_positive_by_user: dict[int, list[int]] = defaultdict(list)
    test_positive_by_user: dict[int, set[int]] = defaultdict(set)

    for row in train_rows:
        user_index = user_to_index[row.user_id]
        item_index = item_to_index[row.video_id]
        category_index = int(item_categories[item_index])
        train_seen_by_user[user_index].add(item_index)
        item_exposure_counts[item_index] += 1
        item_watch_sum[item_index] += row.watch_ratio
        item_duration_sum[item_index] += row.video_duration
        category_exposure_counts[category_index] += 1
        user_exposure_counts[user_index] += 1
        user_watch_sum[user_index] += row.watch_ratio
        if row.label:
            item_positive_counts[item_index] += 1
            category_positive_counts[category_index] += 1
            user_positive_counts[user_index] += 1
            user_category_positive[user_index, category_index] += 1
            train_positive_by_user[user_index].append(item_index)

    for row in test_rows:
        if row.label:
            test_positive_by_user[user_to_index[row.user_id]].add(item_to_index[row.video_id])

    eval_user_indices = sorted(user for user, positives in test_positive_by_user.items() if positives)
    max_popularity = float(torch.log1p(item_positive_counts).max())
    item_popularity_norm = torch.log1p(item_positive_counts) / max_popularity if max_popularity else item_positive_counts
    item_avg_watch = item_watch_sum / item_exposure_counts.clamp_min(1)
    max_item_watch = float(item_avg_watch.max())
    item_avg_watch_norm = item_avg_watch / max_item_watch if max_item_watch else item_avg_watch
    item_avg_duration = item_duration_sum / item_exposure_counts.clamp_min(1)
    max_duration = float(item_avg_duration.max())
    item_duration_norm = item_avg_duration / max_duration if max_duration else item_avg_duration
    category_ctr = category_positive_counts / category_exposure_counts.clamp_min(1)
    max_category_ctr = float(category_ctr.max())
    category_ctr_norm = category_ctr / max_category_ctr if max_category_ctr else category_ctr
    user_positive_rate = user_positive_counts / user_exposure_counts.clamp_min(1)
    user_avg_watch = user_watch_sum / user_exposure_counts.clamp_min(1)
    max_user_watch = float(user_avg_watch.max())
    user_avg_watch_norm = user_avg_watch / max_user_watch if max_user_watch else user_avg_watch
    user_category_pref = user_category_positive / user_positive_counts.clamp_min(1).unsqueeze(1)

    return PreparedData(
        source_path=source_path,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_user=user_ids,
        index_to_item=item_ids,
        item_info=item_info,
        item_categories=item_categories,
        item_text_indices=item_text_indices,
        item_text_lengths=item_text_lengths,
        item_duration_norm=item_duration_norm,
        item_popularity_norm=item_popularity_norm,
        item_avg_watch_norm=item_avg_watch_norm,
        category_ctr_norm=category_ctr_norm,
        user_positive_rate=user_positive_rate,
        user_avg_watch_norm=user_avg_watch_norm,
        user_category_pref=user_category_pref,
        train_seen_by_user=train_seen_by_user,
        train_positive_by_user=train_positive_by_user,
        test_positive_by_user=test_positive_by_user,
        eval_user_indices=eval_user_indices,
        positive_threshold=args.positive_threshold,
    )


def dense_for_pairs(data: PreparedData, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
    categories = data.item_categories[items]
    category_pref = data.user_category_pref[users, categories]
    category_ctr = data.category_ctr_norm[categories]
    return torch.stack(
        [
            data.item_popularity_norm[items],
            data.item_avg_watch_norm[items],
            data.item_duration_norm[items],
            category_pref,
            data.user_positive_rate[users],
            data.user_avg_watch_norm[users],
            category_ctr,
        ],
        dim=1,
    ).float()


def build_feature_tensors(
    data: PreparedData,
    rows: list[RawInteraction],
    max_rows: int,
    seed: int,
) -> FeatureTensors:
    if max_rows and len(rows) > max_rows:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_rows)

    users = torch.tensor([data.user_to_index[row.user_id] for row in rows], dtype=torch.long)
    items = torch.tensor([data.item_to_index[row.video_id] for row in rows], dtype=torch.long)
    labels = torch.tensor([row.label for row in rows], dtype=torch.float32)
    categories = data.item_categories[items]
    text_indices = data.item_text_indices[items]
    text_lengths = data.item_text_lengths[items]
    dense = dense_for_pairs(data, users, items)
    return FeatureTensors(users, items, labels, categories, text_indices, text_lengths, dense)


def move_batch(batch: FeatureTensors, indices: torch.Tensor, device: torch.device) -> FeatureTensors:
    return FeatureTensors(
        users=batch.users[indices].to(device),
        items=batch.items[indices].to(device),
        labels=batch.labels[indices].to(device),
        categories=batch.categories[indices].to(device),
        text_indices=batch.text_indices[indices].to(device),
        text_lengths=batch.text_lengths[indices].to(device),
        dense=batch.dense[indices].to(device),
    )


def train_neural_model(
    model: nn.Module,
    train_data: FeatureTensors,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    positive_weight: float,
) -> tuple[nn.Module, float, float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = torch.Generator().manual_seed(seed)
    started = time.perf_counter()
    final_loss = 0.0
    n = len(train_data.labels)

    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=rng)
        total_loss = 0.0
        batches = 0
        model.train()
        for start in range(0, n, batch_size):
            batch_indices = order[start : start + batch_size]
            batch = move_batch(train_data, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            if positive_weight == 1.0:
                loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
            else:
                weights = torch.where(
                    batch.labels > 0.5,
                    torch.full_like(batch.labels, positive_weight),
                    torch.ones_like(batch.labels),
                )
                loss = F.binary_cross_entropy_with_logits(logits, batch.labels, weight=weights)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        final_loss = total_loss / batches if batches else 0.0
        print(f"epoch={epoch} loss={final_loss:.6f}", flush=True)
    return model, final_loss, time.perf_counter() - started


def score_mf_matrix(
    model: MatrixFactorization,
    data: PreparedData,
    eval_users: list[int],
    device: torch.device,
    batch_users: int,
) -> torch.Tensor:
    model.eval()
    item_indices = torch.arange(len(data.index_to_item), dtype=torch.long, device=device)
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(eval_users), batch_users):
            users = torch.tensor(eval_users[start : start + batch_users], dtype=torch.long, device=device)
            chunks.append(model.score_matrix(users, item_indices).detach().cpu())
    return torch.cat(chunks, dim=0)


def score_two_tower_matrix(
    model: KuaiRecTwoTower,
    data: PreparedData,
    eval_users: list[int],
    device: torch.device,
    batch_users: int,
    batch_items: int,
) -> torch.Tensor:
    model.eval()
    all_items = torch.arange(len(data.index_to_item), dtype=torch.long)
    item_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(all_items), batch_items):
            items = all_items[start : start + batch_items]
            dense = dense_for_pairs(data, torch.zeros_like(items), items).to(device)
            item_vectors = model.encode_item(
                items.to(device),
                data.item_categories[items].to(device),
                data.item_text_indices[items].to(device),
                data.item_text_lengths[items].to(device),
                dense,
            )
            item_chunks.append(item_vectors.detach().cpu())
        item_vectors_cpu = torch.cat(item_chunks, dim=0).to(device)

        rows: list[torch.Tensor] = []
        for start in range(0, len(eval_users), batch_users):
            user_indices = torch.tensor(eval_users[start : start + batch_users], dtype=torch.long)
            dummy_items = torch.zeros(len(user_indices), dtype=torch.long)
            dense = dense_for_pairs(data, user_indices, dummy_items).to(device)
            user_vectors = model.encode_user(user_indices.to(device), dense)
            rows.append((user_vectors @ item_vectors_cpu.T).detach().cpu())
    return torch.cat(rows, dim=0)


def score_ranker_matrix(
    model: KuaiRecRanker,
    data: PreparedData,
    eval_users: list[int],
    device: torch.device,
    batch_users: int,
) -> torch.Tensor:
    model.eval()
    num_items = len(data.index_to_item)
    item_indices = torch.arange(num_items, dtype=torch.long)
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(eval_users), batch_users):
            users = torch.tensor(eval_users[start : start + batch_users], dtype=torch.long)
            repeated_users = users.repeat_interleave(num_items)
            tiled_items = item_indices.repeat(len(users))
            batch = FeatureTensors(
                users=repeated_users.to(device),
                items=tiled_items.to(device),
                labels=torch.zeros(len(repeated_users), dtype=torch.float32, device=device),
                categories=data.item_categories[tiled_items].to(device),
                text_indices=data.item_text_indices[tiled_items].to(device),
                text_lengths=data.item_text_lengths[tiled_items].to(device),
                dense=dense_for_pairs(data, repeated_users, tiled_items).to(device),
            )
            scores = model(batch).detach().cpu().reshape(len(users), num_items)
            rows.append(scores)
    return torch.cat(rows, dim=0)


def dcg_at_k(recommended: list[int], positives: set[int], k: int) -> float:
    value = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in positives:
            value += 1.0 / math.log2(rank + 1)
    return value


def ideal_dcg(num_positive: int, k: int) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, min(num_positive, k) + 1))


def evaluate_topk(
    score_matrix: torch.Tensor,
    data: PreparedData,
    model: str,
    scope: str,
    k: int,
    train_rows: int,
    train_seconds: float,
    eval_seconds: float,
    device: str,
    notes: str,
) -> ExperimentMetric:
    total_recall = 0.0
    total_hit = 0.0
    total_precision = 0.0
    total_ndcg = 0.0
    recommended_items: set[int] = set()
    evaluated = 0

    for row_index, user_index in enumerate(data.eval_user_indices):
        positives = data.test_positive_by_user[user_index]
        if not positives:
            continue
        scores = score_matrix[row_index].clone()
        for seen_item in data.train_seen_by_user.get(user_index, set()):
            scores[seen_item] = -float("inf")
        top_k = min(k, len(scores))
        recommended = torch.topk(scores, k=top_k).indices.tolist()
        hits = len(set(recommended) & positives)
        total_recall += hits / len(positives)
        total_hit += 1.0 if hits else 0.0
        total_precision += hits / k
        idcg = ideal_dcg(len(positives), k)
        total_ndcg += dcg_at_k(recommended, positives, k) / idcg if idcg else 0.0
        recommended_items.update(recommended[:k])
        evaluated += 1

    if evaluated == 0:
        raise ValueError("No evaluable users with positive test rows.")

    return ExperimentMetric(
        model=model,
        scope=scope,
        k=k,
        train_rows=train_rows,
        eval_users=evaluated,
        recall=total_recall / evaluated,
        hit_rate=total_hit / evaluated,
        precision=total_precision / evaluated,
        ndcg=total_ndcg / evaluated,
        coverage=len(recommended_items) / len(data.index_to_item),
        auc=0.0,
        logloss=0.0,
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        device=device,
        notes=notes,
    )


def auc_score(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        rank_sum += avg_rank * sum(label for _, label in ranked[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def logloss_score(labels: list[int], logits: list[float]) -> float:
    if not labels:
        return 0.0
    total = 0.0
    for label, logit in zip(labels, logits):
        if logit >= 0:
            total += (1 - label) * logit + math.log1p(math.exp(-logit))
        else:
            total += -label * logit + math.log1p(math.exp(logit))
    return total / len(labels)


def add_auc_logloss(metric: ExperimentMetric, rows: list[RawInteraction], score_matrix: torch.Tensor, data: PreparedData) -> ExperimentMetric:
    user_position = {user_index: position for position, user_index in enumerate(data.eval_user_indices)}
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        user_index = data.user_to_index[row.user_id]
        item_index = data.item_to_index[row.video_id]
        position = user_position.get(user_index)
        if position is None:
            continue
        score = float(score_matrix[position, item_index])
        if not math.isfinite(score):
            continue
        labels.append(row.label)
        scores.append(score)
    return ExperimentMetric(
        model=metric.model,
        scope=metric.scope,
        k=metric.k,
        train_rows=metric.train_rows,
        eval_users=metric.eval_users,
        recall=metric.recall,
        hit_rate=metric.hit_rate,
        precision=metric.precision,
        ndcg=metric.ndcg,
        coverage=metric.coverage,
        auc=auc_score(labels, scores),
        logloss=logloss_score(labels, scores),
        train_seconds=metric.train_seconds,
        eval_seconds=metric.eval_seconds,
        device=metric.device,
        notes=metric.notes,
    )


def score_popularity_matrix(data: PreparedData) -> torch.Tensor:
    return data.item_popularity_norm.unsqueeze(0).repeat(len(data.eval_user_indices), 1)


def score_category_matrix(data: PreparedData) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    item_indices = torch.arange(len(data.index_to_item), dtype=torch.long)
    categories = data.item_categories[item_indices]
    base = data.item_popularity_norm * 0.45 + data.item_avg_watch_norm * 0.10 + data.category_ctr_norm[categories] * 0.10
    for user_index in data.eval_user_indices:
        category_pref = data.user_category_pref[user_index, categories]
        rows.append(base + category_pref * 0.35)
    return torch.stack(rows, dim=0)


def score_itemcf_matrix(data: PreparedData, max_history: int, max_neighbors: int) -> torch.Tensor:
    item_positive_counts = Counter()
    co_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for history in data.train_positive_by_user.values():
        unique_history = list(dict.fromkeys(history[-max_history:]))
        for item in unique_history:
            item_positive_counts[item] += 1
        for i, item_i in enumerate(unique_history):
            for item_j in unique_history[i + 1 :]:
                co_counts[item_i][item_j] += 1
                co_counts[item_j][item_i] += 1

    neighbors: dict[int, list[tuple[int, float]]] = {}
    for item, counter in co_counts.items():
        sims = []
        for other, co_count in counter.items():
            denom = math.sqrt(item_positive_counts[item] * item_positive_counts[other])
            if denom:
                sims.append((other, co_count / denom))
        sims.sort(key=lambda pair: (-pair[1], pair[0]))
        neighbors[item] = sims[:max_neighbors]

    rows: list[torch.Tensor] = []
    for user_index in data.eval_user_indices:
        scores = data.item_popularity_norm.clone() * 0.01
        for item in data.train_positive_by_user.get(user_index, [])[-max_history:]:
            for neighbor, sim in neighbors.get(item, []):
                scores[neighbor] += sim
        rows.append(scores)
    return torch.stack(rows, dim=0)


def normalize_score_rows(score_matrix: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(score_matrix)
    safe_scores = torch.where(finite, score_matrix, torch.zeros_like(score_matrix))
    counts = finite.sum(dim=1, keepdim=True).clamp_min(1)
    means = safe_scores.sum(dim=1, keepdim=True) / counts
    centered = torch.where(finite, score_matrix - means, torch.zeros_like(score_matrix))
    variances = (centered * centered).sum(dim=1, keepdim=True) / counts
    std = variances.sqrt().clamp_min(1e-6)
    return torch.where(finite, centered / std, torch.full_like(score_matrix, -float("inf")))


def evaluate_pipeline(
    two_tower_scores: torch.Tensor,
    ranker_scores: torch.Tensor,
    data: PreparedData,
    candidate_k: int,
    blend_alpha: float,
    k: int,
    train_rows: int,
    train_seconds: float,
    eval_seconds: float,
    device: str,
    scope: str,
) -> tuple[ExperimentMetric, torch.Tensor]:
    normalized_two_tower = normalize_score_rows(two_tower_scores)
    normalized_ranker = normalize_score_rows(ranker_scores)
    pipeline_scores = torch.full_like(ranker_scores, -float("inf"))
    for row_index, user_index in enumerate(data.eval_user_indices):
        scores = two_tower_scores[row_index].clone()
        for seen_item in data.train_seen_by_user.get(user_index, set()):
            scores[seen_item] = -float("inf")
        top_candidates = torch.topk(scores, k=min(candidate_k, len(scores))).indices
        blended_scores = (
            blend_alpha * normalized_ranker[row_index, top_candidates]
            + (1.0 - blend_alpha) * normalized_two_tower[row_index, top_candidates]
        )
        pipeline_scores[row_index, top_candidates] = blended_scores
    model_name = (
        f"TwoTower+DNN-Rerank@{candidate_k}"
        if blend_alpha == 1.0
        else f"TwoTower+DNN-Blend@{candidate_k}a{blend_alpha:g}"
    )
    return (
        evaluate_topk(
            score_matrix=pipeline_scores,
            data=data,
            model=model_name,
            scope=scope,
            k=k,
            train_rows=train_rows,
            train_seconds=train_seconds,
            eval_seconds=eval_seconds,
            device=device,
            notes=(
                f"Two-Tower 先取 Top{candidate_k} 候选，再按 "
                f"{blend_alpha:g}*Ranker + {1.0 - blend_alpha:g}*TwoTower 融合重排 Top{k}。"
            ),
        ),
        pipeline_scores,
    )


def write_metric_csv(path: Path, metrics: list[ExperimentMetric]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "scope",
        "k",
        "train_rows",
        "eval_users",
        "recall",
        "hit_rate",
        "precision",
        "ndcg",
        "coverage",
        "auc",
        "logloss",
        "train_seconds",
        "eval_seconds",
        "device",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {
                    "model": metric.model,
                    "scope": metric.scope,
                    "k": metric.k,
                    "train_rows": metric.train_rows,
                    "eval_users": metric.eval_users,
                    "recall": f"{metric.recall:.8f}",
                    "hit_rate": f"{metric.hit_rate:.8f}",
                    "precision": f"{metric.precision:.8f}",
                    "ndcg": f"{metric.ndcg:.8f}",
                    "coverage": f"{metric.coverage:.8f}",
                    "auc": format_csv_float(metric.auc, 8),
                    "logloss": format_csv_float(metric.logloss, 8),
                    "train_seconds": f"{metric.train_seconds:.4f}",
                    "eval_seconds": f"{metric.eval_seconds:.4f}",
                    "device": metric.device,
                    "notes": metric.notes,
                }
            )


def format_csv_float(value: float, digits: int) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def format_metric(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.6f}"


def write_single_report(path: Path, title: str, metrics: list[ExperimentMetric], description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "| 模型 | Scope | 训练样本 | 评估用户 | Recall@20 | HitRate@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 评估秒 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        lines.append(
            "| "
            f"{metric.model} | {metric.scope} | {metric.train_rows:,} | {metric.eval_users:,} | "
            f"{format_metric(metric.recall)} | {format_metric(metric.hit_rate)} | "
            f"{format_metric(metric.ndcg)} | {format_metric(metric.coverage)} | "
            f"{format_metric(metric.auc)} | {format_metric(metric.logloss)} | "
            f"{metric.train_seconds:.2f} | {metric.eval_seconds:.2f} |"
        )
    lines.extend(["", "## 说明", ""])
    for metric in metrics:
        lines.append(f"- {metric.model}：{metric.notes}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all_report(path: Path, data: PreparedData, metrics: list[ExperimentMetric], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positives_train = sum(row.label for row in data.train_rows)
    positives_test = sum(row.label for row in data.test_rows)
    best = max(metrics, key=lambda metric: metric.ndcg)
    neural_train_rows = next((metric.train_rows for metric in metrics if metric.model == "MF"), args.neural_train_rows)
    lines = [
        "# KuaiRec 全套训练报告",
        "",
        "## 实验范围",
        "",
        "本报告汇总 KuaiRec 短视频推荐训练。当前跑通 Popularity、Category、ItemCF、MF、",
        "Two-Tower、DNN Ranker 和两阶段 pipeline，并支持标签阈值、训练规模、候选集和融合权重消融。",
        "",
        "## 数据切分",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 数据源 | `{data.source_path.name}` |",
        f"| 用户数 | {len(data.index_to_user):,} |",
        f"| 视频数 | {len(data.index_to_item):,} |",
        f"| 训练交互 | {len(data.train_rows):,} |",
        f"| 验证交互 | {len(data.valid_rows):,} |",
        f"| 测试交互 | {len(data.test_rows):,} |",
        f"| 训练正样本 | {positives_train:,} |",
        f"| 测试正样本 | {positives_test:,} |",
        f"| 正反馈阈值 | `watch_ratio >= {args.positive_threshold}` |",
        f"| 神经模型训练样本 | {neural_train_rows:,} |",
        f"| Epochs | {args.epochs} |",
        f"| Batch size | {args.batch_size:,} |",
        f"| Ranker 正样本权重 | {args.ranker_positive_weight:g} |",
        f"| 融合重排 alpha | `{','.join(str(item) for item in args.rerank_blend_alphas)}` |",
        "",
        "## 结果汇总",
        "",
        "| 模型 | Recall@20 | HitRate@20 | Precision@20 | NDCG@20 | Coverage@20 | AUC | LogLoss | 训练秒 | 设备 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for metric in metrics:
        lines.append(
            "| "
            f"{metric.model} | {format_metric(metric.recall)} | {format_metric(metric.hit_rate)} | "
            f"{format_metric(metric.precision)} | {format_metric(metric.ndcg)} | "
            f"{format_metric(metric.coverage)} | {format_metric(metric.auc)} | "
            f"{format_metric(metric.logloss)} | {metric.train_seconds:.2f} | `{metric.device}` |"
        )
    lines.extend(
        [
            "",
            "## 当前结论",
            "",
            f"- 当前 `NDCG@20` 最好的模型是 `{best.model}`，数值为 `{best.ndcg:.6f}`。",
            "- Popularity 是最低基线，只利用视频全局完播热度。",
            "- Category baseline 开始利用用户历史类别偏好，适合解释短视频兴趣迁移。",
            "- ItemCF 利用用户共同完播视频构造 item-item 相似度，是第一版个性化召回。",
            "- MF、Two-Tower 和 DNN Ranker 已完成 MLU 训练链路验证。",
            "- 两阶段 pipeline 模拟企业推荐的召回后排序流程，本轮支持 Ranker-only 与 Two-Tower/Ranker 融合重排。",
            "- Pipeline 只对召回候选打重排分；候选外视频没有概率分数，因此候选外样本不参与 `AUC` 和 `LogLoss` 计算。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(data: PreparedData, metrics: list[ExperimentMetric], args: argparse.Namespace) -> None:
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    write_metric_csv(args.outputs_dir / "experiment_results.csv", metrics)
    for model_key, filename, title, description in [
        ("Popularity", "popularity", "KuaiRec Popularity Baseline 报告", "按视频全局正反馈热度排序。"),
        ("Category", "category", "KuaiRec Category Baseline 报告", "融合视频热度、类别 CTR 和用户历史类别偏好。"),
        ("ItemCF", "itemcf", "KuaiRec ItemCF 报告", "基于用户共同完播视频计算物品相似度。"),
        ("MF", "mf", "KuaiRec Matrix Factorization 报告", "使用用户和视频 embedding 的点积预测完播正反馈。"),
        ("Two-Tower", "two_tower", "KuaiRec Two-Tower 报告", "用户塔和视频塔分别编码后点积召回。"),
        ("DNNRanker", "ranker", "KuaiRec DNN Ranker 报告", "融合 ID、类别、文本哈希和统计特征预测正反馈概率。"),
    ]:
        selected = [metric for metric in metrics if metric.model == model_key]
        if selected:
            write_metric_csv(args.outputs_dir / f"{filename}_results.csv", selected)
            write_single_report(args.reports_dir / f"{filename}_report.md", title, selected, description)
    pipeline_metrics = [metric for metric in metrics if metric.model.startswith("TwoTower+DNN-")]
    if pipeline_metrics:
        write_metric_csv(args.outputs_dir / "pipeline_results.csv", pipeline_metrics)
        write_metric_csv(args.outputs_dir / "candidate_ablation_results.csv", pipeline_metrics)
        write_single_report(
            args.reports_dir / "pipeline_report.md",
            "KuaiRec Two-Tower + DNN Ranker Pipeline 报告",
            pipeline_metrics,
            "先用 Two-Tower 召回候选，再用 DNN Ranker 重排。",
        )
        write_single_report(
            args.reports_dir / "candidate_ablation_report.md",
            "KuaiRec Candidate K 消融报告",
            pipeline_metrics,
            "比较不同 `candidate_k` 对最终 Top20 结果的影响。",
        )
    write_all_report(args.reports_dir / "all_experiments_report.md", data, metrics, args)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--matrix", choices=["small_matrix.csv", "big_matrix.csv"], default="small_matrix.csv")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for smoke tests. 0 means full matrix.")
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--candidate-ks", type=parse_int_list, default=(50, 100, 200))
    parser.add_argument(
        "--rerank-blend-alphas",
        type=parse_float_list,
        default=(1.0,),
        help="Comma-separated alpha values for alpha*Ranker + (1-alpha)*TwoTower rerank scores.",
    )
    parser.add_argument("--itemcf-history", type=int, default=80)
    parser.add_argument("--itemcf-neighbors", type=int, default=100)
    parser.add_argument("--neural-train-rows", type=int, default=1_200_000)
    parser.add_argument("--auc-rows", type=int, default=300_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--score-batch-users", type=int, default=64)
    parser.add_argument("--score-batch-items", type=int, default=4096)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--tower-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ranker-hidden-dims", type=str, default="128,64")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--ranker-positive-weight", type=float, default=1.0)
    parser.add_argument("--text-buckets", type=int, default=8192)
    parser.add_argument("--max-text-tokens", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mlu.")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}", flush=True)

    data = build_prepared_data(args)
    print(
        f"Loaded {args.matrix}: train={len(data.train_rows):,} "
        f"valid={len(data.valid_rows):,} test={len(data.test_rows):,} "
        f"eval_users={len(data.eval_user_indices):,}",
        flush=True,
    )
    scope_name = args.matrix.replace(".csv", "")

    metrics: list[ExperimentMetric] = []

    start = time.perf_counter()
    popularity_scores = score_popularity_matrix(data)
    popularity_metric = evaluate_topk(
        popularity_scores,
        data,
        "Popularity",
        scope_name,
        args.k,
        len(data.train_rows),
        time.perf_counter() - start,
        0.0,
        "cpu",
        "视频全局完播正反馈热度。",
    )
    metrics.append(add_auc_logloss(popularity_metric, data.test_rows[: args.auc_rows], popularity_scores, data))

    start = time.perf_counter()
    category_scores = score_category_matrix(data)
    category_metric = evaluate_topk(
        category_scores,
        data,
        "Category",
        scope_name,
        args.k,
        len(data.train_rows),
        time.perf_counter() - start,
        0.0,
        "cpu",
        "视频热度 + 类别 CTR + 用户历史类别偏好。",
    )
    metrics.append(add_auc_logloss(category_metric, data.test_rows[: args.auc_rows], category_scores, data))

    start = time.perf_counter()
    itemcf_scores = score_itemcf_matrix(data, args.itemcf_history, args.itemcf_neighbors)
    itemcf_metric = evaluate_topk(
        itemcf_scores,
        data,
        "ItemCF",
        scope_name,
        args.k,
        len(data.train_rows),
        time.perf_counter() - start,
        0.0,
        "cpu",
        "基于共同完播视频的余弦归一化共现相似度。",
    )
    metrics.append(add_auc_logloss(itemcf_metric, data.test_rows[: args.auc_rows], itemcf_scores, data))

    train_features = build_feature_tensors(data, data.train_rows, args.neural_train_rows, args.seed)
    print(f"Neural train rows: {len(train_features.labels):,}", flush=True)

    mf_model = MatrixFactorization(len(data.index_to_user), len(data.index_to_item), args.embedding_dim)
    mf_model, _, mf_train_seconds = train_neural_model(
        mf_model,
        train_features,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed + 1,
        1.0,
    )
    start = time.perf_counter()
    mf_scores = score_mf_matrix(mf_model, data, data.eval_user_indices, device, args.score_batch_users)
    mf_eval_seconds = time.perf_counter() - start
    mf_metric = evaluate_topk(
        mf_scores,
        data,
        "MF",
        f"{scope_name}_sample",
        args.k,
        len(train_features.labels),
        mf_train_seconds,
        mf_eval_seconds,
        str(device),
        "用户和视频 embedding 点积 + bias，使用 BCE 学习完播正反馈。",
    )
    metrics.append(add_auc_logloss(mf_metric, data.test_rows[: args.auc_rows], mf_scores, data))

    two_tower_model = KuaiRecTwoTower(
        num_users=len(data.index_to_user),
        num_items=len(data.index_to_item),
        num_categories=int(data.item_categories.max()),
        text_buckets=args.text_buckets,
        embedding_dim=args.embedding_dim,
        tower_dim=args.tower_dim,
        hidden_dim=args.hidden_dim,
        dense_dim=train_features.dense.shape[1],
        dropout=args.dropout,
        temperature=args.temperature,
    )
    two_tower_model, _, two_tower_train_seconds = train_neural_model(
        two_tower_model,
        train_features,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed + 2,
        1.0,
    )
    start = time.perf_counter()
    two_tower_scores = score_two_tower_matrix(
        two_tower_model,
        data,
        data.eval_user_indices,
        device,
        args.score_batch_users,
        args.score_batch_items,
    )
    two_tower_eval_seconds = time.perf_counter() - start
    two_tower_metric = evaluate_topk(
        two_tower_scores,
        data,
        "Two-Tower",
        f"{scope_name}_sample",
        args.k,
        len(train_features.labels),
        two_tower_train_seconds,
        two_tower_eval_seconds,
        str(device),
        "用户塔和视频塔分别编码，视频塔融合类别、caption 哈希文本和统计特征。",
    )
    metrics.append(add_auc_logloss(two_tower_metric, data.test_rows[: args.auc_rows], two_tower_scores, data))

    hidden_dims = tuple(int(part.strip()) for part in args.ranker_hidden_dims.split(",") if part.strip())
    ranker_model = KuaiRecRanker(
        num_users=len(data.index_to_user),
        num_items=len(data.index_to_item),
        num_categories=int(data.item_categories.max()),
        text_buckets=args.text_buckets,
        embedding_dim=args.embedding_dim,
        hidden_dims=hidden_dims,
        dense_dim=train_features.dense.shape[1],
        dropout=args.dropout,
    )
    ranker_model, _, ranker_train_seconds = train_neural_model(
        ranker_model,
        train_features,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
        args.weight_decay,
        args.seed + 3,
        args.ranker_positive_weight,
    )
    start = time.perf_counter()
    ranker_scores = score_ranker_matrix(ranker_model, data, data.eval_user_indices, device, args.score_batch_users)
    ranker_eval_seconds = time.perf_counter() - start
    ranker_metric = evaluate_topk(
        ranker_scores,
        data,
        "DNNRanker",
        f"{scope_name}_sample",
        args.k,
        len(train_features.labels),
        ranker_train_seconds,
        ranker_eval_seconds,
        str(device),
        "融合用户、视频、类别、caption 哈希文本和统计特征的 MLP 排序模型。",
    )
    metrics.append(add_auc_logloss(ranker_metric, data.test_rows[: args.auc_rows], ranker_scores, data))

    for candidate_k in args.candidate_ks:
        for blend_alpha in args.rerank_blend_alphas:
            pipeline_metric, pipeline_scores = evaluate_pipeline(
                two_tower_scores,
                ranker_scores,
                data,
                candidate_k,
                blend_alpha,
                args.k,
                len(train_features.labels),
                two_tower_train_seconds + ranker_train_seconds,
                two_tower_eval_seconds + ranker_eval_seconds,
                str(device),
                scope_name,
            )
            metrics.append(add_auc_logloss(pipeline_metric, data.test_rows[: args.auc_rows], pipeline_scores, data))

    write_outputs(data, metrics, args)
    print("Wrote KuaiRec experiment suite outputs.")
    for metric in metrics:
        print(
            f"{metric.model}: Recall@{args.k}={metric.recall:.6f} "
            f"NDCG@{args.k}={metric.ndcg:.6f} AUC={metric.auc:.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
