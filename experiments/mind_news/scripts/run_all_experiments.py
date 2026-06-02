"""Run the remaining MIND-small recommendation experiments.

The full RecZoo MIND files are used for deterministic statistics baselines.
Neural models use the processed 100k-row samples by default so the experiment
suite remains reproducible on a local CPU and can later be moved to MLU.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


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
    title_entities: str
    abstract_entities: str
    title_word_count: int
    abstract_word_count: int


@dataclass(frozen=True)
class InteractionStats:
    train_path: Path
    rows: int
    positive_rows: int
    negative_rows: int
    ctr: float
    click_counts: Counter[str]
    exposure_counts: Counter[str]
    category_clicks: Counter[str]
    category_exposures: Counter[str]
    subcategory_clicks: Counter[str]
    subcategory_exposures: Counter[str]
    max_log_clicks: float
    max_category_ctr: float
    max_subcategory_ctr: float


@dataclass(frozen=True)
class EvalMetrics:
    model: str
    scope: str
    train_rows: int
    valid_rows: int
    evaluated_impressions: int
    avg_candidates: float
    auc: float
    mrr: float
    ndcg5: float
    ndcg10: float
    hitrate5: float
    hitrate10: float
    coverage5: float
    coverage10: float
    train_seconds: float
    eval_seconds: float
    notes: str


@dataclass
class MetricAccumulator:
    model: str
    scope: str
    train_rows: int
    train_seconds: float
    notes: str
    valid_rows: int = 0
    evaluated_impressions: int = 0
    auc_impressions: int = 0
    auc_total: float = 0.0
    mrr_total: float = 0.0
    ndcg5_total: float = 0.0
    ndcg10_total: float = 0.0
    hitrate5_total: float = 0.0
    hitrate10_total: float = 0.0
    recommended5: set[str] | None = None
    recommended10: set[str] | None = None

    def __post_init__(self) -> None:
        self.recommended5 = set()
        self.recommended10 = set()

    def add_group(self, labels: list[int], news_ids: list[str], scores: list[float]) -> None:
        self.evaluated_impressions += 1
        group_auc = auc_score(labels, scores)
        if group_auc is not None:
            self.auc_impressions += 1
            self.auc_total += group_auc

        ranked = sorted(zip(scores, news_ids, labels), key=lambda row: (-row[0], row[1]))
        ranked_news_ids = [news_id for _, news_id, _ in ranked]
        ranked_labels = [label for _, _, label in ranked]
        self.mrr_total += mrr_score(ranked_labels)
        self.ndcg5_total += ndcg_at_k(ranked_labels, 5)
        self.ndcg10_total += ndcg_at_k(ranked_labels, 10)
        self.hitrate5_total += 1.0 if any(ranked_labels[:5]) else 0.0
        self.hitrate10_total += 1.0 if any(ranked_labels[:10]) else 0.0
        assert self.recommended5 is not None
        assert self.recommended10 is not None
        self.recommended5.update(ranked_news_ids[:5])
        self.recommended10.update(ranked_news_ids[:10])

    def finish(self, total_news: int, eval_seconds: float) -> EvalMetrics:
        if self.evaluated_impressions == 0:
            raise ValueError(f"No impressions evaluated for {self.model}")
        assert self.recommended5 is not None
        assert self.recommended10 is not None
        return EvalMetrics(
            model=self.model,
            scope=self.scope,
            train_rows=self.train_rows,
            valid_rows=self.valid_rows,
            evaluated_impressions=self.evaluated_impressions,
            avg_candidates=self.valid_rows / self.evaluated_impressions,
            auc=self.auc_total / self.auc_impressions if self.auc_impressions else 0.0,
            mrr=self.mrr_total / self.evaluated_impressions,
            ndcg5=self.ndcg5_total / self.evaluated_impressions,
            ndcg10=self.ndcg10_total / self.evaluated_impressions,
            hitrate5=self.hitrate5_total / self.evaluated_impressions,
            hitrate10=self.hitrate10_total / self.evaluated_impressions,
            coverage5=len(self.recommended5) / total_news if total_news else 0.0,
            coverage10=len(self.recommended10) / total_news if total_news else 0.0,
            train_seconds=self.train_seconds,
            eval_seconds=eval_seconds,
            notes=self.notes,
        )


@dataclass(frozen=True)
class FeatureData:
    rows: list[dict[str, str]]
    labels: torch.Tensor
    user_indices: torch.Tensor
    news_indices: torch.Tensor
    category_indices: torch.Tensor
    subcategory_indices: torch.Tensor
    hour_indices: torch.Tensor
    history_category_indices: torch.Tensor
    history_subcategory_indices: torch.Tensor
    history_category_lengths: torch.Tensor
    history_subcategory_lengths: torch.Tensor
    entity_indices: torch.Tensor
    entity_lengths: torch.Tensor
    dense_features: torch.Tensor


class FeatureBuilder:
    def __init__(
        self,
        rows: list[dict[str, str]],
        stats: InteractionStats,
        news_metadata: dict[str, NewsInfo],
        max_history: int,
        max_entities: int,
        entity_buckets: int,
    ) -> None:
        self.stats = stats
        self.news_metadata = news_metadata
        self.max_history = max_history
        self.max_entities = max_entities
        self.entity_buckets = entity_buckets
        self.user_to_index = self._build_mapping(row["user_id"] for row in rows)
        self.news_to_index = self._build_mapping(row["news_id"] for row in rows)
        self.category_to_index = self._build_mapping(
            token
            for row in rows
            for token in [row["cat"], *split_history(row.get("cat_his", ""))]
            if token
        )
        self.subcategory_to_index = self._build_mapping(
            token
            for row in rows
            for token in [row["sub_cat"], *split_history(row.get("subcat_his", ""))]
            if token
        )
        self.hour_to_index = self._build_mapping(parse_hour_bucket(row.get("hour", "")) for row in rows)

    @staticmethod
    def _build_mapping(values: object) -> dict[str, int]:
        unique_values = sorted({str(value) for value in values if str(value)})
        return {value: index + 1 for index, value in enumerate(unique_values)}

    def _map(self, mapping: dict[str, int], value: str) -> int:
        return mapping.get(value, 0)

    def _history_indices(self, mapping: dict[str, int], value: str) -> tuple[list[int], int]:
        tokens = split_history(value)[-self.max_history :]
        indices = [self._map(mapping, token) for token in tokens]
        length = len(indices)
        if len(indices) < self.max_history:
            indices.extend([0] * (self.max_history - len(indices)))
        return indices, length

    def _entity_indices(self, row: dict[str, str]) -> tuple[list[int], int]:
        news_info = self.news_metadata.get(row["news_id"])
        raw_entities = []
        if news_info:
            raw_entities.extend(split_history(news_info.title_entities))
            raw_entities.extend(split_history(news_info.abstract_entities))
        raw_entities.extend(split_history(row.get("title_entities", "")))
        raw_entities.extend(split_history(row.get("abstract_entities", "")))

        seen: set[str] = set()
        indices: list[int] = []
        for token in raw_entities:
            if not token or token in seen:
                continue
            seen.add(token)
            bucket = zlib.crc32(token.encode("utf-8")) % self.entity_buckets
            indices.append(bucket + 1)
            if len(indices) == self.max_entities:
                break
        length = len(indices)
        if len(indices) < self.max_entities:
            indices.extend([0] * (self.max_entities - len(indices)))
        return indices, length

    def _dense_features(self, row: dict[str, str]) -> list[float]:
        news_id = row["news_id"]
        category = row["cat"]
        subcategory = row["sub_cat"]
        news_info = self.news_metadata.get(news_id)
        log_click = math.log1p(self.stats.click_counts[news_id])
        news_popularity = log_click / self.stats.max_log_clicks if self.stats.max_log_clicks else 0.0
        category_ctr = smoothed_ctr(
            self.stats.category_clicks[category],
            self.stats.category_exposures[category],
            self.stats.ctr,
        )
        subcategory_ctr = smoothed_ctr(
            self.stats.subcategory_clicks[subcategory],
            self.stats.subcategory_exposures[subcategory],
            self.stats.ctr,
        )
        category_ctr_norm = category_ctr / self.stats.max_category_ctr if self.stats.max_category_ctr else 0.0
        subcategory_ctr_norm = (
            subcategory_ctr / self.stats.max_subcategory_ctr if self.stats.max_subcategory_ctr else 0.0
        )
        history_categories = split_history(row.get("cat_his", ""))
        history_subcategories = split_history(row.get("subcat_his", ""))
        hist_cat_pref = history_preference(history_categories, category)
        hist_subcat_pref = history_preference(history_subcategories, subcategory)
        history_len_norm = min(len(split_history(row.get("news_his", ""))) / 50.0, 1.0)
        title_words = news_info.title_word_count if news_info else 0
        abstract_words = news_info.abstract_word_count if news_info else 0
        title_norm = min(title_words / 30.0, 1.0)
        abstract_norm = min(abstract_words / 120.0, 1.0)
        return [
            news_popularity,
            category_ctr_norm,
            subcategory_ctr_norm,
            hist_cat_pref,
            hist_subcat_pref,
            history_len_norm,
            title_norm,
            abstract_norm,
        ]

    def transform(self, rows: list[dict[str, str]]) -> FeatureData:
        labels = []
        user_indices = []
        news_indices = []
        category_indices = []
        subcategory_indices = []
        hour_indices = []
        history_category_indices = []
        history_subcategory_indices = []
        history_category_lengths = []
        history_subcategory_lengths = []
        entity_indices = []
        entity_lengths = []
        dense_features = []

        for row in rows:
            labels.append(float(row["click"]))
            user_indices.append(self._map(self.user_to_index, row["user_id"]))
            news_indices.append(self._map(self.news_to_index, row["news_id"]))
            category_indices.append(self._map(self.category_to_index, row["cat"]))
            subcategory_indices.append(self._map(self.subcategory_to_index, row["sub_cat"]))
            hour_indices.append(self._map(self.hour_to_index, parse_hour_bucket(row.get("hour", ""))))

            cat_hist, cat_len = self._history_indices(self.category_to_index, row.get("cat_his", ""))
            subcat_hist, subcat_len = self._history_indices(
                self.subcategory_to_index,
                row.get("subcat_his", ""),
            )
            ent_indices, ent_len = self._entity_indices(row)
            history_category_indices.append(cat_hist)
            history_subcategory_indices.append(subcat_hist)
            history_category_lengths.append(cat_len)
            history_subcategory_lengths.append(subcat_len)
            entity_indices.append(ent_indices)
            entity_lengths.append(ent_len)
            dense_features.append(self._dense_features(row))

        return FeatureData(
            rows=rows,
            labels=torch.tensor(labels, dtype=torch.float32),
            user_indices=torch.tensor(user_indices, dtype=torch.long),
            news_indices=torch.tensor(news_indices, dtype=torch.long),
            category_indices=torch.tensor(category_indices, dtype=torch.long),
            subcategory_indices=torch.tensor(subcategory_indices, dtype=torch.long),
            hour_indices=torch.tensor(hour_indices, dtype=torch.long),
            history_category_indices=torch.tensor(history_category_indices, dtype=torch.long),
            history_subcategory_indices=torch.tensor(history_subcategory_indices, dtype=torch.long),
            history_category_lengths=torch.tensor(history_category_lengths, dtype=torch.float32),
            history_subcategory_lengths=torch.tensor(history_subcategory_lengths, dtype=torch.float32),
            entity_indices=torch.tensor(entity_indices, dtype=torch.long),
            entity_lengths=torch.tensor(entity_lengths, dtype=torch.float32),
            dense_features=torch.tensor(dense_features, dtype=torch.float32),
        )


class MindDNNRanker(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_news: int,
        num_categories: int,
        num_subcategories: int,
        num_hours: int,
        dense_dim: int,
        embedding_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.news_embedding = nn.Embedding(num_news + 1, embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, embedding_dim // 2, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(num_subcategories + 1, embedding_dim // 2, padding_idx=0)
        self.hour_embedding = nn.Embedding(num_hours + 1, 4, padding_idx=0)
        input_dim = embedding_dim * 2 + embedding_dim + 4 + dense_dim
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        features = torch.cat(
            [
                self.user_embedding(batch["user_indices"]),
                self.news_embedding(batch["news_indices"]),
                self.category_embedding(batch["category_indices"]),
                self.subcategory_embedding(batch["subcategory_indices"]),
                self.hour_embedding(batch["hour_indices"]),
                batch["dense_features"],
            ],
            dim=1,
        )
        return self.mlp(features).squeeze(1)


class MindContentTwoTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_news: int,
        num_categories: int,
        num_subcategories: int,
        entity_buckets: int,
        dense_dim: int,
        embedding_dim: int,
        tower_dim: int,
        hidden_dim: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        half_dim = embedding_dim // 2
        self.temperature = temperature
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.news_embedding = nn.Embedding(num_news + 1, embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories + 1, half_dim, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(num_subcategories + 1, half_dim, padding_idx=0)
        self.entity_embedding = nn.Embedding(entity_buckets + 1, embedding_dim, padding_idx=0)
        self.news_dense_projection = nn.Linear(dense_dim, embedding_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim + half_dim + half_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tower_dim),
        )
        self.news_tower = nn.Sequential(
            nn.Linear(embedding_dim * 3 + half_dim + half_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tower_dim),
        )

    def _mean_embedding(
        self,
        embedding: nn.Embedding,
        indices: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths = lengths.clamp_min(1).unsqueeze(1)
        return embedding(indices).sum(dim=1) / lengths

    def encode_user(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_vector = self.user_embedding(batch["user_indices"])
        cat_history = self._mean_embedding(
            self.category_embedding,
            batch["history_category_indices"],
            batch["history_category_lengths"],
        )
        subcat_history = self._mean_embedding(
            self.subcategory_embedding,
            batch["history_subcategory_indices"],
            batch["history_subcategory_lengths"],
        )
        return F.normalize(self.user_tower(torch.cat([user_vector, cat_history, subcat_history], dim=1)), dim=1)

    def encode_news(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        news_vector = self.news_embedding(batch["news_indices"])
        category_vector = self.category_embedding(batch["category_indices"])
        subcategory_vector = self.subcategory_embedding(batch["subcategory_indices"])
        entity_vector = self._mean_embedding(
            self.entity_embedding,
            batch["entity_indices"],
            batch["entity_lengths"],
        )
        dense_vector = self.news_dense_projection(batch["dense_features"])
        return F.normalize(
            self.news_tower(
                torch.cat(
                    [
                        news_vector,
                        category_vector,
                        subcategory_vector,
                        entity_vector,
                        dense_vector,
                    ],
                    dim=1,
                )
            ),
            dim=1,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_vectors = self.encode_user(batch)
        news_vectors = self.encode_news(batch)
        return (user_vectors * news_vectors).sum(dim=1) / self.temperature


def resolve_default_train_path(raw_dir: Path, processed_dir: Path) -> Path:
    raw_path = raw_dir / "MIND_small_x1" / "train.csv"
    if raw_path.exists():
        return raw_path
    return processed_dir / "train_sample.csv"


def resolve_default_valid_path(raw_dir: Path, processed_dir: Path) -> Path:
    raw_path = raw_dir / "MIND_small_x1" / "valid.csv"
    if raw_path.exists():
        return raw_path
    return processed_dir / "valid_sample.csv"


def validate_schema(path: Path, fieldnames: list[str] | None) -> None:
    required = {"imp_id", "click", "hour", "user_id", "news_id", "cat", "sub_cat"}
    if not fieldnames or not required.issubset(set(fieldnames)):
        raise ValueError(f"Unexpected MIND interaction schema in {path}: {fieldnames}")


def split_history(value: str) -> list[str]:
    return [token for token in value.split("^") if token]


def parse_hour_bucket(value: str) -> str:
    return value.strip() or "unknown"


def smoothed_ctr(clicks: int, exposures: int, global_ctr: float, alpha: float = 20.0) -> float:
    return (clicks + alpha * global_ctr) / (exposures + alpha) if exposures else global_ctr


def history_preference(history_tokens: list[str], current_token: str) -> float:
    if not history_tokens or not current_token:
        return 0.0
    return sum(1 for token in history_tokens if token == current_token) / len(history_tokens)


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
    return dcg_at_k(labels, k) / ideal_dcg if ideal_dcg else 0.0


def mrr_score(labels: list[int]) -> float:
    for index, label in enumerate(labels, start=1):
        if label == 1:
            return 1.0 / index
    return 0.0


def read_news_metadata(path: Path) -> dict[str, NewsInfo]:
    news: dict[str, NewsInfo] = {}
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {
            "news_id",
            "category",
            "subcategory",
            "title",
            "title_entities",
            "abstract_entities",
            "title_word_count",
            "abstract_word_count",
        }
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Unexpected news metadata schema in {path}: {reader.fieldnames}")
        for row in reader:
            news[row["news_id"]] = NewsInfo(
                category=row["category"],
                subcategory=row["subcategory"],
                title=row["title"],
                title_entities=row["title_entities"],
                abstract_entities=row["abstract_entities"],
                title_word_count=int(row["title_word_count"] or 0),
                abstract_word_count=int(row["abstract_word_count"] or 0),
            )
    return news


def build_interaction_stats(train_path: Path) -> InteractionStats:
    start = time.perf_counter()
    rows = 0
    positive_rows = 0
    negative_rows = 0
    click_counts: Counter[str] = Counter()
    exposure_counts: Counter[str] = Counter()
    category_clicks: Counter[str] = Counter()
    category_exposures: Counter[str] = Counter()
    subcategory_clicks: Counter[str] = Counter()
    subcategory_exposures: Counter[str] = Counter()

    with train_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        validate_schema(train_path, reader.fieldnames)
        for row in reader:
            rows += 1
            click = row["click"]
            news_id = row["news_id"]
            category = row["cat"]
            subcategory = row["sub_cat"]
            exposure_counts[news_id] += 1
            category_exposures[category] += 1
            subcategory_exposures[subcategory] += 1
            if click == "1":
                positive_rows += 1
                click_counts[news_id] += 1
                category_clicks[category] += 1
                subcategory_clicks[subcategory] += 1
            elif click == "0":
                negative_rows += 1

    labeled_rows = positive_rows + negative_rows
    ctr = positive_rows / labeled_rows if labeled_rows else 0.0
    max_log_clicks = max((math.log1p(count) for count in click_counts.values()), default=1.0)
    max_category_ctr = max(
        (
            smoothed_ctr(category_clicks[key], category_exposures[key], ctr)
            for key in category_exposures
        ),
        default=1.0,
    )
    max_subcategory_ctr = max(
        (
            smoothed_ctr(subcategory_clicks[key], subcategory_exposures[key], ctr)
            for key in subcategory_exposures
        ),
        default=1.0,
    )
    print(f"Built full interaction stats in {time.perf_counter() - start:.1f}s")
    return InteractionStats(
        train_path=train_path,
        rows=rows,
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        ctr=ctr,
        click_counts=click_counts,
        exposure_counts=exposure_counts,
        category_clicks=category_clicks,
        category_exposures=category_exposures,
        subcategory_clicks=subcategory_clicks,
        subcategory_exposures=subcategory_exposures,
        max_log_clicks=max_log_clicks,
        max_category_ctr=max_category_ctr,
        max_subcategory_ctr=max_subcategory_ctr,
    )


def popularity_score(row: dict[str, str], stats: InteractionStats) -> float:
    news_id = row["news_id"]
    return float(stats.click_counts[news_id]) + 1e-6 * math.log1p(stats.exposure_counts[news_id])


def category_score(row: dict[str, str], stats: InteractionStats) -> float:
    news_id = row["news_id"]
    category = row["cat"]
    subcategory = row["sub_cat"]
    news_popularity = (
        math.log1p(stats.click_counts[news_id]) / stats.max_log_clicks
        if stats.max_log_clicks
        else 0.0
    )
    category_ctr = smoothed_ctr(
        stats.category_clicks[category],
        stats.category_exposures[category],
        stats.ctr,
    )
    subcategory_ctr = smoothed_ctr(
        stats.subcategory_clicks[subcategory],
        stats.subcategory_exposures[subcategory],
        stats.ctr,
    )
    category_ctr_norm = category_ctr / stats.max_category_ctr if stats.max_category_ctr else 0.0
    subcategory_ctr_norm = (
        subcategory_ctr / stats.max_subcategory_ctr if stats.max_subcategory_ctr else 0.0
    )
    history_categories = split_history(row.get("cat_his", ""))
    history_subcategories = split_history(row.get("subcat_his", ""))
    hist_cat_pref = history_preference(history_categories, category)
    hist_subcat_pref = history_preference(history_subcategories, subcategory)
    return (
        0.50 * news_popularity
        + 0.16 * category_ctr_norm
        + 0.14 * subcategory_ctr_norm
        + 0.14 * hist_cat_pref
        + 0.06 * hist_subcat_pref
        + 1e-9 * stats.exposure_counts[news_id]
    )


def evaluate_streaming_scorers(
    valid_path: Path,
    scorers: dict[str, Callable[[dict[str, str]], float]],
    total_news: int,
    train_rows: int,
    scope: str,
    notes_by_model: dict[str, str],
    train_seconds_by_model: dict[str, float] | None = None,
) -> list[EvalMetrics]:
    start = time.perf_counter()
    train_seconds_by_model = train_seconds_by_model or {}
    accumulators = {
        model: MetricAccumulator(
            model=model,
            scope=scope,
            train_rows=train_rows,
            train_seconds=train_seconds_by_model.get(model, 0.0),
            notes=notes_by_model.get(model, ""),
        )
        for model in scorers
    }
    current_rows: list[dict[str, str]] = []
    current_key: tuple[str, str] | None = None

    def consume_group(rows: list[dict[str, str]]) -> None:
        if not rows:
            return
        labels = [int(row["click"]) for row in rows]
        news_ids = [row["news_id"] for row in rows]
        for model, scorer in scorers.items():
            scores = [scorer(row) for row in rows]
            accumulators[model].add_group(labels, news_ids, scores)

    with valid_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        validate_schema(valid_path, reader.fieldnames)
        for row in reader:
            key = (row["imp_id"], row["user_id"])
            if current_key is not None and key != current_key:
                consume_group(current_rows)
                current_rows = []
            current_key = key
            current_rows.append(row)
            for accumulator in accumulators.values():
                accumulator.valid_rows += 1
    consume_group(current_rows)

    eval_seconds = time.perf_counter() - start
    return [accumulator.finish(total_news, eval_seconds) for accumulator in accumulators.values()]


def read_rows(path: Path, max_rows: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        validate_schema(path, reader.fieldnames)
        for row in reader:
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


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


def batch_from_features(data: FeatureData, indices: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "user_indices": data.user_indices[indices].to(device),
        "news_indices": data.news_indices[indices].to(device),
        "category_indices": data.category_indices[indices].to(device),
        "subcategory_indices": data.subcategory_indices[indices].to(device),
        "hour_indices": data.hour_indices[indices].to(device),
        "history_category_indices": data.history_category_indices[indices].to(device),
        "history_subcategory_indices": data.history_subcategory_indices[indices].to(device),
        "history_category_lengths": data.history_category_lengths[indices].to(device),
        "history_subcategory_lengths": data.history_subcategory_lengths[indices].to(device),
        "entity_indices": data.entity_indices[indices].to(device),
        "entity_lengths": data.entity_lengths[indices].to(device),
        "dense_features": data.dense_features[indices].to(device),
    }


def train_model(
    model: nn.Module,
    train_data: FeatureData,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[nn.Module, float, float]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    positives = float(train_data.labels.sum().item())
    negatives = len(train_data.rows) - positives
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator().manual_seed(seed)
    final_loss = 0.0
    start = time.perf_counter()

    for epoch in range(epochs):
        permutation = torch.randperm(len(train_data.rows), generator=generator)
        epoch_loss = 0.0
        batches = 0
        model.train()
        for start_index in range(0, len(permutation), batch_size):
            batch_indices = permutation[start_index : start_index + batch_size]
            labels = train_data.labels[batch_indices].to(device)
            batch = batch_from_features(train_data, batch_indices, device)
            logits = model(batch)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        final_loss = epoch_loss / max(batches, 1)
        print(f"{model.__class__.__name__} epoch {epoch + 1}/{epochs} loss={final_loss:.6f}")

    return model, time.perf_counter() - start, final_loss


def predict_scores(
    model: nn.Module,
    data: FeatureData,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for start_index in range(0, len(data.rows), batch_size):
            batch_indices = torch.arange(start_index, min(start_index + batch_size, len(data.rows)))
            batch = batch_from_features(data, batch_indices, device)
            logits = model(batch)
            scores.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return scores


def evaluate_rows_with_scores(
    rows: list[dict[str, str]],
    scores: list[float],
    model: str,
    scope: str,
    total_news: int,
    train_rows: int,
    train_seconds: float,
    eval_seconds: float,
    notes: str,
) -> EvalMetrics:
    accumulator = MetricAccumulator(
        model=model,
        scope=scope,
        train_rows=train_rows,
        train_seconds=train_seconds,
        notes=notes,
    )
    current_key: tuple[str, str] | None = None
    group_rows: list[dict[str, str]] = []
    group_scores: list[float] = []

    def consume_group(rows_in_group: list[dict[str, str]], scores_in_group: list[float]) -> None:
        if not rows_in_group:
            return
        labels = [int(row["click"]) for row in rows_in_group]
        news_ids = [row["news_id"] for row in rows_in_group]
        accumulator.add_group(labels, news_ids, scores_in_group)

    for row, score in zip(rows, scores):
        key = (row["imp_id"], row["user_id"])
        if current_key is not None and key != current_key:
            consume_group(group_rows, group_scores)
            group_rows = []
            group_scores = []
        current_key = key
        group_rows.append(row)
        group_scores.append(score)
        accumulator.valid_rows += 1
    consume_group(group_rows, group_scores)
    return accumulator.finish(total_news, eval_seconds)


def evaluate_pipeline(
    rows: list[dict[str, str]],
    two_tower_scores: list[float],
    ranker_scores: list[float],
    candidate_k: int,
    total_news: int,
    train_rows: int,
    train_seconds: float,
    eval_seconds: float,
) -> EvalMetrics:
    accumulator = MetricAccumulator(
        model="TwoTower+DNN-Rerank",
        scope="sample",
        train_rows=train_rows,
        train_seconds=train_seconds,
        notes=f"Two-Tower 在每个曝光组内筛选 Top {candidate_k} 候选，DNN Ranker 再重排。",
    )
    current_key: tuple[str, str] | None = None
    group_indices: list[int] = []

    def consume_group(indices: list[int]) -> None:
        if not indices:
            return
        selected = set(
            sorted(indices, key=lambda index: (-two_tower_scores[index], rows[index]["news_id"]))[
                :candidate_k
            ]
        )
        labels = [int(rows[index]["click"]) for index in indices]
        news_ids = [rows[index]["news_id"] for index in indices]
        final_scores = [
            ranker_scores[index] if index in selected else -1e9 + two_tower_scores[index]
            for index in indices
        ]
        accumulator.add_group(labels, news_ids, final_scores)

    for index, row in enumerate(rows):
        key = (row["imp_id"], row["user_id"])
        if current_key is not None and key != current_key:
            consume_group(group_indices)
            group_indices = []
        current_key = key
        group_indices.append(index)
        accumulator.valid_rows += 1
    consume_group(group_indices)
    return accumulator.finish(total_news, eval_seconds)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_seconds(value: float) -> str:
    return f"{value:.2f}"


def write_metric_csv(path: Path, metrics: list[EvalMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "scope",
                "train_rows",
                "valid_rows",
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
                "train_seconds",
                "eval_seconds",
                "notes",
            ]
        )
        for metric in metrics:
            writer.writerow(
                [
                    metric.model,
                    metric.scope,
                    metric.train_rows,
                    metric.valid_rows,
                    metric.evaluated_impressions,
                    f"{metric.avg_candidates:.8f}",
                    f"{metric.auc:.8f}",
                    f"{metric.mrr:.8f}",
                    f"{metric.ndcg5:.8f}",
                    f"{metric.ndcg10:.8f}",
                    f"{metric.hitrate5:.8f}",
                    f"{metric.hitrate10:.8f}",
                    f"{metric.coverage5:.8f}",
                    f"{metric.coverage10:.8f}",
                    f"{metric.train_seconds:.4f}",
                    f"{metric.eval_seconds:.4f}",
                    metric.notes,
                ]
            )


def write_report(
    path: Path,
    metrics: list[EvalMetrics],
    args: argparse.Namespace,
    full_stats: InteractionStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MIND-small 全量试验报告",
        "",
        "## 实验范围",
        "",
        "本报告汇总当前 MIND-small 阶段的剩余试验。统计类 baseline 使用全量 RecZoo",
        "`train.csv` 和 `valid.csv`；神经网络模型默认使用 `train_sample.csv` 和",
        "`valid_sample.csv`，保证本地 CPU 可复现，后续可以无缝迁移到 MLU。",
        "",
        "## 数据规模",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
        f"| 全量训练样本 | {full_stats.rows:,} |",
        f"| 全量训练正样本 | {full_stats.positive_rows:,} |",
        f"| 全量训练 CTR | {full_stats.ctr * 100:.4f}% |",
        f"| 神经模型训练样本 | {args.sample_train_rows:,} |",
        f"| 神经模型验证样本 | {args.sample_valid_rows:,} |",
        "",
        "## 结果汇总",
        "",
        "| 模型 | 范围 | AUC | MRR | NDCG@5 | NDCG@10 | HitRate@5 | HitRate@10 | Coverage@10 | 说明 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for metric in metrics:
        lines.append(
            "| "
            f"{metric.model} | "
            f"{metric.scope} | "
            f"{format_float(metric.auc)} | "
            f"{format_float(metric.mrr)} | "
            f"{format_float(metric.ndcg5)} | "
            f"{format_float(metric.ndcg10)} | "
            f"{format_float(metric.hitrate5)} | "
            f"{format_float(metric.hitrate10)} | "
            f"{format_float(metric.coverage10)} | "
            f"{metric.notes} |"
        )

    lines.extend(
        [
            "",
            "## 模型说明",
            "",
            "| 模型 | 使用的信息 | 小白解释 |",
            "|---|---|---|",
            "| Popularity | 新闻全局点击次数 | 只看新闻是不是热门，不看用户是谁 |",
            "| Category | 新闻热度、类别点击率、用户历史类别 | 开始利用用户过去喜欢的新闻类别 |",
            "| DNNRanker | 用户、新闻、类别、小时、热度、历史偏好、标题/摘要长度 | 用 MLP 学习点击概率 |",
            "| ContentTwoTower | 用户历史类别、新闻 ID、类别、实体和内容统计 | 将用户和新闻映射到同一向量空间后点积打分 |",
            "| TwoTower+DNN-Rerank | Two-Tower 候选选择 + DNN 排序 | 模拟企业推荐中的召回后排序流程 |",
            "",
            "## 当前结论",
            "",
            "- 全量 Popularity 和 Category 是当前全数据口径的最低基线。",
            "- 神经模型结果使用样本口径，主要证明 MIND 阶段的排序、双塔和重排代码路径已经跑通。",
            "- 后续要追求更强指标，应把样本训练切换到全量训练，并优先迁移到 MLU 容器运行。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_single_report(path: Path, title: str, metrics: list[EvalMetrics], description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "| 模型 | 范围 | 训练样本 | 验证样本 | 曝光组 | AUC | MRR | NDCG@5 | NDCG@10 | HitRate@10 | 训练耗时秒 | 评估耗时秒 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        lines.append(
            "| "
            f"{metric.model} | "
            f"{metric.scope} | "
            f"{metric.train_rows:,} | "
            f"{metric.valid_rows:,} | "
            f"{metric.evaluated_impressions:,} | "
            f"{format_float(metric.auc)} | "
            f"{format_float(metric.mrr)} | "
            f"{format_float(metric.ndcg5)} | "
            f"{format_float(metric.ndcg10)} | "
            f"{format_float(metric.hitrate10)} | "
            f"{format_seconds(metric.train_seconds)} | "
            f"{format_seconds(metric.eval_seconds)} |"
        )
    lines.extend(["", "## 说明", "", *[f"- {metric.notes}" for metric in metrics]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--full-train-path", type=Path, default=None)
    parser.add_argument("--full-valid-path", type=Path, default=None)
    parser.add_argument("--sample-train-path", type=Path, default=None)
    parser.add_argument("--sample-valid-path", type=Path, default=None)
    parser.add_argument("--sample-train-rows", type=int, default=100_000)
    parser.add_argument("--sample-valid-rows", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dims", type=str, default="64,32")
    parser.add_argument("--tower-dim", type=int, default=32)
    parser.add_argument("--tower-hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-history", type=int, default=30)
    parser.add_argument("--max-entities", type=int, default=8)
    parser.add_argument("--entity-buckets", type=int, default=2048)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    full_train_path = args.full_train_path or resolve_default_train_path(args.raw_dir, args.processed_dir)
    full_valid_path = args.full_valid_path or resolve_default_valid_path(args.raw_dir, args.processed_dir)
    sample_train_path = args.sample_train_path or args.processed_dir / "train_sample.csv"
    sample_valid_path = args.sample_valid_path or args.processed_dir / "valid_sample.csv"
    news_metadata_path = args.processed_dir / "news_metadata.csv"

    for required_path in [full_train_path, full_valid_path, sample_train_path, sample_valid_path, news_metadata_path]:
        if not required_path.exists():
            print(f"Missing required path: {required_path}", file=sys.stderr)
            return 1

    news_metadata = read_news_metadata(news_metadata_path)
    total_news = len(news_metadata)
    stats_start = time.perf_counter()
    full_stats = build_interaction_stats(full_train_path)
    stats_seconds = time.perf_counter() - stats_start

    full_metrics = evaluate_streaming_scorers(
        valid_path=full_valid_path,
        scorers={
            "Popularity": lambda row: popularity_score(row, full_stats),
            "Category": lambda row: category_score(row, full_stats),
        },
        total_news=total_news,
        train_rows=full_stats.rows,
        scope="full",
        notes_by_model={
            "Popularity": "全量新闻点击热度基线。",
            "Category": "全量新闻热度 + 类别 CTR + 用户历史类别偏好。",
        },
        train_seconds_by_model={"Popularity": stats_seconds, "Category": stats_seconds},
    )

    train_rows = read_rows(sample_train_path, args.sample_train_rows)
    valid_rows = read_rows(sample_valid_path, args.sample_valid_rows)
    sample_metrics = evaluate_streaming_scorers(
        valid_path=sample_valid_path,
        scorers={
            "Popularity-Sample": lambda row: popularity_score(row, full_stats),
            "Category-Sample": lambda row: category_score(row, full_stats),
        },
        total_news=total_news,
        train_rows=len(train_rows),
        scope="sample",
        notes_by_model={
            "Popularity-Sample": "全量热度分数在样本验证集上的结果。",
            "Category-Sample": "全量类别增强分数在样本验证集上的结果。",
        },
        train_seconds_by_model={"Popularity-Sample": stats_seconds, "Category-Sample": stats_seconds},
    )

    feature_builder = FeatureBuilder(
        rows=[*train_rows, *valid_rows],
        stats=full_stats,
        news_metadata=news_metadata,
        max_history=args.max_history,
        max_entities=args.max_entities,
        entity_buckets=args.entity_buckets,
    )
    train_data = feature_builder.transform(train_rows)
    valid_data = feature_builder.transform(valid_rows)
    device = choose_device(args.device)
    hidden_dims = tuple(int(part.strip()) for part in args.hidden_dims.split(",") if part.strip())

    ranker = MindDNNRanker(
        num_users=len(feature_builder.user_to_index),
        num_news=len(feature_builder.news_to_index),
        num_categories=len(feature_builder.category_to_index),
        num_subcategories=len(feature_builder.subcategory_to_index),
        num_hours=len(feature_builder.hour_to_index),
        dense_dim=train_data.dense_features.shape[1],
        embedding_dim=args.embedding_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
    )
    ranker, ranker_train_seconds, _ = train_model(
        model=ranker,
        train_data=train_data,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    ranker_eval_start = time.perf_counter()
    ranker_scores = predict_scores(ranker, valid_data, device, args.batch_size)
    ranker_eval_seconds = time.perf_counter() - ranker_eval_start
    ranker_metric = evaluate_rows_with_scores(
        rows=valid_rows,
        scores=ranker_scores,
        model="DNNRanker",
        scope="sample",
        total_news=total_news,
        train_rows=len(train_rows),
        train_seconds=ranker_train_seconds,
        eval_seconds=ranker_eval_seconds,
        notes="MLP Ranker，融合用户、新闻、类别、小时 embedding 和内容/统计 dense 特征。",
    )

    two_tower = MindContentTwoTower(
        num_users=len(feature_builder.user_to_index),
        num_news=len(feature_builder.news_to_index),
        num_categories=len(feature_builder.category_to_index),
        num_subcategories=len(feature_builder.subcategory_to_index),
        entity_buckets=args.entity_buckets,
        dense_dim=train_data.dense_features.shape[1],
        embedding_dim=args.embedding_dim,
        tower_dim=args.tower_dim,
        hidden_dim=args.tower_hidden_dim,
        dropout=args.dropout,
        temperature=args.temperature,
    )
    two_tower, two_tower_train_seconds, _ = train_model(
        model=two_tower,
        train_data=train_data,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 7,
    )
    two_tower_eval_start = time.perf_counter()
    two_tower_scores = predict_scores(two_tower, valid_data, device, args.batch_size)
    two_tower_eval_seconds = time.perf_counter() - two_tower_eval_start
    two_tower_metric = evaluate_rows_with_scores(
        rows=valid_rows,
        scores=two_tower_scores,
        model="ContentTwoTower",
        scope="sample",
        total_news=total_news,
        train_rows=len(train_rows),
        train_seconds=two_tower_train_seconds,
        eval_seconds=two_tower_eval_seconds,
        notes="内容感知双塔，使用用户历史类别、新闻类别、实体和内容统计特征。",
    )

    pipeline_metric = evaluate_pipeline(
        rows=valid_rows,
        two_tower_scores=two_tower_scores,
        ranker_scores=ranker_scores,
        candidate_k=args.candidate_k,
        total_news=total_news,
        train_rows=len(train_rows),
        train_seconds=ranker_train_seconds + two_tower_train_seconds,
        eval_seconds=ranker_eval_seconds + two_tower_eval_seconds,
    )

    all_metrics = [*full_metrics, *sample_metrics, ranker_metric, two_tower_metric, pipeline_metric]
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    write_metric_csv(args.outputs_dir / "experiment_results.csv", all_metrics)
    write_metric_csv(args.outputs_dir / "category_results.csv", [metric for metric in all_metrics if "Category" in metric.model])
    write_metric_csv(args.outputs_dir / "ranker_results.csv", [ranker_metric])
    write_metric_csv(args.outputs_dir / "two_tower_results.csv", [two_tower_metric])
    write_metric_csv(args.outputs_dir / "pipeline_results.csv", [pipeline_metric])

    write_report(args.reports_dir / "all_experiments_report.md", all_metrics, args, full_stats)
    write_single_report(
        args.reports_dir / "category_report.md",
        "MIND Category Baseline 报告",
        [metric for metric in all_metrics if "Category" in metric.model],
        "Category baseline 在新闻热度之外加入类别点击率和用户历史类别偏好。",
    )
    write_single_report(
        args.reports_dir / "ranker_report.md",
        "MIND DNN Ranker 报告",
        [ranker_metric],
        "DNN Ranker 使用 embedding 和 dense 特征预测候选新闻点击概率。",
    )
    write_single_report(
        args.reports_dir / "two_tower_report.md",
        "MIND 内容感知 Two-Tower 报告",
        [two_tower_metric],
        "ContentTwoTower 将用户历史和新闻内容特征编码到同一向量空间。",
    )
    write_single_report(
        args.reports_dir / "pipeline_report.md",
        "MIND Two-Tower + DNN Ranker Pipeline 报告",
        [pipeline_metric],
        "Pipeline 先用 Two-Tower 在曝光候选内筛选 TopK，再用 DNN Ranker 重排。",
    )

    print("Wrote MIND experiment suite outputs.")
    for metric in all_metrics:
        print(
            f"{metric.model} [{metric.scope}] "
            f"AUC={metric.auc:.6f} MRR={metric.mrr:.6f} "
            f"NDCG@10={metric.ndcg10:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
