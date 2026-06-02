"""Benchmark KuaiRec Two-Tower training throughput with one or two MLU cards."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from run_all_experiments import (
    DEFAULT_RAW_DIR,
    FeatureTensors,
    KuaiRecTwoTower,
    build_feature_tensors,
    build_prepared_data,
    move_batch,
)


class InBatchTwoTowerWrapper(nn.Module):
    def __init__(self, model: KuaiRecTwoTower) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: FeatureTensors) -> torch.Tensor:
        user_vectors = self.model.encode_user(batch.users, batch.dense)
        item_vectors = self.model.encode_item(
            batch.items,
            batch.categories,
            batch.text_indices,
            batch.text_lengths,
            batch.dense,
        )
        return (user_vectors @ item_vectors.T) / self.model.temperature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--matrix", choices=["small_matrix.csv", "big_matrix.csv"], default="big_matrix.csv")
    parser.add_argument("--positive-threshold", type=float, default=0.8)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--train-rows", type=int, default=2_000_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--tower-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--loss", choices=["inbatch"], default="inbatch")
    parser.add_argument("--text-buckets", type=int, default=8192)
    parser.add_argument("--max-text-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def init_runtime() -> tuple[torch.device, int, int, int, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    try:
        import torch_mlu  # noqa: F401

        if hasattr(torch, "mlu") and torch.mlu.is_available():
            if hasattr(torch.mlu, "set_device"):
                torch.mlu.set_device(local_rank)
            device = torch.device(f"mlu:{local_rank}")
            if distributed:
                dist.init_process_group(backend="cncl")
            return device, rank, local_rank, world_size, distributed
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if distributed:
            dist.init_process_group(backend="nccl")
        return device, rank, local_rank, world_size, distributed

    device = torch.device("cpu")
    if distributed:
        dist.init_process_group(backend="gloo")
    return device, rank, local_rank, world_size, distributed


def train(
    model: nn.Module,
    features: FeatureTensors,
    device: torch.device,
    rank: int,
    world_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> tuple[float, int, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = torch.Generator().manual_seed(seed)
    n = len(features.labels)
    processed = 0
    final_loss = 0.0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=rng)
        local_order = order[rank::world_size]
        total_loss = 0.0
        batches = 0
        model.train()
        for start in range(0, len(local_order), batch_size):
            batch_indices = local_order[start : start + batch_size]
            if len(batch_indices) < 2:
                continue
            batch = move_batch(features, batch_indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            labels = torch.arange(len(batch_indices), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
            processed += len(batch_indices)
        final_loss = total_loss / batches if batches else 0.0
        if rank == 0:
            print(f"epoch={epoch} ddp_inbatch_loss={final_loss:.6f}", flush=True)

    elapsed = time.perf_counter() - started
    processed_tensor = torch.tensor(float(processed), device=device)
    if world_size > 1:
        dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
    return final_loss, int(processed_tensor.item()), elapsed


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device, rank, local_rank, world_size, distributed = init_runtime()

    data = build_prepared_data(args)
    features = build_feature_tensors(data, data.train_rows, args.train_rows, args.seed, positive_only=True)
    if rank == 0:
        print(
            f"Loaded {args.matrix}: positive_train_rows={len(features.labels):,} "
            f"world_size={world_size} device={device}",
            flush=True,
        )

    tower = KuaiRecTwoTower(
        num_users=len(data.index_to_user),
        num_items=len(data.index_to_item),
        num_categories=int(data.item_categories.max()),
        text_buckets=args.text_buckets,
        embedding_dim=args.embedding_dim,
        tower_dim=args.tower_dim,
        hidden_dim=args.hidden_dim,
        dense_dim=features.dense.shape[1],
        dropout=args.dropout,
        temperature=args.temperature,
    )
    model: nn.Module = InBatchTwoTowerWrapper(tower).to(device)
    if distributed:
        model = DistributedDataParallel(model)

    final_loss, processed_samples, train_seconds = train(
        model=model,
        features=features,
        device=device,
        rank=rank,
        world_size=world_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed + 1,
    )

    if rank == 0:
        result = {
            "matrix": args.matrix,
            "positive_threshold": args.positive_threshold,
            "loss": args.loss,
            "device": str(device),
            "world_size": world_size,
            "local_rank": local_rank,
            "train_rows": len(features.labels),
            "processed_samples": processed_samples,
            "epochs": args.epochs,
            "per_process_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "embedding_dim": args.embedding_dim,
            "tower_dim": args.tower_dim,
            "hidden_dim": args.hidden_dim,
            "train_seconds": train_seconds,
            "samples_per_second": processed_samples / train_seconds if train_seconds else 0.0,
            "final_loss": final_loss,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
