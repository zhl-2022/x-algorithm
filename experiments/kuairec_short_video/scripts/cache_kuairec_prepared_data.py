"""Build a reusable KuaiRec PreparedData cache for upgrade experiments."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

from run_all_experiments import DEFAULT_RAW_DIR, build_prepared_data


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--matrix", choices=["small_matrix.csv", "big_matrix.csv"], default="big_matrix.csv")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--positive-threshold", type=float, default=0.8)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--text-buckets", type=int, default=8192)
    parser.add_argument("--max-text-tokens", type=int, default=16)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.output_cache.exists() and not args.force:
        print(f"Prepared cache already exists: {args.output_cache}", flush=True)
        return 0

    started = time.perf_counter()
    data = build_prepared_data(args)
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    with args.output_cache.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.perf_counter() - started
    print(
        f"Wrote prepared cache: {args.output_cache} "
        f"train={len(data.train_rows):,} test={len(data.test_rows):,} "
        f"eval_users={len(data.eval_user_indices):,} seconds={elapsed:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
