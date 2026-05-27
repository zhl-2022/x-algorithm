"""Download and extract MovieLens 1M for the recall experiment."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = EXPERIMENT_DIR / "data" / "raw"


def download_file(url: str, target: Path, force: bool = False) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        print(f"Using existing archive: {target}")
        return target

    temp_target = target.with_suffix(target.suffix + ".tmp")
    if temp_target.exists():
        temp_target.unlink()

    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        with temp_target.open("wb") as output:
            shutil.copyfileobj(response, output)

    temp_target.replace(target)
    print(f"Saved archive: {target}")
    return target


def extract_zip(archive: Path, raw_dir: Path, force: bool = False) -> Path:
    dataset_dir = raw_dir / "ml-1m"
    if dataset_dir.exists() and not force:
        print(f"Using existing extracted dataset: {dataset_dir}")
        return dataset_dir

    if dataset_dir.exists() and force:
        shutil.rmtree(dataset_dir)

    print(f"Extracting {archive} to {raw_dir}")
    with zipfile.ZipFile(archive) as zip_ref:
        zip_ref.extractall(raw_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Expected extracted directory not found: {dataset_dir}")

    print(f"Extracted dataset: {dataset_dir}")
    return dataset_dir


def download_and_extract(
    url: str = DEFAULT_URL,
    raw_dir: Path = DEFAULT_RAW_DIR,
    force: bool = False,
) -> Path:
    archive = download_file(url=url, target=raw_dir / "ml-1m.zip", force=force)
    return extract_zip(archive=archive, raw_dir=raw_dir, force=force)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="MovieLens 1M zip URL.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory used for the zip archive and extracted files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract the dataset.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    download_and_extract(url=args.url, raw_dir=args.raw_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
