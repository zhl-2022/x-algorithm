"""Download and extract MIND-small for the news recommendation experiment."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


OFFICIAL_BASE_URL = "https://mind201910small.blob.core.windows.net/release"
RECZOO_URL = "https://huggingface.co/datasets/reczoo/MIND_small_x1/resolve/main/MIND_small_x1.zip"
ARCHIVES = {
    "train": "MINDsmall_train.zip",
    "dev": "MINDsmall_dev.zip",
}
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
    with urllib.request.urlopen(url, timeout=300) as response:
        with temp_target.open("wb") as output:
            shutil.copyfileobj(response, output)

    temp_target.replace(target)
    print(f"Saved archive: {target}")
    return target


def extract_zip(
    archive: Path,
    target_dir: Path,
    force: bool = False,
    required_files: list[str] | None = None,
) -> Path:
    if target_dir.exists() and not force:
        print(f"Using existing extracted dataset: {target_dir}")
        return target_dir

    if target_dir.exists() and force:
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} to {target_dir}")
    with zipfile.ZipFile(archive) as zip_ref:
        zip_ref.extractall(target_dir)

    if required_files:
        missing = [filename for filename in required_files if not (target_dir / filename).exists()]
        if missing:
            raise FileNotFoundError(f"Missing files after extraction: {missing}")

    print(f"Extracted dataset: {target_dir}")
    return target_dir


def download_official(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False) -> None:
    for split, archive_name in ARCHIVES.items():
        archive = download_file(
            url=f"{OFFICIAL_BASE_URL}/{archive_name}",
            target=raw_dir / archive_name,
            force=force,
        )
        extract_zip(
            archive=archive,
            target_dir=raw_dir / f"MINDsmall_{split}",
            force=force,
            required_files=["news.tsv", "behaviors.tsv"],
        )


def download_reczoo(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False) -> None:
    archive = download_file(
        url=RECZOO_URL,
        target=raw_dir / "MIND_small_x1.zip",
        force=force,
    )
    extract_zip(
        archive=archive,
        target_dir=raw_dir / "MIND_small_x1",
        force=force,
        required_files=["news_corpus.tsv", "train.csv", "valid.csv"],
    )


def download_and_extract(
    raw_dir: Path = DEFAULT_RAW_DIR,
    force: bool = False,
    source: str = "auto",
) -> None:
    if source == "official":
        download_official(raw_dir=raw_dir, force=force)
        return

    if source == "reczoo":
        download_reczoo(raw_dir=raw_dir, force=force)
        return

    try:
        download_official(raw_dir=raw_dir, force=force)
    except Exception as exc:
        print(f"Official MIND download failed: {exc}")
        print("Falling back to RecZoo MIND_small_x1 mirror.")
        download_reczoo(raw_dir=raw_dir, force=force)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory used for archives and extracted files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract MIND-small.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "official", "reczoo"],
        default="auto",
        help="Download source. auto tries official first, then RecZoo mirror.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    download_and_extract(raw_dir=args.raw_dir, force=args.force, source=args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
