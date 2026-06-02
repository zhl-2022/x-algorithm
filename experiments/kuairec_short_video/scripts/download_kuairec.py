"""Download the KuaiRec dataset from Zenodo.

The downloaded files are intentionally stored under data/raw/, which is ignored
by git in this experiment directory.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"

ZENODO_FILES = {
    "KuaiRec.zip": {
        "url": "https://zenodo.org/api/records/18164998/files/KuaiRec.zip/content",
        "md5": "261550d472c48eff4990fb13c0e5bcf7",
        "required": True,
    },
    "kuairec_caption_category.csv": {
        "url": "https://zenodo.org/api/records/18164998/files/kuairec_caption_category.csv/content",
        "md5": "31bc38cdccdf75a71df137779035f8cb",
        "required": False,
    },
    "video_raw_categories_multi.csv": {
        "url": "https://zenodo.org/api/records/18164998/files/video_raw_categories_multi.csv/content",
        "md5": "d05eea147135d2cdf7759fba5c0d70d4",
        "required": False,
    },
    "user_features_raw.csv": {
        "url": "https://zenodo.org/api/records/18164998/files/user_features_raw.csv/content",
        "md5": "3969b8120035e7ced36d56926a7cbd24",
        "required": False,
    },
}


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, force: bool) -> None:
    if target.exists() and not force:
        print(f"[skip] {target} already exists")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"[download] {url}")
    print(f"[target]   {target}")

    request = urllib.request.Request(url, headers={"User-Agent": "x-algorithm-kuiarec"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as file:
        total = int(response.headers.get("Content-Length", "0") or "0")
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 / total
                print(f"\r  {downloaded / 1024 / 1024:.1f} MiB / {total / 1024 / 1024:.1f} MiB ({percent:.1f}%)", end="")
            else:
                print(f"\r  {downloaded / 1024 / 1024:.1f} MiB", end="")
        print()

    os.replace(tmp, target)


def extract_zip(zip_path: Path, extract_dir: Path, force: bool) -> None:
    marker = extract_dir / ".extracted"
    if marker.exists() and not force:
        print(f"[skip] {zip_path.name} already extracted to {extract_dir}")
        return

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    marker.write_text(zip_path.name + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download KuaiRec data from Zenodo.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory for raw data files.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract existing files.")
    parser.add_argument("--skip-extract", action="store_true", help="Only download files, do not extract KuaiRec.zip.")
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Only download KuaiRec.zip. By default, auxiliary raw feature files are also downloaded.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    selected = {
        name: meta
        for name, meta in ZENODO_FILES.items()
        if not args.main_only or meta["required"]
    }

    for filename, meta in selected.items():
        target = raw_dir / filename
        download(meta["url"], target, args.force)
        expected_md5 = meta["md5"]
        actual_md5 = md5sum(target)
        if actual_md5 != expected_md5:
            print(f"[error] MD5 mismatch for {filename}: expected {expected_md5}, got {actual_md5}", file=sys.stderr)
            return 1
        print(f"[ok] {filename} md5={actual_md5}")

    zip_path = raw_dir / "KuaiRec.zip"
    if zip_path.exists() and not args.skip_extract:
        extract_zip(zip_path, raw_dir / "KuaiRec", args.force)

    print("[done] KuaiRec download step finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
