"""Download the 60 UORED-VAFCLS v5 CSV recordings from Mendeley Data."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_downloads(path: str | Path) -> list[tuple[str, str]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, url = line.split("\t", 1)
        if (
            Path(name).name != name
            or "/" in name
            or "\\" in name
            or name.startswith(".")
            or not name.endswith(".csv")
            or not url.startswith("https://data.mendeley.com/")
        ):
            raise ValueError(f"invalid download row for {name!r}")
        rows.append((name, url))
    if len(rows) != 60 or len({name for name, _ in rows}) != 60:
        raise ValueError("the pinned UORED v5 list must contain exactly 60 unique CSV files")
    return rows


def download_one(name: str, url: str, destination: Path, *, retries: int = 5) -> Path:
    target = destination / name
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    partial = target.with_suffix(".csv.part")
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FabGuard-AI/0.1"})
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                partial.open("wb") as sink,
            ):
                while chunk := response.read(1024 * 1024):
                    sink.write(chunk)
            if partial.stat().st_size <= 1_000_000:
                raise OSError("downloaded file is unexpectedly small")
            partial.replace(target)
            return target
        except (OSError, urllib.error.URLError):
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", default="configs/uored_v5_downloads.tsv")
    parser.add_argument("--output", default="data/raw/csv")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    downloads = load_downloads(args.files)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        futures = {
            executor.submit(download_one, name, url, destination): name for name, url in downloads
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"[{completed:02d}/60] {futures[future]}")
    print(f"downloaded {completed} files to {destination.resolve()}")


if __name__ == "__main__":
    main()
