from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request
import zipfile

from tqdm import tqdm


COCO_URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


class _DownloadProgress:
    def __init__(self, description: str):
        self.progress = tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=description)

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            self.progress.total = total_size
        downloaded = block_num * block_size
        self.progress.update(downloaded - self.progress.n)

    def close(self) -> None:
        self.progress.close()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"[skip] {destination} already exists")
        return
    progress = _DownloadProgress(destination.name)
    try:
        urllib.request.urlretrieve(url, destination, reporthook=progress)
    finally:
        progress.close()


def _extract(zip_path: Path, root: Path, expected_dir: str) -> None:
    marker = root / expected_dir
    if marker.exists() and any(marker.iterdir()):
        print(f"[skip] {marker} already extracted")
        return
    print(f"[extract] {zip_path} -> {root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)


def download_coco(root: Path, split: str = "val2017") -> None:
    root = Path(root)
    if split != "val2017":
        raise ValueError("This helper currently supports val2017 only")

    image_zip = root / f"{split}.zip"
    annotation_zip = root / "annotations_trainval2017.zip"
    _download(COCO_URLS[split], image_zip)
    _download(COCO_URLS["annotations"], annotation_zip)
    _extract(image_zip, root, split)
    _extract(annotation_zip, root, "annotations")

    required = [
        root / split,
        root / "annotations" / f"instances_{split}.json",
        root / "annotations" / f"captions_{split}.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"COCO download incomplete. Missing:\n{joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO val2017 for region experiments.")
    parser.add_argument("--root", type=Path, required=True, help="COCO root directory")
    parser.add_argument("--split", default="val2017", choices=["val2017"])
    args = parser.parse_args()
    download_coco(args.root, args.split)


if __name__ == "__main__":
    main()
