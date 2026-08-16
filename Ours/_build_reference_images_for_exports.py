import csv
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_ROOT = REPO_ROOT / "Ours" / "experiment_exports"
REFERENCE_ROOT = EXPORTS_ROOT / "reference_images"
COCO_CANDIDATES = [
    REPO_ROOT / "COCO" / "val2017",
    REPO_ROOT / "COCO" / "val2017" / "val2017",
    REPO_ROOT / "datasets" / "coco" / "val2017",
    REPO_ROOT / "datasets" / "coco" / "images" / "val2017",
]


def find_coco_root() -> Path:
    for path in COCO_CANDIDATES:
        if (path / "000000000776.jpg").exists():
            return path
    checked = "\n".join(str(p) for p in COCO_CANDIDATES)
    raise FileNotFoundError(f"Cannot find COCO val2017 root. Checked:\n{checked}")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_image_id_from_name(name: str) -> int | None:
    match = re.search(r"coco_val2017_(\d{12})", name)
    if match:
        return int(match.group(1))
    match = re.search(r"coco_(\d{12})", name)
    if match:
        return int(match.group(1))
    return None


def generated_to_reference_name(generated_name: str) -> str:
    stem = Path(generated_name).stem
    if stem.endswith("__generated"):
        stem = stem[: -len("__generated")] + "__reference"
    elif stem.endswith("_generated"):
        stem = stem[: -len("_generated")] + "_reference"
    else:
        stem = stem + "_reference"
    return stem + ".png"


def load_records(export_dir: Path) -> list[dict[str, Any]]:
    jsonl_path = export_dir / "metric_generated_manifest.jsonl"
    if jsonl_path.exists():
        return read_jsonl(jsonl_path)

    summary_path = export_dir / "generation_summary.json"
    if summary_path.exists():
        data = read_json(summary_path)
        if isinstance(data, list):
            return data

    records: list[dict[str, Any]] = []
    generated_dir = export_dir / "generated_images"
    for idx, image_path in enumerate(sorted(generated_dir.glob("*.png"))):
        image_id = parse_image_id_from_name(image_path.name)
        if image_id is None:
            raise ValueError(f"Cannot parse COCO image id from {image_path.name}")
        records.append(
            {
                "index": idx,
                "image_id": image_id,
                "file_name": f"{image_id:012d}.jpg",
                "generated_image_relative_path": f"generated_images/{image_path.name}",
            }
        )
    return records


def local_generated_path(export_dir: Path, record: dict[str, Any]) -> Path:
    for key in ("generated_image_relative_path",):
        rel = record.get(key)
        if rel:
            path = export_dir / str(rel)
            if path.exists():
                return path

    generated_path = record.get("generated_path") or record.get("generated_image_path")
    if generated_path:
        path = export_dir / "generated_images" / Path(str(generated_path)).name
        if path.exists():
            return path

    image_id = int(record["image_id"])
    candidates = sorted((export_dir / "generated_images").glob(f"*{image_id:012d}*generated*.png"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Cannot find generated image for record image_id={image_id}")


def infer_target_size(generated_path: Path, record: dict[str, Any]) -> tuple[int, int]:
    target_size = record.get("target_size")
    if isinstance(target_size, list) and len(target_size) == 2:
        return int(target_size[0]), int(target_size[1])

    with Image.open(generated_path) as image:
        width, height = image.size
    return height, width


def is_valid_image(path: Path, expected_size: tuple[int, int]) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        expected_h, expected_w = expected_size
        return (width, height) == (expected_w, expected_h)
    except Exception:
        return False


def save_resized_reference(source_path: Path, reference_path: Path, target_size: tuple[int, int]) -> None:
    target_h, target_w = target_size
    tmp_path = reference_path.with_suffix(reference_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with Image.open(source_path) as image:
        image = image.convert("RGB")
        image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
        image.save(tmp_path, format="PNG")
    tmp_path.replace(reference_path)


def main() -> None:
    coco_root = find_coco_root()
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)

    export_dirs = [
        path
        for path in sorted(EXPORTS_ROOT.iterdir())
        if path.is_dir() and path.name != REFERENCE_ROOT.name and (path / "generated_images").exists()
    ]

    summary_rows: list[dict[str, Any]] = []
    for export_dir in export_dirs:
        records = load_records(export_dir)
        reference_dir = REFERENCE_ROOT / export_dir.name
        reference_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for metric_index, record in enumerate(records):
            generated_path = local_generated_path(export_dir, record)
            target_h, target_w = infer_target_size(generated_path, record)

            image_id = int(record["image_id"])
            file_name = str(record.get("file_name") or f"{image_id:012d}.jpg")
            source_path = coco_root / file_name
            if not source_path.exists():
                missing.append(file_name)
                continue

            reference_name = generated_to_reference_name(generated_path.name)
            reference_path = reference_dir / reference_name
            if not is_valid_image(reference_path, (target_h, target_w)):
                save_resized_reference(source_path, reference_path, (target_h, target_w))

            manifest_rows.append(
                {
                    "metric_index": record.get("metric_index", record.get("index", metric_index)),
                    "experiment": export_dir.name,
                    "image_id": image_id,
                    "file_name": file_name,
                    "sample_id": record.get("sample_id", ""),
                    "target_height": target_h,
                    "target_width": target_w,
                    "generated_image": str(generated_path.relative_to(export_dir)),
                    "reference_image": str(reference_path.relative_to(reference_dir)),
                    "coco_source": str(source_path),
                }
            )

        manifest_jsonl = reference_dir / "reference_manifest.jsonl"
        with manifest_jsonl.open("w", encoding="utf-8", newline="\n") as f:
            for row in manifest_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest_csv = reference_dir / "reference_manifest.csv"
        with manifest_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
            if manifest_rows:
                writer.writeheader()
                writer.writerows(manifest_rows)

        export_summary = {
            "experiment": export_dir.name,
            "generated_dir": str((export_dir / "generated_images").resolve()),
            "reference_dir": str(reference_dir.resolve()),
            "coco_root": str(coco_root.resolve()),
            "num_input_records": len(records),
            "num_reference_images": len(manifest_rows),
            "num_missing_coco_images": len(missing),
            "missing_coco_images": missing[:20],
        }
        with (reference_dir / "reference_summary.json").open("w", encoding="utf-8") as f:
            json.dump(export_summary, f, ensure_ascii=False, indent=2)

        summary_rows.append(export_summary)
        print(
            f"[OK] {export_dir.name}: "
            f"{len(manifest_rows)}/{len(records)} reference images -> {reference_dir}"
        )

    with (REFERENCE_ROOT / "reference_exports_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Common reference parent: {REFERENCE_ROOT}")


if __name__ == "__main__":
    main()
