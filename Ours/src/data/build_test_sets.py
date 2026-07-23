from __future__ import annotations

import argparse
from collections import Counter
import json
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


JsonDict = Dict[str, object]


MAIN_MANIFESTS = {
    "sd15_512": {
        "source": "coco_val2017_multidiffusion_coco_all_512x512_all.jsonl",
        "smoke": "coco_val2017_multidiffusion_coco_all_512x512_smoke_bs8.jsonl",
        "mini": "coco_val2017_multidiffusion_coco_all_512x512_mini32.jsonl",
        "batch_size": 8,
        "smoke_counts": {2: 3, 3: 3, 4: 2},
        "mini_counts": {2: 16, 3: 10, 4: 6},
    },
    "sdxl_1024": {
        "source": "coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_all.jsonl",
        "smoke": "coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_smoke_bs2.jsonl",
        "mini": "coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_mini32.jsonl",
        "batch_size": 2,
        "smoke_counts": {2: 1, 4: 1},
        "mini_counts": {2: 16, 3: 10, 4: 6},
    },
    "sd3_1024": {
        "source": "coco_val2017_multidiffusion_coco_all_sd3_1024x1024_all.jsonl",
        "smoke": "coco_val2017_multidiffusion_coco_all_sd3_1024x1024_smoke_bs2.jsonl",
        "mini": "coco_val2017_multidiffusion_coco_all_sd3_1024x1024_mini32.jsonl",
        "batch_size": 2,
        "smoke_counts": {2: 1, 4: 1},
        "mini_counts": {2: 16, 3: 10, 4: 6},
    },
}


def read_jsonl(path: Path) -> List[JsonDict]:
    records: List[JsonDict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def write_jsonl(records: Sequence[JsonDict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def write_json(data: JsonDict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def object_count(record: JsonDict) -> int:
    return len(record["annotation_ids"])  # type: ignore[arg-type]


def _evenly_spaced(records: Sequence[JsonDict], count: int) -> List[JsonDict]:
    if count <= 0:
        return []
    if count >= len(records):
        return list(records)
    if count == 1:
        return [records[0]]

    indexes = [round(i * (len(records) - 1) / (count - 1)) for i in range(count)]
    selected: List[JsonDict] = []
    used = set()
    for index in indexes:
        cursor = index
        while cursor in used and cursor + 1 < len(records):
            cursor += 1
        while cursor in used and cursor - 1 >= 0:
            cursor -= 1
        used.add(cursor)
        selected.append(records[cursor])
    return selected


def select_subset(records: Sequence[JsonDict], count_by_objects: Dict[int, int]) -> List[JsonDict]:
    groups: Dict[int, List[JsonDict]] = {}
    for record in sorted(records, key=lambda item: int(item["image_id"])):
        groups.setdefault(object_count(record), []).append(record)

    selected: List[JsonDict] = []
    for count, target_count in sorted(count_by_objects.items()):
        selected.extend(_evenly_spaced(groups.get(count, []), target_count))

    expected = sum(count_by_objects.values())
    if len(selected) != expected:
        raise ValueError(f"Expected {expected} selected records, got {len(selected)}")
    return sorted(selected, key=lambda item: int(item["image_id"]))


def build_summary(
    records: Sequence[JsonDict],
    source_manifest: Path,
    output_manifest: Path,
    subset_name: str,
    batch_size: int,
) -> JsonDict:
    object_counts = Counter(object_count(record) for record in records)
    categories = Counter(
        category
        for record in records
        for category in record["category_names"]  # type: ignore[union-attr]
    )
    target_sizes = Counter(tuple(record["target_size"]) for record in records)  # type: ignore[arg-type]
    model_families = Counter(str(record["model_family"]) for record in records)
    protocols = Counter(str(record["protocol"]) for record in records)
    pmax = max(object_counts) if object_counts else 0
    target_size = next(iter(target_sizes)) if target_sizes else (0, 0)

    return {
        "subset_name": subset_name,
        "source_manifest": str(source_manifest),
        "output_manifest": str(output_manifest),
        "num_records": len(records),
        "batch_size": batch_size,
        "expected_num_dataloader_batches": ceil(len(records) / batch_size) if batch_size else None,
        "expected_first_batch_mask_shape": [min(len(records), batch_size), pmax, 1, target_size[0], target_size[1]],
        "object_count_distribution": {str(k): v for k, v in sorted(object_counts.items())},
        "model_family_distribution": dict(model_families),
        "protocol_distribution": dict(protocols),
        "target_size_distribution": {f"{k[0]}x{k[1]}": v for k, v in sorted(target_sizes.items())},
        "top_categories": [{"category": key, "count": value} for key, value in categories.most_common(20)],
        "image_ids": [int(record["image_id"]) for record in records],
        "sample_ids": [str(record["sample_id"]) for record in records],
    }


def find_default_instances_json(repo_root: Path) -> Optional[Path]:
    candidates = [
        repo_root / "COCO" / "annotations_trainval2017" / "annotations" / "instances_val2017.json",
        repo_root / "annotations_trainval2017" / "annotations" / "instances_val2017.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def find_default_image_dir(repo_root: Path) -> Optional[Path]:
    candidates = [
        repo_root / "COCO" / "val2017" / "val2017",
        repo_root / "COCO" / "val2017",
        repo_root / "annotations_trainval2017" / "val2017",
        repo_root / "val2017",
    ]
    return next((path for path in candidates if path.is_dir()), None)


def load_annotations_by_id(instances_json: Path) -> Dict[int, JsonDict]:
    with instances_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(annotation["id"]): annotation for annotation in data.get("annotations", [])}


def make_sample_for_preview(
    record: JsonDict,
    annotations_by_id: Dict[int, JsonDict],
    image_dir: Path,
) -> dict:
    from PIL import Image

    from .coco_mask_utils import decode_coco_mask, mask_to_tensor, resize_mask_nearest, stack_masks

    target_size = tuple(int(v) for v in record["target_size"])  # type: ignore[arg-type]
    original_size = tuple(int(v) for v in record["original_size"])  # type: ignore[arg-type]
    masks = []
    for annotation_id in record["annotation_ids"]:  # type: ignore[union-attr]
        annotation = annotations_by_id[int(annotation_id)]
        mask_np = decode_coco_mask(annotation, original_size)
        mask_np = resize_mask_nearest(mask_np, target_size)
        masks.append(mask_to_tensor(mask_np))

    image_path = image_dir / str(record["file_name"])
    return {
        "sample_id": str(record["sample_id"]),
        "background_prompt": str(record["caption"]),
        "foreground_prompts": list(record["foreground_prompts"]),  # type: ignore[arg-type]
        "category_names": list(record["category_names"]),  # type: ignore[arg-type]
        "masks": stack_masks(masks),
        "target_size": target_size,
        "metadata": record,
        "image": Image.open(image_path).convert("RGB"),
    }


def export_previews(
    records: Sequence[JsonDict],
    output_dir: Path,
    annotations_by_id: Dict[int, JsonDict],
    image_dir: Path,
    limit: int,
) -> int:
    from .visualize import export_sample_preview

    exported = 0
    for record in records[: max(limit, 0)]:
        sample = make_sample_for_preview(record, annotations_by_id, image_dir)
        export_sample_preview(sample, output_dir)
        exported += 1
    return exported


def build_test_sets(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root)
    manifest_dir = Path(args.manifest_dir)
    output_dir = Path(args.output_dir)

    instances_json = Path(args.instances_json) if args.instances_json else find_default_instances_json(repo_root)
    image_dir = Path(args.image_dir) if args.image_dir else find_default_image_dir(repo_root)
    can_preview = instances_json is not None and image_dir is not None and args.preview_limit > 0
    annotations_by_id = load_annotations_by_id(instances_json) if can_preview and instances_json else {}

    for family, spec in MAIN_MANIFESTS.items():
        source_manifest = manifest_dir / str(spec["source"])
        records = read_jsonl(source_manifest)

        for subset_name, manifest_name, counts in [
            ("smoke", str(spec["smoke"]), spec["smoke_counts"]),
            ("mini32", str(spec["mini"]), spec["mini_counts"]),
        ]:
            subset = select_subset(records, counts)  # type: ignore[arg-type]
            subset_manifest = output_dir / "manifests" / subset_name / manifest_name
            write_jsonl(subset, subset_manifest)

            summary = build_summary(
                subset,
                source_manifest=source_manifest,
                output_manifest=subset_manifest,
                subset_name=subset_name,
                batch_size=int(spec["batch_size"]),
            )
            summary["preview_status"] = {
                "enabled": bool(can_preview),
                "instances_json": str(instances_json) if instances_json else None,
                "image_dir": str(image_dir) if image_dir else None,
                "preview_limit": args.preview_limit,
                "exported": 0,
                "error": None,
            }

            if can_preview and instances_json and image_dir:
                preview_dir = output_dir / "previews" / subset_name / family
                try:
                    exported = export_previews(
                        subset,
                        preview_dir,
                        annotations_by_id,
                        image_dir,
                        limit=args.preview_limit,
                    )
                    summary["preview_status"]["exported"] = exported  # type: ignore[index]
                except Exception as exc:
                    summary["preview_status"]["enabled"] = False  # type: ignore[index]
                    summary["preview_status"]["error"] = str(exc)  # type: ignore[index]

            report_path = output_dir / "reports" / subset_name / manifest_name.replace(".jsonl", "_summary.json")
            write_json(summary, report_path)
            print(f"{subset_manifest}\t{len(subset)} records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic smoke/mini test sets from full COCO manifests.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--manifest-dir", type=Path, default=Path("Ours") / "data_manifests")
    parser.add_argument("--output-dir", type=Path, default=Path("Ours") / "test_sets")
    parser.add_argument("--instances-json", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--preview-limit", type=int, default=8)
    args = parser.parse_args()
    build_test_sets(args)


if __name__ == "__main__":
    main()
