"""Expose a verified full run to the older standalone FID programs without copying images."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import shutil
from pathlib import Path

from verify_run import MANIFESTS, verify_run


def prepare_metric_export(profile: str, run_dir: Path) -> Path:
    verify_run(profile, run_dir)
    family = "sdxl" if "sdxl" in profile else "sd15"
    manifest = [json.loads(line) for line in MANIFESTS[family].read_text(encoding="utf-8").splitlines() if line.strip()]
    summary_path = next(run_dir.rglob("generation_summary.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    image_paths = [Path(row["generated_path"]).resolve() for row in summary]
    original_paths = image_paths[:]
    generated_dirs = {path.parent for path in image_paths}
    if len(generated_dirs) != 1:
        raise RuntimeError(f"Expected one generated image folder, found {generated_dirs}")
    source_dir = next(iter(generated_dirs))
    storage_modes = {"source"}
    if source_dir.name == "generated_images":
        export_dir = source_dir.parent
    else:
        storage_modes.clear()
        export_dir = run_dir / "metric_export"
        target_images = export_dir / "generated_images"
        target_images.mkdir(parents=True, exist_ok=True)
        for image in image_paths:
            target_image = target_images / image.name
            if target_image.exists():
                if not os.path.samefile(image, target_image) and not filecmp.cmp(image, target_image, shallow=False):
                    raise RuntimeError(f"Existing metric image differs from source: {target_image}")
                storage_modes.add("existing")
            else:
                try:
                    os.link(image, target_image)
                    storage_modes.add("hardlink")
                except OSError:
                    try:
                        os.symlink(image, target_image)
                        storage_modes.add("symlink")
                    except OSError:
                        shutil.copy2(image, target_image)
                        storage_modes.add("copy")
        image_paths = [target_images / image.name for image in image_paths]
    target = export_dir / "metric_generated_manifest.jsonl"
    records = []
    for index, (source, row, image) in enumerate(zip(manifest, summary, image_paths)):
        records.append({
            "metric_index": index,
            "experiment_id": profile,
            "sample_id": source["sample_id"],
            "image_id": int(source["image_id"]),
            "file_name": source["file_name"],
            "generated_image_path": str(image),
            "generated_image_relative_path": str(image.relative_to(export_dir)),
            "source_generated_path": str(original_paths[index]),
            "source_manifest_path": str(MANIFESTS[family]),
            "source_generation_summary": str(summary_path),
            "coco_original_path": str(Path(os.environ["COCO_ROOT"]) / "val2017" / source["file_name"]),
            "background_prompt": source.get("caption"),
            "foreground_prompts": source.get("foreground_prompts"),
            "category_names": source.get("category_names"),
            "annotation_ids": source.get("annotation_ids"),
            "target_size": source["target_size"],
            "seed": row["seed"],
            "elapsed_sec": row.get("elapsed_sec"),
            "generation_metadata": row,
        })
    content = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    if target.exists():
        if target.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Existing metric manifest differs from verified run: {target}")
    else:
        target.write_text(content, encoding="utf-8")
    csv_path = export_dir / "metric_generated_manifest.csv"
    fields = ("metric_index", "experiment_id", "sample_id", "image_id", "file_name",
              "generated_image_relative_path", "source_generated_path", "seed", "elapsed_sec")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})
    export_summary = {
        "experiment_id": profile,
        "num_generated_images": len(records),
        "generated_images_dir": str(export_dir / "generated_images"),
        "manifest_jsonl": str(target),
        "manifest_csv": str(csv_path),
        "source_manifest_path": str(MANIFESTS[family]),
        "source_generation_summary": str(summary_path),
        "image_storage": sorted(storage_modes),
    }
    (export_dir / "metric_export_summary.json").write_text(
        json.dumps(export_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(prepare_metric_export(args.profile, args.run_dir))


if __name__ == "__main__":
    main()
