"""Fail before model loading when the fixed full1073 input set is incomplete."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def verify_inputs(profile: str, coco_root: Path, run_profile: str | None = None) -> dict:
    family = "sdxl" if "sdxl" in profile else "sd15"
    suffix = "sdxl_1024x1024" if family == "sdxl" else "512x512"
    manifest = ROOT / "Ours/data_manifests" / f"coco_val2017_multidiffusion_coco_all_{suffix}_all.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1073:
        raise RuntimeError(f"Expected 1073 manifest records, found {len(rows)}: {manifest}")
    ids = [str(row["sample_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate sample_id in full manifest")
    annotations = {}
    annotation_hashes = {}
    for annotation in ("instances_val2017.json", "captions_val2017.json"):
        path = coco_root / "annotations" / annotation
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise RuntimeError(f"Invalid COCO annotation JSON: {path}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("images"), list) or not isinstance(payload.get("annotations"), list):
            raise RuntimeError(f"Unexpected COCO annotation structure: {path}")
        annotations[annotation] = payload
        annotation_hashes[annotation] = hashlib.sha256(path.read_bytes()).hexdigest()
    instances = annotations["instances_val2017.json"]
    captions = annotations["captions_val2017.json"]
    instance_images = {int(image["id"]): image for image in instances["images"]}
    caption_image_ids = {int(image["id"]) for image in captions["images"]}
    object_annotations = {int(annotation["id"]): annotation for annotation in instances["annotations"]}
    caption_annotations = {int(annotation["id"]): annotation for annotation in captions["annotations"]}
    categories = {int(category["id"]): category["name"] for category in instances["categories"]}
    for row in rows:
        image_id = int(row["image_id"])
        image_metadata = instance_images.get(image_id)
        if (image_metadata is None or image_metadata["file_name"] != row["file_name"]
                or [int(image_metadata["width"]), int(image_metadata["height"])] != row["original_size"]
                or image_id not in caption_image_ids):
            raise RuntimeError(f"Manifest image absent or mismatched in COCO annotations: {row['sample_id']}")
        if not (len(row["annotation_ids"]) == len(row["category_ids"]) == len(row["category_names"])):
            raise RuntimeError(f"Manifest region metadata has different lengths: {row['sample_id']}")
        for annotation_id, category_id, category_name in zip(
                row["annotation_ids"], row["category_ids"], row["category_names"]):
            obj = object_annotations.get(int(annotation_id))
            if (obj is None or int(obj["image_id"]) != image_id
                    or int(obj["category_id"]) != int(category_id)
                    or categories.get(int(category_id)) != category_name):
                raise RuntimeError(f"Manifest object annotation differs from COCO instances: {row['sample_id']}")
        caption = caption_annotations.get(int(row["caption_id"]))
        if (caption is None or int(caption["image_id"]) != image_id
                or caption["caption"] != row["caption"]):
            raise RuntimeError(f"Manifest caption differs from COCO captions: {row['sample_id']}")
    names_to_check = {row["file_name"] for row in rows}
    if profile == "diag_sem_sd15_lcm_semanticfull" and run_profile != "smoke":
        names_to_check.update(image["file_name"] for image in instance_images.values())
    missing = [name for name in sorted(names_to_check) if not (coco_root / "val2017" / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} COCO images; first five: {missing[:5]}")
    image_set_hash = hashlib.sha256()
    manifest_image_sizes = {row["file_name"]: tuple(row["original_size"]) for row in rows}
    for name in sorted(names_to_check):
        path = coco_root / "val2017" / name
        try:
            with Image.open(path) as image:
                if image.size[0] <= 0 or image.size[1] <= 0:
                    raise RuntimeError(f"Empty COCO image: {path}")
                if name in manifest_image_sizes and image.size != manifest_image_sizes[name]:
                    raise RuntimeError(f"COCO image dimensions differ from manifest: {path}")
                image.verify()
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Corrupt COCO image: {path}") from exc
        if name in manifest_image_sizes:
            image_set_hash.update(name.encode("utf-8"))
            image_set_hash.update(hashlib.sha256(path.read_bytes()).digest())
    return {"profile": profile, "run_profile_override": run_profile, "family": family,
            "coco_root": str(coco_root.resolve()),
            "records": len(rows), "checked_images": len(names_to_check),
            "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "instances_sha256": annotation_hashes["instances_val2017.json"],
            "captions_sha256": annotation_hashes["captions_val2017.json"],
            "manifest_images_sha256": image_set_hash.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("coco_root", type=Path)
    parser.add_argument("run_profile", nargs="?", default=None)
    args = parser.parse_args()
    print(json.dumps(verify_inputs(args.profile, args.coco_root, args.run_profile), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
