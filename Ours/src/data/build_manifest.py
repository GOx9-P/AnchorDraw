from __future__ import annotations

import argparse
from pathlib import Path

from .coco_profiles import (
    multidiffusion_coco_all,
    multidiffusion_coco_1k,
    ours_overlap_stress,
    ours_weighted_mask,
    semanticdraw_sd15,
    semanticdraw_sd3,
    semanticdraw_sdxl,
)
from .coco_region_config import COCORegionConfig
from .coco_region_sampler import build_coco_manifest


def _make_config(args: argparse.Namespace) -> COCORegionConfig:
    common = {
        "seed": args.seed,
        "subset_size": args.subset_size,
        "manifest_path": args.manifest_path,
    }
    if args.profile == "semanticdraw":
        if args.model_family == "sd15":
            config = semanticdraw_sd15(args.coco_root, **common)
        elif args.model_family == "sdxl":
            config = semanticdraw_sdxl(args.coco_root, **common)
        elif args.model_family == "sd3":
            config = semanticdraw_sd3(args.coco_root, **common)
        else:
            raise ValueError(f"Unsupported model_family: {args.model_family}")
    elif args.profile == "multidiffusion_coco_all":
        config = multidiffusion_coco_all(args.coco_root, model_family=args.model_family, **common)
    elif args.profile == "multidiffusion_coco_1k":
        config = multidiffusion_coco_1k(args.coco_root, model_family=args.model_family, **common)
    elif args.profile == "ours_weighted_mask":
        config = ours_weighted_mask(args.coco_root, model_family=args.model_family, **common)
    elif args.profile == "ours_overlap_stress":
        config = ours_overlap_stress(args.coco_root, model_family=args.model_family, **common)
    else:
        raise ValueError(f"Unsupported profile: {args.profile}")

    overrides = {}
    if args.min_objects is not None:
        overrides["min_objects"] = args.min_objects
    if args.max_objects is not None:
        overrides["max_objects"] = args.max_objects
    if args.min_mask_area_ratio is not None:
        overrides["min_mask_area_ratio"] = args.min_mask_area_ratio
    if args.truncate_objects:
        overrides["truncate_objects"] = True
    if args.caption_policy is not None:
        overrides["caption_policy"] = args.caption_policy
    if args.object_policy is not None:
        overrides["object_policy"] = args.object_policy
    return config.copy_with(**overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fixed COCO manifest for region experiments.")
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=[
            "semanticdraw",
            "multidiffusion_coco_all",
            "multidiffusion_coco_1k",
            "ours_weighted_mask",
            "ours_overlap_stress",
        ],
        default="multidiffusion_coco_all",
    )
    parser.add_argument("--model-family", choices=["sd15", "sdxl", "sd3"], default="sd15")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of valid images to sample. Omit to keep all valid images.",
    )
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--min-objects", type=int, default=None)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--min-mask-area-ratio", type=float, default=None)
    parser.add_argument("--truncate-objects", action="store_true")
    parser.add_argument("--caption-policy", choices=["first", "seeded_random"], default=None)
    parser.add_argument("--object-policy", choices=["largest", "seeded_random"], default=None)

    args = parser.parse_args()
    config = _make_config(args)
    records = build_coco_manifest(config, overwrite=args.overwrite)
    print(f"Manifest: {config.resolved_manifest_path()}")
    print(f"Records: {len(records)}")


if __name__ == "__main__":
    main()
