from __future__ import annotations

from pathlib import Path

from .coco_region_config import COCORegionConfig, ModelFamily, PathLike


def semanticdraw_sd15(coco_root: PathLike, **overrides: object) -> COCORegionConfig:
    config = COCORegionConfig(
        coco_root=Path(coco_root),
        profile="semanticdraw",
        model_family="sd15",
        target_size=(512, 512),
        min_objects=1,
        max_objects=8,
        truncate_objects=True,
        exclude_categories=(),
        min_mask_area_ratio=0.0,
        batch_size=8,
    )
    return config.copy_with(**overrides)


def semanticdraw_sdxl(coco_root: PathLike, **overrides: object) -> COCORegionConfig:
    config = COCORegionConfig(
        coco_root=Path(coco_root),
        profile="semanticdraw",
        model_family="sdxl",
        target_size=(1024, 1024),
        min_objects=1,
        max_objects=8,
        truncate_objects=True,
        exclude_categories=(),
        min_mask_area_ratio=0.0,
        batch_size=2,
    )
    return config.copy_with(**overrides)


def semanticdraw_sd3(coco_root: PathLike, **overrides: object) -> COCORegionConfig:
    config = semanticdraw_sdxl(coco_root).copy_with(model_family="sd3")
    return config.copy_with(**overrides)


def multidiffusion_coco_all(coco_root: PathLike, **overrides: object) -> COCORegionConfig:
    config = COCORegionConfig(
        coco_root=Path(coco_root),
        profile="multidiffusion_coco_all",
        model_family="sd15",
        target_size=(512, 512),
        seed=42,
        subset_size=None,
        min_objects=2,
        max_objects=4,
        truncate_objects=False,
        exclude_categories=("person",),
        min_mask_area_ratio=0.05,
        drop_iscrowd=True,
        prompt_template="a {label}",
        caption_policy="first",
        object_policy="largest",
        batch_size=8,
    )
    return config.copy_with(**overrides)


def multidiffusion_coco_1k(coco_root: PathLike, **overrides: object) -> COCORegionConfig:
    """Backward-compatible name. It now returns all valid COCO val samples."""
    return multidiffusion_coco_all(coco_root, **overrides)


def ours_weighted_mask(
    coco_root: PathLike,
    model_family: ModelFamily = "sd15",
    **overrides: object,
) -> COCORegionConfig:
    target_size = (512, 512) if model_family == "sd15" else (1024, 1024)
    batch_size = 8 if model_family == "sd15" else 2
    config = COCORegionConfig(
        coco_root=Path(coco_root),
        profile="ours_weighted_mask",
        model_family=model_family,
        target_size=target_size,
        seed=42,
        subset_size=None,
        min_objects=2,
        max_objects=4,
        truncate_objects=False,
        exclude_categories=("person",),
        min_mask_area_ratio=0.05,
        drop_iscrowd=True,
        prompt_template="a {label}",
        caption_policy="first",
        object_policy="largest",
        batch_size=batch_size,
    )
    return config.copy_with(**overrides)


def ours_overlap_stress(coco_root: PathLike, **overrides: object) -> COCORegionConfig:
    config = COCORegionConfig(
        coco_root=Path(coco_root),
        profile="ours_overlap_stress",
        model_family="sd15",
        target_size=(512, 512),
        seed=42,
        subset_size=None,
        min_objects=2,
        max_objects=4,
        truncate_objects=False,
        exclude_categories=("person",),
        min_mask_area_ratio=0.02,
        drop_iscrowd=True,
        prompt_template="a {label}",
        caption_policy="first",
        object_policy="largest",
        batch_size=8,
    )
    return config.copy_with(**overrides)
