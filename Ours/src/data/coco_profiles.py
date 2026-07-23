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


def _target_size_for_model(model_family: ModelFamily) -> tuple[int, int]:
    return (512, 512) if model_family == "sd15" else (1024, 1024)


def _batch_size_for_model(model_family: ModelFamily) -> int:
    return 8 if model_family == "sd15" else 2


def multidiffusion_coco_all(
    coco_root: PathLike,
    model_family: ModelFamily = "sd15",
    **overrides: object,
) -> COCORegionConfig:
    config = COCORegionConfig(
        coco_root=Path(coco_root),
        profile="multidiffusion_coco_all",
        model_family=model_family,
        target_size=_target_size_for_model(model_family),
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
        batch_size=_batch_size_for_model(model_family),
    )
    return config.copy_with(**overrides)


def multidiffusion_coco_1k(
    coco_root: PathLike,
    model_family: ModelFamily = "sd15",
    **overrides: object,
) -> COCORegionConfig:
    """Backward-compatible name. It now returns all valid COCO val samples."""
    return multidiffusion_coco_all(coco_root, model_family=model_family, **overrides)


def ours_weighted_mask(
    coco_root: PathLike,
    model_family: ModelFamily = "sd15",
    **overrides: object,
) -> COCORegionConfig:
    config = COCORegionConfig(
        coco_root=Path(coco_root),
        profile="ours_weighted_mask",
        model_family=model_family,
        target_size=_target_size_for_model(model_family),
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
        batch_size=_batch_size_for_model(model_family),
    )
    return config.copy_with(**overrides)


def ours_overlap_stress(
    coco_root: PathLike,
    model_family: ModelFamily = "sd15",
    **overrides: object,
) -> COCORegionConfig:
    config = COCORegionConfig(
        coco_root=Path(coco_root),
        profile="ours_overlap_stress",
        model_family=model_family,
        target_size=_target_size_for_model(model_family),
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
        batch_size=_batch_size_for_model(model_family),
    )
    return config.copy_with(**overrides)
