from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]

ProfileName = Literal[
    "semanticdraw",
    "multidiffusion_coco_all",
    "multidiffusion_coco_1k",
    "ours_weighted_mask",
    "ours_overlap_stress",
]
ModelFamily = Literal["sd15", "sdxl", "sd3"]
CaptionPolicy = Literal["first", "seeded_random"]
ObjectPolicy = Literal["largest", "seeded_random"]
MaskDType = Literal["uint8", "bool"]


@dataclass(frozen=True)
class COCORegionConfig:
    coco_root: PathLike
    split: str = "val2017"
    instances_json: Optional[PathLike] = None
    captions_json: Optional[PathLike] = None

    profile: ProfileName = "semanticdraw"
    model_family: ModelFamily = "sd15"
    target_size: Optional[Tuple[int, int]] = None

    manifest_path: Optional[PathLike] = None
    manifest_dir: PathLike = Path("Ours") / "data_manifests"
    build_manifest_if_missing: bool = False
    manifest_version: int = 1
    seed: int = 42
    subset_size: Optional[int] = None

    min_objects: int = 2
    max_objects: int = 4
    truncate_objects: bool = False
    exclude_categories: Sequence[str] = ("person",)
    min_mask_area_ratio: float = 0.05
    drop_iscrowd: bool = True

    prompt_template: str = "a {label}"
    caption_policy: CaptionPolicy = "first"
    object_policy: ObjectPolicy = "largest"

    mask_resize_mode: Literal["nearest"] = "nearest"
    mask_dtype: MaskDType = "uint8"
    return_image: bool = False
    cache_index: bool = True
    cache_resized_masks: bool = True
    cache_dir: Optional[PathLike] = Path("Ours") / "cache"

    batch_size: int = 8
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    def copy_with(self, **overrides: object) -> "COCORegionConfig":
        return replace(self, **overrides)

    @property
    def target_hw(self) -> Tuple[int, int]:
        if self.target_size is not None:
            return self.target_size
        if self.model_family == "sd15":
            return (512, 512)
        if self.model_family in ("sdxl", "sd3"):
            return (1024, 1024)
        raise ValueError(f"Unsupported model_family: {self.model_family}")

    @property
    def target_h(self) -> int:
        return self.target_hw[0]

    @property
    def target_w(self) -> int:
        return self.target_hw[1]

    def resolved_coco_root(self) -> Path:
        return Path(self.coco_root)

    def resolved_image_dir(self) -> Path:
        return self.resolved_coco_root() / self.split

    def resolved_instances_json(self) -> Path:
        if self.instances_json is not None:
            return Path(self.instances_json)
        return self.resolved_coco_root() / "annotations" / f"instances_{self.split}.json"

    def resolved_captions_json(self) -> Path:
        if self.captions_json is not None:
            return Path(self.captions_json)
        return self.resolved_coco_root() / "annotations" / f"captions_{self.split}.json"

    def resolved_manifest_dir(self) -> Path:
        return Path(self.manifest_dir)

    def resolved_cache_dir(self) -> Path:
        if self.cache_dir is None:
            return Path("Ours") / "cache"
        return Path(self.cache_dir)

    def resolved_manifest_path(self) -> Path:
        if self.manifest_path is not None:
            return Path(self.manifest_path)

        subset = "all" if self.subset_size is None else str(self.subset_size)
        h, w = self.target_hw
        profile = self.profile
        if profile == "semanticdraw":
            profile = f"semanticdraw_{self.model_family}"
        elif profile == "multidiffusion_coco_all" and self.model_family != "sd15":
            profile = f"multidiffusion_coco_all_{self.model_family}"
        elif profile == "ours_weighted_mask":
            profile = f"ours_weighted_mask_{self.model_family}"
        elif profile == "ours_overlap_stress":
            profile = f"ours_overlap_stress_{self.model_family}"

        uses_seeded_selection = (
            self.subset_size is not None or
            self.caption_policy == "seeded_random" or
            self.object_policy == "seeded_random"
        )
        seed_part = f"_seed{self.seed}" if uses_seeded_selection else ""
        filename = f"coco_{self.split}_{profile}_{h}x{w}{seed_part}_{subset}.jsonl"
        return self.resolved_manifest_dir() / filename

    def validate(self) -> None:
        if self.min_objects < 0:
            raise ValueError("min_objects must be non-negative")
        if self.max_objects < self.min_objects:
            raise ValueError("max_objects must be >= min_objects")
        if self.min_mask_area_ratio < 0:
            raise ValueError("min_mask_area_ratio must be >= 0")
        if self.subset_size is not None and self.subset_size <= 0:
            raise ValueError("subset_size must be positive or None")
        if self.mask_resize_mode != "nearest":
            raise ValueError("Only nearest mask resizing is supported")
        if self.mask_dtype not in ("uint8", "bool"):
            raise ValueError("mask_dtype must be 'uint8' or 'bool'")
