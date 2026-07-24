from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union


PathLike = Union[str, Path]
MetricName = Literal["fid", "is", "clip_fg", "clip_pg", "time"]
ModelFamily = Literal["sd15", "sdxl", "sd3"]

SUPPORTED_METRICS = {"fid", "is", "clip_fg", "clip_pg", "time"}


@dataclass(frozen=True)
class MetricEvaluationConfig:
    manifest_path: PathLike
    coco_root: PathLike
    generated_dir: Optional[PathLike] = None
    generation_summary: Optional[PathLike] = None
    output_dir: PathLike = Path("Ours") / "eval_outputs"

    split: str = "val2017"
    instances_json: Optional[PathLike] = None
    captions_json: Optional[PathLike] = None
    model_family: ModelFamily = "sd15"
    target_size: Optional[Tuple[int, int]] = None

    metrics: Sequence[str] = ("fid", "is", "clip_fg", "clip_pg", "time")
    max_samples: Optional[int] = None
    error_on_missing_generated: bool = True

    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    cache_resized_masks: bool = True
    cache_dir: Optional[PathLike] = Path("Ours") / "cache"

    device: str = "auto"
    fid_feature: int = 2048
    is_splits: int = 10
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    clip_batch_size: int = 32
    foreground_crop_padding_ratio: float = 0.08
    mask_foreground_for_clip: bool = True

    def selected_metrics(self) -> tuple[str, ...]:
        return tuple(metric.strip().lower() for metric in self.metrics)

    @property
    def target_hw(self) -> Tuple[int, int]:
        if self.target_size is not None:
            return self.target_size
        if self.model_family == "sd15":
            return (512, 512)
        if self.model_family in ("sdxl", "sd3"):
            return (1024, 1024)
        raise ValueError(f"Unsupported model_family: {self.model_family}")

    def resolved_manifest_path(self) -> Path:
        return Path(self.manifest_path)

    def resolved_coco_root(self) -> Path:
        return Path(self.coco_root)

    def resolved_generated_dir(self) -> Optional[Path]:
        if self.generated_dir is None:
            return None
        return Path(self.generated_dir)

    def resolved_generation_summary(self) -> Optional[Path]:
        if self.generation_summary is not None:
            return Path(self.generation_summary)
        generated_dir = self.resolved_generated_dir()
        if generated_dir is None:
            return None
        for filename in ("generation_summary.json", "smoke_generation_summary.json"):
            candidate = generated_dir / filename
            if candidate.exists():
                return candidate
        return generated_dir / "generation_summary.json"

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

    def resolved_instances_json(self) -> Optional[Path]:
        if self.instances_json is None:
            return None
        return Path(self.instances_json)

    def resolved_captions_json(self) -> Optional[Path]:
        if self.captions_json is None:
            return None
        return Path(self.captions_json)

    def validate(self) -> None:
        manifest_path = self.resolved_manifest_path()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.clip_batch_size <= 0:
            raise ValueError("clip_batch_size must be positive")
        if self.is_splits <= 0:
            raise ValueError("is_splits must be positive")
        if self.foreground_crop_padding_ratio < 0:
            raise ValueError("foreground_crop_padding_ratio must be non-negative")

        unsupported = set(self.selected_metrics()) - SUPPORTED_METRICS
        if unsupported:
            raise ValueError(f"Unsupported metrics: {sorted(unsupported)}")

        if self.resolved_generated_dir() is None and self.resolved_generation_summary() is None:
            raise ValueError("Provide generated_dir and/or generation_summary to locate generated images")
