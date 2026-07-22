from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .coco_mask_utils import (
    cache_path_for_mask,
    decode_coco_mask,
    mask_to_tensor,
    rescale_bbox_xyxy,
    resize_mask_nearest,
    stack_masks,
)
from .coco_region_collate import collate_coco_region_batch
from .coco_region_config import COCORegionConfig
from .coco_region_manifest import JsonDict, load_coco_index, load_manifest
from .coco_region_sampler import build_coco_manifest


class COCORegionDataset(Dataset):
    def __init__(self, config: COCORegionConfig):
        config.validate()
        self.config = config
        self.manifest_path = config.resolved_manifest_path()

        if not self.manifest_path.exists():
            if config.build_manifest_if_missing:
                build_coco_manifest(config)
            else:
                raise FileNotFoundError(
                    f"Manifest not found: {self.manifest_path}. "
                    "Build it with build_coco_manifest(config), or set "
                    "build_manifest_if_missing=True for explicit auto-build."
                )

        self.records = load_manifest(self.manifest_path)
        self.index = load_coco_index(
            config.resolved_instances_json(),
            config.resolved_captions_json(),
        )

    def __len__(self) -> int:
        return len(self.records)

    def _mask_cache_path(self, image_id: int, annotation_id: int, target_size: tuple[int, int]) -> Path:
        return cache_path_for_mask(
            self.config.resolved_cache_dir(),
            self.config.split,
            target_size,
            image_id,
            annotation_id,
        )

    def _load_or_decode_mask(
        self,
        record: JsonDict,
        annotation: JsonDict,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        image_id = int(record["image_id"])
        annotation_id = int(annotation["id"])
        cache_path = self._mask_cache_path(image_id, annotation_id, target_size)

        if self.config.cache_resized_masks and cache_path.exists():
            mask = torch.load(cache_path, map_location="cpu")
            if self.config.mask_dtype == "bool":
                return mask.bool()
            return (mask > 0).to(torch.uint8)

        original_size = tuple(int(v) for v in record["original_size"])
        mask_np = decode_coco_mask(annotation, original_size)
        mask_np = resize_mask_nearest(mask_np, target_size)
        mask = mask_to_tensor(mask_np, dtype=self.config.mask_dtype)

        if self.config.cache_resized_masks:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mask.cpu(), cache_path)

        return mask

    def _load_image(self, record: JsonDict) -> Image.Image:
        image_path = self.config.resolved_image_dir() / str(record["file_name"])
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        target_size = self.config.target_hw
        original_size = tuple(int(v) for v in record["original_size"])

        masks = []
        boxes = []
        annotation_ids = [int(v) for v in record["annotation_ids"]]
        for annotation_id in annotation_ids:
            annotation = self.index.annotations_by_id[annotation_id]
            masks.append(self._load_or_decode_mask(record, annotation, target_size))
            boxes.append(rescale_bbox_xyxy(annotation["bbox"], original_size, target_size))

        sample = {
            "sample_id": str(record["sample_id"]),
            "image_id": int(record["image_id"]),
            "file_name": str(record["file_name"]),
            "background_prompt": str(record["caption"]),
            "foreground_prompts": list(record["foreground_prompts"]),
            "category_names": list(record["category_names"]),
            "category_ids": [int(v) for v in record["category_ids"]],
            "masks": stack_masks(masks),
            "boxes_xyxy": torch.as_tensor(boxes, dtype=torch.float32),
            "area_ratios": torch.as_tensor(record["area_ratios"], dtype=torch.float32),
            "original_size": original_size,
            "target_size": target_size,
            "annotation_ids": annotation_ids,
            "metadata": record,
        }
        if self.config.return_image:
            sample["image"] = self._load_image(record)
        return sample


def build_coco_region_dataloader(
    config: COCORegionConfig,
    shuffle: bool = False,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    dataset = COCORegionDataset(config)
    kwargs = {
        "dataset": dataset,
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "collate_fn": collate_coco_region_batch,
        "generator": generator,
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(**kwargs)

