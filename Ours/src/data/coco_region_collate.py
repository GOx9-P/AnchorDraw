from __future__ import annotations

from typing import Dict, List

import torch


def collate_coco_region_batch(samples: List[dict]) -> dict:
    if not samples:
        raise ValueError("Cannot collate an empty batch")

    batch_size = len(samples)
    pmax = max(int(sample["masks"].shape[0]) for sample in samples)
    _, h, w = samples[0]["masks"].shape[-3:]
    mask_dtype = samples[0]["masks"].dtype

    masks = torch.zeros(batch_size, pmax, 1, h, w, dtype=mask_dtype)
    valid_regions = torch.zeros(batch_size, pmax, dtype=torch.bool)
    category_ids = torch.full((batch_size, pmax), -1, dtype=torch.long)
    boxes = torch.zeros(batch_size, pmax, 4, dtype=torch.float32)
    area_ratios = torch.zeros(batch_size, pmax, dtype=torch.float32)

    foreground_prompts = []
    category_names = []
    images = []
    has_images = "image" in samples[0]

    for b, sample in enumerate(samples):
        p = int(sample["masks"].shape[0])
        masks[b, :p] = sample["masks"]
        valid_regions[b, :p] = True
        category_ids[b, :p] = torch.as_tensor(sample["category_ids"], dtype=torch.long)
        boxes[b, :p] = sample["boxes_xyxy"]
        area_ratios[b, :p] = sample["area_ratios"]
        foreground_prompts.append(sample["foreground_prompts"])
        category_names.append(sample["category_names"])
        if has_images:
            images.append(sample["image"])

    batch: Dict[str, object] = {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "image_ids": [sample["image_id"] for sample in samples],
        "file_names": [sample["file_name"] for sample in samples],
        "background_prompts": [sample["background_prompt"] for sample in samples],
        "foreground_prompts": foreground_prompts,
        "category_names": category_names,
        "category_ids": category_ids,
        "masks": masks,
        "valid_regions": valid_regions,
        "boxes_xyxy": boxes,
        "area_ratios": area_ratios,
        "original_sizes": torch.as_tensor([sample["original_size"] for sample in samples], dtype=torch.long),
        "target_sizes": torch.as_tensor([sample["target_size"] for sample in samples], dtype=torch.long),
        "annotation_ids": [sample["annotation_ids"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
    }
    if has_images:
        batch["images"] = images
    return batch

