from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch


_COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (42, 157, 143),
    (244, 162, 97),
    (131, 56, 236),
    (255, 190, 11),
    (58, 134, 255),
    (6, 214, 160),
]


def _mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[0]
    return (mask.detach().cpu().numpy() > 0)


def make_mask_overlay(
    image: Image.Image,
    masks: torch.Tensor,
    labels: Sequence[str],
    alpha: float = 0.45,
) -> Image.Image:
    base = image.convert("RGB")
    overlay = np.asarray(base).astype(np.float32)

    for index, mask in enumerate(masks):
        color = np.asarray(_COLORS[index % len(_COLORS)], dtype=np.float32)
        mask_np = _mask_to_numpy(mask)
        overlay[mask_np] = (1.0 - alpha) * overlay[mask_np] + alpha * color

    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    for index, label in enumerate(labels):
        mask_np = _mask_to_numpy(masks[index])
        ys, xs = np.where(mask_np)
        if len(xs) == 0:
            continue
        x, y = int(xs.min()), int(ys.min())
        color = _COLORS[index % len(_COLORS)]
        draw.rectangle((x, y, x + 8 + 7 * len(label), y + 18), fill=color)
        draw.text((x + 4, y + 3), label, fill=(255, 255, 255))
    return out


def export_sample_preview(sample: dict, output_dir: Path, image: Optional[Image.Image] = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image is None:
        if "image" not in sample:
            raise ValueError("sample has no image; set return_image=True or pass image explicitly")
        image = sample["image"]

    target_h, target_w = sample["target_size"]
    image = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
    overlay = make_mask_overlay(image, sample["masks"], sample["category_names"])

    stem = str(sample["sample_id"])
    image_path = output_dir / f"{stem}_overlay.png"
    meta_path = output_dir / f"{stem}.json"
    overlay.save(image_path)

    metadata = dict(sample["metadata"])
    metadata["background_prompt"] = sample["background_prompt"]
    metadata["foreground_prompts"] = sample["foreground_prompts"]
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return image_path


def export_dataset_previews(dataset: Iterable[dict], output_dir: Path, limit: int = 20) -> None:
    for index, sample in enumerate(dataset):
        if index >= limit:
            break
        export_sample_preview(sample, output_dir)

