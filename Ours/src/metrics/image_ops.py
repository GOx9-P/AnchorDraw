from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image
import torch


def resize_rgb(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    target_h, target_w = int(target_size[0]), int(target_size[1])
    if image.size == (target_w, target_h):
        return image.convert("RGB")
    return image.convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)


def pil_list_to_uint8_tensor(images: list[Image.Image], target_size: Tuple[int, int]) -> torch.Tensor:
    tensors = []
    for image in images:
        resized = resize_rgb(image, target_size)
        array = np.asarray(resized, dtype=np.uint8)
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
    return torch.stack(tensors, dim=0)


def mask_to_bbox_xyxy(mask: torch.Tensor, padding_ratio: float = 0.0) -> tuple[int, int, int, int]:
    if mask.ndim == 3:
        mask_2d = mask[0]
    elif mask.ndim == 2:
        mask_2d = mask
    else:
        raise ValueError(f"Expected mask shape (1,H,W) or (H,W), got {tuple(mask.shape)}")

    mask_2d = mask_2d.detach().cpu() > 0
    ys, xs = torch.where(mask_2d)
    height, width = mask_2d.shape
    if ys.numel() == 0:
        return (0, 0, int(width), int(height))

    x1 = int(xs.min().item())
    x2 = int(xs.max().item()) + 1
    y1 = int(ys.min().item())
    y2 = int(ys.max().item()) + 1

    pad_x = int(round((x2 - x1) * padding_ratio))
    pad_y = int(round((y2 - y1) * padding_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(int(width), x2 + pad_x)
    y2 = min(int(height), y2 + pad_y)
    return (x1, y1, x2, y2)


def make_masked_foreground_crop(
    image: Image.Image,
    mask: torch.Tensor,
    padding_ratio: float = 0.08,
    apply_mask: bool = True,
) -> Image.Image:
    resized = resize_rgb(image, (mask.shape[-2], mask.shape[-1]))
    bbox = mask_to_bbox_xyxy(mask, padding_ratio=padding_ratio)

    if not apply_mask:
        return resized.crop(bbox)

    mask_np = (mask.detach().cpu().squeeze(0).numpy() > 0)
    image_np = np.asarray(resized, dtype=np.uint8)
    white_np = np.full_like(image_np, 255)
    masked_np = np.where(mask_np[..., None], image_np, white_np).astype(np.uint8)
    return Image.fromarray(masked_np, mode="RGB").crop(bbox)


def make_masked_background_image(
    image: Image.Image,
    foreground_masks: torch.Tensor,
    apply_mask: bool = True,
) -> Image.Image:
    if foreground_masks.ndim != 4 or foreground_masks.shape[1] != 1:
        raise ValueError(
            "Expected foreground_masks shape (P,1,H,W), "
            f"got {tuple(foreground_masks.shape)}"
        )

    height, width = int(foreground_masks.shape[-2]), int(foreground_masks.shape[-1])
    resized = resize_rgb(image, (height, width))
    if not apply_mask:
        return resized

    foreground_union = foreground_masks.detach().cpu().to(dtype=torch.float32).amax(dim=0).squeeze(0) > 0
    background_mask_np = ~foreground_union.numpy()
    image_np = np.asarray(resized, dtype=np.uint8)
    white_np = np.full_like(image_np, 255)
    masked_np = np.where(background_mask_np[..., None], image_np, white_np).astype(np.uint8)
    return Image.fromarray(masked_np, mode="RGB")
