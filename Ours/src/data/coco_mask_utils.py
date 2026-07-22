from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
from PIL import Image
import torch


def _require_pycocotools():
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError(
            "pycocotools is required for COCO mask decoding. "
            "Install it with: pip install pycocotools"
        ) from exc
    return mask_utils


def decode_coco_mask(annotation: dict, original_size: Tuple[int, int]) -> np.ndarray:
    mask_utils = _require_pycocotools()
    height, width = int(original_size[0]), int(original_size[1])
    segmentation = annotation.get("segmentation")

    if segmentation is None:
        raise ValueError(f"Annotation {annotation.get('id')} has no segmentation")

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)!r}")

    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(np.uint8)


def resize_mask_nearest(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = int(target_size[0]), int(target_size[1])
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    image = image.resize((target_w, target_h), Image.Resampling.NEAREST)
    return (np.asarray(image) > 0).astype(np.uint8)


def mask_to_tensor(mask: np.ndarray, dtype: str = "uint8") -> torch.Tensor:
    tensor = torch.from_numpy((mask > 0).astype(np.uint8)).unsqueeze(0)
    if dtype == "bool":
        return tensor.bool()
    if dtype == "uint8":
        return tensor.to(torch.uint8)
    raise ValueError(f"Unsupported mask dtype: {dtype}")


def coco_bbox_xywh_to_xyxy(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return (x, y, x + w, y + h)


def rescale_bbox_xyxy(
    bbox_xywh: Sequence[float],
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    orig_h, orig_w = int(original_size[0]), int(original_size[1])
    target_h, target_w = int(target_size[0]), int(target_size[1])
    x1, y1, x2, y2 = coco_bbox_xywh_to_xyxy(bbox_xywh)
    sx = target_w / max(orig_w, 1)
    sy = target_h / max(orig_h, 1)
    return (x1 * sx, y1 * sy, x2 * sx, y2 * sy)


def annotation_area_ratio(annotation: dict, original_size: Tuple[int, int]) -> float:
    height, width = int(original_size[0]), int(original_size[1])
    return float(annotation.get("area", 0.0)) / max(float(height * width), 1.0)


def ensure_binary_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    return (mask > 0).to(mask.dtype)


def cache_path_for_mask(
    cache_root: Path,
    split: str,
    target_size: Tuple[int, int],
    image_id: int,
    annotation_id: int,
) -> Path:
    h, w = target_size
    directory = Path(cache_root) / "coco_masks" / f"{split}_{h}x{w}"
    return directory / f"image_{int(image_id):012d}_ann_{int(annotation_id)}.pt"


def stack_masks(masks: Iterable[torch.Tensor]) -> torch.Tensor:
    masks = list(masks)
    if not masks:
        raise ValueError("Cannot stack an empty mask list")
    return torch.stack(masks, dim=0)

