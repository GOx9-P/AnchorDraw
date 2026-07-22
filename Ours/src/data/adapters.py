from __future__ import annotations

from typing import Dict, Iterator

import torch


def batch_item_to_semanticdraw_inputs(batch: dict, index: int) -> Dict[str, object]:
    valid = batch["valid_regions"][index]
    p = int(valid.sum().item())
    masks = batch["masks"][index, :p].to(dtype=torch.float32)
    target_h = int(batch["target_sizes"][index, 0].item())
    target_w = int(batch["target_sizes"][index, 1].item())

    return {
        "background_prompt": batch["background_prompts"][index],
        "prompts": list(batch["foreground_prompts"][index][:p]),
        "masks": masks,
        "height": target_h,
        "width": target_w,
        "metadata": batch["metadata"][index],
    }


def iter_semanticdraw_inputs(batch: dict) -> Iterator[Dict[str, object]]:
    for index in range(len(batch["sample_ids"])):
        yield batch_item_to_semanticdraw_inputs(batch, index)

