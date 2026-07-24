from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .config import MetricEvaluationConfig
from .io import (
    build_summary_index,
    candidate_generated_paths,
    load_generation_summary,
    load_rgb_image,
    resolve_generated_image_path,
)
from .time_metrics import summarize_generation_time


def _resolve_device(device_name: str):
    import torch

    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _build_loader(config: MetricEvaluationConfig):
    import torch
    from data import COCORegionConfig, build_coco_region_dataloader

    coco_config = COCORegionConfig(
        coco_root=config.resolved_coco_root(),
        split=config.split,
        instances_json=config.resolved_instances_json(),
        captions_json=config.resolved_captions_json(),
        manifest_path=config.resolved_manifest_path(),
        profile="multidiffusion_coco_all",
        model_family=config.model_family,
        target_size=config.target_hw,
        return_image=True,
        cache_resized_masks=config.cache_resized_masks,
        cache_dir=config.cache_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    return build_coco_region_dataloader(coco_config, shuffle=False, drop_last=False)


def _resize_for_metric(image: Image.Image, target_hw: tuple[int, int]) -> Image.Image:
    target_h, target_w = target_hw
    return image.convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)


def run_evaluation(config: MetricEvaluationConfig) -> dict[str, Any]:
    config.validate()
    selected_metrics = set(config.selected_metrics())

    import torch
    from data import batch_item_to_semanticdraw_inputs

    device = _resolve_device(config.device)
    generation_summary_path = config.resolved_generation_summary()
    summary_rows = load_generation_summary(generation_summary_path)
    summary_index = build_summary_index(summary_rows)
    generated_dir = config.resolved_generated_dir()

    loader = _build_loader(config)
    dataset_size = len(loader.dataset)
    effective_size = dataset_size if config.max_samples is None else min(dataset_size, config.max_samples)
    target_hw = config.target_hw

    inception = None
    if "fid" in selected_metrics or "is" in selected_metrics:
        from .inception_metrics import InceptionMetricsAccumulator

        inception = InceptionMetricsAccumulator(
            device=device,
            compute_fid="fid" in selected_metrics,
            compute_is="is" in selected_metrics,
            fid_feature=config.fid_feature,
            is_splits=min(config.is_splits, max(1, effective_size)),
        )

    clip = None
    if "clip_fg" in selected_metrics or "clip_bg" in selected_metrics or "clip_pg" in selected_metrics:
        from .clip_metrics import CLIPScoreAccumulator

        clip = CLIPScoreAccumulator(
            device=device,
            model_name=config.clip_model_name,
            pretrained=config.clip_pretrained,
            batch_size=config.clip_batch_size,
        )

    evaluated_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    global_index = 0
    stop = False

    for batch in loader:
        real_images: list[Image.Image] = []
        generated_images: list[Image.Image] = []

        for local_index, sample_id in enumerate(batch["sample_ids"]):
            if config.max_samples is not None and global_index >= config.max_samples:
                stop = True
                break

            payload = batch_item_to_semanticdraw_inputs(batch, local_index)
            record = payload["metadata"]
            generated_path = resolve_generated_image_path(
                record,
                generated_dir=generated_dir,
                summary_index=summary_index,
                global_index=global_index,
            )

            if generated_path is None:
                candidates = [
                    str(path)
                    for path in candidate_generated_paths(
                        record,
                        generated_dir=generated_dir,
                        summary_index=summary_index,
                        global_index=global_index,
                    )
                ]
                missing = {
                    "index": global_index,
                    "sample_id": str(sample_id),
                    "image_id": int(record["image_id"]),
                    "candidates": candidates,
                }
                missing_rows.append(missing)
                if config.error_on_missing_generated:
                    raise FileNotFoundError(f"Generated image not found for sample {sample_id}: {candidates}")
                global_index += 1
                continue

            generated = _resize_for_metric(load_rgb_image(generated_path), target_hw)
            real = _resize_for_metric(batch["images"][local_index], target_hw)

            if inception is not None:
                real_images.append(real)
                generated_images.append(generated)

            if clip is not None:
                if "clip_pg" in selected_metrics:
                    clip.update_prompt_global([generated], [str(payload["background_prompt"])])
                if "clip_bg" in selected_metrics:
                    clip.update_background_region(
                        generated,
                        payload["masks"].to(dtype=torch.float32).cpu(),
                        str(payload["background_prompt"]),
                        apply_mask=config.mask_background_for_clip,
                    )
                if "clip_fg" in selected_metrics:
                    clip.update_foreground_regions(
                        generated,
                        payload["masks"].to(dtype=torch.float32).cpu(),
                        payload["prompts"],
                        padding_ratio=config.foreground_crop_padding_ratio,
                        apply_mask=config.mask_foreground_for_clip,
                    )

            evaluated_rows.append(
                {
                    "index": global_index,
                    "sample_id": str(sample_id),
                    "image_id": int(record["image_id"]),
                    "file_name": str(record["file_name"]),
                    "generated_path": str(generated_path),
                    "num_foreground_regions": len(payload["prompts"]),
                }
            )
            global_index += 1

        if inception is not None and generated_images:
            inception.update(real_images, generated_images, target_hw)

        if stop:
            break

    metrics: dict[str, float] = {}
    if "time" in selected_metrics:
        metrics.update(summarize_generation_time(summary_rows))
    if inception is not None:
        metrics.update(inception.compute())
    if clip is not None:
        metrics.update(clip.compute())

    return {
        "manifest_path": str(config.resolved_manifest_path()),
        "coco_root": str(config.resolved_coco_root()),
        "generated_dir": None if generated_dir is None else str(generated_dir),
        "generation_summary": None if generation_summary_path is None else str(generation_summary_path),
        "model_family": config.model_family,
        "target_size": list(target_hw),
        "selected_metrics": sorted(selected_metrics),
        "device": str(device),
        "num_manifest_records": dataset_size,
        "num_evaluated": len(evaluated_rows),
        "num_missing_generated": len(missing_rows),
        "metrics": metrics,
        "evaluated": evaluated_rows,
        "missing_generated": missing_rows,
    }
