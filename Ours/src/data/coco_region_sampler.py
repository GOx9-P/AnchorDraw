from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from .coco_region_config import COCORegionConfig
from .coco_region_manifest import JsonDict, load_coco_index, load_manifest, save_manifest


def _stable_rng(seed: int, *parts: object) -> random.Random:
    text = "|".join([str(seed), *[str(part) for part in parts]])
    return random.Random(text)


def _select_caption(captions: Sequence[JsonDict], config: COCORegionConfig, image_id: int) -> JsonDict:
    if not captions:
        raise ValueError(f"Image {image_id} has no COCO caption")
    captions = sorted(captions, key=lambda item: int(item["id"]))
    if config.caption_policy == "first":
        return captions[0]
    if config.caption_policy == "seeded_random":
        return _stable_rng(config.seed, image_id, "caption").choice(list(captions))
    raise ValueError(f"Unsupported caption_policy: {config.caption_policy}")


def _select_objects(
    valid_objects: Sequence[Tuple[JsonDict, float, str]],
    config: COCORegionConfig,
    image_id: int,
) -> List[Tuple[JsonDict, float, str]]:
    if config.object_policy == "largest":
        objects = sorted(valid_objects, key=lambda item: (-item[1], int(item[0]["id"])))
    elif config.object_policy == "seeded_random":
        objects = sorted(valid_objects, key=lambda item: int(item[0]["id"]))
        objects = list(objects)
        _stable_rng(config.seed, image_id, "objects").shuffle(objects)
    else:
        raise ValueError(f"Unsupported object_policy: {config.object_policy}")

    return list(objects[: config.max_objects])


def _prompt_for_category(category_name: str, config: COCORegionConfig) -> str:
    return config.prompt_template.format(label=category_name, category=category_name)


def _profile_protocol(config: COCORegionConfig) -> str:
    if config.profile == "semanticdraw":
        return f"semanticdraw_{config.model_family}"
    if config.profile == "multidiffusion_coco_all":
        return "multidiffusion_coco_all"
    if config.profile == "multidiffusion_coco_1k":
        return "multidiffusion_coco_all"
    if config.profile == "ours_weighted_mask":
        return f"ours_weighted_mask_{config.model_family}"
    if config.profile == "ours_overlap_stress":
        return f"ours_overlap_stress_{config.model_family}"
    return config.profile


def build_coco_manifest(config: COCORegionConfig, overwrite: bool = False) -> List[JsonDict]:
    config.validate()
    manifest_path = config.resolved_manifest_path()
    if manifest_path.exists() and not overwrite:
        return load_manifest(manifest_path)

    index = load_coco_index(
        config.resolved_instances_json(),
        config.resolved_captions_json(),
    )
    excluded = {name.lower() for name in config.exclude_categories}
    candidates: List[JsonDict] = []
    target_h, target_w = config.target_hw
    protocol = _profile_protocol(config)

    for image in index.iter_images():
        image_id = int(image["id"])
        original_size = (int(image["height"]), int(image["width"]))
        valid_objects: List[Tuple[JsonDict, float, str]] = []

        for annotation in index.annotations_for_image(image_id):
            if config.drop_iscrowd and int(annotation.get("iscrowd", 0)) != 0:
                continue

            category_name = index.category_name(int(annotation["category_id"]))
            if category_name.lower() in excluded:
                continue

            area_ratio = float(annotation.get("area", 0.0)) / max(
                float(original_size[0] * original_size[1]),
                1.0,
            )
            if area_ratio < config.min_mask_area_ratio:
                continue

            if not annotation.get("segmentation"):
                continue

            valid_objects.append((annotation, area_ratio, category_name))

        if len(valid_objects) < config.min_objects:
            continue
        if len(valid_objects) > config.max_objects and not config.truncate_objects:
            continue

        selected_objects = _select_objects(valid_objects, config, image_id)
        if len(selected_objects) < config.min_objects:
            continue

        captions = index.captions_for_image(image_id)
        if not captions:
            continue
        caption = _select_caption(captions, config, image_id)

        annotation_ids = [int(obj[0]["id"]) for obj in selected_objects]
        category_ids = [int(obj[0]["category_id"]) for obj in selected_objects]
        category_names = [obj[2] for obj in selected_objects]
        area_ratios = [float(obj[1]) for obj in selected_objects]
        foreground_prompts = [_prompt_for_category(name, config) for name in category_names]

        uses_seeded_selection = (
            config.subset_size is not None or
            config.caption_policy == "seeded_random" or
            config.object_policy == "seeded_random"
        )
        seed_part = f"_seed{config.seed}" if uses_seeded_selection else ""
        sample_id = f"coco_{config.split}_{image_id:012d}_{protocol}{seed_part}"
        record: JsonDict = {
            "sample_id": sample_id,
            "image_id": image_id,
            "file_name": str(image.get("file_name", f"{image_id:012d}.jpg")),
            "caption_id": int(caption["id"]),
            "caption": str(caption["caption"]),
            "annotation_ids": annotation_ids,
            "category_ids": category_ids,
            "category_names": category_names,
            "foreground_prompts": foreground_prompts,
            "original_size": [original_size[0], original_size[1]],
            "target_size": [target_h, target_w],
            "area_ratios": area_ratios,
            "protocol": protocol,
            "profile": config.profile,
            "model_family": config.model_family,
            "seed": config.seed if uses_seeded_selection else None,
            "version": config.manifest_version,
            "sampling": {
                "min_objects": config.min_objects,
                "max_objects": config.max_objects,
                "truncate_objects": config.truncate_objects,
                "exclude_categories": list(config.exclude_categories),
                "min_mask_area_ratio": config.min_mask_area_ratio,
                "drop_iscrowd": config.drop_iscrowd,
                "caption_policy": config.caption_policy,
                "object_policy": config.object_policy,
                "prompt_template": config.prompt_template,
            },
        }
        candidates.append(record)

    candidates.sort(key=lambda item: int(item["image_id"]))
    if config.subset_size is not None:
        rng = random.Random(config.seed)
        rng.shuffle(candidates)
        candidates = candidates[: config.subset_size]

    save_manifest(candidates, manifest_path)
    return candidates
