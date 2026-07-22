from importlib import import_module

from .coco_profiles import (
    multidiffusion_coco_all,
    multidiffusion_coco_1k,
    ours_overlap_stress,
    ours_weighted_mask,
    semanticdraw_sd15,
    semanticdraw_sd3,
    semanticdraw_sdxl,
)
from .coco_region_config import COCORegionConfig
from .coco_region_manifest import load_manifest, save_manifest
from .coco_region_sampler import build_coco_manifest

_LAZY_EXPORTS = {
    "COCORegionDataset": (".coco_region_dataset", "COCORegionDataset"),
    "batch_item_to_semanticdraw_inputs": (".adapters", "batch_item_to_semanticdraw_inputs"),
    "build_coco_region_dataloader": (".coco_region_dataset", "build_coco_region_dataloader"),
    "iter_semanticdraw_inputs": (".adapters", "iter_semanticdraw_inputs"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, package=__name__)
    return getattr(module, attr_name)


__all__ = [
    "COCORegionConfig",
    "COCORegionDataset",
    "batch_item_to_semanticdraw_inputs",
    "build_coco_manifest",
    "build_coco_region_dataloader",
    "iter_semanticdraw_inputs",
    "load_manifest",
    "multidiffusion_coco_all",
    "multidiffusion_coco_1k",
    "ours_overlap_stress",
    "ours_weighted_mask",
    "save_manifest",
    "semanticdraw_sd15",
    "semanticdraw_sd3",
    "semanticdraw_sdxl",
]
