#!/usr/bin/env python3
"""Standalone SSH benchmark; generated from the source notebook named below."""
from __future__ import annotations
import importlib.util
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
WORK_DIR = Path(os.environ["ANCHORDRAW_WORK_DIR"]).resolve()
SSH_COCO_ROOT = Path(os.environ["COCO_ROOT"]).resolve()
COCO_ROOT = SSH_COCO_ROOT
WORK_DIR.mkdir(parents=True, exist_ok=True)

def Markdown(value):
    return value

def display(value):
    if isinstance(value, str):
        print(value)

# Notebook: Ours/colab_multidiffusion_experiments/multidiffusion_sd3_flashflowmatch_naive_colab.ipynb
# Notebook SHA256: a7c4d346ff21d1cce8c2150788b829393014d668288c42eaf6731333e125277f

# --- Source notebook cell 4 ---
# Cấu hình Colab cho MultiDiffusion naive SD3 + Flash Flow Match.
from pathlib import Path
import os

RUN_PROFILE = os.environ.get('RUN_PROFILE', 'smoke_bs2')      # choices: "smoke_bs2", "mini32", "mini128", "full1073"
COLAB_GPU_MODE = os.environ.get('COLAB_GPU_MODE', 'low_vram')    # choices: "low_vram", "high_vram_24gb", "a100_80gb"
SD3_VIEW_PROFILE = "native_full_v128"  # choices: "native_full_v128", "diagnostic_v64s8"

PROFILE_CONFIGS = {
    "smoke_bs2": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "smoke" / "coco_val2017_multidiffusion_coco_all_sd3_1024x1024_smoke_bs2.jsonl",
        "expected_samples": 2,
        "label": "smoke_bs2",
        "run_metrics": False,
        "max_display_results": 2,
    },
    "mini32": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "mini32" / "coco_val2017_multidiffusion_coco_all_sd3_1024x1024_mini32.jsonl",
        "expected_samples": 32,
        "label": "mini32",
        "run_metrics": True,
        "max_display_results": 4,
    },
    "mini128": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "mini128" / "coco_val2017_multidiffusion_coco_all_sd3_1024x1024_mini128.jsonl",
        "expected_samples": 128,
        "label": "mini128",
        "run_metrics": True,
        "max_display_results": 6,
    },
    "full1073": {
        "manifest": REPO_ROOT / "Ours" / "data_manifests" / "coco_val2017_multidiffusion_coco_all_sd3_1024x1024_all.jsonl",
        "expected_samples": 1073,
        "label": "full1073",
        "run_metrics": True,
        "max_display_results": 8,
    },
}

GPU_MODE_CONFIGS = {
    "low_vram": {"batch_size": 1, "num_workers": 0, "bootstrapping": 2, "metric_batch_size": 1, "clip_batch_size": 4, "safe_vae": True},
    "high_vram_24gb": {"batch_size": 1, "num_workers": 1, "bootstrapping": 2, "metric_batch_size": 2, "clip_batch_size": 8, "safe_vae": True},
    "a100_80gb": {"batch_size": 2, "num_workers": 2, "bootstrapping": 2, "metric_batch_size": 4, "clip_batch_size": 8, "safe_vae": True},
}

SD3_VIEW_PROFILES = {
    "native_full_v128": {"window_size": 128, "stride": 128, "label": "v128full", "expected_view_count_1024": 1, "checklist_id": "CFG-MD-SD3-FFM-G0-B2-V128FULL", "paper_use": True},
    "diagnostic_v64s8": {"window_size": 64, "stride": 8, "label": "v64s8", "expected_view_count_1024": 81, "checklist_id": "CFG-MD-SD3-FFM-G0-B2-V64S8", "paper_use": False},
}

assert RUN_PROFILE in PROFILE_CONFIGS, f"Unknown RUN_PROFILE={RUN_PROFILE!r}"
assert COLAB_GPU_MODE in GPU_MODE_CONFIGS, f"Unknown COLAB_GPU_MODE={COLAB_GPU_MODE!r}"
assert SD3_VIEW_PROFILE in SD3_VIEW_PROFILES, f"Unknown SD3_VIEW_PROFILE={SD3_VIEW_PROFILE!r}"

RUN_CONFIG = PROFILE_CONFIGS[RUN_PROFILE]
GPU_CONFIG = GPU_MODE_CONFIGS[COLAB_GPU_MODE]
SD3_VIEW_CONFIG = SD3_VIEW_PROFILES[SD3_VIEW_PROFILE]

RUN_MANIFEST = RUN_CONFIG["manifest"]
EXPECTED_SAMPLES = RUN_CONFIG["expected_samples"]
RUN_LABEL = RUN_CONFIG["label"]
MAX_DISPLAY_RESULTS = 0
RUN_METRICS_OVERRIDE = None  # None = theo RUN_PROFILE; True/False = ép bật/tắt metric.
RUN_METRICS_AFTER_GENERATION = bool(RUN_CONFIG["run_metrics"] if RUN_METRICS_OVERRIDE is None else RUN_METRICS_OVERRIDE)

COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'datasets/coco')))
DIFFUSERS_COMMIT_EXPECTED = "53dcfcf3f3009a79fd484eed9b09ea76189da434"
MODEL_FAMILY = "sd3"
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
FLASH_SD3_REPO_ID = "jasperai/flash-sd3"
MODEL_REVISION = "ea42f8cef0f178587cf766dc8129abd379c90671"
FLASH_SD3_REVISION = "fb184d916f9193a81a4264758eb0ebbff1c56b69"
SAMPLER_NAME = "FlashFlowMatchEulerDiscreteScheduler"
TARGET_SIZE = (1024, 1024)
SD3_BASE_SEEDS = (2024, 10002024, 20002024)
BASE_SEED = int(os.environ.get("ANCHORDRAW_BASE_SEED", "2024"))
assert BASE_SEED in SD3_BASE_SEEDS, f"Use one of the paired SD3 seeds: {SD3_BASE_SEEDS}"
NEGATIVE_PROMPT = ""

FFM_SCHEDULE_STEPS = 50
FFM_T_INDEX_LIST = [0, 4, 12, 25, 37]
FFM_GUIDANCE_SCALE = 0.0

BATCH_SIZE = int(GPU_CONFIG["batch_size"])
NUM_WORKERS = int(GPU_CONFIG["num_workers"])
BOOTSTRAPPING = int(GPU_CONFIG["bootstrapping"])
METRIC_BATCH_SIZE = int(GPU_CONFIG["metric_batch_size"])
CLIP_BATCH_SIZE = int(GPU_CONFIG["clip_batch_size"])
SAFE_VAE = bool(GPU_CONFIG["safe_vae"])

SD3_VIEW_WINDOW_SIZE = int(SD3_VIEW_CONFIG["window_size"])
SD3_VIEW_STRIDE = int(SD3_VIEW_CONFIG["stride"])
SD3_VIEW_LABEL = str(SD3_VIEW_CONFIG["label"])
SD3_VIEW_CHECKLIST_ID = str(SD3_VIEW_CONFIG["checklist_id"])
SD3_VIEW_EXPECTED_COUNT_1024 = int(SD3_VIEW_CONFIG["expected_view_count_1024"])
SD3_VIEW_PAPER_USE = bool(SD3_VIEW_CONFIG["paper_use"])

RUN_SANITY_CHECK = True
RUN_EXPORT_ZIP = True
SAVE_EXPORT_TO_GOOGLE_DRIVE = False
GOOGLE_DRIVE_EXPORT_ROOT = Path(str(WORK_DIR / 'drive/MyDrive/SemanticDraw_Results/MD_SD3_FFM_naive'))
COPY_ORIGINAL_IMAGES_FOR_METRIC_EXPORT = False
SKIP_EXISTING = True
STOP_ON_BAD_OUTPUT = True

METRIC_NAMES = ("fid", "is", "clip_fg", "clip_bg", "time")
IS_SPLITS = 10

EXPERIMENT_ID = "MD-SD3-FFM-NAIVE-F1073"
CODE_VERSION_TAG = "md_sd3_flashflowmatch_naive_v2_integrity"
RUN_NAME = f"md_naive_runtime_safe_sd3_flashflowmatch_1024x1024_{RUN_LABEL}_{COLAB_GPU_MODE}_{SD3_VIEW_LABEL}_b{BATCH_SIZE}_bt{BOOTSTRAPPING}_v2_seed{BASE_SEED}"
RUN_BASE_DIR = Path(str(WORK_DIR / 'anchordraw_runs'))
OUTPUT_DIR = RUN_BASE_DIR / RUN_NAME
GENERATED_IMAGES_DIR = OUTPUT_DIR / "generated_images"
OVERLAY_IMAGES_DIR = OUTPUT_DIR / "mask_overlays"
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"
MASK_CACHE_ROOT = Path(str(WORK_DIR / 'anchordraw_mask_cache')) / MODEL_FAMILY
MASK_CACHE_DIR = MASK_CACHE_ROOT
RUN_SUMMARY_PATH = OUTPUT_DIR / "generation_summary.json"
RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"
METRICS_REPORT_PREFIX = f"{EXPERIMENT_ID.lower().replace('-', '_')}_{RUN_LABEL}"

for path in [OUTPUT_DIR, GENERATED_IMAGES_DIR, OVERLAY_IMAGES_DIR, METRICS_OUTPUT_DIR, MASK_CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

print("Experiment ID    :", EXPERIMENT_ID)
print("Run profile      :", RUN_PROFILE)
print("GPU mode         :", COLAB_GPU_MODE)
print("Model            :", MODEL_ID)
print("Acceleration     :", FLASH_SD3_REPO_ID)
print("Sampler          :", SAMPLER_NAME)
print("Resolution       :", f"{TARGET_SIZE[1]}x{TARGET_SIZE[0]}")
print("Manifest         :", RUN_MANIFEST)
print("Expected samples :", EXPECTED_SAMPLES)
print("Batch size       :", BATCH_SIZE)
print("Bootstrap        :", BOOTSTRAPPING)
print("Guidance         :", FFM_GUIDANCE_SCALE)
print("Schedule steps   :", FFM_SCHEDULE_STEPS)
print("t_index_list     :", FFM_T_INDEX_LIST)
print("View profile     :", SD3_VIEW_PROFILE, SD3_VIEW_CONFIG)
print("Run metrics      :", RUN_METRICS_AFTER_GENERATION)
print("Save to Drive    :", SAVE_EXPORT_TO_GOOGLE_DRIVE)
print("Output dir       :", OUTPUT_DIR)
COCO_ROOT = SSH_COCO_ROOT
MAX_DISPLAY_RESULTS = 0

# --- Source notebook cell 6 ---
# Import dataloader, metrics và wrapper MultiDiffusion SD3 Flash Flow Match.
import csv
import gc
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import shutil
import sys
import time
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# terminal display provided above
from PIL import Image

OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_REGION_FILE = REPO_ROOT / "Baseline" / "MultiDiffusion-master" / "MultiDiffusion-master" / "region_based.py"
BASELINE_SD3_FILE = REPO_ROOT / "Baseline" / "semantic-draw-main" / "src" / "model" / "pipeline_semantic_draw_3.py"

sys.path = [str(OURS_SRC)] + [p for p in sys.path if p != str(OURS_SRC)]
for module_name in list(sys.modules):
    if module_name == "data" or module_name.startswith("data.") or module_name == "baselines" or module_name.startswith("baselines."):
        del sys.modules[module_name]

importlib.invalidate_caches()
if "torchao" in sys.modules:
    raise RuntimeError("torchao đã được import trong session này. Restart runtime rồi Run All.")
if importlib.util.find_spec("torchao") is not None:
    raise RuntimeError("torchao vẫn import được trong runtime. Chạy cell dependency, restart runtime, rồi Run All.")

from baselines import MultiDiffusionSD3FlashFlowMatch, get_sd3_views
from data import COCORegionConfig, batch_item_to_semanticdraw_inputs, build_coco_region_dataloader
from data.visualize import make_mask_overlay

WRAPPER_SOURCE_FILE = Path(inspect.getsourcefile(MultiDiffusionSD3FlashFlowMatch) or "")
assert WRAPPER_SOURCE_FILE.exists(), "Cannot locate MultiDiffusionSD3FlashFlowMatch source file."
wrapper_source = WRAPPER_SOURCE_FILE.read_text(encoding="utf-8")
wrapper_sha256 = hashlib.sha256(WRAPPER_SOURCE_FILE.read_bytes()).hexdigest()
assert "StableDiffusion3Pipeline" in wrapper_source, "Wrapper không load SD3 pipeline."
assert "FlashFlowMatchEulerDiscreteScheduler" in wrapper_source, "Wrapper không dùng Flash Flow Match scheduler."
assert "text_encoder_3=None" in wrapper_source, "Wrapper không match cách load SD3 của SemanticDraw baseline."
assert "latent_shift_factor" in wrapper_source, "Wrapper thiếu SD3 VAE shift_factor."

import importlib.metadata
direct_url_text = importlib.metadata.distribution("diffusers").read_text("direct_url.json")
direct_url = json.loads(direct_url_text or "{}")
diffusers_commit = direct_url.get("vcs_info", {}).get("commit_id")
if diffusers_commit != DIFFUSERS_COMMIT_EXPECTED:
    raise RuntimeError(f"Wrong diffusers commit: {diffusers_commit!r}; expected {DIFFUSERS_COMMIT_EXPECTED}.")
dependency_versions = {
    package: importlib.metadata.version(package)
    for package in (
        "torch", "torchvision", "transformers", "peft", "accelerate",
        "huggingface_hub", "diffusers", "numpy", "Pillow", "pycocotools", "einops",
        "torchmetrics", "torch-fidelity", "open-clip-torch",
    )
}
data_source_dir = REPO_ROOT / "Ours" / "src" / "data"
data_source_hash = hashlib.sha256()
for data_file in sorted(data_source_dir.glob("*.py")):
    data_source_hash.update(data_file.name.encode("utf-8"))
    data_source_hash.update(data_file.read_bytes())
data_source_sha256 = data_source_hash.hexdigest()
metrics_source_hash = hashlib.sha256()
for metric_file in sorted((REPO_ROOT / "Ours" / "src" / "metrics").glob("*.py")):
    metrics_source_hash.update(metric_file.name.encode("utf-8"))
    metrics_source_hash.update(metric_file.read_bytes())
metrics_source_sha256 = metrics_source_hash.hexdigest()
MASK_CACHE_DIR = MASK_CACHE_ROOT / data_source_sha256
manifest_sha256 = hashlib.sha256(Path(RUN_MANIFEST).read_bytes()).hexdigest()
run_scientific_config = {
    "code_version": CODE_VERSION_TAG,
    "diffusers_commit": diffusers_commit,
    "dependency_versions": dependency_versions,
    "run_profile": RUN_PROFILE,
    "expected_samples": EXPECTED_SAMPLES,
    "model_family": MODEL_FAMILY,
    "model_id": MODEL_ID,
    "acceleration": FLASH_SD3_REPO_ID,
    "model_revision": MODEL_REVISION,
    "flash_sd3_revision": FLASH_SD3_REVISION,
    "sampler": SAMPLER_NAME,
    "target_size": list(TARGET_SIZE),
    "base_seed": BASE_SEED,
    "negative_prompt": NEGATIVE_PROMPT,
    "dtype": "float16",
    "data_source_sha256": data_source_sha256,
    "metrics_source_sha256": metrics_source_sha256,
    "guidance_scale": FFM_GUIDANCE_SCALE,
    "schedule_steps": FFM_SCHEDULE_STEPS,
    "t_index_list": FFM_T_INDEX_LIST,
    "bootstrapping": BOOTSTRAPPING,
    "safe_vae": SAFE_VAE,
    "view_profile": SD3_VIEW_PROFILE,
    "view_window_size": SD3_VIEW_WINDOW_SIZE,
    "view_stride": SD3_VIEW_STRIDE,
    "manifest_sha256": manifest_sha256,
    "wrapper_sha256": wrapper_sha256,
}
run_signature = hashlib.sha256(
    json.dumps(run_scientific_config, sort_keys=True).encode("utf-8")
).hexdigest()
run_scientific_config["run_signature"] = run_signature
if RUN_CONFIG_PATH.exists():
    previous_config = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
    if previous_config.get("run_signature") != run_signature:
        raise RuntimeError(
            f"Existing output uses a different scientific configuration: {RUN_CONFIG_PATH}. "
            "Select a new RUN_NAME; do not mix results."
        )
    if previous_config.get("input_masks_sha256"):
        run_scientific_config["input_masks_sha256"] = previous_config["input_masks_sha256"]
    if previous_config.get("coco_annotations_sha256"):
        run_scientific_config["coco_annotations_sha256"] = previous_config["coco_annotations_sha256"]
RUN_CONFIG_PATH.write_text(
    json.dumps(run_scientific_config, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("[OK] Imports are ready.")
print("[OK] Wrapper source file:", WRAPPER_SOURCE_FILE)
print("[OK] Wrapper source SHA256:", wrapper_sha256)
if BASELINE_SD3_FILE.exists():
    print("[OK] SemanticDraw SD3 baseline file:", BASELINE_SD3_FILE)
    print("[OK] SemanticDraw SD3 SHA256:", hashlib.sha256(BASELINE_SD3_FILE.read_bytes()).hexdigest())
if BASELINE_REGION_FILE.exists():
    print("[OK] MultiDiffusion region file:", BASELINE_REGION_FILE)
    print("[OK] MultiDiffusion region_based.py SHA256:", hashlib.sha256(BASELINE_REGION_FILE.read_bytes()).hexdigest())
print("[INFO] Number of SD3 views:", len(get_sd3_views(1024, 1024, window_size=SD3_VIEW_WINDOW_SIZE, stride=SD3_VIEW_STRIDE)))

# --- Source notebook cell 7 ---
# Tạo dataloader theo RUN_PROFILE đã chọn.
annotation_hash = hashlib.sha256()
for annotation_path in (
    COCO_ROOT / "annotations" / "instances_val2017.json",
    COCO_ROOT / "annotations" / "captions_val2017.json",
):
    annotation_hash.update(annotation_path.name.encode("utf-8"))
    annotation_hash.update(annotation_path.read_bytes())
coco_annotations_sha256 = annotation_hash.hexdigest()
if run_scientific_config.get("coco_annotations_sha256") not in (None, coco_annotations_sha256):
    raise RuntimeError("COCO annotation files changed since this run started.")
run_scientific_config["coco_annotations_sha256"] = coco_annotations_sha256
MASK_CACHE_DIR = MASK_CACHE_ROOT / data_source_sha256 / coco_annotations_sha256
RUN_CONFIG_PATH.write_text(json.dumps(run_scientific_config, ensure_ascii=False, indent=2), encoding="utf-8")
print("[CHECK] COCO annotations SHA256:", coco_annotations_sha256)
config = COCORegionConfig(
    coco_root=COCO_ROOT,
    split="val2017",
    instances_json=COCO_ROOT / "annotations" / "instances_val2017.json",
    captions_json=COCO_ROOT / "annotations" / "captions_val2017.json",
    manifest_path=RUN_MANIFEST,
    profile="multidiffusion_coco_all",
    model_family=MODEL_FAMILY,
    target_size=TARGET_SIZE,
    return_image=True,
    cache_resized_masks=True,
    cache_dir=MASK_CACHE_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=NUM_WORKERS > 0,
)

loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
dataset_size = len(loader.dataset)
if EXPECTED_SAMPLES is not None:
    assert dataset_size == EXPECTED_SAMPLES, f"Profile {RUN_PROFILE} expected {EXPECTED_SAMPLES} samples, got {dataset_size}."
preview_batch = next(iter(loader))

print(f"[OK] Manifest records: {dataset_size}")
print(f"[OK] Dataloader batches: {len(loader)} batch(es) x up to {BATCH_SIZE} sample(s)")
print(f"[OK] First batch masks shape: {tuple(preview_batch['masks'].shape)}")
print("First batch sample IDs:")
for sample_id in preview_batch["sample_ids"]:
    print(" -", sample_id)

# --- Source notebook cell 9 ---
# Helper chung cho seed, kiểm tra ảnh, adapter input và hiển thị kết quả.
def md_escape(text: object) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")

def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def image_stats(image: Image.Image) -> dict:
    arr = np.array(image.convert("RGB"))
    return {"min": int(arr.min()), "max": int(arr.max()), "mean": float(arr.mean()), "std": float(arr.std())}

def is_bad_generated_image(image: Image.Image) -> bool:
    stats = image_stats(image)
    near_black = stats["max"] <= 3 or stats["mean"] <= 2.0
    static_noise_like = 95.0 <= stats["std"] and 95.0 <= stats["mean"] <= 160.0
    return bool(near_black or static_noise_like)

def make_multidiffusion_payload(batch: dict, index: int) -> dict:
    item = batch_item_to_semanticdraw_inputs(batch, index)
    fg_masks = item["masks"].float().cpu()
    fg_union = fg_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    background_mask = (1.0 - fg_union).clamp(0, 1)
    all_masks = torch.cat([background_mask, fg_masks], dim=0)
    prompts = [item["background_prompt"], *item["prompts"]]
    negative_prompts = [NEGATIVE_PROMPT for _ in prompts]
    metadata = item["metadata"]
    assert len(prompts) == int(all_masks.shape[0]), f"prompts/masks mismatch: {len(prompts)} vs {tuple(all_masks.shape)}"
    return {
        "sample_id": metadata["sample_id"],
        "image_id": metadata["image_id"],
        "file_name": metadata["file_name"],
        "height": item["height"],
        "width": item["width"],
        "background_prompt": item["background_prompt"],
        "prompts": prompts,
        "negative_prompts": negative_prompts,
        "foreground_prompts": item["prompts"],
        "category_names": metadata["category_names"],
        "annotation_ids": metadata["annotation_ids"],
        "area_ratios": metadata["area_ratios"],
        "foreground_masks": fg_masks,
        "all_masks": all_masks,
        "metadata": metadata,
    }

def display_multidiffusion_result(payload: dict, original: Image.Image, overlay: Image.Image, generated: Image.Image, elapsed: float, generated_path: Path) -> None:
    rows = ["| Region | Prompt | Annotation | Area ratio |", "|---|---|---:|---:|"]
    rows.append(f"| Background | {md_escape(payload['background_prompt'])} | - | - |")
    for label, prompt, ann_id, area in zip(payload["category_names"], payload["foreground_prompts"], payload["annotation_ids"], payload["area_ratios"]):
        rows.append(f"| {md_escape(label)} | {md_escape(prompt)} | {ann_id} | {float(area):.4f} |")
    display(Markdown(
        f"### `{payload['sample_id']}`\n"
        f"- image_id: `{payload['image_id']}`\n"
        f"- file: `{payload['file_name']}`\n"
        f"- prompt/mask count: `{len(payload['prompts'])}`\n"
        f"- generated path: `{generated_path}`\n"
        f"- elapsed: `{elapsed:.2f}s`\n"
        f"- generated stats: `{image_stats(generated)}`\n\n"
        + "\n".join(rows)
    ))
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(original)
    axes[0].set_title("COCO original resized")
    axes[1].imshow(overlay)
    axes[1].set_title("Foreground mask overlay")
    axes[2].imshow(generated)
    axes[2].set_title("MultiDiffusion naive + SD3 Flash Flow Match generated")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

print("[OK] Helper functions are ready.")

# Validate every decoded mask and manifest position before loading SD3.
preflight_ids = []
preflight_mask_hash = hashlib.sha256()
for preflight_batch in loader:
    for preflight_index in range(len(preflight_batch["sample_ids"])):
        item = make_multidiffusion_payload(preflight_batch, preflight_index)
        masks = item["foreground_masks"]
        if (masks.ndim != 4 or masks.shape[1] != 1 or masks.shape[-2:] != TARGET_SIZE
                or masks.shape[0] == 0 or masks.shape[0] != len(item["foreground_prompts"])
                or masks.shape[0] != len(item["category_names"])
                or not torch.isfinite(masks).all() or (masks.flatten(1).sum(1) <= 0).any()):
            raise RuntimeError(f"Invalid SD3 foreground masks at {item['sample_id']}")
        preflight_ids.append(item["sample_id"])
        preflight_mask_hash.update(item["sample_id"].encode("utf-8"))
        preflight_mask_hash.update(masks.contiguous().numpy().tobytes())
manifest_ids = [json.loads(line)["sample_id"] for line in Path(RUN_MANIFEST).read_text(encoding="utf-8").splitlines() if line.strip()]
if preflight_ids != manifest_ids or len(preflight_ids) != EXPECTED_SAMPLES:
    raise RuntimeError("SD3 dataloader order/count differs from the manifest.")
input_masks_sha256 = preflight_mask_hash.hexdigest()
if run_scientific_config.get("input_masks_sha256") not in (None, input_masks_sha256):
    raise RuntimeError("Decoded COCO masks differ from the existing run. Stop before generation.")
run_scientific_config["input_masks_sha256"] = input_masks_sha256
RUN_CONFIG_PATH.write_text(json.dumps(run_scientific_config, ensure_ascii=False, indent=2), encoding="utf-8")
print("[CHECK] Input masks SHA256:", input_masks_sha256)
print(f"[OK] Preflighted {len(preflight_ids)} ordered SD3 masks.")

# --- Source notebook cell 10 ---
assert torch.cuda.is_available(), "Colab runtime chưa bật GPU. Hãy bật Runtime > Change runtime type > GPU."
device = torch.device("cuda:0")
dtype = torch.float16
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

# --- Source notebook cell 11 ---
# Load MultiDiffusion SD3 + Flash Flow Match.
seed_everything(BASE_SEED)
md_sd3 = MultiDiffusionSD3FlashFlowMatch(
    device=device,
    model_id=MODEL_ID,
    flash_sd3_repo_id=FLASH_SD3_REPO_ID,
    model_revision=MODEL_REVISION,
    flash_sd3_revision=FLASH_SD3_REVISION,
    dtype=dtype,
    t_index_list=FFM_T_INDEX_LIST,
    schedule_steps=FFM_SCHEDULE_STEPS,
    safe_vae=SAFE_VAE,
    view_window_size=SD3_VIEW_WINDOW_SIZE,
    view_stride=SD3_VIEW_STRIDE,
    runtime_checks=True,
    show_progress=True,
)

if hasattr(md_sd3.pipe, "enable_vae_tiling"):
    md_sd3.pipe.enable_vae_tiling()

sd3_views = get_sd3_views(TARGET_SIZE[0], TARGET_SIZE[1], vae_scale_factor=md_sd3.vae_scale_factor, window_size=SD3_VIEW_WINDOW_SIZE, stride=SD3_VIEW_STRIDE)

print("Model family     : SD3")
print("Method variant   : MD-naive runtime-safe")
print("Code version     :", CODE_VERSION_TAG)
print("Model            :", MODEL_ID)
print("Acceleration     :", FLASH_SD3_REPO_ID)
print("Scheduler        :", type(md_sd3.scheduler).__name__)
print("Schedule steps   :", FFM_SCHEDULE_STEPS)
print("t_index_list     :", FFM_T_INDEX_LIST)
print("timesteps        :", [int(t.item()) for t in md_sd3.timesteps])
print("sigmas           :", [round(float(s.item()), 4) for s in md_sd3.sigmas])
print("guidance_scale   :", FFM_GUIDANCE_SCALE)
print("bootstrapping    :", BOOTSTRAPPING)
print("safe_vae         :", SAFE_VAE)
print("SD3 view profile :", SD3_VIEW_PROFILE)
print("SD3 checklist ID :", SD3_VIEW_CHECKLIST_ID)
print("SD3 paper-use    :", SD3_VIEW_PAPER_USE)
print("SD3 view count   :", len(sd3_views), sd3_views[:3])
assert type(md_sd3.scheduler).__name__ == SAMPLER_NAME, f"Expected {SAMPLER_NAME}."
assert TARGET_SIZE != (1024, 1024) or len(sd3_views) == SD3_VIEW_EXPECTED_COUNT_1024, f"Expected {SD3_VIEW_EXPECTED_COUNT_1024} views for {SD3_VIEW_PROFILE}, got {len(sd3_views)}."

# --- Source notebook cell 12 ---
# Sanity check nhẹ: kiểm tra checkpoint/scheduler/VAE bằng pipeline SD3 thường.
# Cell này KHÔNG dùng MultiDiffusion fusion.
if RUN_SANITY_CHECK:
    seed_everything(BASE_SEED)
    # Exercise the same safe-VAE and Flash scheduler path as the experiment.
    sanity_image = md_sd3.generate(
        masks=torch.ones((1, 1, TARGET_SIZE[0], TARGET_SIZE[1]), dtype=torch.float32),
        prompts=["a studio photo of a teddy bear on a clean table"],
        negative_prompts=[NEGATIVE_PROMPT],
        height=TARGET_SIZE[0],
        width=TARGET_SIZE[1],
        guidance_scale=FFM_GUIDANCE_SCALE,
        bootstrapping=0,
        t_index_list=FFM_T_INDEX_LIST,
        schedule_steps=FFM_SCHEDULE_STEPS,
    ).convert("RGB")
    stats = image_stats(sanity_image)
    print("[SANITY] image stats:", stats)
    display(sanity_image.resize((512, 512)))
    if STOP_ON_BAD_OUTPUT and is_bad_generated_image(sanity_image):
        raise RuntimeError("Sanity image looks invalid. Check model/token/scheduler/VAE before manifest generation.")
else:
    print("[INFO] Sanity check skipped.")

# --- Source notebook cell 13 ---
# Chạy generation cho mọi sample trong manifest và hiển thị kết quả.
manifest_rows = [
    json.loads(line) for line in Path(RUN_MANIFEST).read_text(encoding="utf-8").splitlines() if line.strip()
]
assert len(manifest_rows) == dataset_size == EXPECTED_SAMPLES
expected_ids = [row["sample_id"] for row in manifest_rows]
assert len(set(expected_ids)) == len(expected_ids), "Manifest has duplicate sample IDs."

summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8")) if RUN_SUMMARY_PATH.exists() else []
if not isinstance(summary, list):
    raise RuntimeError("Existing generation summary must be a list")
existing_by_index = {}
for old_row in summary:
    position = int(old_row.get("index", -1))
    if position < 0 or position >= len(expected_ids) or position in existing_by_index:
        raise RuntimeError(f"Invalid or duplicate existing summary index: {position}")
    expected_path = GENERATED_IMAGES_DIR / f"{position:04d}_{expected_ids[position]}_generated.png"
    if (old_row.get("sample_id") != expected_ids[position]
            or old_row.get("seed") != BASE_SEED + position
            or old_row.get("run_signature") != run_signature
            or Path(old_row.get("generated_path", "")).resolve() != expected_path.resolve()):
        raise RuntimeError(f"Existing summary row {position} does not match this run")
    if expected_path.is_file():
        try:
            with Image.open(expected_path) as check_image:
                check_image.verify()
            with Image.open(expected_path) as check_image:
                if check_image.size != TARGET_SIZE or check_image.mode != "RGB":
                    raise RuntimeError("Wrong image dimensions or mode")
        except Exception as exc:
            print(f"[RETRY] invalid existing image at index={position}: {exc}", flush=True)
            continue
        existing_by_index[position] = old_row
summary = sorted(existing_by_index.values(), key=lambda row: row["index"])
summary_tmp = RUN_SUMMARY_PATH.with_suffix(".tmp")
summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
summary_tmp.replace(RUN_SUMMARY_PATH)
print(f"[RESUME] Verified {len(summary)} existing summary rows")
failed_images = []
failure_log = OUTPUT_DIR / "failed_images.jsonl"
previous_retry_seconds = {}
if failure_log.exists():
    for failure_line in failure_log.read_text(encoding="utf-8").splitlines():
        if failure_line.strip():
            previous_failure = json.loads(failure_line)
            previous_index = int(previous_failure["index"])
            previous_retry_seconds[previous_index] = previous_retry_seconds.get(previous_index, 0.0) + float(previous_failure["attempt_elapsed_sec"])
summary = []  # Rebuild in manifest order, including verified resumed rows.
global_index = 0
for batch_index, batch in enumerate(loader):
    print(f"[BATCH] {batch_index + 1}/{len(loader)} - {len(batch['sample_ids'])} sample(s)")
    for local_index, sample_id in enumerate(batch["sample_ids"]):
        retry_overhead_sec = previous_retry_seconds.get(global_index, 0.0)
        for attempt in range(1, 4):
            attempt_started = time.perf_counter()
            try:
                payload = make_multidiffusion_payload(batch, local_index)
                assert payload["sample_id"] == expected_ids[global_index], (
                    global_index, payload["sample_id"], expected_ids[global_index]
                )
                original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
                overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)
                stem = f"{global_index:04d}_{payload['sample_id']}"
                generated_path = GENERATED_IMAGES_DIR / f"{stem}_generated.png"
                overlay_path = OVERLAY_IMAGES_DIR / f"{stem}_overlay.png"
                old_row = existing_by_index.get(global_index)

                if SKIP_EXISTING and generated_path.exists() and old_row is not None:
                    with Image.open(generated_path) as check_img:
                        check_img.verify()
                    with Image.open(generated_path) as existing_img:
                        generated = existing_img.convert("RGB")
                    if generated.size != (payload["width"], payload["height"]):
                        raise RuntimeError(f"Existing PNG has wrong dimensions: {generated_path}")
                    overlay.save(overlay_path)
                    elapsed = float(old_row.get("elapsed_sec", 0.0))
                    row = dict(old_row)
                    row["skipped_existing"] = True
                    row["generated_path"] = str(generated_path)
                    row["overlay_path"] = str(overlay_path)
                else:
                    seed = BASE_SEED + global_index
                    seed_everything(seed)
                    tic = time.perf_counter()
                    generated = md_sd3.generate(
                        masks=payload["all_masks"],
                        prompts=payload["prompts"],
                        negative_prompts=payload["negative_prompts"],
                        height=payload["height"],
                        width=payload["width"],
                        guidance_scale=FFM_GUIDANCE_SCALE,
                        bootstrapping=BOOTSTRAPPING,
                        t_index_list=FFM_T_INDEX_LIST,
                        schedule_steps=FFM_SCHEDULE_STEPS,
                    ).convert("RGB")
                    elapsed = time.perf_counter() - tic + retry_overhead_sec
                    generated_tmp = generated_path.with_suffix(".tmp")
                    generated.save(generated_tmp, format="PNG")
                    generated_tmp.replace(generated_path)
                    overlay.save(overlay_path)
                    with Image.open(generated_path) as check_img:
                        check_img.verify()
                    row = {
                        "index": global_index,
                        "run_signature": run_signature,
                        "batch_index": batch_index,
                        "local_index": local_index,
                        "sample_id": payload["sample_id"],
                        "image_id": int(payload["image_id"]),
                        "file_name": payload["file_name"],
                        "seed": seed,
                        "method": "multidiffusion",
                        "variant": "naive_runtime_safe",
                        "model_family": MODEL_FAMILY,
                        "model_id": MODEL_ID,
                        "flash_sd3_repo_id": FLASH_SD3_REPO_ID,
                        "sampler": "flash_flow_match",
                        "scheduler": type(md_sd3.scheduler).__name__,
                        "t_index_list": FFM_T_INDEX_LIST,
                        "num_schedule_steps": FFM_SCHEDULE_STEPS,
                        "guidance_scale": FFM_GUIDANCE_SCALE,
                        "bootstrapping": BOOTSTRAPPING,
                        "safe_vae": SAFE_VAE,
                        "sd3_view_profile": SD3_VIEW_PROFILE,
                        "sd3_view_checklist_id": SD3_VIEW_CHECKLIST_ID,
                        "sd3_view_window_size": SD3_VIEW_WINDOW_SIZE,
                        "sd3_view_stride": SD3_VIEW_STRIDE,
                        "sd3_view_count": len(sd3_views),
                        "num_regions_including_background": len(payload["prompts"]),
                        "num_foreground_regions": len(payload["foreground_prompts"]),
                        "elapsed_sec": elapsed,
                        "generated_path": str(generated_path),
                        "overlay_path": str(overlay_path),
                        "skipped_existing": False,
                    }

                stats = image_stats(generated)
                if STOP_ON_BAD_OUTPUT and is_bad_generated_image(generated):
                    raise RuntimeError(f"Generated image at index {global_index} looks invalid: {stats}")
                summary.append(row)
                summary.sort(key=lambda item: item["index"])
                summary_tmp = RUN_SUMMARY_PATH.with_suffix(".tmp")
                summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                summary_tmp.replace(RUN_SUMMARY_PATH)
                if MAX_DISPLAY_RESULTS is None or global_index < MAX_DISPLAY_RESULTS:
                    display_multidiffusion_result(payload, original, overlay, generated, elapsed, generated_path)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                import traceback
                new_row = next((item for item in summary if item["index"] == global_index), None)
                if new_row is not None:
                    summary = [item for item in summary if item["index"] != global_index]
                    summary_tmp = RUN_SUMMARY_PATH.with_suffix(".tmp")
                    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    summary_tmp.replace(RUN_SUMMARY_PATH)
                existing_by_index.pop(global_index, None)
                attempt_elapsed_sec = time.perf_counter() - attempt_started
                retry_overhead_sec += attempt_elapsed_sec
                failure = {"index": global_index, "sample_id": expected_ids[global_index],
                           "seed": BASE_SEED + global_index, "attempt": attempt,
                           "attempt_elapsed_sec": attempt_elapsed_sec,
                           "error": repr(exc), "traceback": traceback.format_exc()}
                with failure_log.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(f"[IMAGE ERROR] index={global_index} attempt={attempt}/3: {exc}", flush=True)
                if attempt == 3:
                    failed_images.append(global_index)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                break
        global_index += 1

if failed_images:
    print(f"[INCOMPLETE] {len(failed_images)} indices exhausted three attempts: {failed_images}", flush=True)

assert len(summary) == dataset_size == EXPECTED_SAMPLES
assert [row["index"] for row in summary] == list(range(EXPECTED_SAMPLES))
assert [row["sample_id"] for row in summary] == expected_ids
summary_df = pd.DataFrame(summary)
display(Markdown(f"## Done\nGenerated `{len(summary)}` image(s) from `{dataset_size}` manifest record(s).\n\nSummary saved to `{RUN_SUMMARY_PATH}`."))
display(summary_df.tail(min(5, len(summary_df))))
summary[:3]

# --- Source notebook cell 14 ---
# Export ảnh đã sinh sang folder chuẩn để đo metric/reproduce sau này và nén thành zip.
METRIC_EXPORT_ROOT = Path(str(WORK_DIR / 'anchordraw_metric_exports'))
METRIC_EXPORT_DIR = METRIC_EXPORT_ROOT / RUN_NAME
METRIC_EXPORT_GENERATED_DIR = METRIC_EXPORT_DIR / "generated_images"
METRIC_EXPORT_ORIGINAL_DIR = METRIC_EXPORT_DIR / "original_images"
METRIC_EXPORT_MANIFEST_JSONL = METRIC_EXPORT_DIR / "metric_generated_manifest.jsonl"
METRIC_EXPORT_MANIFEST_CSV = METRIC_EXPORT_DIR / "metric_generated_manifest.csv"
METRIC_EXPORT_SUMMARY_JSON = METRIC_EXPORT_DIR / "export_summary.json"
METRIC_EXPORT_ZIP_PATH = METRIC_EXPORT_ROOT / f"{RUN_NAME}__metric_export.zip"

for path in [METRIC_EXPORT_DIR, METRIC_EXPORT_GENERATED_DIR]:
    path.mkdir(parents=True, exist_ok=True)
if COPY_ORIGINAL_IMAGES_FOR_METRIC_EXPORT:
    METRIC_EXPORT_ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)

def safe_file_name(text: object, max_len: int = 120) -> str:
    name = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(text)).strip("_")
    return name[:max_len] or "sample"

def load_manifest_records_by_sample_id(manifest_path: Path) -> dict:
    records = {}
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[record["sample_id"]] = record
    return records

if "summary" not in globals() or not summary:
    with RUN_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        summary = json.load(f)

# A partial or mixed run must never be exported as the full experiment.
summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
expected_ids = [
    json.loads(line)["sample_id"]
    for line in Path(RUN_MANIFEST).read_text(encoding="utf-8").splitlines() if line.strip()
]
assert len(summary) == len(expected_ids) == EXPECTED_SAMPLES
assert [row["index"] for row in summary] == list(range(EXPECTED_SAMPLES))
assert [row["sample_id"] for row in summary] == expected_ids
assert all(row.get("run_signature") == run_signature for row in summary)
expected_pngs = {
    f"{i:04d}_{sample_id}_generated.png" for i, sample_id in enumerate(expected_ids)
}
actual_pngs = {path.name for path in GENERATED_IMAGES_DIR.glob("*_generated.png")}
assert actual_pngs == expected_pngs, (
    f"Missing PNGs: {sorted(expected_pngs - actual_pngs)[:5]}; "
    f"extra PNGs: {sorted(actual_pngs - expected_pngs)[:5]}"
)

manifest_by_sample_id = load_manifest_records_by_sample_id(Path(RUN_MANIFEST))
metric_records = []
for row_position, gen in enumerate(summary):
    sample_id = gen.get("sample_id")
    manifest_record = manifest_by_sample_id.get(sample_id, {})
    image_id = int(gen.get("image_id", manifest_record.get("image_id", -1)))
    file_name = gen.get("file_name", manifest_record.get("file_name"))
    source_generated_path = Path(gen["generated_path"])
    if not source_generated_path.exists():
        raise FileNotFoundError(source_generated_path)
    with Image.open(source_generated_path) as check_img:
        check_img.verify()

    metric_index = int(gen.get("index", row_position))
    canonical_name = f"{metric_index:06d}__coco_{image_id:012d}__{safe_file_name(sample_id)}__generated.png"
    metric_generated_path = METRIC_EXPORT_GENERATED_DIR / canonical_name
    if source_generated_path.resolve() != metric_generated_path.resolve():
        shutil.copy2(source_generated_path, metric_generated_path)

    coco_original_path = Path(COCO_ROOT) / "val2017" / str(file_name) if file_name else None
    copied_original_path = None
    if COPY_ORIGINAL_IMAGES_FOR_METRIC_EXPORT and coco_original_path is not None and coco_original_path.exists():
        original_name = f"{metric_index:06d}__coco_{image_id:012d}__{safe_file_name(sample_id)}__original.jpg"
        copied_original_path = METRIC_EXPORT_ORIGINAL_DIR / original_name
        shutil.copy2(coco_original_path, copied_original_path)

    metric_records.append({
        "metric_index": metric_index,
        "experiment_id": EXPERIMENT_ID,
        "sample_id": sample_id,
        "image_id": image_id,
        "file_name": file_name,
        "generated_image_path": str(metric_generated_path),
        "generated_image_relative_path": str(metric_generated_path.relative_to(METRIC_EXPORT_DIR)),
        "source_generated_path": str(source_generated_path),
        "coco_original_path": str(coco_original_path) if coco_original_path is not None else None,
        "copied_original_path": str(copied_original_path) if copied_original_path is not None else None,
        "source_manifest_path": str(RUN_MANIFEST),
        "source_output_dir": str(OUTPUT_DIR),
        "background_prompt": manifest_record.get("caption"),
        "foreground_prompts": manifest_record.get("foreground_prompts"),
        "category_names": manifest_record.get("category_names"),
        "annotation_ids": manifest_record.get("annotation_ids"),
        "elapsed_sec": gen.get("elapsed_sec"),
        "model_family": MODEL_FAMILY,
        "model_id": MODEL_ID,
        "acceleration": FLASH_SD3_REPO_ID,
        "sampler": SAMPLER_NAME,
        "guidance_scale": FFM_GUIDANCE_SCALE,
        "bootstrapping": BOOTSTRAPPING,
        "view_profile": SD3_VIEW_PROFILE,
        "view_window_size": SD3_VIEW_WINDOW_SIZE,
        "view_stride": SD3_VIEW_STRIDE,
        "view_count": len(sd3_views),
    })

expected_export_pngs = {
    f"{i:06d}__coco_{int(row['image_id']):012d}__{safe_file_name(row['sample_id'])}__generated.png"
    for i, row in enumerate(summary)
}
assert {path.name for path in METRIC_EXPORT_GENERATED_DIR.glob("*_generated.png")} == expected_export_pngs

with METRIC_EXPORT_MANIFEST_JSONL.open("w", encoding="utf-8") as f:
    for record in metric_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

fieldnames = list(metric_records[0].keys()) if metric_records else []
with METRIC_EXPORT_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metric_records)

export_summary = {
    "experiment_id": EXPERIMENT_ID,
    "run_name": RUN_NAME,
    "run_profile": RUN_PROFILE,
    "gpu_mode": COLAB_GPU_MODE,
    "num_generated_images": len(metric_records),
    "run_signature": run_signature,
    "manifest_sha256": manifest_sha256,
    "wrapper_sha256": wrapper_sha256,
    "input_masks_sha256": run_scientific_config["input_masks_sha256"],
    "coco_annotations_sha256": run_scientific_config["coco_annotations_sha256"],
    "run_config": "run_config.json",
    "generation_summary": "generation_summary.json",
    "generated_images_dir": str(METRIC_EXPORT_GENERATED_DIR),
    "manifest_jsonl": str(METRIC_EXPORT_MANIFEST_JSONL),
    "manifest_csv": str(METRIC_EXPORT_MANIFEST_CSV),
    "source_generation_summary": str(RUN_SUMMARY_PATH),
    "source_manifest_path": str(RUN_MANIFEST),
    "model_family": MODEL_FAMILY,
    "model_id": MODEL_ID,
    "acceleration": FLASH_SD3_REPO_ID,
    "sampler": SAMPLER_NAME,
    "guidance_scale": FFM_GUIDANCE_SCALE,
    "bootstrapping": BOOTSTRAPPING,
    "view_profile": SD3_VIEW_PROFILE,
}
with METRIC_EXPORT_SUMMARY_JSON.open("w", encoding="utf-8") as f:
    json.dump(export_summary, f, ensure_ascii=False, indent=2)
shutil.copy2(RUN_CONFIG_PATH, METRIC_EXPORT_DIR / "run_config.json")
shutil.copy2(RUN_SUMMARY_PATH, METRIC_EXPORT_DIR / "generation_summary.json")

if RUN_EXPORT_ZIP:
    if METRIC_EXPORT_ZIP_PATH.exists():
        METRIC_EXPORT_ZIP_PATH.unlink()
    with zipfile.ZipFile(METRIC_EXPORT_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in METRIC_EXPORT_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(METRIC_EXPORT_DIR.parent))

if SAVE_EXPORT_TO_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount(str(WORK_DIR / 'drive'))
    drive_target_dir = GOOGLE_DRIVE_EXPORT_ROOT / RUN_NAME
    drive_target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(METRIC_EXPORT_DIR, drive_target_dir, dirs_exist_ok=True)
    if RUN_EXPORT_ZIP and METRIC_EXPORT_ZIP_PATH.exists():
        shutil.copy2(METRIC_EXPORT_ZIP_PATH, GOOGLE_DRIVE_EXPORT_ROOT / METRIC_EXPORT_ZIP_PATH.name)
    print("[OK] Copied export to Google Drive:", drive_target_dir)

print("[OK] Metric export dir:", METRIC_EXPORT_DIR)
print("[OK] Metric generated images:", METRIC_EXPORT_GENERATED_DIR)
print("[OK] Metric manifest JSONL:", METRIC_EXPORT_MANIFEST_JSONL)
print("[OK] Metric manifest CSV:", METRIC_EXPORT_MANIFEST_CSV)
if RUN_EXPORT_ZIP:
    print("[OK] Metric export zip:", METRIC_EXPORT_ZIP_PATH)
display(pd.DataFrame(metric_records).head(min(5, len(metric_records))))

# --- Source notebook cell 16 ---
# Giải phóng VRAM trước khi load Inception/CLIP cho metric.
for var_name in ("md_sd3", "sanity_image", "generated", "payload", "overlay", "original", "batch", "loader", "preview_batch"):
    if var_name in globals():
        del globals()[var_name]

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass

print("[OK] Released generation objects before metric evaluation.")

# --- Source notebook cell 17 ---
# Đo FID, IS, CLIP(fg), CLIP(bg), Time(s) nếu RUN_METRICS_AFTER_GENERATION=True.
if not RUN_METRICS_AFTER_GENERATION:
    display(Markdown(
        "## Metric Skipped\n"
        f"RUN_PROFILE hiện tại là `{RUN_PROFILE}`. "
        "Đổi sang `mini32`, `mini128`, hoặc `full1073` để bật metric."
    ))
else:
    from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report

    assert RUN_SUMMARY_PATH.exists(), f"Missing generation summary: {RUN_SUMMARY_PATH}"
    checked_summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
    manifest_ids = [
        json.loads(line)["sample_id"]
        for line in Path(RUN_MANIFEST).read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(checked_summary) == len(manifest_ids) == EXPECTED_SAMPLES
    assert [row["index"] for row in checked_summary] == list(range(EXPECTED_SAMPLES))
    assert [row["sample_id"] for row in checked_summary] == manifest_ids
    assert all(row.get("run_signature") == run_signature for row in checked_summary)
    metric_annotation_hash = hashlib.sha256()
    for annotation_path in (COCO_ROOT / "annotations" / "instances_val2017.json", COCO_ROOT / "annotations" / "captions_val2017.json"):
        metric_annotation_hash.update(annotation_path.name.encode("utf-8"))
        metric_annotation_hash.update(annotation_path.read_bytes())
    assert metric_annotation_hash.hexdigest() == run_scientific_config["coco_annotations_sha256"]
    metric_source_check = hashlib.sha256()
    for metric_file in sorted((REPO_ROOT / "Ours" / "src" / "metrics").glob("*.py")):
        metric_source_check.update(metric_file.name.encode("utf-8"))
        metric_source_check.update(metric_file.read_bytes())
    assert metric_source_check.hexdigest() == run_scientific_config["metrics_source_sha256"]
    metric_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    metric_config = MetricEvaluationConfig(
        manifest_path=RUN_MANIFEST,
        coco_root=COCO_ROOT,
        generated_dir=GENERATED_IMAGES_DIR,
        generation_summary=RUN_SUMMARY_PATH,
        output_dir=METRICS_OUTPUT_DIR,
        model_family=MODEL_FAMILY,
        target_size=TARGET_SIZE,
        metrics=METRIC_NAMES,
        batch_size=METRIC_BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        device=metric_device,
        clip_batch_size=CLIP_BATCH_SIZE,
        is_splits=IS_SPLITS,
    )

    metric_report = run_evaluation(metric_config)
    assert metric_report["num_evaluated"] == EXPECTED_SAMPLES
    assert metric_report["num_missing_generated"] == 0
    metrics_json, metrics_csv = write_metrics_report(metric_report, METRICS_OUTPUT_DIR, prefix=METRICS_REPORT_PREFIX)
    if SAVE_EXPORT_TO_GOOGLE_DRIVE:
        shutil.copy2(metrics_json, GOOGLE_DRIVE_EXPORT_ROOT / f"{RUN_NAME}__metrics.json")
    values = metric_report["metrics"]

    def fmt(value: object, digits: int = 4) -> str:
        if value is None:
            return "-"
        try:
            value = float(value)
            if math.isnan(value):
                return "-"
            return f"{value:.{digits}f}"
        except Exception:
            return str(value)

    metrics_table = pd.DataFrame([
        {"Metric": "FID↓", "Value": fmt(values.get("fid"))},
        {"Metric": "IS↑", "Value": fmt(values.get("is_mean"))},
        {"Metric": "IS std", "Value": fmt(values.get("is_std"))},
        {"Metric": "CLIP(fg)↑", "Value": fmt(values.get("clip_fg_x100"))},
        {"Metric": "CLIP(bg)↑", "Value": fmt(values.get("clip_bg_x100"))},
        {"Metric": "Time(s)↓", "Value": fmt(values.get("time_mean_sec"))},
        {"Metric": "Total time(s)", "Value": fmt(values.get("time_total_sec"))},
    ])

    display(Markdown(
        f"## Metric Done\n"
        f"- evaluated: `{metric_report['num_evaluated']}` / `{metric_report['num_manifest_records']}` samples\n"
        f"- missing generated images: `{metric_report['num_missing_generated']}`\n"
        f"- metrics JSON: `{metrics_json}`\n"
        f"- metrics CSV: `{metrics_csv}`"
    ))
    display(metrics_table)
    metric_report

# Keep the verified report alongside the ZIP for three-seed aggregation.
sd3_export_zip = METRIC_EXPORT_ZIP_PATH
assert sd3_export_zip.is_file(), sd3_export_zip
sd3_sidecar = sd3_export_zip.with_name(sd3_export_zip.name.replace("__metric_export.zip", "__metrics.json"))
shutil.copy2(metrics_json, sd3_sidecar)
print("[OK] SD3 report sidecar:", sd3_sidecar)
