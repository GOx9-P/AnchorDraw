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

# Notebook: Ours/colab_multidiffusion_experiments/multidiffusion_sdxl_euler_colab.ipynb
# Notebook SHA256: e9ab1ed87683e4c248ece1fcbe6194a04fd4583b1d920d3dd4a6ec12d0713b87

# --- Source notebook cell 4 ---
# Cấu hình Colab cho MultiDiffusion naive SDXL + Euler Discrete.
from pathlib import Path
import os

RUN_PROFILE = os.environ.get('RUN_PROFILE', 'smoke_bs2')  # choices: "smoke_bs2", "mini32", "mini128", "full1073"
COLAB_GPU_MODE = os.environ.get('COLAB_GPU_MODE', 'low_vram')  # choices: "low_vram", "high_vram_24gb", "a100_80gb"

PROFILE_CONFIGS = {
    "smoke_bs2": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "smoke" / "coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_smoke_bs2.jsonl",
        "expected_samples": 2,
        "label": "smoke_bs2",
        "run_metrics": False,
        "max_display_results": 2,
    },
    "mini32": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "mini32" / "coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_mini32.jsonl",
        "expected_samples": 32,
        "label": "mini32",
        "run_metrics": True,
        "max_display_results": 4,
    },
    "mini128": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "mini128" / "coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_mini128.jsonl",
        "expected_samples": 128,
        "label": "mini128",
        "run_metrics": True,
        "max_display_results": 6,
    },
    "full1073": {
        "manifest": REPO_ROOT / "Ours" / "data_manifests" / "coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_all.jsonl",
        "expected_samples": 1073,
        "label": "full1073",
        "run_metrics": True,
        "max_display_results": 8,
    },
}

# MultiDiffusion gốc dùng bootstrapping để tăng độ bám tight masks.
# Với SDXL-Lightning 5 selected timesteps, bootstrapping=2 là cấu hình validate/benchmark chính.
GPU_MODE_CONFIGS = {
    "low_vram": {"batch_size": 1, "bootstrapping": 2, "metric_batch_size": 1, "clip_batch_size": 4, "safe_vae": True},
    "high_vram_24gb": {"batch_size": 2, "bootstrapping": 2, "metric_batch_size": 2, "clip_batch_size": 8, "safe_vae": True},
    "a100_80gb": {"batch_size": 4, "bootstrapping": 2, "metric_batch_size": 8, "clip_batch_size": 16, "safe_vae": True},
}

assert RUN_PROFILE in PROFILE_CONFIGS, f"Unknown RUN_PROFILE={RUN_PROFILE!r}"
assert COLAB_GPU_MODE in GPU_MODE_CONFIGS, f"Unknown COLAB_GPU_MODE={COLAB_GPU_MODE!r}"
RUN_CONFIG = PROFILE_CONFIGS[RUN_PROFILE]
GPU_CONFIG = GPU_MODE_CONFIGS[COLAB_GPU_MODE]

RUN_MANIFEST = RUN_CONFIG["manifest"]
EXPECTED_SAMPLES = RUN_CONFIG["expected_samples"]
RUN_LABEL = RUN_CONFIG["label"]
RUN_METRICS_AFTER_GENERATION = bool(RUN_CONFIG["run_metrics"])
MAX_DISPLAY_RESULTS = 0

COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'datasets/coco')))
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO_ID = "ByteDance/SDXL-Lightning"
LIGHTNING_WEIGHT_NAME = "sdxl_lightning_4step_unet.safetensors"
TARGET_SIZE = (1024, 1024)
BASE_SEED = 2024
BASE_SEED = int(os.environ.get("ANCHORDRAW_BASE_SEED", str(BASE_SEED)))
NEGATIVE_PROMPT = ""

EULER_TIMESTEP_SPACING = "trailing"
EULER_SCHEDULE_STEPS = 50
EULER_T_INDEX_LIST = [0, 4, 12, 25, 37]

# SDXL view profile.
# native_full_v128 là cấu hình chính cho SDXL 1024: 1 full latent view 128x128.
# panorama_v64s8 là stress/debug profile theo sliding-window 64/8; smoke test dễ tạo mosaic nên không dùng làm paper main metric.
SDXL_VIEW_PROFILE = "native_full_v128"  # choices: "native_full_v128", "panorama_v64s8"
SDXL_VIEW_PROFILES = {
    "native_full_v128": {
        "window_size": 128,
        "stride": 128,
        "label": "v128full",
        "expected_view_count_1024": 1,
        "checklist_id": "CFG-MD-SDXL-EULER-G1-B2-V128FULL",
        "paper_use": True,
    },
    "panorama_v64s8": {
        "window_size": 64,
        "stride": 8,
        "label": "v64s8",
        "expected_view_count_1024": 81,
        "checklist_id": "CFG-MD-SDXL-EULER-G1-B2-V64S8",
        "paper_use": False,
    },
}
assert SDXL_VIEW_PROFILE in SDXL_VIEW_PROFILES, f"Unknown SDXL_VIEW_PROFILE={SDXL_VIEW_PROFILE!r}"
SDXL_VIEW_CONFIG = SDXL_VIEW_PROFILES[SDXL_VIEW_PROFILE]
SDXL_VIEW_WINDOW_SIZE = int(SDXL_VIEW_CONFIG["window_size"])
SDXL_VIEW_STRIDE = int(SDXL_VIEW_CONFIG["stride"])
SDXL_VIEW_LABEL = str(SDXL_VIEW_CONFIG["label"])
SDXL_VIEW_EXPECTED_COUNT_1024 = int(SDXL_VIEW_CONFIG["expected_view_count_1024"])
SDXL_VIEW_CHECKLIST_ID = str(SDXL_VIEW_CONFIG["checklist_id"])
SDXL_VIEW_PAPER_USE = bool(SDXL_VIEW_CONFIG["paper_use"])

# Manual CFG trong wrapper: 1.0 = conditional-only/no-CFG.
# Không dùng 0.0 ở đây, vì 0.0 sẽ lấy nhánh unconditional trong công thức thủ công.
EULER_GUIDANCE_SCALE = 1.0

BATCH_SIZE = int(GPU_CONFIG["batch_size"])
BOOTSTRAPPING = int(GPU_CONFIG["bootstrapping"])
SAFE_VAE = bool(GPU_CONFIG["safe_vae"])
METRIC_BATCH_SIZE = int(GPU_CONFIG["metric_batch_size"])
CLIP_BATCH_SIZE = int(GPU_CONFIG["clip_batch_size"])
IS_SPLITS = 10
METRIC_NAMES = ("fid", "is", "clip_fg", "clip_bg", "time")

# Code version tag để tránh lẫn output cũ.
# global_time_ids_v2: SDXL time_ids giữ crop (0,0) cho toàn canvas, không set theo từng tile.
CODE_VERSION_TAG = "global_time_ids_v2"

EXPERIMENT_ID = f"md_naive_runtime_safe_sdxl_lightning4_euler_1024x1024_{SDXL_VIEW_LABEL}_{RUN_LABEL}_b{BATCH_SIZE}_bt{BOOTSTRAPPING}_{COLAB_GPU_MODE}_{CODE_VERSION_TAG}"
RUNS_ROOT = Path(str(WORK_DIR / 'anchordraw_runs'))
OUTPUT_DIR = RUNS_ROOT / EXPERIMENT_ID
GENERATED_IMAGES_DIR = OUTPUT_DIR / "generated_images"
OVERLAY_IMAGES_DIR = OUTPUT_DIR / "mask_overlays"
MASK_CACHE_DIR = Path(str(WORK_DIR / 'multidiffusion_sdxl_mask_cache'))
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"
RUN_SUMMARY_PATH = OUTPUT_DIR / "generation_summary.json"
METRICS_REPORT_PREFIX = f"{EXPERIMENT_ID}_metrics"

assert RUN_MANIFEST.exists(), f"Missing manifest: {RUN_MANIFEST}"
for folder in (OUTPUT_DIR, GENERATED_IMAGES_DIR, OVERLAY_IMAGES_DIR, MASK_CACHE_DIR, METRICS_OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

print(f"[OK] Run profile: {RUN_PROFILE} ({RUN_LABEL})")
print(f"[OK] Colab GPU mode: {COLAB_GPU_MODE}")
print(f"[OK] Manifest: {RUN_MANIFEST}")
print(f"[OK] Expected samples: {EXPECTED_SAMPLES}")
print(f"[OK] COCO root: {COCO_ROOT}")
print(f"[OK] Output dir: {OUTPUT_DIR}")
print(f"[OK] Generated dir: {GENERATED_IMAGES_DIR}")
print(f"[OK] Overlay dir: {OVERLAY_IMAGES_DIR}")
print(f"[OK] Experiment ID: {EXPERIMENT_ID}")
print(f"[OK] Code version tag: {CODE_VERSION_TAG}")
print(f"[OK] BATCH_SIZE: {BATCH_SIZE}")
print(f"[OK] BOOTSTRAPPING: {BOOTSTRAPPING}")
print(f"[OK] METRIC_BATCH_SIZE: {METRIC_BATCH_SIZE}")
print(f"[OK] CLIP_BATCH_SIZE: {CLIP_BATCH_SIZE}")
print(f"[OK] RUN_METRICS_AFTER_GENERATION: {RUN_METRICS_AFTER_GENERATION}")
print(f"[OK] Euler t_index_list: {EULER_T_INDEX_LIST} over {EULER_SCHEDULE_STEPS} schedule steps")
print(f"[OK] Euler guidance scale: {EULER_GUIDANCE_SCALE}")
print(f"[OK] SDXL view profile: {SDXL_VIEW_PROFILE}")
print(f"[OK] SDXL view checklist ID: {SDXL_VIEW_CHECKLIST_ID}")
print(f"[OK] SDXL view paper-use: {SDXL_VIEW_PAPER_USE}")
print(f"[OK] SDXL view window/stride: {SDXL_VIEW_WINDOW_SIZE}/{SDXL_VIEW_STRIDE}")
print(f"[OK] Expected view count at 1024: {SDXL_VIEW_EXPECTED_COUNT_1024}")
print(f"[OK] Safe VAE encode/decode: {SAFE_VAE}")
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 6 ---
# Import dataloader, metrics và wrapper MultiDiffusion SDXL Euler.
import csv
import gc
import hashlib
import importlib
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
sys.path = [str(OURS_SRC)] + [p for p in sys.path if p != str(OURS_SRC)]

from baselines import MultiDiffusionSDXLEuler, get_sdxl_views
from data import COCORegionConfig, batch_item_to_semanticdraw_inputs, build_coco_region_dataloader
from data.visualize import make_mask_overlay

WRAPPER_SOURCE_FILE = Path(inspect.getsourcefile(MultiDiffusionSDXLEuler) or "")
assert WRAPPER_SOURCE_FILE.exists(), "Cannot locate MultiDiffusionSDXLEuler source file."
wrapper_source = WRAPPER_SOURCE_FILE.read_text(encoding="utf-8")
wrapper_sha256 = hashlib.sha256(WRAPPER_SOURCE_FILE.read_bytes()).hexdigest()
assert "add_time_ids_input = add_time_ids" in wrapper_source, "Wrapper is not global_time_ids_v2. Pull latest repo and restart runtime."
assert "add_time_ids_input[:, 2]" not in wrapper_source, "Old tile-crop time_ids logic is still loaded. Pull latest repo and restart runtime."

print("[OK] Imports are ready.")
print("[OK] Wrapper source file:", WRAPPER_SOURCE_FILE)
print("[OK] Wrapper source SHA256:", wrapper_sha256)
print("[OK] Wrapper code version: global_time_ids_v2")
if BASELINE_REGION_FILE.exists():
    region_sha256 = hashlib.sha256(BASELINE_REGION_FILE.read_bytes()).hexdigest()
    print("[OK] Baseline MultiDiffusion region file:", BASELINE_REGION_FILE)
    print("[OK] Baseline region_based.py SHA256:", region_sha256)
else:
    print("[WARN] Baseline MultiDiffusion region_based.py not found; using Ours/src/baselines wrapper only.")

# --- Source notebook cell 7 ---
# Tạo dataloader theo RUN_PROFILE đã chọn.
config = COCORegionConfig(
    coco_root=COCO_ROOT,
    split="val2017",
    instances_json=COCO_ROOT / "annotations" / "instances_val2017.json",
    captions_json=COCO_ROOT / "annotations" / "captions_val2017.json",
    manifest_path=RUN_MANIFEST,
    profile="multidiffusion_coco_all",
    model_family="sdxl",
    target_size=TARGET_SIZE,
    return_image=True,
    cache_resized_masks=True,
    cache_dir=MASK_CACHE_DIR,
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
)

loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
dataset_size = len(loader.dataset)
num_batches = len(loader)
if EXPECTED_SAMPLES is not None:
    assert dataset_size == EXPECTED_SAMPLES, f"Profile {RUN_PROFILE} expected {EXPECTED_SAMPLES} samples, got {dataset_size}."
preview_batch = next(iter(loader))

print(f"[OK] Manifest records: {dataset_size}")
print(f"[OK] Dataloader batches: {num_batches} batch(es) x up to {BATCH_SIZE} sample(s)")
print(f"[OK] First batch size: {len(preview_batch['sample_ids'])}")
print(f"[OK] First batch masks shape: {tuple(preview_batch['masks'].shape)}  # (B, Pmax, C, H, W)")
print("First batch sample IDs:")
for sample_id in preview_batch["sample_ids"]:
    print(" -", sample_id)

# --- Source notebook cell 9 ---
def md_escape(text: object) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_stats(image: Image.Image) -> dict:
    import numpy as np

    arr = torch.from_numpy(np.array(image.convert("RGB")))
    return {
        "min": int(arr.min().item()),
        "max": int(arr.max().item()),
        "mean": float(arr.float().mean().item()),
        "std": float(arr.float().std().item()),
    }


def make_multidiffusion_payload(batch: dict, index: int) -> dict:
    item = batch_item_to_semanticdraw_inputs(batch, index)
    fg_masks = item["masks"].float().cpu()
    fg_union = fg_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    background_mask = (1.0 - fg_union).clamp(0, 1)
    all_masks = torch.cat([background_mask, fg_masks], dim=0)

    prompts = [item["background_prompt"], *item["prompts"]]
    negative_prompts = [NEGATIVE_PROMPT for _ in prompts]
    metadata = item["metadata"]

    return {
        "sample_id": metadata["sample_id"],
        "image_id": metadata["image_id"],
        "file_name": metadata["file_name"],
        "height": item["height"],
        "width": item["width"],
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
    rows.append(f"| Background | {md_escape(payload['prompts'][0])} | - | - |")
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original)
    axes[0].set_title("COCO original resized")
    axes[1].imshow(overlay)
    axes[1].set_title("Foreground mask overlay")
    axes[2].imshow(generated)
    axes[2].set_title("MultiDiffusion naive + SDXL Euler generated")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


print("[OK] Helper functions are ready.")

# --- Source notebook cell 10 ---
# Login Hugging Face nếu có token trong Colab Secrets hoặc biến môi trường.
def maybe_login_to_huggingface() -> None:
    token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from google.colab import userdata
            token = userdata.get("HF_TOKEN")
        except Exception:
            token = None
    if token:
        from huggingface_hub import login
        login(token=token)
        print("[OK] Hugging Face token loaded.")
    else:
        print("[INFO] No HF_TOKEN found. Public/gated model access depends on your Hugging Face permissions.")


assert torch.cuda.is_available(), "Colab runtime chưa bật GPU. Hãy bật Runtime > Change runtime type > GPU."
device = torch.device("cuda:0")
dtype = torch.float16
maybe_login_to_huggingface()
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

# --- Source notebook cell 11 ---
print("[OK] Không cần torchao/PEFT guard: SDXL-Lightning 4-step ở notebook này load bằng UNet checkpoint, không load LoRA.")

# --- Source notebook cell 12 ---
# Load MultiDiffusion + SDXL-Lightning + Euler Discrete.
seed_everything(BASE_SEED)
md_euler = MultiDiffusionSDXLEuler(
    device=device,
    model_id=MODEL_ID,
    lightning_repo_id=LIGHTNING_REPO_ID,
    lightning_weight_name=LIGHTNING_WEIGHT_NAME,
    dtype=dtype,
    variant="fp16",
    timestep_spacing=EULER_TIMESTEP_SPACING,
    t_index_list=EULER_T_INDEX_LIST,
    schedule_steps=EULER_SCHEDULE_STEPS,
    safe_vae=SAFE_VAE,
    view_window_size=SDXL_VIEW_WINDOW_SIZE,
    view_stride=SDXL_VIEW_STRIDE,
)

print("Model family     : SDXL")
print("Method variant   : MD-naive runtime-safe")
print("Code version     :", CODE_VERSION_TAG)
print("Model            :", MODEL_ID)
print("Acceleration     :", f"{LIGHTNING_REPO_ID}/{LIGHTNING_WEIGHT_NAME}")
print("Scheduler        :", type(md_euler.scheduler).__name__)
print("Timestep spacing :", getattr(md_euler.scheduler.config, "timestep_spacing", EULER_TIMESTEP_SPACING))
print("Schedule steps   :", EULER_SCHEDULE_STEPS)
print("t_index_list     :", EULER_T_INDEX_LIST)
print("timesteps        :", [int(t.item()) for t in md_euler.timesteps])
print("sigmas           :", [round(float(s.item()), 4) for s in md_euler.sigmas])
print("guidance_scale   :", EULER_GUIDANCE_SCALE)
print("bootstrapping    :", BOOTSTRAPPING)
sdxl_views = get_sdxl_views(
    TARGET_SIZE[0],
    TARGET_SIZE[1],
    vae_scale_factor=md_euler.vae_scale_factor,
    window_size=SDXL_VIEW_WINDOW_SIZE,
    stride=SDXL_VIEW_STRIDE,
)
print("SDXL view profile:", SDXL_VIEW_PROFILE)
print("SDXL checklist ID:", SDXL_VIEW_CHECKLIST_ID)
print("SDXL paper-use   :", SDXL_VIEW_PAPER_USE)
print("SDXL view window :", SDXL_VIEW_WINDOW_SIZE)
print("SDXL view stride :", SDXL_VIEW_STRIDE)
print("SDXL view count  :", len(sdxl_views), sdxl_views[:3])
assert type(md_euler.scheduler).__name__ == "EulerDiscreteScheduler", "Expected EulerDiscreteScheduler."
assert TARGET_SIZE != (1024, 1024) or len(sdxl_views) == SDXL_VIEW_EXPECTED_COUNT_1024, f"Expected {SDXL_VIEW_EXPECTED_COUNT_1024} views for {SDXL_VIEW_PROFILE}, got {len(sdxl_views)}."

# --- Source notebook cell 13 ---
# Plain sanity check: kiểm tra checkpoint/scheduler/VAE trước khi chạy MultiDiffusion views.
# Cell này KHÔNG dùng MultiDiffusion sliding-window, nên nó phải ra ảnh bình thường.
RUN_PLAIN_SANITY_CHECK = True

if RUN_PLAIN_SANITY_CHECK:
    seed_everything(BASE_SEED)
    plain_sanity_image = md_euler.pipe(
        prompt="a studio photo of a teddy bear on a clean table",
        negative_prompt=NEGATIVE_PROMPT,
        height=TARGET_SIZE[0],
        width=TARGET_SIZE[1],
        num_inference_steps=4,
        guidance_scale=0.0,
    ).images[0].convert("RGB")
    print("[PLAIN SANITY] image stats:", image_stats(plain_sanity_image))
    display(plain_sanity_image.resize((512, 512)))
else:
    print("[INFO] Plain sanity check skipped.")

# Optional: kiểm tra riêng MD full-mask. Bật True nếu muốn debug wrapper, nhưng rất chậm vì chạy 81 views.
RUN_MD_FULL_MASK_SANITY_CHECK = False

if RUN_MD_FULL_MASK_SANITY_CHECK:
    seed_everything(BASE_SEED)
    sanity_mask = torch.ones(1, 1, TARGET_SIZE[0], TARGET_SIZE[1])
    md_sanity_image = md_euler.generate(
        masks=sanity_mask,
        prompts=["a studio photo of a teddy bear on a clean table"],
        negative_prompts=[NEGATIVE_PROMPT],
        height=TARGET_SIZE[0],
        width=TARGET_SIZE[1],
        guidance_scale=EULER_GUIDANCE_SCALE,
        bootstrapping=BOOTSTRAPPING,
    )
    print("[MD FULL-MASK SANITY] image stats:", image_stats(md_sanity_image))
    display(md_sanity_image.resize((512, 512)))
else:
    print("[INFO] MD full-mask sanity skipped. Manifest generation will test MD behavior.")

# --- Source notebook cell 14 ---
failed_images = []
# Chạy generation cho mọi sample trong manifest và hiển thị kết quả.
summary = []
global_index = 0

for batch_index, batch in enumerate(loader):
    print(f"[BATCH] {batch_index + 1}/{len(loader)} - {len(batch['sample_ids'])} sample(s)")

    for local_index, sample_id in enumerate(batch["sample_ids"]):
        retry_overhead_sec = 0.0
        for attempt in range(1, 4):
            attempt_started = time.perf_counter()
            tracked_lists = ('summary',)
            list_lengths_before = {name: len(globals()[name]) for name in tracked_lists}
            try:
                payload = make_multidiffusion_payload(batch, local_index)
                original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
                overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)

                seed = BASE_SEED + global_index
                seed_everything(seed)

                tic = time.perf_counter()
                generated = md_euler.generate(
                    masks=payload["all_masks"],
                    prompts=payload["prompts"],
                    negative_prompts=payload["negative_prompts"],
                    height=payload["height"],
                    width=payload["width"],
                    guidance_scale=EULER_GUIDANCE_SCALE,
                    bootstrapping=BOOTSTRAPPING,
                )
                elapsed = time.perf_counter() - tic + retry_overhead_sec

                stem = f"{global_index:04d}_{payload['sample_id']}"
                generated_path = GENERATED_IMAGES_DIR / f"{stem}_generated.png"
                overlay_path = OVERLAY_IMAGES_DIR / f"{stem}_overlay.png"
                generated.save(generated_path)
                overlay.save(overlay_path)

                with Image.open(generated_path) as check_img:
                    check_img.verify()

                summary.append({
                    "index": global_index,
                    "batch_index": batch_index,
                    "local_index": local_index,
                    "sample_id": payload["sample_id"],
                    "image_id": payload["image_id"],
                    "file_name": payload["file_name"],
                    "seed": seed,
                    "method": "multidiffusion",
                    "model_family": "sdxl",
                    "model_id": MODEL_ID,
                    "sampler": "euler_discrete",
                    "scheduler": type(md_euler.scheduler).__name__,
                    "lightning_repo": LIGHTNING_REPO_ID,
                    "lightning_weight": LIGHTNING_WEIGHT_NAME,
                    "euler_t_index_list": EULER_T_INDEX_LIST,
                    "num_schedule_steps": EULER_SCHEDULE_STEPS,
                    "guidance_scale": EULER_GUIDANCE_SCALE,
                    "bootstrapping": BOOTSTRAPPING,
                    "sdxl_view_profile": SDXL_VIEW_PROFILE,
                    "sdxl_view_checklist_id": SDXL_VIEW_CHECKLIST_ID,
                    "sdxl_view_window_size": SDXL_VIEW_WINDOW_SIZE,
                    "sdxl_view_stride": SDXL_VIEW_STRIDE,
                    "sdxl_view_count": len(sdxl_views),
                    "num_regions_including_background": len(payload["prompts"]),
                    "elapsed_sec": elapsed,
                    "generated_path": str(generated_path),
                    "overlay_path": str(overlay_path),
                })

                should_display = MAX_DISPLAY_RESULTS is None or global_index < MAX_DISPLAY_RESULTS
                if should_display:
                    display_multidiffusion_result(payload, original, overlay, generated, elapsed, generated_path)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                import traceback
                rolled_back = any(len(globals()[name]) > list_lengths_before[name] for name in tracked_lists)
                for name in tracked_lists:
                    del globals()[name][list_lengths_before[name]:]
                if rolled_back:
                    try:
                        pass
                    except Exception as checkpoint_error:
                        print(f"[CHECKPOINT ERROR] index={global_index}: {checkpoint_error}", flush=True)
                attempt_elapsed_sec = time.perf_counter() - attempt_started
                retry_overhead_sec += attempt_elapsed_sec
                failure = {"index": global_index, "seed": BASE_SEED + global_index,
                           "sample_id": str(locals().get("sample_id", "")),
                           "attempt": attempt, "attempt_elapsed_sec": attempt_elapsed_sec,
                           "error": repr(exc),
                           "traceback": traceback.format_exc()}
                with (WORK_DIR / "failed_images.jsonl").open("a", encoding="utf-8") as output:
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
    raise RuntimeError(f"Run incomplete: {len(failed_images)} manifest indices failed after three attempts: {failed_images}")

summary_path = OUTPUT_DIR / "generation_summary.json"
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

display(Markdown(
    f"## Done\n"
    f"Generated `{len(summary)}` image(s) from `{dataset_size}` manifest record(s). "
    f"Summary saved to `{summary_path}`."
))
summary[:5]

# --- Source notebook cell 17 ---
# Giải phóng VRAM trước khi load Inception/CLIP cho metric.
import gc

for var_name in ("md_euler", "sanity_image", "generated", "payload", "overlay", "original", "batch", "loader", "preview_batch"):
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

# --- Source notebook cell 18 ---
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
    metric_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    metric_config = MetricEvaluationConfig(
        manifest_path=RUN_MANIFEST,
        coco_root=COCO_ROOT,
        generated_dir=GENERATED_IMAGES_DIR,
        generation_summary=RUN_SUMMARY_PATH,
        output_dir=METRICS_OUTPUT_DIR,
        model_family="sdxl",
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
    metrics_json, metrics_csv = write_metrics_report(metric_report, METRICS_OUTPUT_DIR, prefix=METRICS_REPORT_PREFIX)
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
