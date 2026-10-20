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

# Notebook: Ours/colab_semanticdraw_smoke/semanticdraw_sdxl_euler_smoke_bs2_colab.ipynb
# Notebook SHA256: d33abbbbabcd8d8b58fa30b53573d7b2d255b43aab977dec56621ef56eaed70f

# --- Source notebook cell 4 ---
# Cấu hình Colab cho SemanticDraw SDXL + Euler Discrete.
from pathlib import Path

# Đổi sang "full1073" sau khi smoke chạy ổn.
RUN_PROFILE = os.environ.get('RUN_PROFILE', 'full1073')  # choices: "smoke_bs2", "full1073"

# low_vram: phù hợp Colab T4/L4 khi muốn chạy an toàn.
# high_vram_24gb: dùng khi có GPU nhiều VRAM hơn, ví dụ A100/RTX 4090.
COLAB_GPU_MODE = os.environ.get('COLAB_GPU_MODE', 'high_vram_24gb')  # choices: "low_vram", "high_vram_24gb"
LOW_VRAM_COLAB_MODE = COLAB_GPU_MODE == "low_vram"

MANIFEST_BY_PROFILE = {
    "smoke_bs2": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "smoke" / "coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_smoke_bs2.jsonl",
    "full1073": REPO_ROOT / "Ours" / "data_manifests" / "coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_all.jsonl",
}
EXPECTED_DATASET_SIZE_BY_PROFILE = {
    "smoke_bs2": 2,
    "full1073": 1073,
}

assert RUN_PROFILE in MANIFEST_BY_PROFILE, f"Unknown RUN_PROFILE: {RUN_PROFILE}"
assert COLAB_GPU_MODE in {"low_vram", "high_vram_24gb"}, f"Unknown COLAB_GPU_MODE: {COLAB_GPU_MODE}"

RUN_MANIFEST = MANIFEST_BY_PROFILE[RUN_PROFILE]
EXPECTED_DATASET_SIZE = EXPECTED_DATASET_SIZE_BY_PROFILE[RUN_PROFILE]
COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'datasets/coco')))

# Baseline SDXL dùng SDXL base và thay UNet bằng SDXL-Lightning 4-step weight.
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO_ID = "ByteDance/SDXL-Lightning"
LIGHTNING_WEIGHT_NAME = "sdxl_lightning_4step_unet.safetensors"
EULER_NUM_INFERENCE_STEPS = 4

# Custom SemanticDraw + bootstrap giữ t_index_list mặc định của baseline.
SEMANTICDRAW_T_INDEX_LIST = [0, 4, 12, 25, 37]
SEMANTICDRAW_NUM_INFERENCE_STEPS = None
EULER_TIMESTEP_SPACING = "trailing"
EULER_GUIDANCE_SCALE = 0.0

TARGET_SIZE = (1024, 1024)
BASE_SEED = 2024

# BATCH_SIZE ở dataloader, không phải generation batch của diffusion.
# Baseline vẫn sinh tuần tự từng ảnh trong loop.
if RUN_PROFILE == "smoke_bs2":
    BATCH_SIZE = 2
else:
    BATCH_SIZE = 1 if LOW_VRAM_COLAB_MODE else 2

# smoke_bs2 dùng cấu hình demo-style để validate nhanh.
# full1073 low_vram dùng bootstrap=1 để giảm rủi ro VRAM; high_vram có thể dùng 2.
BOOTSTRAP_STEPS = 2 if RUN_PROFILE == "smoke_bs2" else (1 if LOW_VRAM_COLAB_MODE else 2)

MASK_STD = 0.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.3 if RUN_PROFILE == "smoke_bs2" else 0.0
MASK_TYPE = "discrete"
NEGATIVE_PROMPT = ""

SAMPLE_TAG = "smoke2" if RUN_PROFILE == "smoke_bs2" else "full1073"
RESOLUTION_TAG = str(TARGET_SIZE[0])
SAMPLER_TAG = "euler"
ACCEL_TAG = "lightning4"
RUN_MODE_TAG = "colab_lvram" if LOW_VRAM_COLAB_MODE else "colab_24gb"
EXPERIMENT_ID = (
    f"sdraw_sdxl_{ACCEL_TAG}_{SAMPLER_TAG}_"
    f"{RESOLUTION_TAG}_{SAMPLE_TAG}_"
    f"b{BATCH_SIZE}_bt{BOOTSTRAP_STEPS}_{RUN_MODE_TAG}"
)

RUNS_ROOT = Path(str(WORK_DIR / 'anchordraw_runs'))
OUTPUT_DIR = RUNS_ROOT / EXPERIMENT_ID
MASK_CACHE_DIR = Path(str(WORK_DIR / 'semanticdraw_mask_cache'))
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"

MAX_DISPLAY_RESULTS = 0
RUN_PLAIN_PIPELINE_SANITY = True
RUN_BOOTSTRAP_TRACE = True
TRACE_DECODE_LATENT = False

# Metric mặc định tắt cho smoke vì 2 ảnh không đủ ý nghĩa.
# Khi chuyển sang full1073, flag này tự bật.
RUN_METRICS_AFTER_GENERATION = RUN_PROFILE == "full1073"

# Compatibility cho runtime Colab. Không đổi checkpoint/sampler.
PATCH_SEMANTICDRAW_SAFE_REGION_MIXING = True
PATCH_SEMANTICDRAW_FULL_VAE_UPCAST = True
# Bootstrap dùng latent của ảnh trắng. Trên Colab, SDXL VAE fp16 encode ảnh trắng 1024x1024
# có thể sinh NaN/ảnh đen; flag này chỉ encode latent nền trắng bằng fp32 rồi cast về dtype model.
PATCH_SEMANTICDRAW_SAFE_WHITE_BOOTSTRAP_LATENT = True

STOP_ON_NEAR_BLACK_OUTPUT = True
NEAR_BLACK_MAX_PIXEL = 5
NEAR_BLACK_STD_PIXEL = 1.0

STOP_ON_STATIC_NOISE_OUTPUT = True
STATIC_NOISE_MIN_STD_PIXEL = 90.0
STATIC_NOISE_MEAN_LOW = 95.0
STATIC_NOISE_MEAN_HIGH = 160.0

METRIC_NAMES = ("fid", "is", "clip_fg", "clip_bg", "time")
METRIC_BATCH_SIZE = 1 if LOW_VRAM_COLAB_MODE else 2
CLIP_BATCH_SIZE = 4 if LOW_VRAM_COLAB_MODE else 8
IS_SPLITS = 10
METRICS_REPORT_PREFIX = f"{EXPERIMENT_ID}_metrics"

assert RUN_MANIFEST.exists(), f"Missing manifest: {RUN_MANIFEST}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[OK] Run profile: {RUN_PROFILE}")
print(f"[OK] Colab GPU mode: {COLAB_GPU_MODE}")
print(f"[OK] Manifest: {RUN_MANIFEST}")
print(f"[OK] Expected samples: {EXPECTED_DATASET_SIZE}")
print(f"[OK] COCO root: {COCO_ROOT}")
print(f"[OK] Experiment ID: {EXPERIMENT_ID}")
print(f"[OK] Output dir: {OUTPUT_DIR}")
print(f"[OK] Metrics output dir: {METRICS_OUTPUT_DIR}")
print(f"[OK] Batch size: {BATCH_SIZE}")
print(f"[OK] Bootstrap steps: {BOOTSTRAP_STEPS}")
print(f"[OK] SemanticDraw t_index_list: {SEMANTICDRAW_T_INDEX_LIST}")
print(f"[OK] SemanticDraw num_inference_steps arg: {SEMANTICDRAW_NUM_INFERENCE_STEPS}")
print(f"[OK] Metric batch size: {METRIC_BATCH_SIZE}")
print(f"[OK] CLIP batch size: {CLIP_BATCH_SIZE}")
print(f"[OK] Run metrics after generation: {RUN_METRICS_AFTER_GENERATION}")
print(f"[OK] Safe white bootstrap latent: {PATCH_SEMANTICDRAW_SAFE_WHITE_BOOTSTRAP_LATENT}")
print(f"[OK] SDXL model: {MODEL_ID}")
print(f"[OK] Lightning weight: {LIGHTNING_REPO_ID}/{LIGHTNING_WEIGHT_NAME}")
if LOW_VRAM_COLAB_MODE:
    print("[WARN] Low-VRAM mode ưu tiên validate ổn định. Dùng high_vram_24gb cho benchmark trên GPU mạnh.")
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 5 ---
# Tạo cấu trúc folder output cho run hiện tại.
# Root OUTPUT_DIR giữ summary/log metadata.
# Ảnh generated và overlay được tách riêng để dễ xem, zip, xoá, hoặc dùng cho metric.
GENERATED_IMAGES_DIR = OUTPUT_DIR / "generated_images"
OVERLAY_IMAGES_DIR = OUTPUT_DIR / "mask_overlays"
RUN_SUMMARY_PATH = OUTPUT_DIR / "generation_summary.json"

for folder in (GENERATED_IMAGES_DIR, OVERLAY_IMAGES_DIR):
    folder.mkdir(parents=True, exist_ok=True)

print(f"[OK] Generated images dir: {GENERATED_IMAGES_DIR}")
print(f"[OK] Mask overlays dir: {OVERLAY_IMAGES_DIR}")
print(f"[OK] Generation summary path: {RUN_SUMMARY_PATH}")

# --- Source notebook cell 7 ---
# Import dataloader của Ours và baseline SemanticDrawSDXLPipeline.
import sys
import importlib.util
import json
import time

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
# terminal display provided above

OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_SRC = REPO_ROOT / "Baseline" / "semantic-draw-main" / "src"

sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from data.visualize import make_mask_overlay

# Load trực tiếp file pipeline_semantic_draw_sdxl.py để tránh import toàn bộ model/__init__.py.
sys.path.insert(0, str(BASELINE_SRC))
pipeline_path = BASELINE_SRC / "model" / "pipeline_semantic_draw_sdxl.py"
spec = importlib.util.spec_from_file_location("pipeline_semantic_draw_sdxl", pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)

if PATCH_SEMANTICDRAW_SAFE_REGION_MIXING:
    # Patch runtime cho một điểm dễ sinh NaN trong region mixing:
    # bản gốc dùng `value / count_all`, trong đó count_all có thể bằng 0 hoặc quá nhỏ ở vài pixel/timestep.
    # Ta clamp mẫu số và giữ latent cũ ở vùng chưa được region nào phủ. Khi mask phủ đủ, kết quả tương đương bản gốc.
    source = pipeline_path.read_text(encoding="utf-8")
    old_mixing = "                latent = torch.where(count_all > 0, value / count_all, value)"
    new_mixing = (
        "                denom = count_all.clamp_min(torch.finfo(count_all.dtype).tiny)\n"
        "                latent = torch.where(count_all > 0, value / denom, latent)"
    )
    assert old_mixing in source, "Could not find SemanticDraw SDXL region mixing line to patch."
    source = source.replace(old_mixing, new_mixing)

    # Patch bug trong baseline SDXL khi truyền `background_prompt` nhưng không truyền background image.
    # Pipeline tự prepend background prompt/mask, nhưng các biến đếm `num_*` vẫn giữ số foreground cũ.
    # Kết quả là SDXL `time_ids` có batch size 4 trong khi prompt_embeds có batch size 5.
    # Lỗi Colab thấy: RuntimeError: shape '[5, -1]' is invalid for input of size 6144.
    old_background_count = "                has_background = False # Regard that background does not exist."
    new_background_count = (
        "                has_background = False # Regard that background does not exist.\n"
        "                num_masks = len(prompts)\n"
        "                num_prompts = len(prompts)\n"
        "                num_nprompts = len(negative_prompts)"
    )
    assert old_background_count in source, "Could not find SemanticDraw SDXL background_prompt count line to patch."
    source = source.replace(old_background_count, new_background_count)

    # Patch trace support: baseline SDXL ignores output_type='latent' at the final return
    # and converts the latent to a PIL image. For debugging we need the raw latent tensor.
    old_latent_return = "        else:\n            image = latent\n\n        # Return PIL Image."
    new_latent_return = (
        "        else:\n"
        "            image = latent\n\n"
        "        if output_type == \"latent\":\n"
        "            return image\n\n"
        "        # Return PIL Image."
    )
    assert old_latent_return in source, "Could not find SemanticDraw SDXL latent return block to patch."
    source = source.replace(old_latent_return, new_latent_return)

    pipeline_module.__file__ = str(pipeline_path)
    exec(compile(source, str(pipeline_path), "exec"), pipeline_module.__dict__)
    print("[OK] Patched SemanticDraw SDXL safe region mixing and latent trace return for Colab runtime.")
else:
    assert spec.loader is not None
    spec.loader.exec_module(pipeline_module)

SemanticDrawSDXLPipeline = pipeline_module.SemanticDrawSDXLPipeline

print("[OK] Imports are ready.")

# --- Source notebook cell 8 ---
# Tạo dataloader cho manifest đang chọn.
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
preview_batch = next(iter(loader))

assert config.model_family == "sdxl", f"model_family mismatch: {config.model_family}"
assert tuple(config.target_hw) == TARGET_SIZE, f"target_hw mismatch: {config.target_hw}"
assert dataset_size == EXPECTED_DATASET_SIZE, f"{RUN_PROFILE} manifest should contain {EXPECTED_DATASET_SIZE} records, got {dataset_size}"
assert len(preview_batch["sample_ids"]) == min(BATCH_SIZE, dataset_size), f"First batch size mismatch: {len(preview_batch['sample_ids'])}"
assert tuple(preview_batch["masks"].shape[-2:]) == TARGET_SIZE, f"Mask spatial size mismatch: {preview_batch['masks'].shape}"
assert int(preview_batch["masks"].shape[2]) == 1, f"Mask channel mismatch, expected C=1: {preview_batch['masks'].shape}"

for size in preview_batch["target_sizes"]:
    assert tuple(int(v) for v in size.tolist()) == TARGET_SIZE, f"Batch target size mismatch: {size}"

print(f"[OK] Manifest records: {dataset_size}")
print(f"[OK] Dataloader batches: {num_batches} batch(es) x up to {BATCH_SIZE} sample(s)")
print(f"[OK] First batch size: {len(preview_batch['sample_ids'])}")
print(f"[OK] First batch masks shape: {tuple(preview_batch['masks'].shape)}  # (B, Pmax, C, H, W)")
print("First batch sample IDs:")
for sample_id in preview_batch["sample_ids"]:
    print(" -", sample_id)

# --- Source notebook cell 10 ---
def md_escape(text: object) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")

def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_semanticdraw_payload(batch: dict, index: int) -> dict:
    item = batch_item_to_semanticdraw_inputs(batch, index)
    fg_masks = item["masks"].float().cpu()  # (p, 1, H, W)
    fg_union = fg_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    background_mask = (1.0 - fg_union).clamp(0, 1)
    all_masks = torch.cat([background_mask, fg_masks], dim=0)

    prompts = [item["background_prompt"], *item["prompts"]]
    negative_prompts = [NEGATIVE_PROMPT for _ in prompts]
    metadata = item["metadata"]

    return {
        "sample_id": item["metadata"]["sample_id"],
        "image_id": item["metadata"]["image_id"],
        "file_name": item["metadata"]["file_name"],
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

def display_smoke_result(payload: dict, original: Image.Image, overlay: Image.Image, generated: Image.Image, elapsed: float, generated_path: Path) -> None:
    rows = ["| Region | Prompt | Annotation | Area ratio |", "|---|---|---:|---:|"]
    rows.append(f"| Background | {md_escape(payload['prompts'][0])} | - | - |")
    for label, prompt, ann_id, area in zip(payload["category_names"], payload["foreground_prompts"], payload["annotation_ids"], payload["area_ratios"]):
        rows.append(f"| {md_escape(label)} | {md_escape(prompt)} | {ann_id} | {float(area):.4f} |")

    display(Markdown(
        f"### `{payload['sample_id']}`\n"
        f"- image_id: `{payload['image_id']}`\n"
        f"- file: `{payload['file_name']}`\n"
        f"- generated path: `{generated_path}`\n"
        f"- elapsed: `{elapsed:.2f}s`\n\n"
        + "\n".join(rows)
    ))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original)
    axes[0].set_title("COCO original resized")
    axes[1].imshow(overlay)
    axes[1].set_title("Foreground mask overlay")
    axes[2].imshow(generated)
    axes[2].set_title("SemanticDraw generated")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()



def pil_image_stats(image: Image.Image) -> dict:
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }

def is_near_black_image(image: Image.Image, max_pixel: int = NEAR_BLACK_MAX_PIXEL, std_pixel: float = NEAR_BLACK_STD_PIXEL) -> bool:
    stats = pil_image_stats(image)
    return stats["max"] <= max_pixel and stats["std"] <= std_pixel

def is_static_noise_image(image: Image.Image) -> bool:
    stats = pil_image_stats(image)
    return (
        stats["min"] == 0
        and stats["max"] == 255
        and STATIC_NOISE_MEAN_LOW <= stats["mean"] <= STATIC_NOISE_MEAN_HIGH
        and stats["std"] >= STATIC_NOISE_MIN_STD_PIXEL
    )

print("[OK] Helper functions are ready.")

# --- Source notebook cell 11 ---
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


assert torch.cuda.is_available(), "Colab runtime chưa bật GPU. Hãy bật Runtime > Change runtime type > GPU rồi chạy lại."
device = torch.device("cuda:0")
dtype = torch.float16

maybe_login_to_huggingface()
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

# --- Source notebook cell 12 ---
# Guard chống lỗi torchao/PEFT trước khi load SDXL/Lightning components.
# Nếu bạn vừa gặp lỗi torchao ở cell load pipeline, hãy restart Colab runtime rồi Run All.
import sys
import importlib
import importlib.util

importlib.invalidate_caches()

if "torchao" in sys.modules:
    raise RuntimeError(
        "torchao is already imported in this Python session. Restart the Colab runtime, run the dependency cell, then Run All. "
        "SemanticDraw SDXL + Euler Discrete does not need torchao for this run."
    )

if importlib.util.find_spec("torchao") is not None:
    raise RuntimeError(
        "torchao is still installed/importable in this runtime. Run the dependency cell, then restart the Colab runtime and Run All. "
        "SemanticDraw SDXL + Euler Discrete does not need torchao for this run."
    )

print("[OK] torchao is not importable; PEFT should skip torchao LoRA dispatch if reached.")

# --- Source notebook cell 13 ---
# Load baseline SemanticDraw SDXL pipeline.
# has_i2t=False để không tải BLIP-2 vì COCO caption đã là background prompt.
#
# Fix cho Colab/safetensors:
# pipeline SDXL gốc gọi load_file(..., device=torch.device("cuda:0")).
# Một số runtime Colab báo `SafetensorError: device cuda:0 is invalid`.
# Vì vậy ta đọc checkpoint SDXL-Lightning trên CPU trước, rồi load_state_dict sẽ copy weight vào UNet.
from safetensors.torch import load_file as _safetensors_load_file

def _load_safetensors_on_cpu(filename, device=None):
    if device is not None and str(device).startswith("cuda"):
        print(f"[INFO] Loading safetensors checkpoint on CPU first instead of {device}.")
    return _safetensors_load_file(filename, device="cpu")

pipeline_module.load_file = _load_safetensors_on_cpu

seed_everything(BASE_SEED)
smd = SemanticDrawSDXLPipeline(
    device=device,
    dtype=dtype,
    hf_key=None,
    has_i2t=False,
    t_index_list=SEMANTICDRAW_T_INDEX_LIST,
    default_mask_std=MASK_STD,
    default_mask_strength=MASK_STRENGTH,
    default_preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
    mask_type=MASK_TYPE,
)

if hasattr(smd.pipe, "enable_attention_slicing"):
    smd.pipe.enable_attention_slicing()
if hasattr(smd.pipe, "enable_vae_slicing"):
    smd.pipe.enable_vae_slicing()
if hasattr(smd.pipe, "enable_vae_tiling"):
    smd.pipe.enable_vae_tiling()
    print("[OK] VAE tiling enabled for lower SDXL decode VRAM.")

# Fix cho Colab/diffusers:
# SDXL VAE có `force_upcast=True`. Hàm `upcast_vae()` trong baseline chỉ upcast một phần VAE,
# nhưng một số version diffusers/Colab làm hidden states thành float32 trong khi vài layer vẫn float16,
# gây lỗi: `RuntimeError: expected scalar type Half but found Float`.
# Override này giữ nguyên logic generation/sampler, chỉ đảm bảo toàn bộ VAE đồng nhất float32 khi decode.
def _upcast_full_vae_for_colab():
    smd.vae.to(dtype=torch.float32)

if PATCH_SEMANTICDRAW_FULL_VAE_UPCAST:
    # Không upcast VAE ngay sau khi load, vì plain `smd.pipe(...)` của diffusers vẫn cần tự quản lý VAE dtype.
    # Ta chỉ override hook mà custom SemanticDraw gọi đúng lúc decode.
    smd.upcast_vae = _upcast_full_vae_for_colab
    print("[OK] SemanticDraw custom decode will upcast the full VAE only when needed.")



def _latent_stats_for_bootstrap(name: str, tensor: torch.Tensor) -> dict:
    t = tensor.detach().float()
    finite = bool(torch.isfinite(t).all().item())
    return {
        "name": name,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": finite,
        "min": float(t.min().item()) if finite else float("nan"),
        "max": float(t.max().item()) if finite else float("nan"),
        "mean": float(t.mean().item()) if finite else float("nan"),
        "std": float(t.std().item()) if finite else float("nan"),
    }


if PATCH_SEMANTICDRAW_SAFE_WHITE_BOOTSTRAP_LATENT:
    try:
        original_white_probe = smd.get_white_background(*TARGET_SIZE)
        original_white_stats = _latent_stats_for_bootstrap("original_fp16_white_bootstrap_latent_probe", original_white_probe)
        print("[CHECK] Original bootstrap white latent probe:", original_white_stats)
    except Exception as exc:
        original_white_stats = {"name": "original_fp16_white_bootstrap_latent_probe", "error": repr(exc)}
        print("[CHECK] Original bootstrap white latent probe failed:", repr(exc))

    def _safe_get_white_background(height: int, width: int) -> torch.Tensor:
        """Colab-safe white bootstrap latent; does not change SemanticDraw denoising logic."""
        latent_h = height // smd.vae_scale_factor
        latent_w = width // smd.vae_scale_factor
        cache = getattr(smd, "_safe_white_bootstrap_latent", None)
        if cache is None or cache.shape[-2] < latent_h or cache.shape[-1] < latent_w:
            old_dtype = smd.vae.dtype
            smd.vae.to(dtype=torch.float32)
            white_img = torch.ones(1, 3, height, width, dtype=torch.float32, device=smd.device)
            white_latent = smd.encode_imgs(white_img).to(dtype=smd.dtype)
            if old_dtype != torch.float32:
                smd.vae.to(dtype=old_dtype)
            safe_stats = _latent_stats_for_bootstrap("safe_fp32_white_bootstrap_latent", white_latent)
            if not safe_stats["finite"]:
                raise RuntimeError(f"Safe white bootstrap latent is not finite: {safe_stats}")
            smd._safe_white_bootstrap_latent = white_latent
            smd._safe_white_bootstrap_latent_stats = {
                "original_probe": original_white_stats,
                "safe": safe_stats,
            }
            print("[COMPAT] Safe white bootstrap latent:", safe_stats)
        return smd._safe_white_bootstrap_latent[..., :latent_h, :latent_w]

    # Override chỉ phần tạo latent nền trắng cho bootstrap. Không sửa denoising/mask/latent mixing.
    smd.get_white_background = _safe_get_white_background
    _ = smd.get_white_background(*TARGET_SIZE)

print("[OK] SemanticDrawSDXLPipeline is ready.")

# --- Source notebook cell 14 ---
# Kiểm tra trực tiếp notebook đang dùng sampler/scheduler nào.
print("Model family:", "SDXL")
print("Model:", MODEL_ID)
print("Acceleration:", f"{LIGHTNING_REPO_ID}/{LIGHTNING_WEIGHT_NAME}")
print("Scheduler:", type(smd.scheduler).__name__)
print("Pipeline scheduler:", type(smd.pipe.scheduler).__name__)
print("Timestep spacing:", getattr(smd.scheduler.config, "timestep_spacing", None))
print("SemanticDraw t_index_list:", SEMANTICDRAW_T_INDEX_LIST)
print("SemanticDraw num_inference_steps arg:", SEMANTICDRAW_NUM_INFERENCE_STEPS)
print("Default inference steps:", smd.default_num_inference_steps)
print("Default guidance scale:", smd.default_guidance_scale)
print("Timesteps:", [int(t) for t in smd.timesteps.detach().cpu().tolist()])
print("Sigmas:", [float(x) for x in smd.sigmas.detach().cpu().tolist()])
print("Delta sigmas:", [float(x) for x in smd.dt.detach().cpu().tolist()])
print("SemanticDraw dtype:", smd.dtype)
print("UNet dtype:", smd.unet.dtype)
print("VAE dtype:", smd.vae.dtype)

assert type(smd.scheduler).__name__ == "EulerDiscreteScheduler", "Expected Euler Discrete sampler for SDXL."
assert smd.default_num_inference_steps == EULER_NUM_INFERENCE_STEPS
assert abs(float(smd.default_guidance_scale) - float(EULER_GUIDANCE_SCALE)) < 1e-8
assert torch.isfinite(smd.sigmas).all(), "Euler sigmas contain non-finite values."
assert torch.isfinite(smd.dt).all(), "Euler delta sigmas contain non-finite values."

# --- Source notebook cell 15 ---
# Sanity check: chạy plain SDXL-Lightning không mask để kiểm tra checkpoint/scheduler/VAE.
# Nếu ảnh này đẹp nhưng SemanticDraw bị đen, lỗi nằm ở nhánh SemanticDraw/mask mixing.
# Nếu ảnh này cũng đen, cần kiểm tra lại checkpoint, dtype, VAE hoặc runtime Colab.
if RUN_PLAIN_PIPELINE_SANITY:
    seed_everything(BASE_SEED)
    sanity_prompt = "a studio photo of a teddy bear on a clean table"
    sanity_image = smd.pipe(
        sanity_prompt,
        num_inference_steps=EULER_NUM_INFERENCE_STEPS,
        guidance_scale=EULER_GUIDANCE_SCALE,
        height=TARGET_SIZE[0],
        width=TARGET_SIZE[1],
    ).images[0].convert("RGB")
    sanity_stats = pil_image_stats(sanity_image)
    print("[SANITY] Plain SDXL-Lightning stats:", sanity_stats)
    sanity_preview = sanity_image.resize((512, 512))
    display(sanity_preview)
    if is_near_black_image(sanity_image):
        raise RuntimeError(
            "Plain SDXL-Lightning sanity image is nearly black. "
            "This points to model/checkpoint/VAE/runtime setup, not COCO masks."
        )

    # Dọn VRAM sau sanity check. Diffusers SDXL có thể tạm upcast VAE khi decode,
    # nên ép VAE về dtype chính trước khi đi vào custom SemanticDraw loop.
    del sanity_image, sanity_preview
    import gc
    gc.collect()
    smd.vae.to(dtype=dtype)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print("[OK] Sanity objects released and VAE returned to", smd.vae.dtype)
else:
    print("[INFO] Plain SDXL-Lightning sanity check skipped.")

# --- Source notebook cell 16 ---
# Trace lỗi SemanticDraw SDXL bootstrap trên sample đầu tiên.
# Cell này giúp phân biệt:
# - mask/prompt/dataloader có bất thường không
# - latent sau custom denoising loop có còn giống noise không
# - VAE decode có làm hỏng ảnh không
#
# Chạy cell này sau khi load `smd` và dataloader, trước khi chạy full generation.
RUN_BOOTSTRAP_TRACE = True
# Decode latent bằng VAE ở 1024x1024 rất tốn VRAM trên Colab T4.
# Mặc định trace chỉ in latent/mask/scheduler stats để tránh OOM.
TRACE_DECODE_LATENT = False

if RUN_BOOTSTRAP_TRACE:
    import math
    import pandas as pd

    trace_batch = preview_batch
    trace_payload = make_semanticdraw_payload(trace_batch, 0)
    trace_seed = BASE_SEED
    seed_everything(trace_seed)

    print("[TRACE] sample_id:", trace_payload["sample_id"])
    print("[TRACE] image_id:", trace_payload["image_id"])
    print("[TRACE] background prompt:", trace_payload["prompts"][0])
    print("[TRACE] foreground prompts:", trace_payload["foreground_prompts"])
    print("[TRACE] category names:", trace_payload["category_names"])
    print("[TRACE] annotation ids:", trace_payload["annotation_ids"])
    print("[TRACE] area ratios:", [float(x) for x in trace_payload["area_ratios"]])
    print("[TRACE] bootstrap_steps:", BOOTSTRAP_STEPS)
    print("[TRACE] semanticdraw t_index_list:", SEMANTICDRAW_T_INDEX_LIST)
    print("[TRACE] smd.timesteps:", [int(t) for t in smd.timesteps.detach().cpu().tolist()])
    print("[TRACE] smd.sigmas:", [float(x) for x in smd.sigmas.detach().cpu().tolist()])
    print("[TRACE] smd.dt:", [float(x) for x in smd.dt.detach().cpu().tolist()])

    fg_masks_cpu = trace_payload["foreground_masks"].float().cpu()
    union_mask = fg_masks_cpu.sum(dim=0).clamp(0, 1)
    background_mask = (1.0 - union_mask).clamp(0, 1)
    print("[TRACE] foreground_masks shape:", tuple(fg_masks_cpu.shape))
    print("[TRACE] foreground union min/max/mean/sum:",
          float(union_mask.min()), float(union_mask.max()), float(union_mask.mean()), float(union_mask.sum()))
    print("[TRACE] background mask min/max/mean/sum:",
          float(background_mask.min()), float(background_mask.max()), float(background_mask.mean()), float(background_mask.sum()))

    processed_masks, processed_masks_blurred, processed_std = smd.process_mask(
        trace_payload["foreground_masks"].to(device=device, dtype=torch.float32),
        MASK_STRENGTH,
        MASK_STD,
        height=trace_payload["height"],
        width=trace_payload["width"],
        use_boolean_mask=True,
        timesteps=smd.timesteps,
        preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
    )
    processed_bg_masks = (1 - processed_masks.sum(dim=0)).clamp(0, 1)

    mask_rows = []
    for t_idx in range(processed_masks.shape[1]):
        fg_t = processed_masks[:, t_idx].detach().float().cpu()
        bg_t = processed_bg_masks[t_idx].detach().float().cpu()
        mask_rows.append({
            "t_index": t_idx,
            "timestep": int(smd.timesteps[t_idx].detach().cpu()),
            "sigma": float(smd.sigmas[t_idx].detach().cpu()),
            "fg_sum": float(fg_t.sum()),
            "fg_mean": float(fg_t.mean()),
            "bg_sum": float(bg_t.sum()),
            "bg_mean": float(bg_t.mean()),
            "covered_ratio_latent": float((fg_t.sum(dim=0) > 0).float().mean()),
        })
    display(pd.DataFrame(mask_rows))

    def tensor_stats(name: str, tensor: torch.Tensor) -> dict:
        t = tensor.detach().float()
        stats = {
            "name": name,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "finite": bool(torch.isfinite(t).all().item()),
            "min": float(t.min().item()),
            "max": float(t.max().item()),
            "mean": float(t.mean().item()),
            "std": float(t.std().item()),
            "abs_mean": float(t.abs().mean().item()),
        }
        print("[TRACE]", stats)
        return stats

    foreground_negative_prompts = [NEGATIVE_PROMPT for _ in trace_payload["foreground_prompts"]]

    # 1) Lấy latent output trực tiếp từ custom SemanticDraw loop, chưa qua VAE decode.
    seed_everything(trace_seed)
    trace_latent = smd(
        prompts=trace_payload["foreground_prompts"],
        negative_prompts=foreground_negative_prompts,
        masks=trace_payload["foreground_masks"].to(device=device, dtype=torch.float32),
        background_prompt=trace_payload["prompts"][0],
        background_negative_prompt=NEGATIVE_PROMPT,
        mask_stds=MASK_STD,
        mask_strengths=MASK_STRENGTH,
        height=trace_payload["height"],
        width=trace_payload["width"],
        num_inference_steps=SEMANTICDRAW_NUM_INFERENCE_STEPS,
        guidance_scale=EULER_GUIDANCE_SCALE,
        bootstrap_steps=BOOTSTRAP_STEPS,
        preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
        guidance_rescale=0.0,
        do_blend=False,
        output_type="latent",
    )
    if not torch.is_tensor(trace_latent):
        raise TypeError(
            f"[TRACE] Expected tensor latent from output_type='latent', got {type(trace_latent)!r}. "
            "The runtime patch for latent return did not apply. Restart session and Run All."
        )
    latent_stats = tensor_stats("semanticdraw_output_latent", trace_latent)

    if not latent_stats["finite"]:
        raise RuntimeError("[TRACE] Latent contains NaN/Inf before VAE decode.")

    # 2) Optional decode. On Colab T4, decoding SDXL VAE while the full pipeline is
    # still on GPU often OOMs. The default trace stops at latent stats.
    if not TRACE_DECODE_LATENT:
        del trace_latent
        torch.cuda.empty_cache()
        print("[TRACE] Skipped VAE decode to avoid Colab T4 OOM. Set TRACE_DECODE_LATENT=True only on higher VRAM.")
        print("[TRACE] Bootstrap latent trace passed for the first sample.")
    else:
        decode_latent = trace_latent.detach()
        needs_upcasting = smd.vae.dtype == torch.float16 and smd.vae.config.force_upcast
        if needs_upcasting:
            smd.upcast_vae()
            decode_latent = decode_latent.to(next(iter(smd.vae.post_quant_conv.parameters())).dtype)

        has_latents_mean = hasattr(smd.vae.config, "latents_mean") and smd.vae.config.latents_mean is not None
        has_latents_std = hasattr(smd.vae.config, "latents_std") and smd.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = torch.tensor(smd.vae.config.latents_mean).view(1, 4, 1, 1).to(decode_latent.device, decode_latent.dtype)
            latents_std = torch.tensor(smd.vae.config.latents_std).view(1, 4, 1, 1).to(decode_latent.device, decode_latent.dtype)
            decode_latent = decode_latent * latents_std / smd.vae.config.scaling_factor + latents_mean
        else:
            decode_latent = decode_latent / smd.vae.config.scaling_factor

        torch.cuda.empty_cache()
        tensor_stats("vae_input_latent_after_unscale", decode_latent)
        decoded_tensor = smd.vae.decode(decode_latent, return_dict=False)[0]
        tensor_stats("decoded_tensor_before_pil", decoded_tensor)

        trace_image_tensor = decoded_tensor[0].clip(-1, 1) * 0.5 + 0.5
        trace_image = T.ToPILImage()(trace_image_tensor.detach().cpu()).convert("RGB")
        trace_image_stats = pil_image_stats(trace_image)
        print("[TRACE] decoded image stats:", trace_image_stats)
        display(trace_image.resize((512, 512)))

        # Restore VAE dtype before any later full-generation cell; keeping SDXL VAE in float32
        # after trace is much heavier on Colab T4.
        if needs_upcasting:
            smd.vae.to(dtype=dtype)
            print("[TRACE] VAE restored to", smd.vae.dtype)

        if is_static_noise_image(trace_image):
            raise RuntimeError(
                "[TRACE] Decoded image is static noise. Latent stats above are the key evidence; "
                "bootstrap/mask denoising loop produced an invalid latent."
            )
        if is_near_black_image(trace_image):
            raise RuntimeError(
                "[TRACE] Decoded image is nearly black. Check VAE dtype/upcast and latent stats above."
            )

        del trace_latent, decode_latent, decoded_tensor, trace_image_tensor
        torch.cuda.empty_cache()
        print("[TRACE] Bootstrap trace passed for the first sample.")
else:
    print("[TRACE] Bootstrap trace skipped.")

# --- Source notebook cell 17 ---
# Chạy generation cho mọi sample trong manifest đang chọn và hiển thị kết quả.
summary = []
global_index = 0

for batch_index, batch in enumerate(loader):
    print(f"[BATCH] {batch_index + 1}/{len(loader)} - {len(batch['sample_ids'])} sample(s)")

    for local_index, sample_id in enumerate(batch["sample_ids"]):
        payload = make_semanticdraw_payload(batch, local_index)
        original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
        overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)

        seed = BASE_SEED + global_index
        seed_everything(seed)

        tic = time.perf_counter()
        # SDXL baseline ổn định hơn khi dùng đúng API SemanticDraw:
        # - foreground prompts/masks đi vào `prompts` và `masks`
        # - COCO caption đi vào `background_prompt`
        # Pipeline sẽ tự dựng background mask = 1 - union(foreground masks).
        foreground_negative_prompts = [NEGATIVE_PROMPT for _ in payload["foreground_prompts"]]
        generated = smd(
            prompts=payload["foreground_prompts"],
            negative_prompts=foreground_negative_prompts,
            masks=payload["foreground_masks"].to(device=device, dtype=torch.float32),
            background_prompt=payload["prompts"][0],
            background_negative_prompt=NEGATIVE_PROMPT,
            mask_stds=MASK_STD,
            mask_strengths=MASK_STRENGTH,
            height=payload["height"],
            width=payload["width"],
            num_inference_steps=SEMANTICDRAW_NUM_INFERENCE_STEPS,
            guidance_scale=EULER_GUIDANCE_SCALE,
            bootstrap_steps=BOOTSTRAP_STEPS,
            preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
            guidance_rescale=0.0,
            do_blend=False,
        )
        elapsed = time.perf_counter() - tic
        generated = generated.convert("RGB")
        generated_stats = pil_image_stats(generated)
        print(f"[IMAGE] index={global_index} stats={generated_stats}")
        if STOP_ON_NEAR_BLACK_OUTPUT and is_near_black_image(generated):
            raise RuntimeError(
                f"Generated image at index {global_index} is nearly black: {generated_stats}. "
                "Check the plain sanity image above. If sanity is OK, the issue is in SemanticDraw SDXL mask/latent mixing."
            )
        if STOP_ON_STATIC_NOISE_OUTPUT and is_static_noise_image(generated):
            raise RuntimeError(
                f"Generated image at index {global_index} looks like raw static noise: {generated_stats}. "
                "Plain SDXL-Lightning sanity is OK, so this run is not valid for benchmarking. "
                "The issue is in the custom SemanticDraw SDXL denoising/mask loop or bootstrap schedule."
            )

        stem = f"{global_index:04d}_{payload['sample_id']}"
        generated_path = GENERATED_IMAGES_DIR / f"{stem}_generated.png"
        overlay_path = OVERLAY_IMAGES_DIR / f"{stem}_overlay.png"
        generated.save(generated_path)
        overlay.save(overlay_path)

        summary.append({
            "index": global_index,
            "batch_index": batch_index,
            "local_index": local_index,
            "sample_id": payload["sample_id"],
            "image_id": payload["image_id"],
            "file_name": payload["file_name"],
            "seed": seed,
            "model_family": "sdxl",
            "sampler": "euler_discrete",
            "scheduler": type(smd.scheduler).__name__,
            "timestep_spacing": EULER_TIMESTEP_SPACING,
            "lightning_repo": LIGHTNING_REPO_ID,
            "lightning_weight": LIGHTNING_WEIGHT_NAME,
            "plain_num_inference_steps": EULER_NUM_INFERENCE_STEPS,
            "semanticdraw_num_inference_steps_arg": SEMANTICDRAW_NUM_INFERENCE_STEPS,
            "semanticdraw_t_index_list": SEMANTICDRAW_T_INDEX_LIST,
            "guidance_scale": EULER_GUIDANCE_SCALE,
            "colab_gpu_mode": COLAB_GPU_MODE,
            "low_vram_colab_mode": LOW_VRAM_COLAB_MODE,
            "bootstrap_steps": BOOTSTRAP_STEPS,
            "num_regions_including_background": len(payload["foreground_prompts"]) + 1,
            "elapsed_sec": elapsed,
            "generated_path": str(generated_path),
            "overlay_path": str(overlay_path),
        })

        should_display = MAX_DISPLAY_RESULTS is None or global_index < MAX_DISPLAY_RESULTS
        if should_display:
            display_smoke_result(payload, original, overlay, generated, elapsed, generated_path)

        global_index += 1
        torch.cuda.empty_cache()

summary_path = RUN_SUMMARY_PATH
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

display(Markdown(
    f"## Done\n"
    f"Generated `{len(summary)}` image(s) from `{dataset_size}` manifest record(s). "
    f"Summary saved to `{summary_path}`."
))
summary[:5]

# --- Source notebook cell 20 ---
# Giải phóng VRAM trước khi load Inception/CLIP cho metric.
import gc

for var_name in ("smd", "generated", "payload", "overlay", "original", "batch", "loader", "preview_batch"):
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

# --- Source notebook cell 21 ---
# Đo FID, IS, CLIP(fg), CLIP(bg), Time(s) nếu RUN_METRICS_AFTER_GENERATION=True.
if not RUN_METRICS_AFTER_GENERATION:
    display(Markdown(
        "## Metric Skipped\n"
        "Smoke `bs2` chỉ dùng để validate dataloader + SemanticDraw generation. "
        "Đổi `RUN_PROFILE = \"full1073\"` để đo metric có ý nghĩa."
    ))
    metric_report = None
else:
    from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report
    import pandas as pd
    import math

    generation_summary_path = RUN_SUMMARY_PATH
    assert generation_summary_path.exists(), f"Missing generation summary: {generation_summary_path}"

    metric_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    metric_config = MetricEvaluationConfig(
        manifest_path=RUN_MANIFEST,
        coco_root=COCO_ROOT,
        generated_dir=GENERATED_IMAGES_DIR,
        generation_summary=generation_summary_path,
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
    metrics_json, metrics_csv = write_metrics_report(
        metric_report,
        METRICS_OUTPUT_DIR,
        prefix=METRICS_REPORT_PREFIX,
    )

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
