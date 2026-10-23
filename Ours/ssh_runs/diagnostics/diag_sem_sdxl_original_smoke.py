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

# Notebook: Ours/kaggle_semanticdraw_smoke/semanticdraw_sdxl_euler_original_baseline_smoke_bs2_kaggle.ipynb
# Notebook SHA256: 9c3b0ef3ab96f510b2fb78f133149c9461eb923451f60d9775cd97ec0a6f286b

# --- Source notebook cell 3 ---
# Cấu hình smoke test.
from pathlib import Path

COCO_ROOT = Path(str(WORK_DIR / 'datasets/coco'))
EXPECTED_MANIFEST_NAME = "coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_smoke_bs2.jsonl"
RUN_MANIFEST = REPO_ROOT / "Ours" / "test_sets" / "manifests" / "smoke" / EXPECTED_MANIFEST_NAME

TARGET_SIZE = (1024, 1024)
BATCH_SIZE = 2
EXPECTED_DATASET_SIZE = 2
BASE_SEED = 2024
MAX_DISPLAY_RESULTS = 0

# Theo demo_simple_sdxl.ipynb của baseline, SDXL smoke dùng bootstrap_steps=2 và guidance_scale=0.
BOOTSTRAP_STEPS = 2
GUIDANCE_SCALE = 0.0

# Baseline SDXL source dùng schedule mặc định này khi không truyền t_index_list.
# Source cũng ghi chú schedule [0, 5, 16, 18, 20, 37] là khuyến nghị khi dùng 2-step bootstrap.
BASELINE_DEFAULT_T_INDEX_LIST = [0, 4, 12, 25, 37]
BASELINE_RECOMMENDED_T_INDEX_LIST_FOR_BOOTSTRAP2 = [0, 5, 16, 18, 20, 37]
BASELINE_SCHEDULE_NUM_INFERENCE_STEPS = 50

# Theo demo_simple_sdxl.ipynb và demo/canvas_sdxl/app.py của baseline, SDXL dùng mask_stds=0.0.
MASK_STD = 0.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.3
MASK_TYPE = "discrete"
NEGATIVE_PROMPT = ""

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
ACCEL_REPO = "ByteDance/SDXL-Lightning"
ACCEL_WEIGHT = "sdxl_lightning_4step_unet.safetensors"

# Compatibility flags: chỉ sửa cách runtime load/decode, không sửa denoising/mask/latent mixing.
KAGGLE_COMPAT_LOAD_SAFETENSORS_ON_CPU = True
KAGGLE_COMPAT_FULL_VAE_UPCAST = True
# Bootstrap dùng latent của ảnh trắng. Trên Kaggle, SDXL VAE fp16 encode ảnh trắng 1024x1024
# có thể sinh NaN/ảnh đen; flag này chỉ encode latent nền trắng bằng fp32 rồi cast về dtype model.
KAGGLE_COMPAT_SAFE_WHITE_BOOTSTRAP_LATENT = True
KAGGLE_COMPAT_VAE_TILING = True
KAGGLE_COMPAT_ATTENTION_SLICING = True

OUTPUT_DIR = Path(str(WORK_DIR / 'semanticdraw_sdxl_euler_original_baseline_smoke_bs2_outputs'))
GENERATED_DIR = OUTPUT_DIR / "generated_images"
OVERLAY_DIR = OUTPUT_DIR / "mask_overlays"
MASK_CACHE_DIR = Path(str(WORK_DIR / 'semanticdraw_mask_cache_sdxl_smoke_bs2'))

for path in [OUTPUT_DIR, GENERATED_DIR, OVERLAY_DIR, MASK_CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

assert RUN_MANIFEST.exists(), f"Missing manifest: {RUN_MANIFEST}"
assert RUN_MANIFEST.name == EXPECTED_MANIFEST_NAME, f"Manifest name mismatch: {RUN_MANIFEST.name}"
assert TARGET_SIZE == (1024, 1024), f"SDXL smoke must use 1024x1024, got {TARGET_SIZE}"
assert BATCH_SIZE == 2, f"Smoke bs2 notebook expects BATCH_SIZE=2, got {BATCH_SIZE}"
assert BOOTSTRAP_STEPS == 2, f"Original SDXL demo-style smoke expects bootstrap_steps=2, got {BOOTSTRAP_STEPS}"
assert MASK_STD == 0.0, f"Original SDXL demo-style smoke expects mask_stds=0.0, got {MASK_STD}"
assert BASELINE_DEFAULT_T_INDEX_LIST == [0, 4, 12, 25, 37]
assert BASELINE_RECOMMENDED_T_INDEX_LIST_FOR_BOOTSTRAP2 == [0, 5, 16, 18, 20, 37]

print("[OK] Model:", MODEL_ID)
print("[OK] Acceleration:", f"{ACCEL_REPO}/{ACCEL_WEIGHT}")
print("[OK] Sampler: EulerDiscreteScheduler(timestep_spacing='trailing')")
print("[OK] Manifest:", RUN_MANIFEST)
print("[OK] Output:", OUTPUT_DIR)
print("[OK] Bootstrap steps:", BOOTSTRAP_STEPS)
print("[OK] Mask std/strength:", MASK_STD, MASK_STRENGTH)
print("[OK] Default t-index list:", BASELINE_DEFAULT_T_INDEX_LIST)
print("[OK] Recommended t-index list for bootstrap=2:", BASELINE_RECOMMENDED_T_INDEX_LIST_FOR_BOOTSTRAP2)
print("[OK] Kaggle compatibility flags:", {
    "load_safetensors_on_cpu": KAGGLE_COMPAT_LOAD_SAFETENSORS_ON_CPU,
    "full_vae_upcast": KAGGLE_COMPAT_FULL_VAE_UPCAST,
    "safe_white_bootstrap_latent": KAGGLE_COMPAT_SAFE_WHITE_BOOTSTRAP_LATENT,
    "vae_tiling": KAGGLE_COMPAT_VAE_TILING,
    "attention_slicing": KAGGLE_COMPAT_ATTENTION_SLICING,
})
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 5 ---
# Import dataloader của Ours và pipeline SDXL baseline nguyên bản.
import hashlib
import importlib
import importlib.util
import json
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
import diffusers
import transformers
import accelerate
import safetensors
# terminal display provided above
from PIL import Image

OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_SRC = REPO_ROOT / "Baseline" / "semantic-draw-main" / "src"
BASELINE_PIPELINE_FILE = BASELINE_SRC / "model" / "pipeline_semantic_draw_sdxl.py"

ours_src_str = str(OURS_SRC)
baseline_src_str = str(BASELINE_SRC)

# Path hygiene:
# Baseline có file `src/data.py`, còn Ours có package `Ours/src/data/`.
# Nếu BASELINE_SRC đứng trước OURS_SRC, `import data` sẽ trỏ nhầm sang baseline.
for path_str in [ours_src_str, baseline_src_str]:
    while path_str in sys.path:
        sys.path.remove(path_str)

wrong_data_module = sys.modules.get("data")
if wrong_data_module is not None:
    wrong_data_file = str(getattr(wrong_data_module, "__file__", ""))
    if ours_src_str not in wrong_data_file:
        print("[FIX] Removing previously imported wrong data module:", wrong_data_file)
        for module_name in list(sys.modules):
            if module_name == "data" or module_name.startswith("data."):
                del sys.modules[module_name]

importlib.invalidate_caches()

# Import Ours dataloader trước, bắt buộc `data` phải resolve vào Ours/src/data.
sys.path.insert(0, ours_src_str)
from data import COCORegionConfig, batch_item_to_semanticdraw_inputs, build_coco_region_dataloader
from data.visualize import make_mask_overlay
import data as ours_data_module

data_module_file = str(getattr(ours_data_module, "__file__", ""))
assert ours_src_str in data_module_file, f"`data` imported from wrong location: {data_module_file}"
print("[OK] Ours data module:", data_module_file)

# Sau khi đã import dataloader Ours, thêm baseline src để pipeline gốc import được `util`.
sys.path.insert(1, baseline_src_str)

importlib.invalidate_caches()
if "torchao" in sys.modules:
    raise RuntimeError(
        "torchao đã được import trong session này. Restart Kaggle session rồi Run All. "
        "Notebook này không cần torchao."
    )
if importlib.util.find_spec("torchao") is not None:
    raise RuntimeError(
        "torchao vẫn còn importable trong runtime. Chạy dependency cell, restart Kaggle session, rồi Run All."
    )

# Import file baseline nguyên văn. Không đọc source rồi replace.
spec = importlib.util.spec_from_file_location("pipeline_semantic_draw_sdxl_original", BASELINE_PIPELINE_FILE)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawSDXLPipeline = pipeline_module.SemanticDrawSDXLPipeline

pipeline_sha256 = hashlib.sha256(BASELINE_PIPELINE_FILE.read_bytes()).hexdigest()
print("[OK] Imported original baseline file:", BASELINE_PIPELINE_FILE)
print("[OK] Baseline pipeline SHA256:", pipeline_sha256)
print("[OK] No runtime source patch was applied to denoising/mask/latent-mixing logic.")
print("[OK] Runtime versions:", {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "diffusers": diffusers.__version__,
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "safetensors": safetensors.__version__,
})

# --- Source notebook cell 6 ---
# Tạo dataloader cho smoke manifest.
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
preview_batch = next(iter(loader))

assert config.model_family == "sdxl", f"model_family mismatch: {config.model_family}"
assert tuple(config.target_hw) == TARGET_SIZE, f"target_hw mismatch: {config.target_hw}"
assert dataset_size == EXPECTED_DATASET_SIZE, f"Smoke manifest should contain {EXPECTED_DATASET_SIZE} records, got {dataset_size}"
assert len(preview_batch["sample_ids"]) == BATCH_SIZE, f"First smoke batch should have {BATCH_SIZE} samples"
assert tuple(preview_batch["masks"].shape[-2:]) == TARGET_SIZE, f"Mask spatial size mismatch: {preview_batch['masks'].shape}"
assert int(preview_batch["masks"].shape[2]) == 1, f"Mask channel mismatch, expected C=1: {preview_batch['masks'].shape}"

for size in preview_batch["target_sizes"]:
    assert tuple(int(v) for v in size.tolist()) == TARGET_SIZE, f"Batch target size mismatch: {size}"

print(f"[OK] Manifest records: {dataset_size}")
print(f"[OK] Dataloader batches: {len(loader)} batch(es) x up to {BATCH_SIZE} sample(s)")
print(f"[OK] First batch size: {len(preview_batch['sample_ids'])}")
print(f"[OK] First batch masks shape: {tuple(preview_batch['masks'].shape)}  # (B, Pmax, C, H, W)")
for sample_id in preview_batch["sample_ids"]:
    print(" -", sample_id)

# --- Source notebook cell 7 ---
# Helper functions.
def md_escape(text: object) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_semanticdraw_payload(batch: dict, index: int) -> dict:
    item = batch_item_to_semanticdraw_inputs(batch, index)
    fg_masks = item["masks"].float().cpu()  # (P, 1, H, W)
    fg_union = fg_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    background_mask = (1.0 - fg_union).clamp(0, 1)
    all_masks = torch.cat([background_mask, fg_masks], dim=0)

    prompts = [item["background_prompt"], *item["prompts"]]
    negative_prompts = [NEGATIVE_PROMPT for _ in prompts]
    mask_stds = [MASK_STD for _ in prompts]
    mask_strengths = [MASK_STRENGTH for _ in prompts]
    metadata = item["metadata"]

    payload = {
        "sample_id": metadata["sample_id"],
        "image_id": metadata["image_id"],
        "file_name": metadata["file_name"],
        "height": item["height"],
        "width": item["width"],
        "prompts": prompts,
        "negative_prompts": negative_prompts,
        "mask_stds": mask_stds,
        "mask_strengths": mask_strengths,
        "foreground_prompts": item["prompts"],
        "category_names": metadata["category_names"],
        "annotation_ids": metadata["annotation_ids"],
        "area_ratios": metadata["area_ratios"],
        "foreground_masks": fg_masks,
        "all_masks": all_masks,
        "metadata": metadata,
    }
    validate_semanticdraw_payload(payload)
    return payload


def validate_semanticdraw_payload(payload: dict) -> None:
    masks = payload["all_masks"]
    fg_masks = payload["foreground_masks"]
    prompts = payload["prompts"]
    negative_prompts = payload["negative_prompts"]
    mask_stds = payload["mask_stds"]
    mask_strengths = payload["mask_strengths"]
    height = int(payload["height"])
    width = int(payload["width"])

    assert (height, width) == TARGET_SIZE, f"Payload resolution mismatch: {(height, width)} vs {TARGET_SIZE}"
    assert masks.ndim == 4, f"SemanticDraw masks must be (P, C, H, W), got {tuple(masks.shape)}"
    assert masks.shape[1] == 1, f"SemanticDraw masks must have C=1, got {tuple(masks.shape)}"
    assert tuple(masks.shape[-2:]) == TARGET_SIZE, f"Mask size mismatch: {tuple(masks.shape)}"
    assert masks.shape[0] == len(prompts), f"Prompt/mask mismatch: {len(prompts)} prompts vs {masks.shape[0]} masks"
    assert len(negative_prompts) == len(prompts), f"Negative prompt mismatch: {len(negative_prompts)} vs {len(prompts)}"
    assert len(mask_stds) == len(prompts), f"mask_stds mismatch: {len(mask_stds)} vs {len(prompts)}"
    assert len(mask_strengths) == len(prompts), f"mask_strengths mismatch: {len(mask_strengths)} vs {len(prompts)}"
    assert torch.isfinite(masks).all().item(), "Mask contains NaN/Inf"
    assert float(masks.min()) >= 0.0 and float(masks.max()) <= 1.0, "Mask values must be in [0, 1]"

    fg_union = fg_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    expected_bg = (1.0 - fg_union).clamp(0, 1)
    bg_diff = (masks[0:1] - expected_bg).abs().max().item()
    assert bg_diff <= 1e-6, f"Background mask mismatch, max diff={bg_diff}"


def display_smoke_result(payload: dict, original: Image.Image, overlay: Image.Image, generated: Image.Image, elapsed: float, generated_path: Path) -> None:
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


print("[OK] Helper functions and mismatch guards are ready.")

# --- Source notebook cell 8 ---
# Login Hugging Face nếu có token trong Kaggle Secret hoặc biến môi trường.
def maybe_login_to_huggingface() -> None:
    token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            token = None
    if token:
        from huggingface_hub import login
        login(token=token)
        print("[OK] Hugging Face token loaded.")
    else:
        print("[INFO] No HF_TOKEN found. Public model access will be used.")


assert torch.cuda.is_available(), "Kaggle runtime chưa bật GPU. Hãy bật Accelerator = GPU rồi chạy lại."
device = "cuda"
dtype = torch.float16

maybe_login_to_huggingface()
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

# --- Source notebook cell 9 ---
# Load SemanticDraw SDXL pipeline từ core baseline nguyên bản.
# Các compatibility patch bên dưới chỉ xử lý Kaggle runtime, không sửa core denoising/mask/latent mixing.

if KAGGLE_COMPAT_LOAD_SAFETENSORS_ON_CPU:
    from safetensors.torch import load_file as _safetensors_load_file

    def _load_safetensors_on_cpu(filename, device=None):
        if device is not None and str(device).startswith("cuda"):
            print(f"[COMPAT] Loading safetensors checkpoint on CPU first instead of {device}.")
        return _safetensors_load_file(filename, device="cpu")

    # Baseline module imported `load_file` as a module-level symbol.
    # Rebinding only changes checkpoint I/O, not generation logic.
    pipeline_module.load_file = _load_safetensors_on_cpu

seed_everything(BASE_SEED)
smd = SemanticDrawSDXLPipeline(
    device=device,
    dtype=dtype,
    hf_key=None,
    has_i2t=False,
    default_mask_std=MASK_STD,
    default_mask_strength=MASK_STRENGTH,
    default_preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
    mask_type=MASK_TYPE,
)

if KAGGLE_COMPAT_ATTENTION_SLICING and hasattr(smd.pipe, "enable_attention_slicing"):
    smd.pipe.enable_attention_slicing()
if hasattr(smd.pipe, "enable_vae_slicing"):
    smd.pipe.enable_vae_slicing()
if KAGGLE_COMPAT_VAE_TILING and hasattr(smd.pipe, "enable_vae_tiling"):
    smd.pipe.enable_vae_tiling()
    print("[COMPAT] VAE tiling enabled for lower SDXL decode VRAM.")

if KAGGLE_COMPAT_FULL_VAE_UPCAST:
    def _upcast_full_vae_for_kaggle():
        smd.vae.to(dtype=torch.float32)

    # Baseline already calls `self.upcast_vae()` before SDXL VAE decode when needed.
    # This override only makes that dtype conversion complete for Kaggle/diffusers combinations
    # that otherwise throw Half/Float mismatch. It does not touch denoising or mask logic.
    smd.upcast_vae = _upcast_full_vae_for_kaggle
    print("[COMPAT] SemanticDraw VAE decode hook will upcast full VAE when needed.")



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


if KAGGLE_COMPAT_SAFE_WHITE_BOOTSTRAP_LATENT:
    try:
        original_white_probe = smd.get_white_background(*TARGET_SIZE)
        original_white_stats = _latent_stats_for_bootstrap("original_fp16_white_bootstrap_latent_probe", original_white_probe)
        print("[CHECK] Original bootstrap white latent probe:", original_white_stats)
    except Exception as exc:
        original_white_stats = {"name": "original_fp16_white_bootstrap_latent_probe", "error": repr(exc)}
        print("[CHECK] Original bootstrap white latent probe failed:", repr(exc))

    def _safe_get_white_background(height: int, width: int) -> torch.Tensor:
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

scheduler_name = type(smd.scheduler).__name__
timestep_spacing = getattr(smd.scheduler.config, "timestep_spacing", None)
timesteps = [int(t) for t in smd.timesteps.detach().cpu().tolist()]

assert scheduler_name == "EulerDiscreteScheduler", f"Scheduler mismatch: {scheduler_name}"
assert timestep_spacing == "trailing", f"Timestep spacing mismatch: {timestep_spacing}"
assert smd.default_num_inference_steps == 4, f"SDXL-Lightning default steps mismatch: {smd.default_num_inference_steps}"
assert smd.default_bootstrap_steps == 1, f"Baseline default bootstrap mismatch: {smd.default_bootstrap_steps}"
assert len(timesteps) == 5, f"Baseline default t_index_list should produce 5 timesteps, got {timesteps}"

print("[OK] SemanticDrawSDXLPipeline is ready.")
print("[OK] Scheduler:", scheduler_name)
print("[OK] Timestep spacing:", timestep_spacing)
print("[OK] Baseline default bootstrap steps:", smd.default_bootstrap_steps)
print("[OK] Timesteps:", timesteps)

# --- Source notebook cell 10 ---
# Diagnostic cell: audit config và tìm nguyên nhân ảnh đen trước khi chạy smoke generation.
#
# Cell này KHÔNG sửa core baseline:
# - không source.replace
# - không sửa latent mixing
# - không sửa denoising loop
# - chỉ reset schedule giữa các case để tránh side effect của `smd.pipe(...)`
# - chỉ tạm wrap `smd.vae.decode` để log input/output stats, rồi restore lại sau mỗi case.

RUN_BLACK_OUTPUT_DIAGNOSTIC = True

def reset_semanticdraw_schedule(t_index_list=None):
    if t_index_list is None:
        t_index_list = BASELINE_DEFAULT_T_INDEX_LIST
    smd.prepare_lightning_schedule(
        list(t_index_list),
        BASELINE_SCHEDULE_NUM_INFERENCE_STEPS,
    )
    return {
        "t_index_list": list(t_index_list),
        "timesteps": [int(t) for t in smd.timesteps.detach().cpu().tolist()],
        "sigmas": [float(x) for x in smd.sigmas.detach().cpu().tolist()],
        "dt": [float(x) for x in smd.dt.detach().cpu().tolist()],
    }


def audit_current_config():
    current_schedule = reset_semanticdraw_schedule(BASELINE_DEFAULT_T_INDEX_LIST)
    rows = [
        {
            "field": "device_dtype",
            "current": f"device={device}, dtype={dtype}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
            "baseline/reference": "CUDA + float16 khi chạy demo trên GPU",
            "status": "OK if GPU is enabled",
        },
        {
            "field": "runtime_versions",
            "current": f"torch={torch.__version__}, diffusers={diffusers.__version__}, transformers={transformers.__version__}",
            "baseline/reference": "Notebook pin diffusers==0.30.3 để tương thích SDXL API hiện tại",
            "status": "CHECK if Kaggle preinstalled packages override pins",
        },
        {
            "field": "pipeline_file",
            "current": str(BASELINE_PIPELINE_FILE),
            "baseline/reference": "src/model/pipeline_semantic_draw_sdxl.py",
            "status": "OK",
        },
        {
            "field": "checkpoint",
            "current": MODEL_ID,
            "baseline/reference": "hf_key=None path của pipeline SDXL dùng SDXL base + Lightning UNet",
            "status": "OK for SDXL-base paper-style; khác demo_simple_sdxl dùng Animagine XL",
        },
        {
            "field": "acceleration",
            "current": f"{ACCEL_REPO}/{ACCEL_WEIGHT}",
            "baseline/reference": "ByteDance/SDXL-Lightning/sdxl_lightning_4step_unet.safetensors",
            "status": "OK",
        },
        {
            "field": "scheduler",
            "current": f"{type(smd.scheduler).__name__}(timestep_spacing={getattr(smd.scheduler.config, 'timestep_spacing', None)!r})",
            "baseline/reference": "EulerDiscreteScheduler(timestep_spacing='trailing')",
            "status": "OK",
        },
        {
            "field": "default_t_index_list",
            "current": BASELINE_DEFAULT_T_INDEX_LIST,
            "baseline/reference": "[0, 4, 12, 25, 37], source default; source ghi là hợp với bootstrap=1",
            "status": "CHECK because BOOTSTRAP_STEPS=2",
        },
        {
            "field": "recommended_t_index_for_bootstrap2",
            "current": BASELINE_RECOMMENDED_T_INDEX_LIST_FOR_BOOTSTRAP2,
            "baseline/reference": "[0, 5, 16, 18, 20, 37], source comment khuyến nghị cho 2-step bootstrap",
            "status": "TEST in diagnostic, chưa đổi main generation",
        },
        {
            "field": "bootstrap_steps",
            "current": BOOTSTRAP_STEPS,
            "baseline/reference": "demo_simple_sdxl.ipynb và demo/canvas_sdxl/app.py dùng 2; pipeline default là 1",
            "status": "CHECK",
        },
        {
            "field": "guidance_scale",
            "current": GUIDANCE_SCALE,
            "baseline/reference": "0 trong SDXL-Lightning demo",
            "status": "OK",
        },
        {
            "field": "mask_stds",
            "current": MASK_STD,
            "baseline/reference": "0.0 trong demo_simple_sdxl.ipynb và demo/canvas_sdxl/app.py",
            "status": "OK",
        },
        {
            "field": "mask_strengths",
            "current": MASK_STRENGTH,
            "baseline/reference": "1.0 trong demo_simple_sdxl.ipynb và demo/canvas_sdxl/app.py",
            "status": "OK",
        },
        {
            "field": "preprocess_mask_cover_alpha",
            "current": PREPROCESS_MASK_COVER_ALPHA,
            "baseline/reference": "0.3 default trong pipeline SDXL",
            "status": "OK",
        },
        {
            "field": "white_bootstrap_latent",
            "current": getattr(smd, "_safe_white_bootstrap_latent_stats", "not checked"),
            "baseline/reference": "Bootstrap dùng latent ảnh trắng; compatibility fix encode bằng fp32 nếu fp16 tạo NaN",
            "status": "CHECK",
        },
        {
            "field": "mask_type",
            "current": MASK_TYPE,
            "baseline/reference": "discrete default trong pipeline SDXL",
            "status": "OK",
        },
        {
            "field": "negative_prompt",
            "current": NEGATIVE_PROMPT,
            "baseline/reference": "demo có quality/style negative prefix; paper COCO không ghi rõ phần style demo",
            "status": "DIFF but should not create NaN",
        },
        {
            "field": "current_timesteps_after_reset",
            "current": current_schedule["timesteps"],
            "baseline/reference": "computed from default t-index list over 50 scheduler steps",
            "status": "OK",
        },
        {
            "field": "current_sigmas_after_reset",
            "current": [round(v, 6) for v in current_schedule["sigmas"]],
            "baseline/reference": "Euler sigma schedule used by custom SDXL loop",
            "status": "OK",
        },
    ]
    display(pd.DataFrame(rows))


def tensor_debug_stats(name: str, tensor: torch.Tensor) -> dict:
    t = tensor.detach().float()
    finite = bool(torch.isfinite(t).all().item())
    stats = {
        "name": name,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "finite": finite,
        "min": float(t.min().item()) if finite and t.numel() else float("nan"),
        "max": float(t.max().item()) if finite and t.numel() else float("nan"),
        "mean": float(t.mean().item()) if finite and t.numel() else float("nan"),
        "std": float(t.std().item()) if finite and t.numel() > 1 else (0.0 if finite else float("nan")),
        "abs_mean": float(t.abs().mean().item()) if finite and t.numel() else float("nan"),
    }
    print("[TENSOR]", stats)
    return stats


def pil_debug_stats(image: Image.Image) -> dict:
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    stats = {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
    if stats["max"] <= 5 and stats["std"] <= 2:
        stats["class"] = "near_black"
    elif stats["min"] == 0 and stats["max"] == 255 and 90 <= stats["mean"] <= 160 and stats["std"] >= 80:
        stats["class"] = "static_noise"
    else:
        stats["class"] = "non_black"
    return stats


class CaptureVAEDecode:
    def __init__(self, pipe_obj, label: str):
        self.pipe_obj = pipe_obj
        self.label = label
        self.original_decode = None
        self.records = []

    def __enter__(self):
        self.original_decode = self.pipe_obj.vae.decode

        def wrapped_decode(z, *args, **kwargs):
            input_stats = tensor_debug_stats(f"{self.label}/vae_input", z)
            out = self.original_decode(z, *args, **kwargs)
            if isinstance(out, tuple):
                sample = out[0]
            elif hasattr(out, "sample"):
                sample = out.sample
            else:
                sample = out
            output_stats = tensor_debug_stats(f"{self.label}/vae_output", sample)
            self.records.append({"input": input_stats, "output": output_stats})
            return out

        self.pipe_obj.vae.decode = wrapped_decode
        return self

    def __exit__(self, exc_type, exc, tb):
        self.pipe_obj.vae.decode = self.original_decode
        return False


def run_diag_case(name: str, fn, t_index_list=None):
    print(f"\n[DIAG] Running: {name}")
    torch.cuda.empty_cache()
    schedule = reset_semanticdraw_schedule(t_index_list)
    seed_everything(BASE_SEED)
    result = {
        "case": name,
        "status": "unknown",
        "image_class": None,
        "image_stats": None,
        "vae_decode_records": [],
        "schedule": schedule,
        "error": None,
    }
    try:
        with CaptureVAEDecode(smd, name) as capture:
            image = fn()
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")
        stats = pil_debug_stats(image)
        result.update({
            "status": "ok",
            "image_class": stats["class"],
            "image_stats": stats,
            "vae_decode_records": capture.records,
        })
        print("[IMAGE]", name, stats)
        display(image.resize((384, 384)))
    except Exception as exc:
        result.update({
            "status": "error",
            "error": repr(exc),
        })
        print("[ERROR]", name, repr(exc))
    finally:
        reset_semanticdraw_schedule(BASELINE_DEFAULT_T_INDEX_LIST)
        torch.cuda.empty_cache()
    return result


def make_demo_like_masks(height: int, width: int) -> torch.Tensor:
    m1 = torch.zeros(1, 1, height, width, dtype=torch.float32)
    m2 = torch.zeros(1, 1, height, width, dtype=torch.float32)
    m1[..., height // 5: height * 4 // 5, width // 12: width * 5 // 12] = 1.0
    m2[..., height // 3: height * 5 // 6, width * 7 // 12: width * 11 // 12] = 1.0
    bg = (1.0 - torch.cat([m1, m2], dim=0).sum(dim=0, keepdim=True).clamp(0, 1)).clamp(0, 1)
    return torch.cat([bg, m1, m2], dim=0)


if RUN_BLACK_OUTPUT_DIAGNOSTIC:
    audit_current_config()

    diag_payload = make_semanticdraw_payload(preview_batch, 0)
    h, w = int(diag_payload["height"]), int(diag_payload["width"])
    full_mask = torch.ones(1, 1, h, w, dtype=torch.float32)
    demo_masks = make_demo_like_masks(h, w)
    foreground_negative_prompts = [NEGATIVE_PROMPT for _ in diag_payload["foreground_prompts"]]

    print("[DIAG] sample_id:", diag_payload["sample_id"])
    print("[DIAG] prompt_count/all_mask_count:", len(diag_payload["prompts"]), tuple(diag_payload["all_masks"].shape))
    print("[DIAG] foreground mask count:", tuple(diag_payload["foreground_masks"].shape))
    print("[DIAG] background mask policy: first region = 1 - union(foreground masks), same as demo_simple_sdxl style")

    diagnostic_results = []
    diagnostic_results.append(run_diag_case(
        "plain_pipe_resets_schedule_after",
        lambda: smd.pipe(
            "a studio photo of a teddy bear on a clean table",
            num_inference_steps=smd.default_num_inference_steps,
            guidance_scale=GUIDANCE_SCALE,
            height=h,
            width=w,
        ).images[0].convert("RGB"),
    ))
    diagnostic_results.append(run_diag_case(
        "single_full_mask_boot0_default_schedule",
        lambda: smd(
            ["a studio photo of a teddy bear on a clean table"],
            [NEGATIVE_PROMPT],
            masks=full_mask,
            mask_stds=[MASK_STD],
            mask_strengths=[MASK_STRENGTH],
            height=h,
            width=w,
            bootstrap_steps=0,
            guidance_scale=GUIDANCE_SCALE,
        ).convert("RGB"),
    ))
    diagnostic_results.append(run_diag_case(
        "coco_all_masks_boot0_default_schedule",
        lambda: smd(
            diag_payload["prompts"],
            diag_payload["negative_prompts"],
            masks=diag_payload["all_masks"].float(),
            mask_stds=diag_payload["mask_stds"],
            mask_strengths=diag_payload["mask_strengths"],
            height=h,
            width=w,
            bootstrap_steps=0,
            guidance_scale=GUIDANCE_SCALE,
        ).convert("RGB"),
    ))
    diagnostic_results.append(run_diag_case(
        "demo_like_masks_boot2_default_schedule",
        lambda: smd(
            [
                "purple sky, planets, planets, planets, stars, stars, stars",
                "a photo of the dolomites, masterpiece, absurd quality, background, no humans",
                "1girl, looking at viewer, pretty face, blue hair, fantasy style, witch, magi, robe",
            ],
            [
                "worst quality, bad quality, normal quality, cropped, framed, 1girl, 1boy, humans",
                "worst quality, bad quality, normal quality, cropped, framed, 1girl, 1boy, humans",
                "worst quality, bad quality, normal quality, cropped, framed",
            ],
            masks=demo_masks.float(),
            mask_stds=0.0,
            mask_strengths=1.0,
            height=h,
            width=w,
            bootstrap_steps=BOOTSTRAP_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).convert("RGB"),
    ))
    diagnostic_results.append(run_diag_case(
        "single_full_mask_boot1_default_schedule",
        lambda: smd(
            ["a studio photo of a teddy bear on a clean table"],
            [NEGATIVE_PROMPT],
            masks=full_mask,
            mask_stds=[MASK_STD],
            mask_strengths=[MASK_STRENGTH],
            height=h,
            width=w,
            bootstrap_steps=1,
            guidance_scale=GUIDANCE_SCALE,
        ).convert("RGB"),
    ))
    diagnostic_results.append(run_diag_case(
        "single_full_mask_boot2_default_schedule",
        lambda: smd(
            ["a studio photo of a teddy bear on a clean table"],
            [NEGATIVE_PROMPT],
            masks=full_mask,
            mask_stds=[MASK_STD],
            mask_strengths=[MASK_STRENGTH],
            height=h,
            width=w,
            bootstrap_steps=BOOTSTRAP_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).convert("RGB"),
    ))
    diagnostic_results.append(run_diag_case(
        "single_full_mask_boot2_recommended_2step_schedule",
        lambda: smd(
            ["a studio photo of a teddy bear on a clean table"],
            [NEGATIVE_PROMPT],
            masks=full_mask,
            mask_stds=[MASK_STD],
            mask_strengths=[MASK_STRENGTH],
            height=h,
            width=w,
            bootstrap_steps=BOOTSTRAP_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).convert("RGB"),
        t_index_list=BASELINE_RECOMMENDED_T_INDEX_LIST_FOR_BOOTSTRAP2,
    ))
    diagnostic_results.append(run_diag_case(
        "coco_all_masks_boot1_default_schedule",
        lambda: smd(
            diag_payload["prompts"],
            diag_payload["negative_prompts"],
            masks=diag_payload["all_masks"].float(),
            mask_stds=diag_payload["mask_stds"],
            mask_strengths=diag_payload["mask_strengths"],
            height=h,
            width=w,
            bootstrap_steps=1,
            guidance_scale=GUIDANCE_SCALE,
        ).convert("RGB"),
    ))
    diagnostic_results.append(run_diag_case(
        "coco_all_masks_boot2_default_schedule",
        lambda: smd(
            diag_payload["prompts"],
            diag_payload["negative_prompts"],
            masks=diag_payload["all_masks"].float(),
            mask_stds=diag_payload["mask_stds"],
            mask_strengths=diag_payload["mask_strengths"],
            height=h,
            width=w,
            bootstrap_steps=BOOTSTRAP_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).convert("RGB"),
    ))
    diagnostic_results.append(run_diag_case(
        "coco_all_masks_boot2_recommended_2step_schedule",
        lambda: smd(
            diag_payload["prompts"],
            diag_payload["negative_prompts"],
            masks=diag_payload["all_masks"].float(),
            mask_stds=diag_payload["mask_stds"],
            mask_strengths=diag_payload["mask_strengths"],
            height=h,
            width=w,
            bootstrap_steps=BOOTSTRAP_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).convert("RGB"),
        t_index_list=BASELINE_RECOMMENDED_T_INDEX_LIST_FOR_BOOTSTRAP2,
    ))
    diagnostic_results.append(run_diag_case(
        "coco_foreground_masks_background_prompt_branch_expected_shape_check",
        lambda: smd(
            diag_payload["foreground_prompts"],
            foreground_negative_prompts,
            masks=diag_payload["foreground_masks"].float(),
            mask_stds=[MASK_STD for _ in diag_payload["foreground_prompts"]],
            mask_strengths=[MASK_STRENGTH for _ in diag_payload["foreground_prompts"]],
            background_prompt=diag_payload["prompts"][0],
            background_negative_prompt=NEGATIVE_PROMPT,
            height=h,
            width=w,
            bootstrap_steps=BOOTSTRAP_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        ).convert("RGB"),
    ))

    diagnostic_table = []
    for row in diagnostic_results:
        stats = row.get("image_stats") or {}
        schedule = row.get("schedule") or {}
        diagnostic_table.append({
            "case": row["case"],
            "status": row["status"],
            "class": row.get("image_class"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "decode_calls": len(row.get("vae_decode_records", [])),
            "t_index_list": schedule.get("t_index_list"),
            "error": row.get("error"),
        })
    diagnostic_df = pd.DataFrame(diagnostic_table)
    display(diagnostic_df)

    def ok_non_black(case_name: str) -> bool:
        row = next((r for r in diagnostic_results if r["case"] == case_name), None)
        return bool(row and row["status"] == "ok" and row["image_class"] == "non_black")

    conclusion = []
    if ok_non_black("plain_pipe_resets_schedule_after"):
        conclusion.append("Plain SDXL-Lightning OK: checkpoint/scheduler/VAE cơ bản không hỏng.")
    else:
        conclusion.append("Plain SDXL-Lightning lỗi/đen: kiểm checkpoint, diffusers, VAE, hoặc Kaggle runtime trước.")

    if ok_non_black("single_full_mask_boot0_default_schedule") and not ok_non_black("single_full_mask_boot1_default_schedule"):
        conclusion.append("Single-mask boot0 OK nhưng boot1 lỗi: lỗi nằm ở bootstrap, không phải COCO mask.")
    if ok_non_black("single_full_mask_boot1_default_schedule") and not ok_non_black("single_full_mask_boot2_default_schedule"):
        conclusion.append("Boot1 OK nhưng boot2 lỗi: default schedule [0,4,12,25,37] không ổn với 2-step bootstrap trên runtime/model này.")
    if ok_non_black("single_full_mask_boot2_recommended_2step_schedule") and not ok_non_black("single_full_mask_boot2_default_schedule"):
        conclusion.append("Schedule [0,5,16,18,20,37] sửa được boot2: nên đổi main generation sang schedule khuyến nghị cho bootstrap=2.")
    if ok_non_black("demo_like_masks_boot2_default_schedule") and not ok_non_black("coco_all_masks_boot2_default_schedule"):
        conclusion.append("Demo-like masks OK nhưng COCO masks lỗi: cần trace mask overlap/coverage/order của COCO sample.")
    if ok_non_black("coco_all_masks_boot1_default_schedule") and not ok_non_black("coco_all_masks_boot2_default_schedule"):
        conclusion.append("COCO boot1 OK nhưng boot2 lỗi: smoke SDXL nên chạy bootstrap_steps=1 hoặc schedule khuyến nghị cho boot2.")
    if not ok_non_black("coco_all_masks_boot0_default_schedule"):
        conclusion.append("COCO boot0 đã lỗi: lỗi nằm ở multi-region/mask composition hoặc input prompts/masks, không phải bootstrap.")

    display(Markdown("## Diagnostic Conclusion\n" + "\n".join(f"- {item}" for item in conclusion)))

    reset_semanticdraw_schedule(BASELINE_DEFAULT_T_INDEX_LIST)
else:
    print("[DIAG] Black-output diagnostic skipped.")
    reset_semanticdraw_schedule(BASELINE_DEFAULT_T_INDEX_LIST)

# --- Source notebook cell 11 ---
# Chạy smoke generation cho toàn bộ smoke manifest.
# Reset lại schedule trước generation để tránh side effect từ diagnostic hoặc `smd.pipe(...)`.
reset_semanticdraw_schedule(BASELINE_DEFAULT_T_INDEX_LIST)
summary = []
global_index = 0

for batch_index, batch in enumerate(loader):
    print(f"[BATCH] {batch_index + 1}/{len(loader)} - {len(batch['sample_ids'])} sample(s)")

    for local_index, sample_id in enumerate(batch["sample_ids"]):
        payload = make_semanticdraw_payload(batch, local_index)
        original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
        overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)

        print(
            "[CHECK]",
            payload["sample_id"],
            "prompts/masks/neg/stdev/strength =",
            len(payload["prompts"]),
            payload["all_masks"].shape[0],
            len(payload["negative_prompts"]),
            len(payload["mask_stds"]),
            len(payload["mask_strengths"]),
        )

        seed = BASE_SEED + global_index
        seed_everything(seed)

        tic = time.perf_counter()
        generated = smd(
            payload["prompts"],
            payload["negative_prompts"],
            masks=payload["all_masks"].float(),
            mask_stds=payload["mask_stds"],
            mask_strengths=payload["mask_strengths"],
            height=payload["height"],
            width=payload["width"],
            bootstrap_steps=BOOTSTRAP_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        )
        elapsed = time.perf_counter() - tic

        generated = generated.convert("RGB")
        stem = f"{global_index:04d}_{payload['sample_id']}"
        generated_path = GENERATED_DIR / f"{stem}_generated.png"
        overlay_path = OVERLAY_DIR / f"{stem}_overlay.png"
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
            "checkpoint": MODEL_ID,
            "acceleration": f"{ACCEL_REPO}/{ACCEL_WEIGHT}",
            "sampler": type(smd.scheduler).__name__,
            "timestep_spacing": getattr(smd.scheduler.config, "timestep_spacing", None),
            "bootstrap_steps": BOOTSTRAP_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "prompt_count": len(payload["prompts"]),
            "mask_count": int(payload["all_masks"].shape[0]),
            "negative_prompt_count": len(payload["negative_prompts"]),
            "num_regions_including_background": len(payload["prompts"]),
            "baseline_pipeline_sha256": pipeline_sha256,
            "elapsed_sec": elapsed,
            "generated_path": str(generated_path),
            "overlay_path": str(overlay_path),
        })

        if MAX_DISPLAY_RESULTS is None or global_index < MAX_DISPLAY_RESULTS:
            display_smoke_result(payload, original, overlay, generated, elapsed, generated_path)

        global_index += 1
        torch.cuda.empty_cache()

summary_path = OUTPUT_DIR / "generation_summary.json"
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

display(Markdown(
    f"## Done\n"
    f"Generated `{len(summary)}` image(s) from `{dataset_size}` manifest record(s). "
    f"Summary saved to `{summary_path}`."
))
summary
