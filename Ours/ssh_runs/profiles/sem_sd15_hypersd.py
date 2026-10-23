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

def display(*values):
    for value in values:
        if isinstance(value, str):
            print(value)

# Notebook: Ours/kaggle_semanticdraw_smoke/semanticdraw_sd15_hypersd_full1073_kaggle.ipynb
# Notebook SHA256: 38010410c11d8b859b046151acd8d71fef2498da176864ee5e2851f7270b2e13

# --- Source notebook cell 4 ---
# Cấu hình full manifest 1073 cho benchmark SD1.5 + Hyper-SD.
from pathlib import Path

RUN_MANIFEST = REPO_ROOT / "Ours" / "data_manifests" / "coco_val2017_multidiffusion_coco_all_512x512_all.jsonl"
COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'COCO')))
OUTPUT_DIR = Path(str(WORK_DIR / 'semanticdraw_sd15_hypersd_full1073_maskstd1_rerun_outputs'))
MASK_CACHE_DIR = Path(str(WORK_DIR / 'semanticdraw_mask_cache'))
METRICS_OUTPUT_DIR = Path(str(WORK_DIR / 'semanticdraw_sd15_hypersd_full1073_maskstd1_rerun_metrics'))

# Model theo baseline SD1.5. Có thể đổi sang một SD1.5-compatible checkpoint nếu cần.
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# Hyper-SD SD1.5 LoRA theo model card chính thức.
HYPER_SD_REPO_ID = "ByteDance/Hyper-SD"
HYPER_SD_WEIGHT_NAME = "Hyper-SD15-4steps-lora.safetensors"
HYPER_SD_NUM_INFERENCE_STEPS = 4
HYPER_SD_LORA_SCALE = 1.0

# Trong Diffusers pipeline chuẩn, Hyper-SD 4-step thường dùng guidance_scale=0 để tắt CFG.
# Trong SemanticDraw custom code, công thức luôn là uncond + scale * (cond - uncond).
# Vì vậy scale=1.0 mới tương ứng conditional-only/no-CFG.
SEMANTICDRAW_GUIDANCE_SCALE = 1.0

# BATCH_SIZE chỉ là số sample mỗi lượt dataloader load.
# Tổng số ảnh sinh ra = số record trong manifest vì cell generation sẽ lặp qua toàn bộ loader.
TARGET_SIZE = (512, 512)
BATCH_SIZE = 8
BASE_SEED = 2024
BASE_SEED = int(os.environ.get("ANCHORDRAW_BASE_SEED", str(BASE_SEED)))
BOOTSTRAP_STEPS = 1
MASK_STD = 1.0
BACKGROUND_PROTOCOL = "foreground_then_complement_v1"
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = "discrete"
NEGATIVE_PROMPT = ""

# Chỉ hiển thị 8 preview để Kaggle notebook không bị nặng.
# Ảnh vẫn được sinh và lưu đủ theo full manifest 1073.
MAX_DISPLAY_RESULTS = 0

METRIC_NAMES = ("fid", "is", "clip_fg", "clip_bg", "time")
METRIC_BATCH_SIZE = 8
CLIP_BATCH_SIZE = 16
IS_SPLITS = 10
METRICS_REPORT_PREFIX = "semanticdraw_sd15_hypersd_full1073_maskstd1_rerun_metrics"

assert RUN_MANIFEST.exists(), f"Missing manifest: {RUN_MANIFEST}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[OK] Manifest: {RUN_MANIFEST}")
print(f"[OK] COCO root: {COCO_ROOT}")
print(f"[OK] Output dir: {OUTPUT_DIR}")
print(f"[OK] Metrics output dir: {METRICS_OUTPUT_DIR}")
print(f"[OK] Batch size: {BATCH_SIZE}")
print(f"[OK] Hyper-SD LoRA: {HYPER_SD_REPO_ID}/{HYPER_SD_WEIGHT_NAME}")
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 6 ---
# Import dataloader của Ours và baseline SemanticDrawPipeline.
import sys
import importlib.util
import json
import time

import torch
import matplotlib.pyplot as plt
from PIL import Image
# terminal display provided above

OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_SRC = REPO_ROOT / "Baseline" / "semantic-draw-main" / "src"

sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from data.visualize import make_mask_overlay

# Load trực tiếp file pipeline_semantic_draw.py để tránh import toàn bộ model/__init__.py
# vì các file SDXL/SD3 có thể cần dependency khác không dùng trong smoke SD1.5.
sys.path.insert(0, str(BASELINE_SRC))
pipeline_path = BASELINE_SRC / "model" / "pipeline_semantic_draw.py"
spec = importlib.util.spec_from_file_location("pipeline_semantic_draw", pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline

print("[OK] Imports are ready.")

# --- Source notebook cell 7 ---
# Tạo dataloader cho full manifest 1073.
config = COCORegionConfig(
    coco_root=COCO_ROOT,
    split="val2017",
    instances_json=COCO_ROOT / "annotations" / "instances_val2017.json",
    captions_json=COCO_ROOT / "annotations" / "captions_val2017.json",
    manifest_path=RUN_MANIFEST,
    profile="multidiffusion_coco_all",
    model_family="sd15",
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
assert dataset_size == 1073, f"Expected 1073 manifest records, got {dataset_size}"
num_batches = len(loader)
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

print("[OK] Helper functions are ready.")

# --- Source notebook cell 10 ---
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
        print("[INFO] No HF_TOKEN found. Public/gated model access depends on your Hugging Face permissions.")

assert torch.cuda.is_available(), "Kaggle runtime chưa bật GPU. Hãy bật Accelerator = GPU rồi chạy lại."
device = torch.device("cuda:0")
dtype = torch.float16

maybe_login_to_huggingface()
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

# --- Source notebook cell 11 ---
# Guard chống lỗi torchao/PEFT trước khi load Hyper-SD LoRA.
# Nếu bạn vừa gặp lỗi torchao ở cell load pipeline, hãy restart Kaggle session rồi Run All.
import sys
import importlib
import importlib.util

importlib.invalidate_caches()

if "torchao" in sys.modules:
    raise RuntimeError(
        "torchao is already imported in this Python session. Restart the Kaggle session, run the dependency cell, then Run All. "
        "SemanticDraw SD1.5 + Hyper-SD LoRA does not need torchao for this run."
    )

if importlib.util.find_spec("torchao") is not None:
    raise RuntimeError(
        "torchao is still installed/importable in this runtime. Run the dependency cell, then restart the Kaggle session and Run All. "
        "SemanticDraw SD1.5 + Hyper-SD LoRA does not need torchao for this run."
    )

print("[OK] torchao is not importable; PEFT should skip torchao LoRA dispatch.")

# --- Source notebook cell 12 ---
# Load baseline SemanticDraw SD1.5 pipeline, sau đó chuyển sampler/tăng tốc từ LCM sang Hyper-SD.
# has_i2t=False để không tải BLIP-2 vì COCO caption đã là background prompt.
seed_everything(BASE_SEED)
smd = SemanticDrawPipeline(
    device=device,
    dtype=dtype,
    sd_version="1.5",
    hf_key=MODEL_ID,
    has_i2t=False,
    default_mask_std=MASK_STD,
    default_mask_strength=MASK_STRENGTH,
    default_preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
    mask_type=MASK_TYPE,
)

# Keep attention slicing off to match the AnchorDraw SD1.5 paired run.

from diffusers import DDIMScheduler
from huggingface_hub import hf_hub_download
import types


def clear_existing_lora_adapters(pipe) -> None:
    # SemanticDrawPipeline gốc tự load LCM LoRA. Ta cần bỏ nó trước khi gắn Hyper-SD.
    if hasattr(pipe, "unfuse_lora"):
        try:
            pipe.unfuse_lora()
        except Exception as exc:
            print(f"[WARN] unfuse_lora skipped: {exc}")

    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()
        print("[OK] Existing LoRA weights unloaded.")
        return

    if hasattr(pipe, "delete_adapters"):
        try:
            pipe.delete_adapters(["lcm"])
            print("[OK] Existing LCM adapter deleted.")
            return
        except Exception as exc:
            print(f"[WARN] delete_adapters(['lcm']) skipped: {exc}")

    raise RuntimeError("Could not remove the default LCM-LoRA before loading Hyper-SD.")


def prepare_hyper_sd_ddim_schedule(self, num_inference_steps: int | None = None) -> None:
    if num_inference_steps is None:
        num_inference_steps = self.default_num_inference_steps

    try:
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
    except TypeError:
        self.scheduler.set_timesteps(num_inference_steps)

    self.timesteps = self.scheduler.timesteps.to(dtype=torch.long).cpu()
    shape = (len(self.timesteps), 1, 1, 1)

    alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=torch.float32)
    alpha_prod_t = alphas_cumprod[self.timesteps].view(*shape).clamp(1e-8, 1.0).to(device=self.device)
    beta_prod_t = (1.0 - alpha_prod_t).clamp(0.0, 1.0)

    self.alpha_prod_t_sqrt = alpha_prod_t.sqrt().to(dtype=self.dtype, device=self.device)
    self.beta_prod_t_sqrt = beta_prod_t.sqrt().to(dtype=self.dtype, device=self.device)

    noise_lvs = self.beta_prod_t_sqrt.flatten()
    self.noise_lvs = noise_lvs[None, :, None, None, None]
    self.next_noise_lvs = torch.cat([noise_lvs[1:], noise_lvs.new_zeros(1)])[None, :, None, None, None]


def hyper_sd_ddim_predict_x0_step(self, noise_pred: torch.Tensor, idx: int, latent: torch.Tensor) -> torch.Tensor:
    # Hyper-SD 4-step LoRA dùng DDIM/epsilon-style prediction.
    # SemanticDraw cần x0 prediction để trộn các region rồi re-noise sang timestep kế tiếp.
    alpha = self.alpha_prod_t_sqrt[idx].to(dtype=latent.dtype, device=latent.device).clamp_min(1e-6)
    beta = self.beta_prod_t_sqrt[idx].to(dtype=latent.dtype, device=latent.device)
    return (latent - beta * noise_pred) / alpha


clear_existing_lora_adapters(smd.pipe)

hyper_sd_lora_path = hf_hub_download(
    repo_id=HYPER_SD_REPO_ID,
    filename=HYPER_SD_WEIGHT_NAME,
)
smd.pipe.load_lora_weights(hyper_sd_lora_path, adapter_name="hyper_sd")

if hasattr(smd.pipe, "set_adapters"):
    try:
        smd.pipe.set_adapters(["hyper_sd"], adapter_weights=[HYPER_SD_LORA_SCALE])
    except TypeError:
        smd.pipe.set_adapters(["hyper_sd"])

if hasattr(smd.pipe, "fuse_lora"):
    try:
        smd.pipe.fuse_lora(lora_scale=HYPER_SD_LORA_SCALE, safe_fusing=False)
        print("[OK] Hyper-SD LoRA fused into the pipeline.")
    except TypeError:
        smd.pipe.fuse_lora(lora_scale=HYPER_SD_LORA_SCALE)
        print("[OK] Hyper-SD LoRA fused into the pipeline.")
    except Exception as exc:
        print(f"[WARN] fuse_lora skipped; running with active adapter instead: {exc}")

try:
    smd.pipe.scheduler = DDIMScheduler.from_pretrained(
        MODEL_ID,
        subfolder="scheduler",
        timestep_spacing="trailing",
    )
except Exception as exc:
    print(f"[WARN] DDIMScheduler.from_pretrained skipped, falling back to from_config: {exc}")
    smd.pipe.scheduler = DDIMScheduler.from_config(
        smd.pipe.scheduler.config,
        timestep_spacing="trailing",
    )

smd.scheduler = smd.pipe.scheduler
smd.unet = smd.pipe.unet
smd.default_num_inference_steps = HYPER_SD_NUM_INFERENCE_STEPS
smd.default_guidance_scale = SEMANTICDRAW_GUIDANCE_SCALE

smd.prepare_hyper_sd_ddim_schedule = types.MethodType(prepare_hyper_sd_ddim_schedule, smd)
smd.prepare_lcm_schedule = smd.prepare_hyper_sd_ddim_schedule
smd.scheduler_step = types.MethodType(hyper_sd_ddim_predict_x0_step, smd)
smd.prepare_hyper_sd_ddim_schedule(HYPER_SD_NUM_INFERENCE_STEPS)

print("[OK] SemanticDrawPipeline is ready with SD1.5 + Hyper-SD.")

# --- Source notebook cell 13 ---
# Kiểm tra trực tiếp notebook đang dùng sampler/scheduler nào.
print("Sampler:", "Hyper-SD")
print("Hyper-SD repo:", HYPER_SD_REPO_ID)
print("Hyper-SD LoRA:", HYPER_SD_WEIGHT_NAME)
print("Scheduler:", type(smd.scheduler).__name__)
print("Pipeline scheduler:", type(smd.pipe.scheduler).__name__)
print("Default inference steps:", smd.default_num_inference_steps)
print("SemanticDraw guidance scale:", smd.default_guidance_scale)
print("Timesteps:", [int(t) for t in smd.timesteps.detach().cpu().tolist()])

loaded_adapters = getattr(smd.pipe, "get_active_adapters", None)
if callable(loaded_adapters):
    print("Active LoRA adapters:", loaded_adapters())
else:
    print("Active LoRA adapters: check not available in this diffusers version")

assert type(smd.scheduler).__name__ == "DDIMScheduler", "Expected Hyper-SD sampler via DDIMScheduler trailing."
assert smd.default_num_inference_steps == HYPER_SD_NUM_INFERENCE_STEPS
assert abs(float(smd.default_guidance_scale) - float(SEMANTICDRAW_GUIDANCE_SCALE)) < 1e-8

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
                payload = make_semanticdraw_payload(batch, local_index)
                original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
                overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)

                seed = BASE_SEED + global_index
                seed_everything(seed)

                tic = time.perf_counter()
                generated = smd(
                    prompts=payload["foreground_prompts"],
                    negative_prompts=payload["negative_prompts"][1:],
                    background_prompt=payload["prompts"][0],
                    background_negative_prompt=payload["negative_prompts"][0],
                    masks=payload["foreground_masks"].to(device=device, dtype=torch.float32),
                    mask_stds=MASK_STD,
                    mask_strengths=MASK_STRENGTH,
                    height=payload["height"],
                    width=payload["width"],
                    bootstrap_steps=BOOTSTRAP_STEPS,
                    preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
                    guidance_scale=SEMANTICDRAW_GUIDANCE_SCALE,
                    do_blend=False,
                )
                elapsed = time.perf_counter() - tic + retry_overhead_sec

                stem = f"{global_index:04d}_{payload['sample_id']}"
                generated_path = OUTPUT_DIR / f"{stem}_generated.png"
                overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"
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
                    "mask_std": MASK_STD,
                    "background_protocol": BACKGROUND_PROTOCOL,
                    "bootstrap_steps": BOOTSTRAP_STEPS,
                    "sampler": "hyper-sd",
                    "hyper_sd_repo": HYPER_SD_REPO_ID,
                    "hyper_sd_weight": HYPER_SD_WEIGHT_NAME,
                    "num_inference_steps": HYPER_SD_NUM_INFERENCE_STEPS,
                    "semanticdraw_guidance_scale": SEMANTICDRAW_GUIDANCE_SCALE,
                    "num_regions_including_background": len(payload["prompts"]),
                    "elapsed_sec": elapsed,
                    "generated_path": str(generated_path),
                    "overlay_path": str(overlay_path),
                })

                should_display = MAX_DISPLAY_RESULTS is None or global_index < MAX_DISPLAY_RESULTS
                if should_display:
                    display_smoke_result(payload, original, overlay, generated, elapsed, generated_path)

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

assert len(summary) == dataset_size == 1073, f"Incomplete generation: {len(summary)}/{dataset_size}"
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

# --- Source notebook cell 18 ---
# Đo FID, IS, CLIP(fg), CLIP(bg), Time(s) trên full output vừa generate.
from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report
import pandas as pd
import math

generation_summary_path = OUTPUT_DIR / "generation_summary.json"
assert generation_summary_path.exists(), f"Missing generation summary: {generation_summary_path}"

metric_device = "cuda:0" if torch.cuda.is_available() else "cpu"
metric_config = MetricEvaluationConfig(
    manifest_path=RUN_MANIFEST,
    coco_root=COCO_ROOT,
    generated_dir=OUTPUT_DIR,
    generation_summary=generation_summary_path,
    output_dir=METRICS_OUTPUT_DIR,
    model_family="sd15",
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
