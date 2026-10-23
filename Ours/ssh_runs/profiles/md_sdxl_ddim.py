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

# Notebook: Ours/colab_multidiffusion_experiments/multidiffusion_sdxl_ddim_ref_colab.ipynb
# Notebook SHA256: 87cd915a6207c2a31fc8fe45983de42f03e9fee37e50123a106183dc327cf500

# --- Source notebook cell 3 ---
# =========================
# CẤU HÌNH CHẠY THÍ NGHIỆM
# =========================

# Chọn tập chạy:
# - "smoke_bs2": 2 ảnh SDXL, dùng để validate nhanh.
# - "mini32": 32 ảnh, kiểm tra ổn định hơn smoke.
# - "mini128": 128 ảnh, kiểm tra metric tương đối.
# - "full1073": toàn bộ 1073 ảnh hợp lệ theo protocol hiện tại.
RUN_PROFILE = os.environ.get('RUN_PROFILE', 'full1073')

# Chọn cấu hình VRAM:
# - "low_vram": T4/V100 nhỏ, batch nhỏ.
# - "high_vram_24gb": RTX 4090/A10/L4 khoảng 24GB.
# - "a100_80gb": A100 80GB, batch metric lớn hơn.
COLAB_GPU_MODE = os.environ.get('COLAB_GPU_MODE', 'a100_80gb')

# Chọn cấu hình DDIM:
# - "ref_g75_b20_full_view": profile chính để so với dòng SDXL Ref. DDIM trong bảng paper.
# - "diagnostic_g75_b20_v64s8": sliding-window 64/8, 81 views; rất chậm, chỉ để trace.
# - "debug_g75_b0_full_view": debug nhanh, không bootstrap.
DDIM_EXPERIMENT_PROFILE = os.environ.get('DDIM_EXPERIMENT_PROFILE', 'ref_g75_b20_full_view')

RUN_SANITY_CHECK = True
RUN_METRICS = True
RUN_EXPORT_ZIP = False
SKIP_EXISTING = True

# Toggle này chỉ quyết định có copy export sang Google Drive hay không.
# Nếu False: notebook vẫn tạo local export folder/zip trong /content để tải thủ công.
SAVE_EXPORT_TO_GOOGLE_DRIVE = False
DRIVE_RESULTS_ROOT = Path(str(WORK_DIR / 'drive/MyDrive/SemanticDraw_Results'))

BASE_SEED = 2024
BASE_SEED = int(os.environ.get("ANCHORDRAW_BASE_SEED", str(BASE_SEED)))
TARGET_SIZE = (1024, 1024)
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, low quality"

MANIFESTS = {
    "smoke_bs2": REPO_ROOT / "Ours/test_sets/manifests/smoke/coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_smoke_bs2.jsonl",
    "mini32": REPO_ROOT / "Ours/test_sets/manifests/mini32/coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_mini32.jsonl",
    "mini128": REPO_ROOT / "Ours/test_sets/manifests/mini128/coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_mini128.jsonl",
    "full1073": REPO_ROOT / "Ours/data_manifests/coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_all.jsonl",
}

DDIM_PROFILES = {
    "ref_g75_b20_full_view": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "bootstrapping": 20,
        "view_window_size": 128,
        "view_stride": 128,
        "label": "ref_g75_b20_v128full",
        "for_main_metric": True,
    },
    "diagnostic_g75_b20_v64s8": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "bootstrapping": 20,
        "view_window_size": 64,
        "view_stride": 8,
        "label": "diagnostic_g75_b20_v64s8",
        "for_main_metric": False,
    },
    "debug_g75_b0_full_view": {
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "bootstrapping": 0,
        "view_window_size": 128,
        "view_stride": 128,
        "label": "debug_g75_b0_v128full",
        "for_main_metric": False,
    },
}

GPU_PROFILES = {
    "low_vram": {
        "batch_size": 1,
        "num_workers": 2,
        "metric_batch_size": 1,
        "clip_batch_size": 4,
        "safe_vae": True,
    },
    "high_vram_24gb": {
        "batch_size": 1,
        "num_workers": 2,
        "metric_batch_size": 2,
        "clip_batch_size": 8,
        "safe_vae": True,
    },
    "a100_80gb": {
        "batch_size": 1,
        "num_workers": 4,
        "metric_batch_size": 8,
        "clip_batch_size": 32,
        "safe_vae": True,
    },
}

assert RUN_PROFILE in MANIFESTS, f"RUN_PROFILE không hợp lệ: {RUN_PROFILE}"
assert DDIM_EXPERIMENT_PROFILE in DDIM_PROFILES, f"DDIM_EXPERIMENT_PROFILE không hợp lệ: {DDIM_EXPERIMENT_PROFILE}"
assert COLAB_GPU_MODE in GPU_PROFILES, f"COLAB_GPU_MODE không hợp lệ: {COLAB_GPU_MODE}"

MANIFEST_PATH = MANIFESTS[RUN_PROFILE]
ddim_cfg = DDIM_PROFILES[DDIM_EXPERIMENT_PROFILE]
gpu_cfg = GPU_PROFILES[COLAB_GPU_MODE]

DDIM_NUM_INFERENCE_STEPS = int(ddim_cfg["num_inference_steps"])
DDIM_GUIDANCE_SCALE = float(ddim_cfg["guidance_scale"])
BOOTSTRAPPING = int(ddim_cfg["bootstrapping"])
VIEW_WINDOW_SIZE = int(ddim_cfg["view_window_size"])
VIEW_STRIDE = int(ddim_cfg["view_stride"])
DDIM_CONFIG_LABEL = str(ddim_cfg["label"])
FOR_MAIN_METRIC = bool(ddim_cfg["for_main_metric"])

BATCH_SIZE = int(gpu_cfg["batch_size"])
NUM_WORKERS = int(gpu_cfg["num_workers"])
METRIC_BATCH_SIZE = int(gpu_cfg["metric_batch_size"])
CLIP_BATCH_SIZE = int(gpu_cfg["clip_batch_size"])
SAFE_VAE = bool(gpu_cfg["safe_vae"])

RUN_ROOT = Path(str(WORK_DIR / 'anchordraw_runs')) / f"multidiffusion_sdxl_ddim_ref_{DDIM_CONFIG_LABEL}_{RUN_PROFILE}_{COLAB_GPU_MODE}"
GENERATED_DIR = RUN_ROOT / "generated_images"
OVERLAY_DIR = RUN_ROOT / "mask_overlays"
METRIC_DIR = RUN_ROOT / "metrics"
DIAGNOSTIC_DIR = RUN_ROOT / "diagnostics"
RUN_SUMMARY_PATH = RUN_ROOT / "generation_summary.json"
EXPORT_ROOT = Path(str(WORK_DIR / 'anchordraw_metric_exports')) / RUN_ROOT.name

for directory in (GENERATED_DIR, OVERLAY_DIR, METRIC_DIR, DIAGNOSTIC_DIR, EXPORT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

display(Markdown("\n".join([
    "## Cấu hình đang chạy",
    "",
    "```text",
    "Experiment ID   : MD-SDXL-DDIM-REF-F1073",
    f"Run root        : {RUN_ROOT}",
    f"Manifest        : {MANIFEST_PATH.relative_to(REPO_ROOT)}",
    f"Run profile     : {RUN_PROFILE}",
    f"GPU mode        : {COLAB_GPU_MODE}",
    "Model           : SDXL base",
    f"Checkpoint      : {MODEL_ID}",
    "Sampler         : DDIMScheduler",
    f"Resolution      : {TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
    f"Steps           : {DDIM_NUM_INFERENCE_STEPS}",
    f"Guidance        : {DDIM_GUIDANCE_SCALE}",
    f"Bootstrap       : {BOOTSTRAPPING}",
    f"View window     : {VIEW_WINDOW_SIZE}",
    f"View stride     : {VIEW_STRIDE}",
    f"Safe VAE        : {SAFE_VAE}",
    f"Metrics enabled : {RUN_METRICS}",
    f"Local zip export : {RUN_EXPORT_ZIP}",
    f"Drive export     : {SAVE_EXPORT_TO_GOOGLE_DRIVE}",
    f"Main metric cfg : {FOR_MAIN_METRIC}",
    "```",
])))

# --- Source notebook cell 6 ---
# Import dataloader của Ours và wrapper MultiDiffusion SDXL DDIM.
OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_MD_SRC = REPO_ROOT / "Baseline" / "MultiDiffusion-master" / "MultiDiffusion-master"

# Đưa Ours lên trước để import đúng package data/metrics/baselines của mình,
# tránh bị nhầm với Baseline/MultiDiffusion-master/data.py nếu có.
sys.path.insert(0, str(BASELINE_MD_SRC))
sys.path.insert(0, str(OURS_SRC))

import importlib
importlib.invalidate_caches()
for module_name in list(sys.modules):
    if module_name == "baselines" or module_name.startswith("baselines."):
        del sys.modules[module_name]

from data import COCORegionConfig, build_coco_region_dataloader
from data.visualize import make_mask_overlay

ddim_wrapper_file = OURS_SRC / "baselines" / "multidiffusion_sdxl_ddim.py"
if not ddim_wrapper_file.exists():
    raise FileNotFoundError(
        f"Không tìm thấy {ddim_wrapper_file}. Repo trên Colab có thể chưa được cập nhật/push file wrapper mới."
    )

# Import trực tiếp từ module wrapper để không phụ thuộc vào baselines.__init__ của clone cũ.
from baselines.multidiffusion_sdxl_ddim import MultiDiffusionSDXLDDIM
from baselines.multidiffusion_sdxl_euler import get_sdxl_views

print("[OK] Imported Ours dataloader and MultiDiffusionSDXLDDIM wrapper.")
print("[INFO] Number of active SDXL views:", len(get_sdxl_views(TARGET_SIZE[0], TARGET_SIZE[1], window_size=VIEW_WINDOW_SIZE, stride=VIEW_STRIDE)))

# --- Source notebook cell 7 ---
# Helper chung.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
print("[INFO] Device:", device)
if torch.cuda.is_available():
    print("[INFO] GPU:", torch.cuda.get_device_name(0))
    print("[INFO] VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_stats(image: Image.Image) -> dict:
    arr = np.asarray(image.convert("RGB"))
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def is_near_black_image(image: Image.Image) -> bool:
    stats = image_stats(image)
    return stats["max"] <= 2 or stats["mean"] <= 1.0


def is_static_noise_image(image: Image.Image) -> bool:
    stats = image_stats(image)
    return stats["std"] > 95 and 105 <= stats["mean"] <= 150


def resize_original(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.Resampling.BILINEAR)


def display_generation_result(payload: dict, original: Image.Image, overlay: Image.Image, generated: Image.Image, elapsed: float, generated_path: Path) -> None:
    display(Markdown("\n".join([
        f"### `{payload['sample_id']}`",
        "",
        f"- image_id: `{payload['image_id']}`",
        f"- file: `{payload['file_name']}`",
        f"- prompt/mask count: `{len(payload['prompts'])}`",
        f"- generated path: `{generated_path}`",
        f"- elapsed: `{elapsed:.2f}s`",
        f"- generated stats: `{image_stats(generated)}`",
    ])))

    table_rows = [{"Region": "Background", "Prompt": payload["prompts"][0], "Annotation": "-", "Area ratio": "-"}]
    for name, prompt, ann_id, area_ratio in zip(
        payload["category_names"][1:],
        payload["prompts"][1:],
        payload["annotation_ids"],
        payload["area_ratios"],
    ):
        table_rows.append(
            {
                "Region": name,
                "Prompt": prompt,
                "Annotation": int(ann_id),
                "Area ratio": round(float(area_ratio), 4),
            }
        )
    display(pd.DataFrame(table_rows))

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    panels = [
        (original, "COCO original resized"),
        (overlay, "Foreground mask overlay"),
        (generated, "MultiDiffusion Ref. + SDXL DDIM generated"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# --- Source notebook cell 8 ---
# Dataloader: đọc manifest có sẵn, không build lại manifest.
assert MANIFEST_PATH.exists(), f"Không tìm thấy manifest: {MANIFEST_PATH}"

coco_cfg = COCORegionConfig(
    coco_root=COCO_ROOT,
    split="val2017",
    manifest_path=MANIFEST_PATH,
    profile="multidiffusion_coco_all",
    model_family="sdxl",
    target_size=TARGET_SIZE,
    return_image=True,
    cache_resized_masks=True,
    cache_dir=Path(str(WORK_DIR / 'anchordraw_cache')),
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=2,
)

loader = build_coco_region_dataloader(coco_cfg, shuffle=False, drop_last=False)
dataset_size = len(loader.dataset)

display(Markdown("\n".join([
    "## Dataloader ready",
    "",
    "```text",
    f"Manifest records : {dataset_size}",
    f"Batch size       : {BATCH_SIZE}",
    f"Num workers      : {NUM_WORKERS}",
    f"Target size      : {TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
    "```",
])))

# --- Source notebook cell 9 ---
# Adapter: biến output dataloader foreground-only thành input đúng chuẩn MultiDiffusion.
def make_multidiffusion_payload(batch: dict, local_index: int) -> dict:
    valid = batch["valid_regions"][local_index]
    p = int(valid.sum().item())

    foreground_masks = batch["masks"][local_index, :p].to(dtype=torch.float32)
    if foreground_masks.ndim != 4 or foreground_masks.shape[1] != 1:
        raise ValueError(f"Expected foreground masks shape P x 1 x H x W, got {tuple(foreground_masks.shape)}")

    foreground_union = foreground_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    background_mask = (1.0 - foreground_union).clamp(0, 1)
    masks = torch.cat([background_mask, foreground_masks], dim=0)

    foreground_prompts = list(batch["foreground_prompts"][local_index][:p])
    prompts = [str(batch["background_prompts"][local_index])] + foreground_prompts
    negative_prompts = [NEGATIVE_PROMPT for _ in prompts]

    category_names = ["Background"] + list(batch["category_names"][local_index][:p])
    area_ratios = [float(v) for v in batch["area_ratios"][local_index, :p].tolist()]
    annotation_ids = [int(v) for v in batch["annotation_ids"][local_index][:p]]

    if masks.shape[0] != len(prompts):
        raise ValueError(f"masks/prompts mismatch: {masks.shape[0]} masks vs {len(prompts)} prompts")

    return {
        "sample_id": str(batch["sample_ids"][local_index]),
        "image_id": int(batch["image_ids"][local_index]),
        "file_name": str(batch["file_names"][local_index]),
        "metadata": batch["metadata"][local_index],
        "image": batch["images"][local_index],
        "foreground_masks": foreground_masks,
        "masks": masks,
        "prompts": prompts,
        "negative_prompts": negative_prompts,
        "category_names": category_names,
        "area_ratios": area_ratios,
        "annotation_ids": annotation_ids,
    }


first_batch = next(iter(loader))
first_payload = make_multidiffusion_payload(first_batch, 0)
print("[CHECK] sample_id:", first_payload["sample_id"])
print("[CHECK] masks shape:", tuple(first_payload["masks"].shape))
print("[CHECK] prompts:", first_payload["prompts"])
print("[CHECK] mask sums:", [float(m.sum().item()) for m in first_payload["masks"]])
assert first_payload["masks"].shape[0] == len(first_payload["prompts"])

# --- Source notebook cell 10 ---
# Load MultiDiffusion SDXL DDIM.
seed_everything(BASE_SEED)

md_ddim = MultiDiffusionSDXLDDIM(
    device=device,
    model_id=MODEL_ID,
    dtype=dtype,
    variant="fp16" if dtype == torch.float16 else None,
    safe_vae=SAFE_VAE,
    enable_attention_slicing=True,
    enable_vae_slicing=True,
    view_window_size=VIEW_WINDOW_SIZE,
    view_stride=VIEW_STRIDE,
    runtime_checks=True,
    show_progress=False,
)

print("[OK] MultiDiffusionSDXLDDIM is ready.")
print("[INFO] Scheduler:", type(md_ddim.scheduler).__name__)
print("[INFO] View count:", len(get_sdxl_views(TARGET_SIZE[0], TARGET_SIZE[1], window_size=VIEW_WINDOW_SIZE, stride=VIEW_STRIDE)))

# --- Source notebook cell 11 ---
# Sanity check nhẹ: kiểm tra checkpoint/scheduler/VAE bằng pipeline SDXL thường.
# Không chạy full MultiDiffusion 64/8 ở đây vì profile đó có 81 views và sẽ rất chậm.
if RUN_SANITY_CHECK:
    seed_everything(BASE_SEED)
    sanity_image = md_ddim.pipe(
        prompt="a studio photo of a teddy bear on a clean table",
        negative_prompt=NEGATIVE_PROMPT,
        height=TARGET_SIZE[0],
        width=TARGET_SIZE[1],
        num_inference_steps=min(10, DDIM_NUM_INFERENCE_STEPS),
        guidance_scale=DDIM_GUIDANCE_SCALE,
    ).images[0].convert("RGB")
    stats = image_stats(sanity_image)
    print("[SANITY] image stats:", stats)
    display(sanity_image.resize((512, 512)))
    if is_near_black_image(sanity_image):
        raise RuntimeError("Sanity image gần đen: checkpoint/scheduler/VAE có vấn đề trước khi chạy manifest.")
else:
    print("[INFO] Sanity check skipped.")

# --- Source notebook cell 12 ---
failed_images = []
# Chạy generation cho từng sample trong manifest.
existing_summary = []
if SKIP_EXISTING and RUN_SUMMARY_PATH.exists():
    with RUN_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        existing_summary = json.load(f)
    print(f"[INFO] Found existing summary with {len(existing_summary)} rows. Existing rows will be skipped.")

summary = list(existing_summary)
completed_indices = {int(row["index"]) for row in summary if "index" in row}

MAX_DISPLAY_RESULTS = 0
num_views = len(get_sdxl_views(TARGET_SIZE[0], TARGET_SIZE[1], window_size=VIEW_WINDOW_SIZE, stride=VIEW_STRIDE))
global_index = 0

for batch_index, batch in enumerate(loader):
    current_batch_size = len(batch["sample_ids"])
    print(f"[BATCH] {batch_index + 1}/{len(loader)} - {current_batch_size} sample(s)")

    for local_index in range(current_batch_size):
        retry_overhead_sec = 0.0
        for attempt in range(1, 4):
            attempt_started = time.perf_counter()
            tracked_lists = ('summary',)
            list_lengths_before = {name: len(globals()[name]) for name in tracked_lists}
            try:
                payload = make_multidiffusion_payload(batch, local_index)
                stem = f"{global_index:04d}_{payload['sample_id']}"
                generated_path = GENERATED_DIR / f"{stem}_generated.png"
                overlay_path = OVERLAY_DIR / f"{stem}_overlay.png"

                if SKIP_EXISTING and global_index in completed_indices and generated_path.exists():
                    print(f"[SKIP] index={global_index} sample_id={payload['sample_id']}")
                    break

                seed = BASE_SEED + global_index
                seed_everything(seed)

                original = resize_original(payload["image"])
                overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"][1:])

                start = time.perf_counter()
                generated = md_ddim.generate(
                    masks=payload["masks"],
                    prompts=payload["prompts"],
                    negative_prompts=payload["negative_prompts"],
                    height=TARGET_SIZE[0],
                    width=TARGET_SIZE[1],
                    num_inference_steps=DDIM_NUM_INFERENCE_STEPS,
                    guidance_scale=DDIM_GUIDANCE_SCALE,
                    bootstrapping=BOOTSTRAPPING,
                    view_window_size=VIEW_WINDOW_SIZE,
                    view_stride=VIEW_STRIDE,
                    show_progress=True,
                ).convert("RGB")
                elapsed = time.perf_counter() - start + retry_overhead_sec

                stats = image_stats(generated)
                if is_near_black_image(generated):
                    raise RuntimeError(f"Generated image at index {global_index} is nearly black: {stats}")
                if is_static_noise_image(generated):
                    print(f"[WARN] Generated image at index {global_index} looks static/noisy: {stats}")

                generated.save(generated_path)
                overlay.save(overlay_path)

                row = {
                    "index": global_index,
                    "batch_index": batch_index,
                    "local_index": local_index,
                    "sample_id": payload["sample_id"],
                    "image_id": payload["image_id"],
                    "file_name": payload["file_name"],
                    "seed": seed,
                    "experiment_id": "MD-SDXL-DDIM-REF-F1073",
                    "model_family": "sdxl",
                    "model_id": MODEL_ID,
                    "sampler": "ddim",
                    "scheduler": type(md_ddim.scheduler).__name__,
                    "num_inference_steps": DDIM_NUM_INFERENCE_STEPS,
                    "guidance_scale": DDIM_GUIDANCE_SCALE,
                    "bootstrapping": BOOTSTRAPPING,
                    "view_window_size": VIEW_WINDOW_SIZE,
                    "view_stride": VIEW_STRIDE,
                    "num_views": num_views,
                    "safe_vae": SAFE_VAE,
                    "num_regions_including_background": len(payload["prompts"]),
                    "num_foreground_regions": len(payload["prompts"]) - 1,
                    "background_prompt": payload["prompts"][0],
                    "foreground_prompts": payload["prompts"][1:],
                    "category_names": payload["category_names"][1:],
                    "annotation_ids": payload["annotation_ids"],
                    "area_ratios": payload["area_ratios"],
                    "elapsed_sec": elapsed,
                    "generated_path": str(generated_path),
                    "overlay_path": str(overlay_path),
                    "generated_stats": stats,
                }
                summary.append(row)

                with RUN_SUMMARY_PATH.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

                should_display = MAX_DISPLAY_RESULTS is None or global_index < MAX_DISPLAY_RESULTS
                if should_display:
                    display_generation_result(payload, original, overlay, generated, elapsed, generated_path)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                import traceback
                rolled_back = any(len(globals()[name]) > list_lengths_before[name] for name in tracked_lists)
                for name in tracked_lists:
                    del globals()[name][list_lengths_before[name]:]
                if rolled_back:
                    try:
                        RUN_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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

summary_df = pd.DataFrame(summary)
display(Markdown(f"## Done\nGenerated `{len(summary)}` image(s) from `{dataset_size}` manifest record(s). Summary saved to `{RUN_SUMMARY_PATH}`."))
display(summary_df.tail(min(5, len(summary_df))))

# --- Source notebook cell 13 ---
# Đo metric sau generation.
# Bật RUN_METRICS=True khi muốn đo FID, IS, CLIP(fg), CLIP(bg), Time(s).
if RUN_METRICS:
    from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report

    if "md_ddim" in globals():
        del md_ddim
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metric_cfg = MetricEvaluationConfig(
        manifest_path=MANIFEST_PATH,
        coco_root=COCO_ROOT,
        generated_dir=GENERATED_DIR,
        generation_summary=RUN_SUMMARY_PATH,
        output_dir=METRIC_DIR,
        model_family="sdxl",
        target_size=TARGET_SIZE,
        metrics=("fid", "is", "clip_fg", "clip_bg", "time"),
        batch_size=METRIC_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        cache_resized_masks=True,
        cache_dir=Path(str(WORK_DIR / 'anchordraw_cache')),
        device="auto",
        clip_batch_size=CLIP_BATCH_SIZE,
        is_splits=10,
    )
    report = run_evaluation(metric_cfg)
    json_path, csv_path = write_metrics_report(report, METRIC_DIR, prefix="metrics")
    print("[OK] Metric report:", json_path)
    print("[OK] Metric CSV:", csv_path)

    metric_rows = []
    for key, value in report["metrics"].items():
        if isinstance(value, (int, float)):
            metric_rows.append({"Metric": key, "Value": float(value)})
    display(pd.DataFrame(metric_rows))
else:
    print("[INFO] RUN_METRICS=False, metric evaluation skipped.")
