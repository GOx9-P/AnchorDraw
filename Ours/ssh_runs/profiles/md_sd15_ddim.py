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

# Notebook: Ours/colab_multidiffusion_experiments/multidiffusion_sd15_ddim_ref_colab.ipynb
# Notebook SHA256: 17b8f7689ec5df6bde47d8d92af1f2810d93ed75a3a2c91ab6b9bdfeaf29b971

# --- Source notebook cell 3 ---
# =========================
# CẤU HÌNH CHẠY THÍ NGHIỆM
# =========================

# Chọn tập chạy:
# - "smoke_bs8": 8 ảnh, dùng để kiểm tra end-to-end nhanh.
# - "mini32": 32 ảnh, kiểm tra ổn định hơn smoke.
# - "mini128": 128 ảnh, kiểm tra metric đáng tin hơn nhưng vẫn chưa phải benchmark chính.
# - "full1073": toàn bộ 1073 ảnh hợp lệ theo protocol hiện tại.
RUN_PROFILE = os.environ.get('RUN_PROFILE', 'full1073')

# Chọn cấu hình VRAM:
# - "low_vram": phù hợp T4/V100 nhỏ, dataloader nhỏ, metric batch nhỏ.
# - "high_vram_24gb": phù hợp RTX 4090/A10/L4 24GB.
# - "a100_80gb": phù hợp A100 80GB, load/metric batch lớn hơn.
COLAB_GPU_MODE = os.environ.get('COLAB_GPU_MODE', 'high_vram_24gb')

# Chọn cấu hình DDIM:
# - "ref_g75_b20": gần nhất với MultiDiffusion gốc/paper Ref., guidance=7.5, bootstrap=20.
# - "debug_g75_b0": tắt bootstrap để debug, không dùng làm dòng Ref. chính.
# - "debug_g75_b1": bootstrap rất nhỏ để smoke/debug.
DDIM_EXPERIMENT_PROFILE = os.environ.get('DDIM_EXPERIMENT_PROFILE', 'ref_g75_b20')

# Chọn precision:
# - "fp16_colab": nhẹ VRAM hơn, nên dùng mặc định trên Colab.
# - "fp32_original_autocast": gần cách load gốc hơn: weights fp32, forward có autocast cuda.
PRECISION_PROFILE = os.environ.get('PRECISION_PROFILE', 'fp32_original_autocast')

REBUILD_MANIFEST = False
RUN_SANITY_CHECK = True
RUN_METRICS = True          # Nên bật True khi RUN_PROFILE="full1073" và đã sinh xong ảnh.
RUN_EXPORT_ZIP = False
USE_GOOGLE_DRIVE_EXPORT = False  # Copy export sang Google Drive sau khi nén local xong.
DRIVE_RESULTS_ROOT = Path(str(WORK_DIR / 'drive/MyDrive/SemanticDraw_Results'))
SKIP_EXISTING = False        # Để đo Time(s) sạch, benchmark chính nên để False.
MAX_DISPLAY_RESULTS = 0
BASE_SEED = 2024
BASE_SEED = int(os.environ.get("ANCHORDRAW_BASE_SEED", str(BASE_SEED)))
NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image"

RUN_PROFILES = {
    "smoke_bs8": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "smoke" / "coco_val2017_multidiffusion_coco_all_512x512_smoke_bs8.jsonl",
        "run_label": "smoke_bs8",
        "expected_samples": 8,
    },
    "mini32": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "mini32" / "coco_val2017_multidiffusion_coco_all_512x512_mini32.jsonl",
        "run_label": "mini32",
        "expected_samples": 32,
    },
    "mini128": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "mini128" / "coco_val2017_multidiffusion_coco_all_512x512_mini128.jsonl",
        "run_label": "mini128",
        "expected_samples": 128,
    },
    "full1073": {
        "manifest": REPO_ROOT / "Ours" / "data_manifests" / "coco_val2017_multidiffusion_coco_all_512x512_all.jsonl",
        "run_label": "full1073",
        "expected_samples": 1073,
    },
}

VRAM_PROFILES = {
    "low_vram": {
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        "metric_batch_size": 1,
        "clip_batch_size": 4,
    },
    "high_vram_24gb": {
        "batch_size": 2,
        "num_workers": 2,
        "pin_memory": True,
        "metric_batch_size": 4,
        "clip_batch_size": 16,
    },
    "a100_80gb": {
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,
        "metric_batch_size": 8,
        "clip_batch_size": 32,
    },
}

DDIM_PROFILES = {
    "ref_g75_b20": {
        "config_label": "g75_b20",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "bootstrapping": 20,
    },
    "debug_g75_b0": {
        "config_label": "g75_b0_debug",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "bootstrapping": 0,
    },
    "debug_g75_b1": {
        "config_label": "g75_b1_debug",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "bootstrapping": 1,
    },
}

if RUN_PROFILE not in RUN_PROFILES:
    raise ValueError(f"RUN_PROFILE không hợp lệ: {RUN_PROFILE}")
if COLAB_GPU_MODE not in VRAM_PROFILES:
    raise ValueError(f"COLAB_GPU_MODE không hợp lệ: {COLAB_GPU_MODE}")
if DDIM_EXPERIMENT_PROFILE not in DDIM_PROFILES:
    raise ValueError(f"DDIM_EXPERIMENT_PROFILE không hợp lệ: {DDIM_EXPERIMENT_PROFILE}")

run_cfg = RUN_PROFILES[RUN_PROFILE]
vram_cfg = VRAM_PROFILES[COLAB_GPU_MODE]
ddim_cfg = DDIM_PROFILES[DDIM_EXPERIMENT_PROFILE]

MANIFEST_PATH = run_cfg["manifest"]
RUN_LABEL = run_cfg["run_label"]
EXPECTED_SAMPLES = run_cfg["expected_samples"]

BATCH_SIZE = vram_cfg["batch_size"]
NUM_WORKERS = vram_cfg["num_workers"]
PIN_MEMORY = vram_cfg["pin_memory"]
METRIC_BATCH_SIZE = vram_cfg["metric_batch_size"]
CLIP_BATCH_SIZE = vram_cfg["clip_batch_size"]

MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_FAMILY = "sd15"
TARGET_SIZE = (512, 512)
DDIM_NUM_INFERENCE_STEPS = ddim_cfg["num_inference_steps"]
DDIM_GUIDANCE_SCALE = ddim_cfg["guidance_scale"]
BOOTSTRAPPING = ddim_cfg["bootstrapping"]
DDIM_CONFIG_LABEL = ddim_cfg["config_label"]

if PRECISION_PROFILE == "fp16_colab":
    LOAD_DTYPE = torch.float16
    USE_AUTOCAST = True
elif PRECISION_PROFILE == "fp32_original_autocast":
    LOAD_DTYPE = None
    USE_AUTOCAST = True
else:
    raise ValueError(f"PRECISION_PROFILE không hợp lệ: {PRECISION_PROFILE}")

COCO_ROOT = Path(str(WORK_DIR / 'coco'))
MASK_CACHE_DIR = Path(str(WORK_DIR / 'multidiffusion_sd15_ddim_mask_cache'))
EXPERIMENT_ID = f"multidiffusion_sd15_ddim_ref_{DDIM_CONFIG_LABEL}_{RUN_LABEL}_{COLAB_GPU_MODE}"
OUTPUT_DIR = Path(str(WORK_DIR / 'anchordraw_runs')) / EXPERIMENT_ID
GENERATED_IMAGES_DIR = OUTPUT_DIR / "generated_images"
OVERLAY_DIR = OUTPUT_DIR / "mask_overlays"
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"
METRIC_EXPORT_DIR = Path(str(WORK_DIR / 'anchordraw_metric_exports')) / EXPERIMENT_ID
RUN_SUMMARY_PATH = OUTPUT_DIR / "generation_summary.json"

for path in (OUTPUT_DIR, GENERATED_IMAGES_DIR, OVERLAY_DIR, METRICS_OUTPUT_DIR, METRIC_EXPORT_DIR):
    path.mkdir(parents=True, exist_ok=True)

print("Model     : SD1.5")
print(f"Checkpoint: {MODEL_ID}")
print("Sampler   : DDIMScheduler")
print(f"Steps     : {DDIM_NUM_INFERENCE_STEPS}")
print(f"Guidance  : {DDIM_GUIDANCE_SCALE}")
print(f"Bootstrap : {BOOTSTRAPPING}")
print(f"Resolution: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
print(f"Manifest  : {MANIFEST_PATH}")
print(f"Profile   : {RUN_PROFILE}")
print(f"VRAM mode : {COLAB_GPU_MODE}")
print(f"Batch size: {BATCH_SIZE} (dataloader only, generation vẫn chạy từng ảnh)")
print(f"Precision : {PRECISION_PROFILE}")
print(f"Output    : {OUTPUT_DIR}")
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 5 ---
# Import dataloader của Ours và wrapper MultiDiffusion DDIM Ref.
OURS_SRC = REPO_ROOT / "Ours" / "src"
if str(OURS_SRC) in sys.path:
    sys.path.remove(str(OURS_SRC))
sys.path.insert(0, str(OURS_SRC))

from data import COCORegionConfig, build_coco_region_dataloader
from data.visualize import make_mask_overlay
from baselines import MultiDiffusionDDIM

BASELINE_REGION_FILE = REPO_ROOT / "Baseline" / "MultiDiffusion-master" / "MultiDiffusion-master" / "region_based.py"
if not BASELINE_REGION_FILE.exists():
    raise FileNotFoundError(BASELINE_REGION_FILE)
region_sha256 = hashlib.sha256(BASELINE_REGION_FILE.read_bytes()).hexdigest()
print(f"[OK] Baseline file: {BASELINE_REGION_FILE}")
print(f"[OK] Baseline region_based.py SHA256: {region_sha256}")
print("[OK] Imported Ours wrapper: MultiDiffusionDDIM")

# --- Source notebook cell 6 ---
# Tạo dataloader từ manifest đã build sẵn.
# REBUILD_MANIFEST mặc định False vì manifest chính đã nằm trong Ours/data_manifests hoặc Ours/test_sets/manifests.

coco_config = COCORegionConfig(
    coco_root=COCO_ROOT,
    split="val2017",
    manifest_path=MANIFEST_PATH,
    profile="multidiffusion_coco_all",
    model_family=MODEL_FAMILY,
    target_size=TARGET_SIZE,
    return_image=True,
    build_manifest_if_missing=REBUILD_MANIFEST,
    cache_resized_masks=True,
    cache_dir=MASK_CACHE_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY and torch.cuda.is_available(),
    persistent_workers=NUM_WORKERS > 0,
)
loader = build_coco_region_dataloader(coco_config, shuffle=False, drop_last=False)
dataset_size = len(loader.dataset)

print(f"[OK] Manifest records: {dataset_size}")
print(f"[OK] Expected samples: {EXPECTED_SAMPLES}")
if dataset_size != EXPECTED_SAMPLES:
    print("[WARN] Số record khác expected_samples. Hãy kiểm tra RUN_PROFILE hoặc manifest path.")

# --- Source notebook cell 7 ---
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


def make_multidiffusion_payload(batch: dict, index: int) -> dict:
    valid = batch["valid_regions"][index]
    p = int(valid.sum().item())
    fg_masks = batch["masks"][index, :p].to(dtype=torch.float32)
    fg_union = fg_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    bg_mask = (1 - fg_union).clamp(0, 1)
    masks = torch.cat([bg_mask, fg_masks], dim=0)

    foreground_prompts = list(batch["foreground_prompts"][index][:p])
    category_names = list(batch["category_names"][index][:p])
    background_prompt = str(batch["background_prompts"][index])
    prompts = [background_prompt] + foreground_prompts
    negative_prompts = [NEGATIVE_PROMPT for _ in prompts]
    target_h = int(batch["target_sizes"][index, 0].item())
    target_w = int(batch["target_sizes"][index, 1].item())
    record = batch["metadata"][index]

    return {
        "sample_id": str(batch["sample_ids"][index]),
        "image_id": int(batch["image_ids"][index]),
        "file_name": str(batch["file_names"][index]),
        "background_prompt": background_prompt,
        "foreground_prompts": foreground_prompts,
        "category_names": category_names,
        "annotation_ids": [int(v) for v in batch["annotation_ids"][index][:p]],
        "area_ratios": [float(v) for v in batch["area_ratios"][index, :p].tolist()],
        "masks": masks,
        "foreground_masks": fg_masks,
        "prompts": prompts,
        "negative_prompts": negative_prompts,
        "height": target_h,
        "width": target_w,
        "metadata": record,
    }


def display_multidiffusion_result(payload: dict, original: Image.Image, overlay: Image.Image, generated: Image.Image, elapsed: float, generated_path: Path) -> None:
    rows = [{"Region": "Background", "Prompt": payload["background_prompt"], "Annotation": "-", "Area ratio": "-"}]
    for name, prompt, ann_id, area in zip(
        payload["category_names"],
        payload["foreground_prompts"],
        payload["annotation_ids"],
        payload["area_ratios"],
    ):
        rows.append({"Region": name, "Prompt": prompt, "Annotation": ann_id, "Area ratio": f"{area:.4f}"})

    display(Markdown(
        f"### `{payload['sample_id']}`\n"
        f"- image_id: `{payload['image_id']}`\n"
        f"- file: `{payload['file_name']}`\n"
        f"- prompt/mask count: `{len(payload['prompts'])}`\n"
        f"- generated path: `{generated_path}`\n"
        f"- elapsed: `{elapsed:.2f}s`\n"
        f"- generated stats: `{image_stats(generated)}`"
    ))
    display(pd.DataFrame(rows))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original)
    axes[0].set_title("COCO original resized")
    axes[1].imshow(overlay)
    axes[1].set_title("Foreground mask overlay")
    axes[2].imshow(generated)
    axes[2].set_title("MultiDiffusion Ref. + SD1.5 DDIM generated")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# --- Source notebook cell 8 ---
# Load MultiDiffusion SD1.5 + DDIM Ref.
if not torch.cuda.is_available():
    raise RuntimeError("Notebook này cần GPU CUDA trên Colab.")

device = torch.device("cuda:0")
print(f"[OK] CUDA device: {torch.cuda.get_device_name(0)}")
print(f"[OK] LOAD_DTYPE: {LOAD_DTYPE}")

seed_everything(BASE_SEED)
md_ddim = MultiDiffusionDDIM(
    device=device,
    sd_version="1.5",
    model_id=MODEL_ID,
    dtype=LOAD_DTYPE,
    use_autocast=USE_AUTOCAST,
    enable_attention_slicing=True,
    enable_vae_slicing=True,
    show_progress=False,
)
print("[OK] MultiDiffusionDDIM is ready.")

# --- Source notebook cell 9 ---
# Sanity check nhẹ: full-mask single prompt để kiểm tra checkpoint/scheduler/VAE/wrapper.
# Đây không phải kết quả benchmark, chỉ để phát hiện lỗi đen/NaN/noise rõ ràng trước khi chạy manifest.
RUN_SANITY_STEPS = min(20, DDIM_NUM_INFERENCE_STEPS)

if RUN_SANITY_CHECK:
    seed_everything(BASE_SEED)
    sanity_mask = torch.ones(1, 1, TARGET_SIZE[0], TARGET_SIZE[1])
    t0 = time.perf_counter()
    sanity_image = md_ddim.generate(
        masks=sanity_mask,
        prompts=["a studio photo of a teddy bear on a clean table"],
        negative_prompts=[NEGATIVE_PROMPT],
        height=TARGET_SIZE[0],
        width=TARGET_SIZE[1],
        num_inference_steps=RUN_SANITY_STEPS,
        guidance_scale=DDIM_GUIDANCE_SCALE,
        bootstrapping=0,
        show_progress=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[SANITY] elapsed:", round(time.perf_counter() - t0, 2), "s")
    print("[SANITY] image stats:", image_stats(sanity_image))
    display(sanity_image.resize((384, 384)))
else:
    print("[INFO] Sanity check skipped.")

# --- Source notebook cell 10 ---
failed_images = []
# Chạy generation cho từng sample trong manifest.
# Lưu ý: BATCH_SIZE ở đây chỉ là dataloader batch size. Generation vẫn gọi từng ảnh một để giữ logic MultiDiffusion gốc.

summary = []
global_index = 0

for batch_index, batch in enumerate(loader):
    batch_count = len(batch["sample_ids"])
    print(f"[BATCH] {batch_index + 1}/{len(loader)} - {batch_count} sample(s)")

    for local_index in range(batch_count):
        retry_overhead_sec = 0.0
        for attempt in range(1, 4):
            attempt_started = time.perf_counter()
            tracked_lists = ('summary',)
            list_lengths_before = {name: len(globals()[name]) for name in tracked_lists}
            try:
                payload = make_multidiffusion_payload(batch, local_index)
                seed = BASE_SEED + global_index
                stem = f"{global_index:04d}_{payload['sample_id']}"
                generated_path = GENERATED_IMAGES_DIR / f"{stem}_generated.png"
                overlay_path = OVERLAY_DIR / f"{stem}_overlay.png"

                original = batch["images"][local_index].convert("RGB").resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
                overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"])
                overlay.save(overlay_path)

                if SKIP_EXISTING and generated_path.exists():
                    generated = Image.open(generated_path).convert("RGB")
                    elapsed = 0.0
                    skipped = True
                else:
                    seed_everything(seed)
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    generated = md_ddim.generate(
                        masks=payload["masks"],
                        prompts=payload["prompts"],
                        negative_prompts=payload["negative_prompts"],
                        height=payload["height"],
                        width=payload["width"],
                        num_inference_steps=DDIM_NUM_INFERENCE_STEPS,
                        guidance_scale=DDIM_GUIDANCE_SCALE,
                        bootstrapping=BOOTSTRAPPING,
                        show_progress=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0 + retry_overhead_sec
                    generated.save(generated_path)
                    skipped = False

                stats = image_stats(generated)
                row = {
                    "index": global_index,
                    "batch_index": batch_index,
                    "local_index": local_index,
                    "sample_id": payload["sample_id"],
                    "image_id": payload["image_id"],
                    "file_name": payload["file_name"],
                    "seed": seed,
                    "method": "multidiffusion",
                    "variant": "ref_original_region_based",
                    "model_family": MODEL_FAMILY,
                    "model_id": MODEL_ID,
                    "sampler": "ddim",
                    "scheduler": "DDIMScheduler",
                    "num_inference_steps": DDIM_NUM_INFERENCE_STEPS,
                    "guidance_scale": DDIM_GUIDANCE_SCALE,
                    "bootstrapping": BOOTSTRAPPING,
                    "precision_profile": PRECISION_PROFILE,
                    "load_dtype": str(LOAD_DTYPE),
                    "height": payload["height"],
                    "width": payload["width"],
                    "num_regions_including_background": len(payload["prompts"]),
                    "num_foreground_regions": len(payload["foreground_prompts"]),
                    "background_prompt": payload["background_prompt"],
                    "foreground_prompts": payload["foreground_prompts"],
                    "category_names": payload["category_names"],
                    "annotation_ids": payload["annotation_ids"],
                    "area_ratios": payload["area_ratios"],
                    "elapsed_sec": elapsed,
                    "skipped_existing": skipped,
                    "generated_path": str(generated_path),
                    "overlay_path": str(overlay_path),
                    "generated_stats": stats,
                }
                summary.append(row)

                with RUN_SUMMARY_PATH.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

                if MAX_DISPLAY_RESULTS is None or global_index < MAX_DISPLAY_RESULTS:
                    display_multidiffusion_result(payload, original, overlay, generated, elapsed, generated_path)

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

summary_path = RUN_SUMMARY_PATH
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

metric_manifest_path = OUTPUT_DIR / "metric_generated_manifest.jsonl"
with metric_manifest_path.open("w", encoding="utf-8") as f:
    for row in summary:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

display(Markdown(
    f"## Done\n"
    f"Generated `{len(summary)}` image(s) from `{dataset_size}` manifest record(s).\n\n"
    f"Summary saved to `{summary_path}`.\n\n"
    f"Generated images folder: `{GENERATED_IMAGES_DIR}`."
))
summary[:5]

# --- Source notebook cell 11 ---
# Đo metric sau generation nếu RUN_METRICS=True.
# Với smoke thì metric chỉ để test code, không dùng để báo cáo khoa học.

if RUN_METRICS:
    from metrics.config import MetricEvaluationConfig
    from metrics.evaluator import run_evaluation
    from metrics.reporting import write_metrics_report

    metric_config = MetricEvaluationConfig(
        manifest_path=MANIFEST_PATH,
        coco_root=COCO_ROOT,
        generated_dir=GENERATED_IMAGES_DIR,
        generation_summary=RUN_SUMMARY_PATH,
        output_dir=METRICS_OUTPUT_DIR,
        model_family=MODEL_FAMILY,
        target_size=TARGET_SIZE,
        metrics=("fid", "is", "clip_fg", "clip_bg", "time"),
        batch_size=METRIC_BATCH_SIZE,
        num_workers=0,
        device="auto",
        clip_batch_size=CLIP_BATCH_SIZE,
        is_splits=10,
    )
    report = run_evaluation(metric_config)
    json_path, csv_path = write_metrics_report(report, METRICS_OUTPUT_DIR, prefix="metrics")
    print(f"[OK] Metrics JSON: {json_path}")
    print(f"[OK] Metrics CSV: {csv_path}")
    display(pd.DataFrame([{"Metric": k, "Value": v} for k, v in report["metrics"].items()]))
else:
    print("[INFO] RUN_METRICS=False, bỏ qua metric trong lần chạy này.")
