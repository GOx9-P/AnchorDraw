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

# Notebook: Ours/colab_semanticdraw_smoke/semanticdraw_sd15_lcm_semanticfull_colab.ipynb
# Notebook SHA256: a9ff78cf0db091ff6f61351126b8fb23ba23427701115bda3c29ec80619e8048

# --- Source notebook cell 3 ---
# 2. Cấu hình thí nghiệm.
# Mặc định chạy full trên GPU mạnh như A100 80GB. Nếu chỉ validate nhanh thì đổi về RUN_PROFILE="smoke" và LOW_VRAM=True.
from pathlib import Path
import json

RUN_PROFILE = os.environ.get('RUN_PROFILE', 'full_val2017') # "smoke" hoặc "full_val2017"
LOW_VRAM = False             # False cho GPU mạnh như A100 80GB; True cho T4/L4 hoặc validate nhẹ
REBUILD_MANIFEST = False      # Đặt True nếu vừa đổi rule sampling, ví dụ MAX_OBJECTS_PER_IMAGE.
RUN_SANITY_CHECK = True       # Chạy một ảnh test nhanh để kiểm tra model/scheduler/VAE.
RUN_METRICS = True          # Metric nội bộ; protocol CLIP hiện tại chưa hoàn toàn khớp paper.
RUN_EXPORT_ZIP = False         # Tạo zip chứa ảnh sinh + manifest export để tải về.
SKIP_EXISTING = True          # Nếu ảnh đã tồn tại thì dùng lại, hữu ích khi resume.

COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'COCO')))
BASE_OUTPUT_DIR = Path(str(WORK_DIR / 'anchordraw_runs'))

MODEL_ID = "runwayml/stable-diffusion-v1-5"
TARGET_SIZE = (512, 512)
BASE_SEED = 2024
MANIFEST_SEED = 2026
SMOKE_SUBSET_SIZE = 8
SMOKE_MAX_OBJECTS_PER_IMAGE = 8
FULL_MAX_OBJECTS_PER_IMAGE = 80  # COCO val2017 hiện có tối đa 62 mask object hợp lệ trong một ảnh.
MAX_OBJECTS_PER_IMAGE = SMOKE_MAX_OBJECTS_PER_IMAGE if (RUN_PROFILE == "smoke" or LOW_VRAM) else FULL_MAX_OBJECTS_PER_IMAGE
DROP_ISCROWD = True

BATCH_SIZE = 1 if LOW_VRAM else 8
METRIC_BATCH_SIZE = 1 if LOW_VRAM else 8
CLIP_BATCH_SIZE = 4 if LOW_VRAM else 16
NUM_WORKERS = 0

BOOTSTRAP_STEPS = 1
MASK_STD = 0.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = "discrete"
NEGATIVE_PROMPT = ""
NUM_INFERENCE_STEPS = None   # None nghĩa là dùng default của baseline SemanticDrawPipeline.
GUIDANCE_SCALE = None        # None nghĩa là dùng default của baseline SemanticDrawPipeline.

START_INDEX = 0
MAX_SAMPLES = None           # None nghĩa là chạy hết range đã chọn.
MAX_DISPLAY_RESULTS = 0

# Preset tham chiếu đang khớp với mặc định để chạy full trên A100 80GB.
# Nếu muốn smoke test nhanh, hãy đặt RUN_PROFILE="smoke" và LOW_VRAM=True ở phía trên.
A100_80GB_FULL_PRESET = {
    "RUN_PROFILE": "full_val2017",
    "LOW_VRAM": False,
    "REBUILD_MANIFEST": False,
    "RUN_SANITY_CHECK": True,
    "RUN_METRICS": False,
    "RUN_EXPORT_ZIP": True,
    "SKIP_EXISTING": True,
    "MAX_OBJECTS_PER_IMAGE": FULL_MAX_OBJECTS_PER_IMAGE,
    "BATCH_SIZE": 8,
    "METRIC_BATCH_SIZE": 8,
    "CLIP_BATCH_SIZE": 16,
    "BOOTSTRAP_STEPS": BOOTSTRAP_STEPS,
    "MASK_STD": MASK_STD,
    "MASK_STRENGTH": MASK_STRENGTH,
    "NUM_INFERENCE_STEPS": NUM_INFERENCE_STEPS,
    "GUIDANCE_SCALE": GUIDANCE_SCALE,
    "START_INDEX": 0,
    "MAX_SAMPLES": None,
    "MAX_DISPLAY_RESULTS": 4,
    "ghi_chu": "BATCH_SIZE là batch của dataloader; pipeline baseline vẫn sinh từng sample một.",
}

if RUN_PROFILE == "smoke":
    subset_size = SMOKE_SUBSET_SIZE
elif RUN_PROFILE == "full_val2017":
    subset_size = None
else:
    raise ValueError(f"RUN_PROFILE không được hỗ trợ: {RUN_PROFILE}")

mode_name = "low_vram" if LOW_VRAM else "standard"
run_suffix = "smoke" if RUN_PROFILE == "smoke" else "full_val2017"
RUN_ID = f"semanticdraw_sd15_lcm_semanticfull_512x512_{run_suffix}_{mode_name}"
RUN_ROOT = BASE_OUTPUT_DIR / RUN_ID
GENERATED_IMAGES_DIR = RUN_ROOT / "generated_images"
OVERLAY_DIR = RUN_ROOT / "mask_overlays"
MANIFEST_DIR = RUN_ROOT / "manifests"
MASK_CACHE_DIR = RUN_ROOT / "mask_cache"
METRICS_OUTPUT_DIR = RUN_ROOT / "metrics"
RUN_SUMMARY_PATH = RUN_ROOT / "generation_summary.json"
RUN_CONFIG_PATH = RUN_ROOT / "run_config.json"

MANIFEST_NAME = (
    f"coco_val2017_semanticdraw_sd15_512x512_maxobj{MAX_OBJECTS_PER_IMAGE}_seed{MANIFEST_SEED}_{subset_size}.jsonl"
    if subset_size is not None
    else f"coco_val2017_semanticdraw_sd15_512x512_maxobj{MAX_OBJECTS_PER_IMAGE}_full_val2017.jsonl"
)
RUN_MANIFEST = MANIFEST_DIR / MANIFEST_NAME

for path in [RUN_ROOT, GENERATED_IMAGES_DIR, OVERLAY_DIR, MANIFEST_DIR, MASK_CACHE_DIR, METRICS_OUTPUT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

run_config = {
    "run_id": RUN_ID,
    "run_profile": RUN_PROFILE,
    "low_vram": LOW_VRAM,
    "model_id": MODEL_ID,
    "sampler": "LCMScheduler qua baseline SemanticDrawPipeline",
    "resolution": list(TARGET_SIZE),
    "manifest_path": str(RUN_MANIFEST),
    "coco_root": str(COCO_ROOT),
    "subset_size": subset_size,
    "max_objects_per_image": MAX_OBJECTS_PER_IMAGE,
    "smoke_max_objects_per_image": SMOKE_MAX_OBJECTS_PER_IMAGE,
    "full_max_objects_per_image": FULL_MAX_OBJECTS_PER_IMAGE,
    "drop_iscrowd": DROP_ISCROWD,
    "batch_size": BATCH_SIZE,
    "metric_batch_size": METRIC_BATCH_SIZE,
    "clip_batch_size": CLIP_BATCH_SIZE,
    "bootstrap_steps": BOOTSTRAP_STEPS,
    "mask_std": MASK_STD,
    "mask_strength": MASK_STRENGTH,
    "mask_type": MASK_TYPE,
    "start_index": START_INDEX,
    "max_samples": MAX_SAMPLES,
    "a100_80gb_full_preset": A100_80GB_FULL_PRESET,
}
with RUN_CONFIG_PATH.open("w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)

print("[CẤU HÌNH ĐANG CHẠY]")
print(json.dumps(run_config, ensure_ascii=False, indent=2))
print("\n[PRESET FULL CHO A100 80GB]")
print(json.dumps(A100_80GB_FULL_PRESET, ensure_ascii=False, indent=2))

if RUN_PROFILE == "full_val2017":
    print("[WARN] Output full được lưu trong runtime Colab. Hãy tải file zip export trước khi kết thúc session.")
    if MAX_OBJECTS_PER_IMAGE < FULL_MAX_OBJECTS_PER_IMAGE:
        print("[WARN] LOW_VRAM đang giới hạn số mask mỗi ảnh. Dùng LOW_VRAM=False nếu muốn chạy full object-mask.")
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 5 ---
# 4. Import dataloader của Ours và pipeline SemanticDraw baseline gốc.
import sys
import importlib.util
import time
import math
import csv
import shutil
import gc

import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
# terminal display provided above

OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_SRC = REPO_ROOT / "Baseline" / "semantic-draw-main" / "src"

sys.path.insert(0, str(OURS_SRC))
from data import semanticdraw_sd15, build_coco_manifest, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from data.visualize import make_mask_overlay

sys.path.insert(0, str(BASELINE_SRC))
pipeline_path = BASELINE_SRC / "model" / "pipeline_semantic_draw.py"
spec = importlib.util.spec_from_file_location("pipeline_semantic_draw_original", pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline

print("[OK] Import xong.")
print(f"[OK] File baseline: {pipeline_path}")

# --- Source notebook cell 6 ---
# 5. Tạo manifest SemanticDraw-full và khởi tạo dataloader.
# Cell này dùng profile SemanticDraw, không dùng filter MultiDiffusion COCO-all chặt hơn.
base_config = semanticdraw_sd15(
    COCO_ROOT,
    seed=MANIFEST_SEED,
    subset_size=subset_size,
    manifest_path=RUN_MANIFEST,
)

config = base_config.copy_with(
    instances_json=COCO_ROOT / "annotations" / "instances_val2017.json",
    captions_json=COCO_ROOT / "annotations" / "captions_val2017.json",
    min_objects=1,
    max_objects=MAX_OBJECTS_PER_IMAGE,
    truncate_objects=True,
    exclude_categories=(),
    min_mask_area_ratio=0.0,
    drop_iscrowd=DROP_ISCROWD,
    prompt_template="a {label}",
    caption_policy="first",
    object_policy="largest",
    return_image=True,
    cache_resized_masks=True,
    cache_dir=MASK_CACHE_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=False,
    persistent_workers=False,
)

records = build_coco_manifest(config, overwrite=REBUILD_MANIFEST)
loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
dataset_size = len(loader.dataset)
num_batches = len(loader)
preview_batch = next(iter(loader))

print(f"[OK] Đường dẫn manifest: {RUN_MANIFEST}")
print(f"[OK] Số record trong manifest: {len(records)}")
print(f"[OK] Số record trong dataloader: {dataset_size}")
print(f"[OK] Số batch dataloader: {num_batches} x tối đa {BATCH_SIZE}")
print(f"[OK] Shape tensor mask batch đầu: {tuple(preview_batch['masks'].shape)}")
print("Các sample_id đầu tiên:")
for sample_id in preview_batch["sample_ids"][:8]:
    print(" -", sample_id)

# --- Source notebook cell 7 ---
# 6. Các hàm hỗ trợ: chuẩn bị input baseline, hiển thị, resume và kiểm tra ảnh.
def md_escape(text: object) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def image_stats(image: Image.Image) -> dict:
    import numpy as np
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def make_semanticdraw_payload(batch: dict, index: int) -> dict:
    item = batch_item_to_semanticdraw_inputs(batch, index)
    metadata = item["metadata"]

    # Khớp với notebook SemanticDraw SD1.5 + LCM 1073 trước đó:
    # caption COCO được đưa vào như một vùng background rõ ràng.
    fg_masks = item["masks"].float().cpu()  # (P, 1, H, W)
    fg_union = fg_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    background_mask = (1.0 - fg_union).clamp(0, 1)
    all_masks = torch.cat([background_mask, fg_masks], dim=0)

    prompts = [item["background_prompt"], *item["prompts"]]
    negative_prompts = [NEGATIVE_PROMPT for _ in prompts]

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


def generated_path_for(index: int, sample_id: str) -> Path:
    return GENERATED_IMAGES_DIR / f"{index:06d}_{sample_id}_generated.png"


def overlay_path_for(index: int, sample_id: str) -> Path:
    return OVERLAY_DIR / f"{index:06d}_{sample_id}_overlay.png"


def display_result(payload: dict, original: Image.Image, overlay: Image.Image, generated: Image.Image, elapsed, generated_path: Path) -> None:
    rows = ["| Vùng | Prompt | Annotation | Tỷ lệ diện tích |", "|---|---|---:|---:|"]
    rows.append(f"| Background | {md_escape(payload['background_prompt'])} | - | - |")
    for label, prompt, ann_id, area in zip(payload["category_names"], payload["foreground_prompts"], payload["annotation_ids"], payload["area_ratios"]):
        rows.append(f"| {md_escape(label)} | {md_escape(prompt)} | {ann_id} | {float(area):.4f} |")

    elapsed_text = "dùng lại ảnh cũ" if elapsed is None else f"{elapsed:.2f}s"
    display(Markdown(
        f"### `{payload['sample_id']}`\n"
        f"- image_id: `{payload['image_id']}`\n"
        f"- file: `{payload['file_name']}`\n"
        f"- số prompt/mask: `{len(payload['prompts'])}`\n"
        f"- đường dẫn ảnh sinh: `{generated_path}`\n"
        f"- thời gian: `{elapsed_text}`\n"
        f"- thống kê ảnh sinh: `{image_stats(generated)}`\n\n"
        + "\n".join(rows)
    ))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original)
    axes[0].set_title("Ảnh COCO gốc đã resize")
    axes[1].imshow(overlay)
    axes[1].set_title("Overlay foreground mask")
    axes[2].imshow(generated)
    axes[2].set_title("Ảnh sinh bởi SemanticDraw SD1.5 LCM")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def load_existing_summary() -> dict[int, dict]:
    if not RUN_SUMMARY_PATH.exists():
        return {}
    with RUN_SUMMARY_PATH.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for row in rows:
        if "index" in row:
            out[int(row["index"])] = row
    return out


print("[OK] Các hàm hỗ trợ đã sẵn sàng.")

# --- Source notebook cell 8 ---
# 7. Đăng nhập Hugging Face nếu có token, sau đó load pipeline baseline.
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
        print("[OK] Đã nạp Hugging Face token.")
    else:
        print("[INFO] Không thấy HF_TOKEN. Notebook sẽ dùng quyền truy cập public.")


assert torch.cuda.is_available(), "Colab chưa bật GPU. Vào Runtime > Change runtime type > GPU."
device = torch.device("cuda:0")
dtype = torch.float16
maybe_login_to_huggingface()
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

# Chặn lỗi tương thích torchao/PEFT trước khi nạp LCM LoRA.
import importlib
import importlib.util
importlib.invalidate_caches()
if "torchao" in sys.modules or importlib.util.find_spec("torchao") is not None:
    raise RuntimeError("torchao vẫn còn import được. Hãy restart runtime, chạy lại cell cài thư viện, rồi Run all.")

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

if hasattr(smd.pipe, "enable_attention_slicing"):
    smd.pipe.enable_attention_slicing()
if LOW_VRAM and hasattr(smd.pipe, "enable_vae_slicing"):
    smd.pipe.enable_vae_slicing()

print("[OK] SemanticDrawPipeline đã sẵn sàng.")

# --- Source notebook cell 9 ---
# 8. Kiểm tra notebook đang dùng đúng baseline SD1.5 + LCM.
print("Họ model: SD1.5")
print("Model id:", MODEL_ID)
print("Scheduler:", type(smd.scheduler).__name__)
print("Scheduler trong pipeline:", type(smd.pipe.scheduler).__name__)
print("Số bước inference mặc định:", smd.default_num_inference_steps)
print("Guidance scale mặc định:", smd.default_guidance_scale)
print("Kích thước sinh:", TARGET_SIZE)
print("Manifest:", RUN_MANIFEST)
print("Số record:", dataset_size)

assert type(smd.scheduler).__name__ == "LCMScheduler", "Notebook phải dùng LCMScheduler từ baseline SemanticDraw SD1.5."

# --- Source notebook cell 10 ---
# 9. Sanity check tùy chọn với một mask phủ toàn ảnh.
if RUN_SANITY_CHECK:
    seed_everything(BASE_SEED)
    sanity_mask = torch.ones(1, 1, TARGET_SIZE[0], TARGET_SIZE[1], dtype=torch.float32, device=device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    tic = time.perf_counter()
    sanity_image = smd(
        prompts=["a studio photo of a teddy bear on a clean table"],
        negative_prompts=[NEGATIVE_PROMPT],
        masks=sanity_mask,
        height=TARGET_SIZE[0],
        width=TARGET_SIZE[1],
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        bootstrap_steps=BOOTSTRAP_STEPS,
        mask_stds=MASK_STD,
        mask_strengths=MASK_STRENGTH,
        preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
        do_blend=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[SANITY] Thời gian:", round(time.perf_counter() - tic, 2), "giây")
    print("[SANITY] Thống kê ảnh:", image_stats(sanity_image))
    display(sanity_image.resize((384, 384)))
else:
    print("[INFO] Bỏ qua sanity check.")

# --- Source notebook cell 11 ---
# 10. Sinh ảnh cho manifest đã chọn.
summary_by_index = load_existing_summary()
end_index = dataset_size if MAX_SAMPLES is None else min(dataset_size, START_INDEX + int(MAX_SAMPLES))
print(f"[CHẠY] index {START_INDEX} đến {end_index - 1} trên tổng {dataset_size} record")

stop = False
global_index = 0

for batch_index, batch in enumerate(loader):
    print(f"[BATCH] {batch_index + 1}/{len(loader)} - {len(batch['sample_ids'])} sample")

    for local_index, sample_id in enumerate(batch["sample_ids"]):
        if global_index < START_INDEX:
            global_index += 1
            continue
        if global_index >= end_index:
            stop = True
            break

        payload = make_semanticdraw_payload(batch, local_index)
        sample_id = payload["sample_id"]
        generated_path = generated_path_for(global_index, sample_id)
        overlay_path = overlay_path_for(global_index, sample_id)

        original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
        overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)
        overlay.save(overlay_path)

        seed = BASE_SEED + global_index
        elapsed = None
        skipped_existing = False

        if SKIP_EXISTING and generated_path.exists():
            generated = Image.open(generated_path).convert("RGB")
            skipped_existing = True
        else:
            seed_everything(seed)
            masks = payload["all_masks"].to(device=device, dtype=torch.float32)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            tic = time.perf_counter()
            generated = smd(
                prompts=payload["prompts"],
                negative_prompts=payload["negative_prompts"],
                masks=masks,
                height=payload["height"],
                width=payload["width"],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                mask_stds=MASK_STD,
                mask_strengths=MASK_STRENGTH,
                bootstrap_steps=BOOTSTRAP_STEPS,
                preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
                do_blend=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - tic
            generated.save(generated_path)

        row = {
            "index": global_index,
            "batch_index": batch_index,
            "local_index": local_index,
            "sample_id": sample_id,
            "image_id": payload["image_id"],
            "file_name": payload["file_name"],
            "seed": seed,
            "model_family": "sd15",
            "model_id": MODEL_ID,
            "sampler": "LCMScheduler",
            "scheduler": type(smd.scheduler).__name__,
            "num_regions_including_background": len(payload["prompts"]),
            "num_foreground_regions": len(payload["foreground_prompts"]),
            "background_prompt": payload["background_prompt"],
            "foreground_prompts": payload["foreground_prompts"],
            "category_names": payload["category_names"],
            "annotation_ids": payload["annotation_ids"],
            "area_ratios": payload["area_ratios"],
            "elapsed_sec": elapsed,
            "skipped_existing": skipped_existing,
            "generated_stats": image_stats(generated),
            "generated_path": str(generated_path),
            "overlay_path": str(overlay_path),
            "source_manifest_path": str(RUN_MANIFEST),
        }
        summary_by_index[global_index] = row

        if MAX_DISPLAY_RESULTS is None or global_index < START_INDEX + MAX_DISPLAY_RESULTS:
            display_result(payload, original, overlay, generated, elapsed, generated_path)

        with RUN_SUMMARY_PATH.open("w", encoding="utf-8") as f:
            json.dump([summary_by_index[k] for k in sorted(summary_by_index)], f, ensure_ascii=False, indent=2)

        global_index += 1
        del generated, overlay, original
        torch.cuda.empty_cache()

    if stop:
        break

summary = [summary_by_index[k] for k in sorted(summary_by_index)]
with RUN_SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

num_in_range = max(0, end_index - START_INDEX)
display(Markdown(
    f"## Hoàn tất\n"
    f"Đã sinh hoặc dùng lại `{len([r for r in summary if START_INDEX <= int(r['index']) < end_index])}` dòng cho range đang chọn.\n"
    f"Kích thước range: `{num_in_range}`. Tổng số dòng summary hiện có: `{len(summary)}`.\n"
    f"Summary đã lưu tại `{RUN_SUMMARY_PATH}`."
))
pd.DataFrame(summary).tail()

# --- Source notebook cell 14 ---
# 12. Đo metric tùy chọn.
if RUN_METRICS:
    for var_name in ("smd", "preview_batch", "batch"):
        if var_name in globals():
            del globals()[var_name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report

    metric_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    metric_config = MetricEvaluationConfig(
        manifest_path=RUN_MANIFEST,
        coco_root=COCO_ROOT,
        generated_dir=GENERATED_IMAGES_DIR,
        generation_summary=RUN_SUMMARY_PATH,
        output_dir=METRICS_OUTPUT_DIR,
        model_family="sd15",
        target_size=TARGET_SIZE,
        metrics=("fid", "is", "clip_fg", "clip_bg", "time"),
        batch_size=METRIC_BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        device=metric_device,
        clip_batch_size=CLIP_BATCH_SIZE,
        is_splits=10,
    )

    metric_report = run_evaluation(metric_config)
    metrics_json, metrics_csv = write_metrics_report(metric_report, METRICS_OUTPUT_DIR, prefix=f"{RUN_ID}_metrics")
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
        {"Metric": "FID", "Value": fmt(values.get("fid"))},
        {"Metric": "IS", "Value": fmt(values.get("is_mean"))},
        {"Metric": "IS std", "Value": fmt(values.get("is_std"))},
        {"Metric": "CLIP(fg) x100", "Value": fmt(values.get("clip_fg_x100"))},
        {"Metric": "CLIP(bg) x100", "Value": fmt(values.get("clip_bg_x100"))},
        {"Metric": "Time mean sec", "Value": fmt(values.get("time_mean_sec"))},
        {"Metric": "Total time sec", "Value": fmt(values.get("time_total_sec"))},
    ])

    display(Markdown(
        f"## Đo metric hoàn tất\n"
        f"- đã đánh giá: `{metric_report['num_evaluated']}` / `{metric_report['num_manifest_records']}`\n"
        f"- số ảnh sinh bị thiếu: `{metric_report['num_missing_generated']}`\n"
        f"- metrics JSON: `{metrics_json}`\n"
        f"- metrics CSV: `{metrics_csv}`"
    ))
    display(metrics_table)
else:
    print("[INFO] RUN_METRICS=False nên bỏ qua bước đo metric.")
