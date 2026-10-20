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

# Notebook: Ours/kaggle_semantic_anchor_experiments/semantic_anchor_sd15_lcm_w03_l04_sigmad4_full1073_kaggle.ipynb
# Notebook SHA256: 8dfee8c61078d915f20695e984c816d1982a9d50572e7b9fd955ad21a059b690

# --- Source notebook cell 4 ---
# Cấu hình full manifest 1073 cho benchmark AnchorDraw SD1.5 + LCM + L04 + WM-03 + sigma_D=4.0.
from pathlib import Path

RUN_MANIFEST = REPO_ROOT / "Ours" / "data_manifests" / "coco_val2017_multidiffusion_coco_all_512x512_all.jsonl"
COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'COCO')))
OUTPUT_DIR = Path(str(WORK_DIR / 'semantic_anchor_sd15_lcm_w03_l04_sigmad4_full1073_coord_aligned_r1_outputs'))
MASK_CACHE_DIR = Path(str(WORK_DIR / 'semantic_anchor_sd15_mask_cache'))
METRICS_OUTPUT_DIR = Path(str(WORK_DIR / 'semantic_anchor_sd15_lcm_w03_l04_sigmad4_full1073_coord_aligned_r1_metrics'))

# Model theo baseline SD1.5.
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# --- THÔNG SỐ RUN ANCHORDRAW (MASK_STD KHỚP NOTEBOOK BASELINE CHẠY LẠI) ---
TARGET_SIZE = (512, 512)
BATCH_SIZE = 8
BASE_SEED = 2024
BOOTSTRAP_STEPS = 1
MASK_STD = 1.0  # Chuan sweep: 1.0 de Gaussian lowpass tao gradient lam mo
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = "discrete"
NEGATIVE_PROMPT = ""

# --- CẤU HÌNH PHƯƠNG PHÁP MỚI ANCHORDRAW ---
METHOD_ID = "WM-03-TOPK-L04-SIGMA4"
METHOD_LABEL = "AnchorDraw (Top-k + Adaptive Bilateral, L04, sigmaD=4.0)"
BEST_ANCHOR_MODE = "semantic_topk_anchor"
TOPK_ATTENTION_PERCENT = 10.0
WEIGHT_POLICY = "adaptive_bilateral"
FIXED_LAYER = {
    "layer_id": "L04",
    "layer_name": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor",  # L04 chuan (16x16)
}
SIGMA_D = 4.0
SEMANTIC_SIGMA_SCALE = 1.0

# Chỉ hiển thị 8 preview để Kaggle notebook không bị nặng.
# Ảnh vẫn được sinh và lưu đủ theo full manifest 1073.
MAX_DISPLAY_RESULTS = 0

# Cấu hình đo Metric (cần kiểm tra đồng nhất khi so sánh baseline)
METRIC_NAMES = ("fid", "is", "clip_fg", "clip_bg", "time")
METRIC_BATCH_SIZE = 8
CLIP_BATCH_SIZE = 16
IS_SPLITS = 10
METRICS_REPORT_PREFIX = "semantic_anchor_sd15_lcm_w03_l04_sigmad4_full1073_coord_aligned_r1_metrics"

assert RUN_MANIFEST.exists(), f"Missing manifest: {RUN_MANIFEST}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[OK] Manifest: {RUN_MANIFEST}")
print(f"[OK] COCO root: {COCO_ROOT}")
print(f"[OK] Output dir: {OUTPUT_DIR}")
print(f"[OK] Metrics output dir: {METRICS_OUTPUT_DIR}")
print(f"[OK] Batch size: {BATCH_SIZE} (Sync với Baseline)")
print(f"[OK] Base seed: {BASE_SEED} (Sync với Baseline)")
print(f"[OK] Method: {METHOD_ID} | Layer: {FIXED_LAYER['layer_id']} | sigma_D: {SIGMA_D}")
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 6 ---
# Import dataloader của Ours, SemanticDrawPipeline và core AnchorDraw modules.
import sys
import importlib.util
import json
import time
import types

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
# terminal display provided above

OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_SRC = REPO_ROOT / "Baseline" / "semantic-draw-main" / "src"

sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from data.visualize import make_mask_overlay
from experiments.semantic_anchor import (
    ATTENTION_COORDINATE_POLICY,
    SemanticAnchorCapture,
    SemanticAnchorRuntime,
    find_target_token_indices,
    compute_anchor_measurements,
    aggregate_attention_maps,
)
assert ATTENTION_COORDINATE_POLICY == "inverse_roll_image_after_latent_shift_r1"

# Load trực tiếp file pipeline_semantic_draw.py để tránh import toàn bộ model/__init__.py
# vì các file SDXL/SD3 có thể cần dependency khác không dùng trong SD1.5.
sys.path.insert(0, str(BASELINE_SRC))
pipeline_path = BASELINE_SRC / "model" / "pipeline_semantic_draw.py"
spec = importlib.util.spec_from_file_location("pipeline_semantic_draw", pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline

print("[OK] All imports (DataLoader + Pipeline + AnchorDraw core) are ready.")

# --- Source notebook cell 7 ---
# Tạo dataloader cho full manifest 1073; baseline lịch sử dùng MASK_STD=0, notebook baseline chạy lại dùng 1.
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

def select_layer_attention_maps(captured_maps, output_size, layer_name):
    selected_maps = {}
    for key, layer_maps in captured_maps.items():
        matches = [layer_map for layer_map in layer_maps if layer_map.layer_name == layer_name]
        if len(matches) != 1:
            available = [layer_map.layer_name for layer_map in layer_maps]
            raise RuntimeError(f"Expected exactly one map for {layer_name} at {key}; got {len(matches)}. Available: {available}")
        values = matches[0].values.detach().float().cpu()[None, None]
        values = F.interpolate(values, size=output_size, mode="bilinear", align_corners=False)[0, 0]
        selected_maps[key] = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)
    return selected_maps

def install_runtime_patches(runtime, target_layer_name):
    """Install Layer L04 selection with support for both argmax and topk."""
    def anchors_from_current_step(self, timestep, foreground_masks, *, strategy, topk_percent, layer_index=None, layer_name=None, **kwargs):
        eff_layer = layer_name or target_layer_name
        maps = select_layer_attention_maps(self.attention_capture.maps, self.image_size, eff_layer)
        anchors = []
        for region_index, mask in enumerate(foreground_masks):
            measurement = compute_anchor_measurements(
                maps[(int(timestep), region_index)], mask.cpu(), topk_percent=topk_percent
            )
            if strategy == "argmax":
                anchors.append((float(measurement["anchor_x"]), float(measurement["anchor_y"])))
            else:
                anchors.append((float(measurement["topk_anchor_x"]), float(measurement["topk_anchor_y"])))
        return anchors
    runtime._anchors_from_current_step = types.MethodType(anchors_from_current_step, runtime)

def display_smoke_result(payload: dict, original: Image.Image, overlay: Image.Image, generated: Image.Image, elapsed: float, generated_path: Path) -> None:
    rows = ["| Region | Prompt | Annotation | Area ratio |", "|---|---|---:|---:|"]
    rows.append(f"| Background | {md_escape(payload['prompts'][0])} | - | - |")
    for label, prompt, ann_id, area in zip(payload["category_names"], payload["foreground_prompts"], payload["annotation_ids"], payload["area_ratios"]):
        rows.append(f"| {md_escape(label)} | {md_escape(prompt)} | {ann_id} | {float(area):.4f} |")

    display(Markdown(
        f"### `{payload['sample_id']}`\n"
        f"- image_id: `{payload['image_id']}`\n"
        f"- file: `{payload['file_name']}`\n"
        f"- method: `{METHOD_ID}`\n"
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
    axes[2].set_title(f"AnchorDraw generated ({METHOD_ID})")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

print("[OK] Helper functions (Payload + Layer Selector + Visualization) are ready.")

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
# Guard chống lỗi torchao/PEFT trước khi load LCM LoRA.
import sys
import importlib
import importlib.util

importlib.invalidate_caches()

if "torchao" in sys.modules:
    raise RuntimeError(
        "torchao is already imported in this Python session. Restart the Kaggle session, run the dependency cell, then Run All."
    )

if importlib.util.find_spec("torchao") is not None:
    raise RuntimeError(
        "torchao is still installed/importable in this runtime. Run the dependency cell, then restart the Kaggle session and Run All."
    )

print("[OK] torchao is not importable; PEFT should skip torchao LoRA dispatch.")

# --- Source notebook cell 12 ---
# Load SemanticDraw SD1.5 pipeline làm nền tảng cho AnchorDraw.
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

# Không bật attention slicing để giữ nguyên vẹn attention maps như Colab sweep

print("[OK] SemanticDrawPipeline loaded successfully.")

# --- Source notebook cell 13 ---
# Kiểm tra scheduler và xác nhận layer L04 tồn tại trong UNet.
print("Scheduler:", type(smd.scheduler).__name__)
print("Default guidance scale:", smd.default_guidance_scale)
print("Default timesteps:", [int(t) for t in smd.timesteps.cpu().tolist()])

assert type(smd.scheduler).__name__ == "LCMScheduler", "Expected LCM sampler via LCMScheduler."
available_layer_names = set(smd.unet.attn_processors)
assert FIXED_LAYER["layer_name"] in available_layer_names, f"Missing {FIXED_LAYER['layer_name']} in UNet"
print(f"[OK] Fixed layer {FIXED_LAYER['layer_id']} verified: {FIXED_LAYER['layer_name']}")

# --- Source notebook cell 14 ---
# Chạy generation cho mọi sample trong manifest bằng thuật toán AnchorDraw.
summary = []
global_index = 0

for batch_index, batch in enumerate(loader):
    print(f"[BATCH] {batch_index + 1}/{len(loader)} - {len(batch['sample_ids'])} sample(s)")

    for local_index, sample_id in enumerate(batch["sample_ids"]):
        payload = make_semanticdraw_payload(batch, local_index)
        original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
        overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)

        # Dùng cùng quy tắc seed với baseline: BASE_SEED (2024) + global_index
        seed = BASE_SEED + global_index
        seed_everything(seed)

        # Xác định token vị trí cho từng đối tượng tiền cảnh
        token_indices = [
            find_target_token_indices(smd.tokenizer, prompt, category)
            for prompt, category in zip(payload["foreground_prompts"], payload["category_names"])
        ]

        tic = time.perf_counter()
        with SemanticAnchorCapture(smd.unet) as capture:
            capture.configure(token_indices)
            runtime = SemanticAnchorRuntime(smd, capture, image_size=TARGET_SIZE)
            install_runtime_patches(runtime, FIXED_LAYER["layer_name"])

            generated, runtime_records = runtime.generate(
                prompts=payload["prompts"],
                negative_prompts=payload["negative_prompts"],
                masks=payload["all_masks"].to(device=device, dtype=torch.float32),
                foreground_masks=payload["foreground_masks"],
                mode=BEST_ANCHOR_MODE,
                weight_policy=WEIGHT_POLICY,
                bootstrap_steps=BOOTSTRAP_STEPS,
                topk_percent=TOPK_ATTENTION_PERCENT,
                spatial_sigma_latent=float(SIGMA_D),
                semantic_sigma_scale=float(SEMANTIC_SIGMA_SCALE),
                mask_stds=MASK_STD,
                mask_strengths=MASK_STRENGTH,
                preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
                attention_layer_name=FIXED_LAYER["layer_name"],
            )
        elapsed = time.perf_counter() - tic

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
            "method_id": METHOD_ID,
            "attention_coordinate_policy": ATTENTION_COORDINATE_POLICY,
            "anchor_mode": BEST_ANCHOR_MODE,
            "weight_policy": WEIGHT_POLICY,
            "fixed_layer": FIXED_LAYER["layer_id"],
            "spatial_sigma_latent": float(SIGMA_D),
            "semantic_sigma_scale": float(SEMANTIC_SIGMA_SCALE),
            "bootstrap_steps": BOOTSTRAP_STEPS,
            "num_regions_including_background": len(payload["prompts"]),
            "elapsed_sec": elapsed,
            "generated_path": str(generated_path),
            "overlay_path": str(overlay_path),
        })

        should_display = MAX_DISPLAY_RESULTS is None or global_index < MAX_DISPLAY_RESULTS
        if should_display:
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
summary[:5]

# --- Source notebook cell 17 ---
# Giải phóng VRAM trước khi load Inception/CLIP cho metric.
import gc

for var_name in ("smd", "runtime", "capture", "generated", "payload", "overlay", "original", "batch", "loader", "preview_batch"):
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
    {"Metric": "Method", "Value": METHOD_ID},
    {"Metric": "FID↓", "Value": fmt(values.get("fid"))},
    {"Metric": "IS↑", "Value": fmt(values.get("is_mean"))},
    {"Metric": "IS std", "Value": fmt(values.get("is_std"))},
    {"Metric": "CLIP(fg)↑", "Value": fmt(values.get("clip_fg_x100"))},
    {"Metric": "CLIP(bg)↑", "Value": fmt(values.get("clip_bg_x100"))},
    {"Metric": "Time(s)↓", "Value": fmt(values.get("time_mean_sec"))},
    {"Metric": "Total time(s)", "Value": fmt(values.get("time_total_sec"))},
])

display(Markdown(
    f"## Metric Evaluation Done ({METHOD_ID})\n"
    f"- evaluated: `{metric_report['num_evaluated']}` / `{metric_report['num_manifest_records']}` samples\n"
    f"- missing generated images: `{metric_report['num_missing_generated']}`\n"
    f"- metrics JSON: `{metrics_json}`\n"
    f"- metrics CSV: `{metrics_csv}`"
))
display(metrics_table)

metric_report
