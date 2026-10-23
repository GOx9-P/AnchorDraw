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

# Notebook: Ours/kaggle_multidiffusion_experiments/multidiffusion_sd15_lcm_full1073_kaggle.ipynb
# Notebook SHA256: 0229072683ac25f2eb5cf2a659d3010f356bdcca8389a069bb13fb0fe9c4b058

# --- Source notebook cell 4 ---
# Runtime profile for MultiDiffusion SD1.5 + LCM.
# RUN_PROFILE selects the data split; LCM_EXPERIMENT_PROFILE selects bootstrap on/off.
# RUN_PROFILE options: "smoke", "mini32", "mini128", "full1073".
# LCM_EXPERIMENT_PROFILE options: "g1_b1", "g1_b0".
RUN_PROFILE = os.environ.get('RUN_PROFILE', 'full1073')
LCM_EXPERIMENT_PROFILE = os.environ.get('LCM_EXPERIMENT_PROFILE', 'g1_b0')

PROFILE_CONFIGS = {
    "smoke": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "smoke" / "coco_val2017_multidiffusion_coco_all_512x512_smoke_bs8.jsonl",
        "expected_samples": 8,
        "label": "smoke_bs8",
        "max_display_results": 8,
    },
    "mini32": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "mini32" / "coco_val2017_multidiffusion_coco_all_512x512_mini32.jsonl",
        "expected_samples": 32,
        "label": "mini32",
        "max_display_results": 8,
    },
    "mini128": {
        "manifest": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "mini128" / "coco_val2017_multidiffusion_coco_all_512x512_mini128.jsonl",
        "expected_samples": 128,
        "label": "mini128",
        "max_display_results": 8,
    },
    "full1073": {
        "manifest": REPO_ROOT / "Ours" / "data_manifests" / "coco_val2017_multidiffusion_coco_all_512x512_all.jsonl",
        "expected_samples": 1073,
        "label": "full1073",
        "max_display_results": 8,
    },
}

LCM_EXPERIMENT_CONFIGS = {
    "g1_b1": {
        "checklist_id": "CFG-MD-SD15-LCM-G1-B1",
        "label": "g1_b1",
        "guidance_scale": 1.0,
        "bootstrapping": 1,
        "description": "Guidance 1.0 + random-background bootstrap 1 step. Already measured on full1073.",
    },
    "g1_b0": {
        "checklist_id": "CFG-MD-SD15-LCM-G1-B0",
        "label": "g1_b0",
        "guidance_scale": 1.0,
        "bootstrapping": 0,
        "description": "Guidance 1.0 + bootstrapping disabled. Used to isolate random-background bootstrap effect.",
    },
}

assert RUN_PROFILE in PROFILE_CONFIGS, f"Unknown RUN_PROFILE={RUN_PROFILE!r}. Choose one of {tuple(PROFILE_CONFIGS)}"
assert LCM_EXPERIMENT_PROFILE in LCM_EXPERIMENT_CONFIGS, f"Unknown LCM_EXPERIMENT_PROFILE={LCM_EXPERIMENT_PROFILE!r}. Choose one of {tuple(LCM_EXPERIMENT_CONFIGS)}"
RUN_CONFIG = PROFILE_CONFIGS[RUN_PROFILE]
LCM_EXPERIMENT_CONFIG = LCM_EXPERIMENT_CONFIGS[LCM_EXPERIMENT_PROFILE]

RUN_MANIFEST = RUN_CONFIG["manifest"]
RUN_LABEL = RUN_CONFIG["label"]
EXPECTED_SAMPLES = RUN_CONFIG["expected_samples"]
CONFIG_ID = str(LCM_EXPERIMENT_CONFIG["checklist_id"])
LCM_CONFIG_LABEL = str(LCM_EXPERIMENT_CONFIG["label"])
LCM_GUIDANCE_SCALE = float(LCM_EXPERIMENT_CONFIG["guidance_scale"])
BOOTSTRAPPING = int(LCM_EXPERIMENT_CONFIG["bootstrapping"])

COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'COCO')))
OUTPUT_DIR = (WORK_DIR / f"multidiffusion_sd15_lcm_{LCM_CONFIG_LABEL}_{RUN_LABEL}_outputs")
MASK_CACHE_DIR = Path(str(WORK_DIR / 'multidiffusion_sd15_lcm_mask_cache'))
METRICS_OUTPUT_DIR = (WORK_DIR / f"multidiffusion_sd15_lcm_{LCM_CONFIG_LABEL}_{RUN_LABEL}_metrics")
EXPERIMENT_ID = f"multidiffusion_sd15_lcm_{LCM_CONFIG_LABEL}_{RUN_LABEL}"

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
LCM_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"

TARGET_SIZE = (512, 512)
BATCH_SIZE = 8
BASE_SEED = 2024
BASE_SEED = int(os.environ.get("ANCHORDRAW_BASE_SEED", str(BASE_SEED)))

LCM_NUM_SCHEDULE_STEPS = 50
LCM_T_INDEX_LIST = [0, 4, 12, 25, 37]

SHOW_INNER_PROGRESS = False
CHECKPOINT_EVERY_IMAGES = 1
RESUME_GENERATION = True
SKIP_EXISTING_IMAGES_WITHOUT_SUMMARY = False
LOG_EVERY_IMAGES = 8

NEGATIVE_PROMPT = ""
MAX_DISPLAY_RESULTS = 0

METRIC_NAMES = ("fid", "is", "clip_fg", "clip_bg", "time")
METRIC_BATCH_SIZE = 8
CLIP_BATCH_SIZE = 16
IS_SPLITS = 10
METRICS_REPORT_PREFIX = f"{EXPERIMENT_ID}_metrics"

assert RUN_MANIFEST.exists(), f"Missing manifest: {RUN_MANIFEST}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[OK] Run profile: {RUN_PROFILE} ({RUN_LABEL})")
print(f"[OK] LCM experiment profile: {LCM_EXPERIMENT_PROFILE}")
print(f"[OK] Config ID: {CONFIG_ID}")
print(f"[OK] Config description: {LCM_EXPERIMENT_CONFIG['description']}")
print(f"[OK] Guidance scale: {LCM_GUIDANCE_SCALE}")
print(f"[OK] Bootstrapping: {BOOTSTRAPPING}")
print(f"[OK] Manifest: {RUN_MANIFEST}")
print(f"[OK] Expected samples: {EXPECTED_SAMPLES}")
print(f"[OK] COCO root: {COCO_ROOT}")
print(f"[OK] Output dir: {OUTPUT_DIR}")
print(f"[OK] Metrics output dir: {METRICS_OUTPUT_DIR}")
print(f"[OK] Batch size: {BATCH_SIZE}")
print(f"[OK] Experiment ID: {EXPERIMENT_ID}")
print(f"[OK] Show inner progress: {SHOW_INNER_PROGRESS}")
print(f"[OK] Checkpoint every images: {CHECKPOINT_EVERY_IMAGES}")
print(f"[OK] Resume generation: {RESUME_GENERATION}")
print(f"[OK] Skip existing images without summary: {SKIP_EXISTING_IMAGES_WITHOUT_SUMMARY}")
print(f"[OK] Log every images: {LOG_EVERY_IMAGES}")
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 6 ---
# Import Ours dataloader, metric package và MultiDiffusion LCM wrapper.
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
# terminal display provided above
from PIL import Image

OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_REGION_FILE = REPO_ROOT / "Baseline" / "MultiDiffusion-master" / "MultiDiffusion-master" / "region_based.py"

# Đưa Ours/src lên trước để import đúng package data/metrics/baselines.
sys.path = [str(OURS_SRC)] + [p for p in sys.path if p != str(OURS_SRC)]

from baselines import MultiDiffusionLCM
from data import COCORegionConfig, batch_item_to_semanticdraw_inputs, build_coco_region_dataloader
from data.visualize import make_mask_overlay

print("[OK] Imports are ready.")
if BASELINE_REGION_FILE.exists():
    region_sha256 = hashlib.sha256(BASELINE_REGION_FILE.read_bytes()).hexdigest()
    print("[OK] Baseline MultiDiffusion region file:", BASELINE_REGION_FILE)
    print("[OK] Baseline region_based.py SHA256:", region_sha256)
else:
    print("[WARN] Baseline MultiDiffusion region_based.py not found in this clone; running Ours/src/baselines implementation.")

# --- Source notebook cell 7 ---
# Tạo dataloader theo RUN_PROFILE đã chọn.
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
    axes[2].set_title("MultiDiffusion + LCM generated")
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
# Guard chống lỗi torchao/PEFT trước khi load LCM LoRA.
# Nếu bạn vừa gặp lỗi torchao ở cell load pipeline, hãy restart Kaggle session rồi Run All.
importlib.invalidate_caches()

if "torchao" in sys.modules:
    raise RuntimeError(
        "torchao is already imported in this Python session. Restart the Kaggle session, run the dependency cell, then Run All. "
        "MultiDiffusion SD1.5 + LCM LoRA does not need torchao."
    )

if importlib.util.find_spec("torchao") is not None:
    raise RuntimeError(
        "torchao is still installed/importable in this runtime. Run the dependency cell, then restart the Kaggle session and Run All. "
        "MultiDiffusion SD1.5 + LCM LoRA does not need torchao."
    )

print("[OK] torchao is not importable; PEFT should skip torchao LoRA dispatch.")

# --- Source notebook cell 12 ---
# Load MultiDiffusion + LCM model.
seed_everything(BASE_SEED)
md_lcm = MultiDiffusionLCM(
    model_id=MODEL_ID,
    lcm_lora_id=LCM_LORA_ID,
    lora_weight_name=LCM_LORA_WEIGHT_NAME,
    device=device,
    dtype=dtype,
    t_index_list=LCM_T_INDEX_LIST,
    schedule_steps=LCM_NUM_SCHEDULE_STEPS,
    show_progress=SHOW_INNER_PROGRESS,
)

print("Scheduler:", type(md_lcm.scheduler).__name__)
print("Model:", MODEL_ID)
print("LCM LoRA:", LCM_LORA_ID)
print("Schedule steps:", LCM_NUM_SCHEDULE_STEPS)
print("t_index_list:", LCM_T_INDEX_LIST)
print("timesteps:", [int(t.item()) for t in md_lcm.timesteps])
print("config_id:", CONFIG_ID)
print("lcm_experiment_profile:", LCM_EXPERIMENT_PROFILE)
print("guidance_scale:", LCM_GUIDANCE_SCALE)
print("bootstrapping:", BOOTSTRAPPING)
print("show_progress:", SHOW_INNER_PROGRESS)
assert type(md_lcm.scheduler).__name__ == "LCMScheduler", "Expected LCMScheduler."

# --- Source notebook cell 13 ---
# Sanity check: một prompt đơn giản với full mask để kiểm tra checkpoint/scheduler/LoRA/VAE.
seed_everything(BASE_SEED)
sanity_mask = torch.ones(1, 1, TARGET_SIZE[0], TARGET_SIZE[1])
sanity_image = md_lcm.generate(
    masks=sanity_mask,
    prompts=["a studio photo of a teddy bear on a clean table"],
    negative_prompts=[NEGATIVE_PROMPT],
    height=TARGET_SIZE[0],
    width=TARGET_SIZE[1],
    guidance_scale=LCM_GUIDANCE_SCALE,
    bootstrapping=0,
)
print("[SANITY] image stats:", image_stats(sanity_image))
display(sanity_image.resize((384, 384)))

# --- Source notebook cell 14 ---
failed_images = []
# Run generation for every sample in the manifest and display readable previews.
summary_path = OUTPUT_DIR / "generation_summary.json"
summary: list[dict] = []
resumed_count = 0
existing_image_only_count = 0
new_generated_count = 0


def is_valid_image(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_previous_summary() -> list[dict]:
    if not RESUME_GENERATION or not summary_path.exists():
        return []
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
    except Exception as exc:
        print(f"[WARN] Could not read previous summary, will generate from scratch: {exc!r}")
    return []


def write_generation_checkpoint() -> None:
    tmp_path = summary_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    tmp_path.replace(summary_path)


def build_previous_index(rows: list[dict]) -> dict[str, dict]:
    index = {}
    for row in rows:
        sample_id = row.get("sample_id")
        generated_path = row.get("generated_path")
        if sample_id is None or not generated_path:
            continue
        path = Path(str(generated_path))
        if is_valid_image(path):
            index[str(sample_id)] = row
    return index


def find_existing_generated_image(global_index: int, sample_id: str) -> Path | None:
    candidates = [
        OUTPUT_DIR / f"{global_index:04d}_{sample_id}_generated.png",
        OUTPUT_DIR / f"{global_index:02d}_{sample_id}_generated.png",
    ]
    candidates.extend(sorted(OUTPUT_DIR.glob(f"*{sample_id}*_generated.png")))
    for path in candidates:
        if is_valid_image(path):
            return path
    return None


previous_summary = load_previous_summary()
previous_by_sample_id = build_previous_index(previous_summary)
if previous_by_sample_id:
    print(f"[RESUME] Found {len(previous_by_sample_id)} valid previous generation record(s).")
elif RESUME_GENERATION:
    print("[RESUME] No valid previous summary records found; generation starts from current manifest order.")

for batch_index, batch in enumerate(loader):
    print(f"[BATCH] {batch_index + 1}/{len(loader)} - {len(batch['sample_ids'])} sample(s)")

    for local_index, sample_id in enumerate(batch["sample_ids"]):
        retry_overhead_sec = 0.0
        for attempt in range(1, 4):
            attempt_started = time.perf_counter()
            tracked_lists = ('summary',)
            list_lengths_before = {name: len(globals()[name]) for name in tracked_lists}
            counters_before = {name: globals()[name] for name in ("new_generated_count", "resumed_count", "existing_image_only_count")}
            try:
                global_index = batch_index * BATCH_SIZE + local_index
                sample_id = str(sample_id)
                payload = make_multidiffusion_payload(batch, local_index)
                stem = f"{global_index:04d}_{payload['sample_id']}"
                generated_path = OUTPUT_DIR / f"{stem}_generated.png"
                overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"

                previous_row = previous_by_sample_id.get(sample_id)
                if RESUME_GENERATION and previous_row is not None:
                    row = dict(previous_row)
                    row["resumed"] = True
                    summary.append(row)
                    resumed_count += 1
                    if LOG_EVERY_IMAGES > 0 and resumed_count % LOG_EVERY_IMAGES == 0:
                        print(f"[RESUME] skipped {resumed_count} existing image(s); last={sample_id}")
                    break

                existing_path = find_existing_generated_image(global_index, sample_id)
                if RESUME_GENERATION and SKIP_EXISTING_IMAGES_WITHOUT_SUMMARY and existing_path is not None:
                    row = {
                        "index": global_index,
                        "batch_index": batch_index,
                        "local_index": local_index,
                        "sample_id": payload["sample_id"],
                        "image_id": payload["image_id"],
                        "file_name": payload["file_name"],
                        "seed": BASE_SEED + global_index,
                        "config_id": CONFIG_ID,
                        "lcm_experiment_profile": LCM_EXPERIMENT_PROFILE,
                        "method": "multidiffusion",
                        "model_family": "sd15",
                        "model_id": MODEL_ID,
                        "sampler": "lcm",
                        "scheduler": type(md_lcm.scheduler).__name__,
                        "lcm_lora": LCM_LORA_ID,
                        "lcm_t_index_list": LCM_T_INDEX_LIST,
                        "num_schedule_steps": LCM_NUM_SCHEDULE_STEPS,
                        "guidance_scale": LCM_GUIDANCE_SCALE,
                        "bootstrapping": BOOTSTRAPPING,
                        "num_regions_including_background": len(payload["prompts"]),
                        "elapsed_sec": None,
                        "generated_path": str(existing_path),
                        "overlay_path": str(overlay_path),
                        "resumed": True,
                        "resume_source": "existing_image_without_summary",
                    }
                    summary.append(row)
                    existing_image_only_count += 1
                    if CHECKPOINT_EVERY_IMAGES > 0 and len(summary) % CHECKPOINT_EVERY_IMAGES == 0:
                        write_generation_checkpoint()
                    if LOG_EVERY_IMAGES > 0 and existing_image_only_count % LOG_EVERY_IMAGES == 0:
                        print(f"[RESUME] skipped {existing_image_only_count} image file(s) without previous timing; last={sample_id}")
                    break

                original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
                overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)

                seed = BASE_SEED + global_index
                seed_everything(seed)

                tic = time.perf_counter()
                generated = md_lcm.generate(
                    masks=payload["all_masks"],
                    prompts=payload["prompts"],
                    negative_prompts=payload["negative_prompts"],
                    height=payload["height"],
                    width=payload["width"],
                    guidance_scale=LCM_GUIDANCE_SCALE,
                    bootstrapping=BOOTSTRAPPING,
                    show_progress=SHOW_INNER_PROGRESS,
                )
                elapsed = time.perf_counter() - tic + retry_overhead_sec

                generated.save(generated_path)
                overlay.save(overlay_path)
                if not is_valid_image(generated_path):
                    raise RuntimeError(f"Generated image is invalid: {generated_path}")

                summary.append({
                    "index": global_index,
                    "batch_index": batch_index,
                    "local_index": local_index,
                    "sample_id": payload["sample_id"],
                    "image_id": payload["image_id"],
                    "file_name": payload["file_name"],
                    "seed": seed,
                    "config_id": CONFIG_ID,
                    "lcm_experiment_profile": LCM_EXPERIMENT_PROFILE,
                    "method": "multidiffusion",
                    "model_family": "sd15",
                    "model_id": MODEL_ID,
                    "sampler": "lcm",
                    "scheduler": type(md_lcm.scheduler).__name__,
                    "lcm_lora": LCM_LORA_ID,
                    "lcm_t_index_list": LCM_T_INDEX_LIST,
                    "num_schedule_steps": LCM_NUM_SCHEDULE_STEPS,
                    "guidance_scale": LCM_GUIDANCE_SCALE,
                    "bootstrapping": BOOTSTRAPPING,
                    "num_regions_including_background": len(payload["prompts"]),
                    "elapsed_sec": elapsed,
                    "generated_path": str(generated_path),
                    "overlay_path": str(overlay_path),
                    "resumed": False,
                })
                new_generated_count += 1

                if CHECKPOINT_EVERY_IMAGES > 0 and len(summary) % CHECKPOINT_EVERY_IMAGES == 0:
                    write_generation_checkpoint()

                should_display = MAX_DISPLAY_RESULTS is None or global_index < MAX_DISPLAY_RESULTS
                if should_display:
                    display_multidiffusion_result(payload, original, overlay, generated, elapsed, generated_path)

                if LOG_EVERY_IMAGES > 0 and new_generated_count % LOG_EVERY_IMAGES == 0:
                    print(f"[PROGRESS] new={new_generated_count}, resumed={resumed_count}, existing_image_only={existing_image_only_count}, total_logged={len(summary)}/{dataset_size}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                import traceback
                rolled_back = any(len(globals()[name]) > list_lengths_before[name] for name in tracked_lists)
                for name in tracked_lists:
                    del globals()[name][list_lengths_before[name]:]
                for name, value in counters_before.items():
                    globals()[name] = value
                if rolled_back:
                    try:
                        write_generation_checkpoint()
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

if failed_images:
    raise RuntimeError(f"Run incomplete: {len(failed_images)} manifest indices failed after three attempts: {failed_images}")

write_generation_checkpoint()

missing = dataset_size - len(summary)
display(Markdown(
    f"## Done\n"
    f"Generated new `{new_generated_count}` image(s). "
    f"Resumed `{resumed_count}` summary record(s). "
    f"Skipped `{existing_image_only_count}` image file(s) without timing. "
    f"Summary rows: `{len(summary)}` / `{dataset_size}`. "
    f"Missing: `{missing}`. "
    f"Summary saved to `{summary_path}`."
))
summary[:5]

# --- Source notebook cell 17 ---
# Giải phóng VRAM trước khi load Inception/CLIP cho metric.
import gc

for var_name in ("md_lcm", "sanity_image", "generated", "payload", "overlay", "original", "batch", "loader", "preview_batch"):
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
# Đo FID, IS, CLIP(fg), CLIP(bg), Time(s) trên output theo RUN_PROFILE vừa generate.
from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report

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
