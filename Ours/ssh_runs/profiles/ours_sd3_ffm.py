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

# Notebook: Ours/semantic_anchor_experiments/anchordraw_sd3_flashflowmatch_l04_sigmad4_colab.ipynb
# Notebook SHA256: 889e3d2df2c61994c415e58f6a9829961e93fce1e8c46f7fc9d830b60602b040

# --- Source notebook cell 6 ---
from pathlib import Path
import json
import os
import hashlib

# Mặc định chạy smoke 2 ảnh; chỉ đổi sang full1073 sau khi kiểm tra trên GPU.
RUN_PROFILE = os.environ.get('RUN_PROFILE', 'smoke_bs2')  # choices: "smoke_bs2", "mini32", "full1073"

# SD3 1024 khá nặng. BATCH_SIZE chỉ là dataloader batch, generation vẫn chạy tuần tự từng ảnh.
COLAB_GPU_MODE = os.environ.get('COLAB_GPU_MODE', 'high_vram')  # choices: "low_vram", "high_vram"
LOW_VRAM_MODE = COLAB_GPU_MODE == "low_vram"

MANIFEST_BY_PROFILE = {
    "smoke_bs2": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "smoke" / "coco_val2017_multidiffusion_coco_all_sd3_1024x1024_smoke_bs2.jsonl",
    "mini32": REPO_ROOT / "Ours" / "test_sets" / "manifests" / "mini32" / "coco_val2017_multidiffusion_coco_all_sd3_1024x1024_mini32.jsonl",
    "full1073": REPO_ROOT / "Ours" / "data_manifests" / "coco_val2017_multidiffusion_coco_all_sd3_1024x1024_all.jsonl",
}
EXPECTED_DATASET_SIZE_BY_PROFILE = {
    "smoke_bs2": 2,
    "mini32": 32,
    "full1073": 1073,
}

assert RUN_PROFILE in MANIFEST_BY_PROFILE, f"Unknown RUN_PROFILE: {RUN_PROFILE}"
assert COLAB_GPU_MODE in {"low_vram", "high_vram"}, f"Unknown COLAB_GPU_MODE: {COLAB_GPU_MODE}"

RUN_MANIFEST = MANIFEST_BY_PROFILE[RUN_PROFILE]
EXPECTED_DATASET_SIZE = EXPECTED_DATASET_SIZE_BY_PROFILE[RUN_PROFILE]
COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'datasets/coco')))

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
FLASH_SD3_REPO_ID = "jasperai/flash-sd3"
MODEL_REVISION = "ea42f8cef0f178587cf766dc8129abd379c90671"
FLASH_SD3_REVISION = "fb184d916f9193a81a4264758eb0ebbff1c56b69"
SAMPLER_NAME = "FlashFlowMatchEulerDiscreteScheduler"
DIFFUSERS_COMMIT_EXPECTED = "53dcfcf3f3009a79fd484eed9b09ea76189da434"
MODEL_FAMILY = "sd3"
TARGET_SIZE = (1024, 1024)

SD3_BASE_SEEDS = (2024, 10002024, 20002024)
BASE_SEED = int(os.environ.get("ANCHORDRAW_BASE_SEED", "2024"))
assert BASE_SEED in SD3_BASE_SEEDS, f"Use one of the paired SD3 seeds: {SD3_BASE_SEEDS}"
WHITE_BOOTSTRAP_SEED = BASE_SEED
BATCH_SIZE = 1 if RUN_PROFILE == "full1073" else 2
NUM_WORKERS = 1 if LOW_VRAM_MODE else 2

# Baseline SD3 defaults.
SEMANTICDRAW_T_INDEX_LIST = [0, 4, 12, 25, 37]
SEMANTICDRAW_NUM_INFERENCE_STEPS = None
GUIDANCE_SCALE = 0.0
BOOTSTRAP_STEPS = 2
MASK_STD = 0.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.3
MASK_TYPE = "discrete"
NEGATIVE_PROMPT = ""

# Re-encode the cached white bootstrap latent with VAE float32.
# Keep this setting identical in SemanticDraw and AnchorDraw SD3 runs.
PATCH_SAFE_WHITE_BOOTSTRAP_LATENT = True

MAX_DISPLAY_RESULTS = 0
STOP_ON_BAD_OUTPUT = True

# Bật True để copy metric export folder và file zip sang Google Drive sau khi sinh ảnh.
SAVE_EXPORT_TO_GOOGLE_DRIVE = False
RUN_METRICS_AFTER_GENERATION = RUN_PROFILE == "full1073"
GOOGLE_DRIVE_EXPORT_ROOT = Path(str(WORK_DIR / 'drive/MyDrive/SemanticDraw_Results/SD3_flashflowmatch'))

SAMPLE_TAG = {"smoke_bs2": "smoke2", "mini32": "mini32", "full1073": "full1073"}[RUN_PROFILE]
EXPERIMENT_ID = (
    f"ours_sd3_flashflowmatch_1024_{SAMPLE_TAG}_"
    f"b{BATCH_SIZE}_bt{BOOTSTRAP_STEPS}_l04_sigmad4_v2_seed{BASE_SEED}_colab"
)

RUNS_ROOT = Path(str(WORK_DIR / 'anchordraw_runs'))
OUTPUT_DIR = RUNS_ROOT / EXPERIMENT_ID
GENERATED_IMAGES_DIR = OUTPUT_DIR / "generated_images"
OVERLAY_IMAGES_DIR = OUTPUT_DIR / "mask_overlays"
RUN_SUMMARY_PATH = OUTPUT_DIR / "generation_summary.json"
RUN_CONFIG_PATH = OUTPUT_DIR / "run_config.json"
MASK_CACHE_DIR = Path(str(WORK_DIR / 'semanticdraw_mask_cache'))

METRIC_EXPORT_ROOT = Path(str(WORK_DIR / 'anchordraw_metric_exports'))
METRIC_EXPORT_DIR = METRIC_EXPORT_ROOT / f"{EXPERIMENT_ID}__metric_export"

for folder in (OUTPUT_DIR, GENERATED_IMAGES_DIR, OVERLAY_IMAGES_DIR, METRIC_EXPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

ATTENTION_LAYER_INDEX = 4
TOPK_PERCENT = 10.0
SPATIAL_SIGMA_LATENT = 4.0
SEMANTIC_SIGMA_SCALE = 1.0

run_config = {
    "method": "AnchorDraw SD3 joint-attention",
    "attention_layer_index": ATTENTION_LAYER_INDEX,
    "topk_percent": TOPK_PERCENT,
    "spatial_sigma_latent": SPATIAL_SIGMA_LATENT,
    "semantic_sigma_scale": SEMANTIC_SIGMA_SCALE,
    "experiment_id": EXPERIMENT_ID,
    "code_version": "ours_sd3_v2",
    "base_seed": BASE_SEED,
    "white_bootstrap_seed": WHITE_BOOTSTRAP_SEED,
    "negative_prompt": NEGATIVE_PROMPT,
    "num_inference_steps": SEMANTICDRAW_NUM_INFERENCE_STEPS,
    "dtype": "float16",
    "run_profile": RUN_PROFILE,
    "expected_samples": EXPECTED_DATASET_SIZE,
    "resolution": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}",
    "model_family": MODEL_FAMILY,
    "model_id": MODEL_ID,
    "acceleration": FLASH_SD3_REPO_ID,
    "model_revision": MODEL_REVISION,
    "flash_sd3_revision": FLASH_SD3_REVISION,
    "sampler": SAMPLER_NAME,
    "manifest": str(RUN_MANIFEST),
    "batch_size": BATCH_SIZE,
    "bootstrap_steps": BOOTSTRAP_STEPS,
    "t_index_list": SEMANTICDRAW_T_INDEX_LIST,
    "guidance_scale": GUIDANCE_SCALE,
    "mask_std": MASK_STD,
    "mask_strength": MASK_STRENGTH,
    "preprocess_mask_cover_alpha": PREPROCESS_MASK_COVER_ALPHA,
    "mask_type": MASK_TYPE,
    "patch_safe_white_bootstrap_latent": PATCH_SAFE_WHITE_BOOTSTRAP_LATENT,
    "save_export_to_google_drive": SAVE_EXPORT_TO_GOOGLE_DRIVE,
    "run_metrics_after_generation": RUN_METRICS_AFTER_GENERATION,
    "google_drive_export_root": str(GOOGLE_DRIVE_EXPORT_ROOT),
}
# A resumed run must use exactly the same scientific configuration and source.
pipeline_source = REPO_ROOT / "Baseline" / "semantic-draw-main" / "src" / "model" / "pipeline_semantic_draw_3.py"
import importlib.metadata
direct_url_text = importlib.metadata.distribution("diffusers").read_text("direct_url.json")
direct_url = json.loads(direct_url_text or "{}")
diffusers_commit = direct_url.get("vcs_info", {}).get("commit_id")
if diffusers_commit != DIFFUSERS_COMMIT_EXPECTED:
    raise RuntimeError(f"Wrong diffusers commit: {diffusers_commit!r}; expected {DIFFUSERS_COMMIT_EXPECTED}.")
run_config["diffusers_commit"] = diffusers_commit
run_config["dependency_versions"] = {
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
run_config["data_source_sha256"] = data_source_hash.hexdigest()
metrics_source_hash = hashlib.sha256()
for metric_file in sorted((REPO_ROOT / "Ours" / "src" / "metrics").glob("*.py")):
    metrics_source_hash.update(metric_file.name.encode("utf-8"))
    metrics_source_hash.update(metric_file.read_bytes())
run_config["metrics_source_sha256"] = metrics_source_hash.hexdigest()
MASK_CACHE_DIR = MASK_CACHE_DIR / run_config["data_source_sha256"]
run_config["manifest_sha256"] = hashlib.sha256(RUN_MANIFEST.read_bytes()).hexdigest()
run_config["pipeline_sha256"] = hashlib.sha256(pipeline_source.read_bytes()).hexdigest()
controller_source = REPO_ROOT / "Ours" / "src" / "experiments" / "semantic_anchor_sd3.py"
shared_source = REPO_ROOT / "Ours" / "src" / "experiments" / "semantic_anchor.py"
run_config["controller_sha256"] = hashlib.sha256(controller_source.read_bytes()).hexdigest()
run_config["shared_anchor_sha256"] = hashlib.sha256(shared_source.read_bytes()).hexdigest()
immutable_keys = (
    "method", "attention_layer_index", "topk_percent", "spatial_sigma_latent",
    "semantic_sigma_scale", "controller_sha256", "shared_anchor_sha256",
    "code_version", "base_seed", "white_bootstrap_seed", "negative_prompt", "num_inference_steps",
    "dtype", "data_source_sha256", "metrics_source_sha256",
    "run_profile", "expected_samples", "resolution", "model_family", "model_id",
    "diffusers_commit", "dependency_versions",
    "acceleration", "model_revision", "flash_sd3_revision", "sampler", "manifest_sha256", "pipeline_sha256", "bootstrap_steps",
    "t_index_list", "guidance_scale", "mask_std", "mask_strength",
    "preprocess_mask_cover_alpha", "mask_type", "patch_safe_white_bootstrap_latent",
)
signature_payload = {key: run_config[key] for key in immutable_keys}
run_config["run_signature"] = hashlib.sha256(
    json.dumps(signature_payload, sort_keys=True).encode("utf-8")
).hexdigest()
if RUN_CONFIG_PATH.exists():
    previous_config = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
    if previous_config.get("run_signature") != run_config["run_signature"]:
        raise RuntimeError(
            f"Existing output uses a different scientific configuration: {RUN_CONFIG_PATH}. "
            "Select a new EXPERIMENT_ID; do not mix results."
        )
    if previous_config.get("white_latent_sha256"):
        run_config["white_latent_sha256"] = previous_config["white_latent_sha256"]
    if previous_config.get("input_masks_sha256"):
        run_config["input_masks_sha256"] = previous_config["input_masks_sha256"]
    if previous_config.get("coco_annotations_sha256"):
        run_config["coco_annotations_sha256"] = previous_config["coco_annotations_sha256"]
RUN_CONFIG_PATH.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps(run_config, ensure_ascii=False, indent=2))
print("[OK] OUTPUT_DIR:", OUTPUT_DIR)
print("[OK] METRIC_EXPORT_DIR:", METRIC_EXPORT_DIR)
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 10 ---
import sys
import importlib.util
import time
import shutil
import csv

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
# terminal display provided above

assert torch.cuda.is_available(), "SD3 experiment requires GPU/CUDA."
device = "cuda"
dtype = torch.float16
print("[OK] torch:", torch.__version__)
print("[OK] device:", torch.cuda.get_device_name(0))

# The pinned diffusers branch must not run with a stale torchao import.
if "torchao" in sys.modules or importlib.util.find_spec("torchao") is not None:
    raise RuntimeError("torchao is still present in this runtime. Restart after dependency installation.")
# Verify special diffusers branch.
from diffusers.schedulers import FlashFlowMatchEulerDiscreteScheduler
print("[OK] FlashFlowMatchEulerDiscreteScheduler:", FlashFlowMatchEulerDiscreteScheduler)

OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_SRC = REPO_ROOT / "Baseline" / "semantic-draw-main" / "src"
sys.path.insert(0, str(BASELINE_SRC))
sys.path.insert(0, str(OURS_SRC))

from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from data.visualize import make_mask_overlay

pipeline_path = BASELINE_SRC / "model" / "pipeline_semantic_draw_3.py"
spec = importlib.util.spec_from_file_location("pipeline_semantic_draw_3_original", pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDraw3Pipeline = pipeline_module.SemanticDraw3Pipeline

print("[OK] Imported baseline file:", pipeline_path)
print("[OK] Pipeline class:", SemanticDraw3Pipeline)

# --- Source notebook cell 12 ---
def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def md_escape(text: object) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")


def pil_image_stats(image: Image.Image) -> dict:
    import numpy as np

    arr = np.asarray(image.convert("RGB"))
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def is_bad_output(image: Image.Image) -> bool:
    stats = pil_image_stats(image)
    near_black = stats["max"] <= 5 or stats["std"] <= 1.0
    static_noise_like = stats["std"] >= 95 and 90 <= stats["mean"] <= 165
    return near_black or static_noise_like


def make_semanticdraw_payload(batch: dict, index: int) -> dict:
    item = batch_item_to_semanticdraw_inputs(batch, index)
    fg_masks = item["masks"].float().cpu()
    metadata = item["metadata"]
    return {
        "sample_id": metadata["sample_id"],
        "image_id": metadata["image_id"],
        "file_name": metadata["file_name"],
        "height": item["height"],
        "width": item["width"],
        "background_prompt": item["background_prompt"],
        "foreground_prompts": item["prompts"],
        "category_names": metadata["category_names"],
        "annotation_ids": metadata["annotation_ids"],
        "area_ratios": metadata["area_ratios"],
        "foreground_masks": fg_masks,
        "metadata": metadata,
    }


def make_semanticdraw_region_inputs(payload: dict) -> tuple[list[str], list[str], torch.Tensor]:
    """Pass foreground masks only; the corrected baseline appends the background."""
    foreground_masks = payload["foreground_masks"].float().cpu()
    foreground_prompts = list(payload["foreground_prompts"])
    foreground_negative_prompts = [NEGATIVE_PROMPT for _ in foreground_prompts]
    assert foreground_masks.shape[0] == len(foreground_prompts) == len(foreground_negative_prompts)
    return foreground_prompts, foreground_negative_prompts, foreground_masks


def display_result(payload: dict, original: Image.Image, overlay: Image.Image, generated: Image.Image, elapsed: float, generated_path: Path) -> None:
    rows = ["| Region | Prompt | Annotation | Area ratio |", "|---|---|---:|---:|"]
    rows.append(f"| Background | {md_escape(payload['background_prompt'])} | - | - |")
    for label, prompt, ann_id, area in zip(
        payload["category_names"],
        payload["foreground_prompts"],
        payload["annotation_ids"],
        payload["area_ratios"],
    ):
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
    axes[2].set_title("AnchorDraw SD3 generated")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# --- Source notebook cell 14 ---
annotation_hash = hashlib.sha256()
for annotation_path in (
    COCO_ROOT / "annotations" / "instances_val2017.json",
    COCO_ROOT / "annotations" / "captions_val2017.json",
):
    annotation_hash.update(annotation_path.name.encode("utf-8"))
    annotation_hash.update(annotation_path.read_bytes())
coco_annotations_sha256 = annotation_hash.hexdigest()
if run_config.get("coco_annotations_sha256") not in (None, coco_annotations_sha256):
    raise RuntimeError("COCO annotation files changed since this run started.")
run_config["coco_annotations_sha256"] = coco_annotations_sha256
MASK_CACHE_DIR = Path(str(WORK_DIR / 'semanticdraw_mask_cache')) / run_config["data_source_sha256"] / coco_annotations_sha256
RUN_CONFIG_PATH.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
print("[CHECK] COCO annotations SHA256:", coco_annotations_sha256)
config = COCORegionConfig(
    coco_root=COCO_ROOT,
    manifest_path=RUN_MANIFEST,
    model_family="sd3",
    profile="multidiffusion_coco_all",
    target_size=TARGET_SIZE,
    return_image=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=2,
    cache_dir=MASK_CACHE_DIR,
)

loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
dataset_size = len(loader.dataset)

print("[OK] Manifest:", RUN_MANIFEST)
print("[OK] Dataset size:", dataset_size)
print("[OK] Expected:", EXPECTED_DATASET_SIZE)
print("[OK] Number of dataloader batches:", len(loader))

assert dataset_size == EXPECTED_DATASET_SIZE, (dataset_size, EXPECTED_DATASET_SIZE)

preview_batch = next(iter(loader))
preview_payload = make_semanticdraw_payload(preview_batch, 0)
print("[OK] First sample:", preview_payload["sample_id"])
print("[OK] First foreground prompts:", preview_payload["foreground_prompts"])
print("[OK] First mask shape:", tuple(preview_payload["foreground_masks"].shape))

# Validate every manifest row and decoded mask before loading the SD3 model.
preflight_ids = []
preflight_mask_hash = hashlib.sha256()
for preflight_batch in loader:
    for preflight_index in range(len(preflight_batch["sample_ids"])):
        item = make_semanticdraw_payload(preflight_batch, preflight_index)
        masks = item["foreground_masks"]
        if (masks.ndim != 4 or masks.shape[1] != 1 or masks.shape[-2:] != TARGET_SIZE
                or masks.shape[0] == 0 or masks.shape[0] != len(item["foreground_prompts"])
                or masks.shape[0] != len(item["category_names"])
                or not torch.isfinite(masks).all() or (masks.flatten(1).sum(1) <= 0).any()):
            raise RuntimeError(f"Invalid SD3 foreground masks at {item['sample_id']}")
        preflight_ids.append(item["sample_id"])
        preflight_mask_hash.update(item["sample_id"].encode("utf-8"))
        preflight_mask_hash.update(masks.contiguous().numpy().tobytes())
manifest_ids = [json.loads(line)["sample_id"] for line in RUN_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
if preflight_ids != manifest_ids or len(preflight_ids) != EXPECTED_DATASET_SIZE:
    raise RuntimeError("SD3 dataloader order/count differs from the manifest.")
input_masks_sha256 = preflight_mask_hash.hexdigest()
if run_config.get("input_masks_sha256") not in (None, input_masks_sha256):
    raise RuntimeError("Decoded COCO masks differ from the existing run. Stop before generation.")
run_config["input_masks_sha256"] = input_masks_sha256
RUN_CONFIG_PATH.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
print("[CHECK] Input masks SHA256:", input_masks_sha256)
print(f"[OK] Preflighted {len(preflight_ids)} ordered SD3 masks.")

# --- Source notebook cell 16 ---
seed_everything(BASE_SEED)

smd = SemanticDraw3Pipeline(
    device=device,
    dtype=dtype,
    hf_key=MODEL_ID,
    lora_key=FLASH_SD3_REPO_ID,
    hf_revision=MODEL_REVISION,
    lora_revision=FLASH_SD3_REVISION,
    has_i2t=False,
    t_index_list=SEMANTICDRAW_T_INDEX_LIST,
    default_mask_std=MASK_STD,
    default_mask_strength=MASK_STRENGTH,
    default_preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
    default_bootstrap_steps=BOOTSTRAP_STEPS,
    mask_type=MASK_TYPE,
)

# Keep SD3's JointAttnProcessor2_0. Generic attention slicing is not
# validated for the image/context pair returned by MMDiT joint attention.
if hasattr(smd.pipe, "enable_vae_slicing"):
    smd.pipe.enable_vae_slicing()
    print("[OK] VAE slicing enabled.")
if hasattr(smd.pipe, "enable_vae_tiling"):
    smd.pipe.enable_vae_tiling()
    print("[OK] VAE tiling enabled.")


if PATCH_SAFE_WHITE_BOOTSTRAP_LATENT:
    @torch.no_grad()
    def get_white_background_float32(height: int, width: int) -> torch.Tensor:
        latent_h = (height + smd.vae_scale_factor - 1) // smd.vae_scale_factor
        latent_w = (width + smd.vae_scale_factor - 1) // smd.vae_scale_factor
        if (not hasattr(smd, "white") or smd.white.shape[-2] < latent_h
                or smd.white.shape[-1] < latent_w):
            old_dtype = next(smd.vae.parameters()).dtype
            try:
                smd.vae.to(dtype=torch.float32)
                white = torch.ones(1, 3, height, width, dtype=torch.float32, device=smd.device)
                smd.white = smd.encode_imgs(white).to(dtype=smd.dtype)
            finally:
                smd.vae.to(dtype=old_dtype)
        return smd.white[..., :latent_h, :latent_w]

    smd.get_white_background = get_white_background_float32
    # Constructor has already encoded white in fp16. Replace that cache now.
    del smd.white
    seed_everything(WHITE_BOOTSTRAP_SEED)
    smd.get_white_background(1024, 1024)
    print("[OK] Safe white bootstrap latent patch enabled.")

if not torch.isfinite(smd.white).all():
    raise FloatingPointError("SD3 white bootstrap latent contains NaN/Inf.")
white_latent_sha256 = hashlib.sha256(smd.white.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
if run_config.get("white_latent_sha256") not in (None, white_latent_sha256):
    raise RuntimeError("White bootstrap latent differs from the existing run. Stop before generation.")
run_config["white_latent_sha256"] = white_latent_sha256
RUN_CONFIG_PATH.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
print("[CHECK] White latent SHA256:", white_latent_sha256)

print("[OK] SemanticDraw3Pipeline is ready.")
print("[CHECK] Scheduler:", type(smd.scheduler).__name__)
print("[CHECK] Timesteps:", [int(t) for t in smd.timesteps.detach().cpu().tolist()])
print("[CHECK] Sigmas:", [float(x) for x in smd.sigmas.detach().cpu().tolist()])

# --- Source notebook cell 18 ---
RUN_PLAIN_PIPELINE_SANITY = True

if RUN_PLAIN_PIPELINE_SANITY:
    seed_everything(BASE_SEED)
    sanity_prompt = "a studio photo of a teddy bear on a clean table"
    sanity_image = smd.pipe(
        sanity_prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=1024,
        width=1024,
    ).images[0].convert("RGB")
    print("[SANITY] stats:", pil_image_stats(sanity_image))
    display(sanity_image.resize((512, 512)))
    if STOP_ON_BAD_OUTPUT and is_bad_output(sanity_image):
        raise RuntimeError("Plain SD3/Flash sanity output looks invalid. Check checkpoint/token/dependency before SemanticDraw.")
else:
    print("[INFO] Plain pipeline sanity skipped.")

# --- Source notebook cell 20 ---
RUN_INPUT_TRACE = True

if RUN_INPUT_TRACE:
    payload = preview_payload
    print("[TRACE] sample_id:", payload["sample_id"])
    print("[TRACE] image_id:", payload["image_id"])
    print("[TRACE] background_prompt:", payload["background_prompt"])
    print("[TRACE] foreground_prompts:", payload["foreground_prompts"])
    print("[TRACE] category_names:", payload["category_names"])
    print("[TRACE] annotation_ids:", payload["annotation_ids"])
    print("[TRACE] area_ratios:", [float(x) for x in payload["area_ratios"]])
    print("[TRACE] foreground mask shape:", tuple(payload["foreground_masks"].shape))

    fg = payload["foreground_masks"].float()
    union = fg.sum(dim=0).clamp(0, 1)
    bg = (1.0 - union).clamp(0, 1)
    print("[TRACE] foreground union min/max/mean/sum:", float(union.min()), float(union.max()), float(union.mean()), float(union.sum()))
    print("[TRACE] background mask min/max/mean/sum:", float(bg.min()), float(bg.max()), float(bg.mean()), float(bg.sum()))
    print("[TRACE] smd.timesteps:", [int(t) for t in smd.timesteps.detach().cpu().tolist()])
    print("[TRACE] smd.sigmas:", [float(x) for x in smd.sigmas.detach().cpu().tolist()])
else:
    print("[INFO] Input trace skipped.")
from experiments.semantic_anchor_sd3 import SD3AnchorController, sd3_category_token_positions

# Reject category/token mismatches before spending GPU time on generation.
token_failures = []
with RUN_MANIFEST.open("r", encoding="utf-8") as manifest_file:
    for manifest_index, line in enumerate(manifest_file):
        if not line.strip():
            continue
        manifest_item = json.loads(line)
        try:
            sd3_category_token_positions(
                smd.pipe, manifest_item["foreground_prompts"], manifest_item["category_names"]
            )
        except Exception as exc:
            token_failures.append((manifest_index, manifest_item["sample_id"], str(exc)))
if token_failures:
    raise RuntimeError(f"SD3 category token mapping failed: {token_failures[:10]}")
print(f"[OK] SD3 category token mapping: {EXPECTED_DATASET_SIZE} manifest rows.")


def generate_sd3(payload: dict, controller=None, output_type: str = "pil") -> tuple[Image.Image | torch.Tensor, int]:
    region_prompts, region_negative_prompts, region_masks = make_semanticdraw_region_inputs(payload)
    image = smd(
        prompts=region_prompts,
        negative_prompts=region_negative_prompts,
        masks=region_masks.to(device=device, dtype=torch.float32),
        background_prompt=payload["background_prompt"],
        background_negative_prompt=NEGATIVE_PROMPT,
        mask_stds=MASK_STD,
        mask_strengths=MASK_STRENGTH,
        height=payload["height"],
        width=payload["width"],
        num_inference_steps=SEMANTICDRAW_NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        bootstrap_steps=BOOTSTRAP_STEPS,
        preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
        do_blend=False,
        step_controller=controller,
        output_type=output_type,
    )
    return (image if output_type == "latent" else image.convert("RGB")), len(region_prompts) + 1


def make_controller(payload: dict, mode: str) -> SD3AnchorController:
    return SD3AnchorController(
        smd,
        foreground_prompts=payload["foreground_prompts"],
        category_names=payload["category_names"],
        foreground_masks=payload["foreground_masks"],
        mode=mode,
        attention_layer_index=ATTENTION_LAYER_INDEX,
        topk_percent=TOPK_PERCENT,
        spatial_sigma_latent=SPATIAL_SIGMA_LATENT,
        semantic_sigma_scale=SEMANTIC_SIGMA_SCALE,
    )

# --- Source notebook cell 22 ---
# Gate: the instrumented no-op path must reproduce the corrected baseline.
parity_payload = preview_payload
seed_everything(BASE_SEED)
baseline_latent, _ = generate_sd3(parity_payload, output_type="latent")
parity_controller = make_controller(parity_payload, mode="noop")
seed_everything(BASE_SEED)
with parity_controller:
    noop_latent, _ = generate_sd3(parity_payload, parity_controller, output_type="latent")

if not torch.equal(baseline_latent, noop_latent):
    maximum_difference = float((baseline_latent.float() - noop_latent.float()).abs().max())
    raise RuntimeError(
        f"SD3 no-op latent parity failed: maximum_abs_difference={maximum_difference}. "
        "Stop before AnchorDraw generation."
    )
print("[PASS] SD3 baseline/no-op final latent parity on first manifest sample.")
print("[PASS] Captured joint-attention maps:", sorted(parity_controller.maps))

# --- Source notebook cell 23 ---
manifest_rows = [
    json.loads(line) for line in RUN_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()
]
assert len(manifest_rows) == dataset_size == EXPECTED_DATASET_SIZE
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
            or old_row.get("run_signature") != run_config["run_signature"]
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
global_index = 0

for batch_index, batch in enumerate(loader):
    print(f"[BATCH] {batch_index + 1}/{len(loader)} - {len(batch['sample_ids'])} sample(s)")

    for local_index, sample_id in enumerate(batch["sample_ids"]):
        retry_overhead_sec = previous_retry_seconds.get(global_index, 0.0)
        for attempt in range(1, 4):
            attempt_started = time.perf_counter()
            try:
                payload = make_semanticdraw_payload(batch, local_index)
                original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
                overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)

                assert payload["sample_id"] == expected_ids[global_index], (
                    global_index, payload["sample_id"], expected_ids[global_index]
                )
                stem = f"{global_index:04d}_{payload['sample_id']}"
                generated_path = GENERATED_IMAGES_DIR / f"{stem}_generated.png"
                overlay_path = OVERLAY_IMAGES_DIR / f"{stem}_overlay.png"
                seed = BASE_SEED + global_index

                if global_index in existing_by_index:
                    if not generated_path.exists():
                        raise RuntimeError(f"Summary row exists but PNG is missing: {generated_path}")
                    with Image.open(generated_path) as existing_image:
                        existing_image.verify()
                    with Image.open(generated_path) as existing_image:
                        generated = existing_image.convert("RGB")
                    if generated.size != (payload["width"], payload["height"]) or is_bad_output(generated):
                        raise RuntimeError(f"Existing PNG is invalid: {generated_path}")
                    print(f"[SKIP] index={global_index} sample_id={payload['sample_id']}")
                    break

                seed_everything(seed)
                tic = time.perf_counter()
                controller = make_controller(payload, mode="anchor")
                with controller:
                    generated, num_regions = generate_sd3(payload, controller)
                elapsed = time.perf_counter() - tic + retry_overhead_sec

                stats = pil_image_stats(generated)
                print(f"[IMAGE] index={global_index} stats={stats}")
                if STOP_ON_BAD_OUTPUT and is_bad_output(generated):
                    raise RuntimeError(
                        f"Generated image at index {global_index} looks invalid: {stats}. "
                        "Plain sanity is OK, so inspect SemanticDraw SD3 mask/bootstrap loop."
                    )

                generated_tmp = generated_path.with_suffix(".tmp")
                generated.save(generated_tmp, format="PNG")
                generated_tmp.replace(generated_path)
                overlay.save(overlay_path)

                row = {
                    "index": global_index,
                    "batch_index": batch_index,
                    "local_index": local_index,
                    "sample_id": payload["sample_id"],
                    "image_id": payload["image_id"],
                    "file_name": payload["file_name"],
                    "seed": seed,
                    "model_family": MODEL_FAMILY,
                    "model_id": MODEL_ID,
                    "acceleration": FLASH_SD3_REPO_ID,
                    "sampler": SAMPLER_NAME,
                    "semanticdraw_t_index_list": SEMANTICDRAW_T_INDEX_LIST,
                    "guidance_scale": GUIDANCE_SCALE,
                    "bootstrap_steps": BOOTSTRAP_STEPS,
                    "mask_std": MASK_STD,
                    "mask_strength": MASK_STRENGTH,
                    "preprocess_mask_cover_alpha": PREPROCESS_MASK_COVER_ALPHA,
                    "num_regions_including_background": num_regions,
                    "attention_layer_index": ATTENTION_LAYER_INDEX,
                    "topk_percent": TOPK_PERCENT,
                    "spatial_sigma_latent": SPATIAL_SIGMA_LATENT,
                    "semantic_sigma_scale": SEMANTIC_SIGMA_SCALE,
                    "controller_records": controller.records,
                    "run_signature": run_config["run_signature"],
                    "elapsed_sec": elapsed,
                    "generated_path": str(generated_path),
                    "overlay_path": str(overlay_path),
                    "background_prompt": payload["background_prompt"],
                    "foreground_prompts": payload["foreground_prompts"],
                    "category_names": payload["category_names"],
                    "annotation_ids": payload["annotation_ids"],
                    "area_ratios": payload["area_ratios"],
                    "target_size": list(TARGET_SIZE),
                }
                summary.append(row)
                summary.sort(key=lambda item: item["index"])
                summary_tmp = RUN_SUMMARY_PATH.with_suffix(".tmp")
                summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                summary_tmp.replace(RUN_SUMMARY_PATH)

                if global_index < MAX_DISPLAY_RESULTS:
                    display_result(payload, original, overlay, generated, elapsed, generated_path)

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

assert len(summary) == dataset_size == EXPECTED_DATASET_SIZE
assert [row["sample_id"] for row in summary] == expected_ids
assert [row["index"] for row in summary] == list(range(EXPECTED_DATASET_SIZE))

display(Markdown(
    f"## Done\n"
    f"Generated `{len(summary)}` image(s) from `{dataset_size}` manifest record(s).\n\n"
    f"Summary saved to `{RUN_SUMMARY_PATH}`."
))

summary[:3]

# --- Source notebook cell 25 ---
# Export is only allowed for a complete, ordered, verified generation.
summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
expected_ids = [
    json.loads(line)["sample_id"]
    for line in RUN_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()
]
assert len(summary) == len(expected_ids) == EXPECTED_DATASET_SIZE
assert [row["index"] for row in summary] == list(range(EXPECTED_DATASET_SIZE))
assert [row["sample_id"] for row in summary] == expected_ids
assert all(row.get("run_signature") == run_config["run_signature"] for row in summary)
expected_pngs = {
    f"{i:04d}_{sample_id}_generated.png" for i, sample_id in enumerate(expected_ids)
}
actual_pngs = {path.name for path in GENERATED_IMAGES_DIR.glob("*_generated.png")}
assert actual_pngs == expected_pngs, (
    f"Missing PNGs: {sorted(expected_pngs - actual_pngs)[:5]}; "
    f"extra PNGs: {sorted(actual_pngs - expected_pngs)[:5]}"
)

export_generated_dir = METRIC_EXPORT_DIR / "generated_images"
export_generated_dir.mkdir(parents=True, exist_ok=True)

metric_rows = []
for row in summary:
    src = Path(row["generated_path"])
    dst = export_generated_dir / src.name
    with Image.open(src) as check_img:
        check_img.verify()
    shutil.copy2(src, dst)

    metric_row = dict(row)
    metric_row["metric_index"] = row["index"]
    metric_row["generated_image_relative_path"] = f"generated_images/{dst.name}"
    metric_row["source_manifest_path"] = str(RUN_MANIFEST)
    metric_row["source_output_dir"] = str(OUTPUT_DIR)
    metric_rows.append(metric_row)

assert {path.name for path in export_generated_dir.glob("*_generated.png")} == expected_pngs
summary_export_path = METRIC_EXPORT_DIR / "generation_summary.json"
summary_export_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
run_config_export_path = METRIC_EXPORT_DIR / "run_config.json"
shutil.copy2(RUN_CONFIG_PATH, run_config_export_path)

jsonl_path = METRIC_EXPORT_DIR / "metric_generated_manifest.jsonl"
with jsonl_path.open("w", encoding="utf-8", newline="\n") as f:
    for row in metric_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

csv_path = METRIC_EXPORT_DIR / "metric_generated_manifest.csv"
with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()) if metric_rows else [])
    if metric_rows:
        writer.writeheader()
        writer.writerows(metric_rows)

export_summary = {
    "experiment_id": EXPERIMENT_ID,
    "num_generated_images": len(metric_rows),
    "run_signature": run_config["run_signature"],
    "manifest_sha256": run_config["manifest_sha256"],
    "pipeline_sha256": run_config["pipeline_sha256"],
    "white_latent_sha256": run_config["white_latent_sha256"],
    "input_masks_sha256": run_config["input_masks_sha256"],
    "coco_annotations_sha256": run_config["coco_annotations_sha256"],
    "run_config": "run_config.json",
    "export_dir": str(METRIC_EXPORT_DIR),
    "generated_images_dir": str(export_generated_dir),
    "generation_summary": str(summary_export_path),
    "manifest_jsonl": str(jsonl_path),
    "manifest_csv": str(csv_path),
    "source_manifest_path": str(RUN_MANIFEST),
    "source_output_dir": str(OUTPUT_DIR),
}
export_summary["zip_path"] = f"{METRIC_EXPORT_DIR}.zip"
(METRIC_EXPORT_DIR / "export_summary.json").write_text(json.dumps(export_summary, ensure_ascii=False, indent=2), encoding="utf-8")

zip_path = shutil.make_archive(
    base_name=str(METRIC_EXPORT_DIR),
    format="zip",
    root_dir=str(METRIC_EXPORT_DIR),
)
assert zip_path == export_summary["zip_path"]

if SAVE_EXPORT_TO_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount(str(WORK_DIR / 'drive'))
    GOOGLE_DRIVE_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    drive_export_dir = GOOGLE_DRIVE_EXPORT_ROOT / METRIC_EXPORT_DIR.name
    drive_zip_path = GOOGLE_DRIVE_EXPORT_ROOT / Path(zip_path).name

    export_summary["google_drive_export_dir"] = str(drive_export_dir)
    export_summary["google_drive_zip_path"] = str(drive_zip_path)
    (METRIC_EXPORT_DIR / "export_summary.json").write_text(json.dumps(export_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.copytree(METRIC_EXPORT_DIR, drive_export_dir, dirs_exist_ok=True)
    shutil.copy2(zip_path, drive_zip_path)

    print("[OK] Google Drive export dir:", drive_export_dir)
    print("[OK] Google Drive ZIP:", drive_zip_path)
else:
    print("[INFO] Google Drive export skipped. Set SAVE_EXPORT_TO_GOOGLE_DRIVE=True to enable it.")

print(json.dumps(export_summary, ensure_ascii=False, indent=2))
print("[OK] ZIP:", zip_path)

# --- Source notebook cell 26 ---
# Metrics run only after the complete generation and export gates have passed.
if RUN_METRICS_AFTER_GENERATION:
    import gc
    from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report

    checked_summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert len(checked_summary) == EXPECTED_DATASET_SIZE
    assert [row["index"] for row in checked_summary] == list(range(EXPECTED_DATASET_SIZE))
    assert all(row["run_signature"] == run_config["run_signature"] for row in checked_summary)
    metric_annotation_hash = hashlib.sha256()
    for annotation_path in (COCO_ROOT / "annotations" / "instances_val2017.json", COCO_ROOT / "annotations" / "captions_val2017.json"):
        metric_annotation_hash.update(annotation_path.name.encode("utf-8"))
        metric_annotation_hash.update(annotation_path.read_bytes())
    assert metric_annotation_hash.hexdigest() == run_config["coco_annotations_sha256"]
    metric_source_check = hashlib.sha256()
    for metric_file in sorted((REPO_ROOT / "Ours" / "src" / "metrics").glob("*.py")):
        metric_source_check.update(metric_file.name.encode("utf-8"))
        metric_source_check.update(metric_file.read_bytes())
    assert metric_source_check.hexdigest() == run_config["metrics_source_sha256"]

    # Release the SD3 model before loading Inception and CLIP.
    for variable in ("controller", "parity_controller", "smd", "sanity_image",
                     "generated", "baseline_latent", "noop_latent", "batch",
                     "loader", "preview_batch"):
        if variable in globals():
            del globals()[variable]
    gc.collect()
    torch.cuda.empty_cache()

    metrics_dir = OUTPUT_DIR / "metrics"
    metric_config = MetricEvaluationConfig(
        manifest_path=RUN_MANIFEST,
        coco_root=COCO_ROOT,
        generated_dir=GENERATED_IMAGES_DIR,
        generation_summary=RUN_SUMMARY_PATH,
        output_dir=metrics_dir,
        model_family=MODEL_FAMILY,
        target_size=TARGET_SIZE,
        metrics=("fid", "is", "clip_fg", "clip_bg", "time"),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        device="cuda:0",
        clip_batch_size=8,
        is_splits=10,
    )
    metric_report = run_evaluation(metric_config)
    assert metric_report["num_manifest_records"] == EXPECTED_DATASET_SIZE
    assert metric_report["num_evaluated"] == EXPECTED_DATASET_SIZE
    assert metric_report["num_missing_generated"] == 0
    metric_json, metric_csv = write_metrics_report(
        metric_report, metrics_dir, prefix="ours_sd3_flashflowmatch_metrics"
    )
    if SAVE_EXPORT_TO_GOOGLE_DRIVE:
        shutil.copy2(metric_json, GOOGLE_DRIVE_EXPORT_ROOT / f"{EXPERIMENT_ID}__metrics.json")
    print("[OK] SD3 AnchorDraw metrics:", metric_json, metric_csv)
    print(json.dumps(metric_report["metrics"], ensure_ascii=False, indent=2))
else:
    print("[INFO] Metrics skipped for smoke profile. Validate images and parity before full1073.")

# Keep the verified report alongside the ZIP for three-seed aggregation.
sd3_export_zip = Path(str(METRIC_EXPORT_DIR) + ".zip")
assert sd3_export_zip.is_file(), sd3_export_zip
sd3_sidecar = sd3_export_zip.with_name(sd3_export_zip.name.replace("__metric_export.zip", "__metrics.json"))
shutil.copy2(metric_json, sd3_sidecar)
print("[OK] SD3 report sidecar:", sd3_sidecar)
