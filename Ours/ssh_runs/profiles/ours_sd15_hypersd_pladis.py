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

# Notebook: Ours/kaggle_semantic_anchor_experiments/anchordraw_sd15_hypersd_w03_l04_sigmad4_pladis_s05_full1073_kaggle.ipynb
# Notebook SHA256: 7c66ca6b19d75ed2db8230f0e8faa48b43bca4f3601e1d8df31dd726e8fe49ce

# --- Source notebook cell 4 ---
# Cấu hình full manifest 1073 cho benchmark AnchorDraw SD1.5 + Hyper-SD + L04 + WM-03 + sigma_D=4.0.
from pathlib import Path
import os

RUN_MANIFEST = REPO_ROOT / "Ours" / "data_manifests" / "coco_val2017_multidiffusion_coco_all_512x512_all.jsonl"
COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'COCO')))
OUTPUT_DIR = Path(str(WORK_DIR / 'anchordraw_sd15_hypersd_w03_l04_sigmad4_pladis_s05_full1073_outputs'))
MASK_CACHE_DIR = Path(str(WORK_DIR / 'semantic_anchor_sd15_mask_cache'))
METRICS_OUTPUT_DIR = Path(str(WORK_DIR / 'anchordraw_sd15_hypersd_w03_l04_sigmad4_pladis_s05_full1073_metrics'))

# Model theo baseline SD1.5.
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

# --- THÔNG SỐ ĐỒNG BỘ 100% VỚI BASELINE ---
TARGET_SIZE = (512, 512)
BATCH_SIZE = 8
BASE_SEED = 2024
BASE_SEED = int(os.environ.get("ANCHORDRAW_BASE_SEED", str(BASE_SEED)))
BOOTSTRAP_STEPS = 1
MASK_STD = 1.0  # Chuẩn sweep: 1.0 để Gaussian lowpass tạo gradient làm mờ
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = "discrete"
NEGATIVE_PROMPT = ""

# --- CẤU HÌNH PHƯƠNG PHÁP MỚI ANCHORDRAW ---
METHOD_ID = "WM-03-TOPK-L04-SIGMA4-PLADIS-S05"
METHOD_LABEL = "AnchorDraw (Top-k + Adaptive Bilateral, L04, sigmaD=4.0, Hyper-SD 4-step, PLADIS scale=0.5)"
BEST_ANCHOR_MODE = "semantic_topk_anchor"
TOPK_ATTENTION_PERCENT = 10.0
WEIGHT_POLICY = "adaptive_bilateral"
FIXED_LAYER = {
    "layer_id": "L04",
    "layer_name": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor",  # L04 chuẩn (16x16)
}
SIGMA_D = 4.0
SEMANTIC_SIGMA_SCALE = 1.0
PLADIS_SCALE = 0.5
ATTENTION_CHUNK_SIZE = 8
EXPECTED_TIMESTEPS = [999, 749, 499, 249]
RUN_CONFIG_PATH = OUTPUT_DIR / 'run_config.json'
assert PLADIS_SCALE == 0.5

# Chỉ hiển thị 8 preview để Kaggle notebook không bị nặng.
# Ảnh vẫn được sinh và lưu đủ theo full manifest 1073.
MAX_DISPLAY_RESULTS = 0

# Cấu hình đo Metric (Đồng bộ 100% với baseline)
METRIC_NAMES = ("fid", "is", "clip_fg", "clip_bg", "time")
METRIC_BATCH_SIZE = 8
CLIP_BATCH_SIZE = 16
IS_SPLITS = 10
METRICS_REPORT_PREFIX = "anchordraw_sd15_hypersd_w03_l04_sigmad4_pladis_s05_full1073_metrics"

assert RUN_MANIFEST.exists(), f"Missing manifest: {RUN_MANIFEST}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[OK] Manifest: {RUN_MANIFEST}")
print(f"[OK] COCO root: {COCO_ROOT}")
print(f"[OK] Output dir: {OUTPUT_DIR}")
print(f"[OK] Metrics output dir: {METRICS_OUTPUT_DIR}")
print(f"[OK] Method ID: {METHOD_ID}")
print(f"[OK] Hyper-SD LoRA: {HYPER_SD_REPO_ID}/{HYPER_SD_WEIGHT_NAME} ({HYPER_SD_NUM_INFERENCE_STEPS} steps)")
print(f"[OK] Anchor: {BEST_ANCHOR_MODE} | Layer: {FIXED_LAYER['layer_id']} | sigma_D: {SIGMA_D}")
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 6 ---
# Import các module nguồn của AnchorDraw và Baseline.
import sys
import importlib.util
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
# terminal display provided above

OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_SRC = REPO_ROOT / "Baseline" / "semantic-draw-main" / "src"

for module_name in list(sys.modules):
    if module_name == "data" or module_name.startswith("data.") or module_name == "experiments" or module_name.startswith("experiments."):
        del sys.modules[module_name]

sys.path.insert(0, str(OURS_SRC))
from data import (
    COCORegionConfig,
    build_coco_region_dataloader,
    batch_item_to_semanticdraw_inputs,
)
from data.visualize import make_mask_overlay
from experiments.semantic_anchor import (
    ATTENTION_COORDINATE_POLICY,
    SemanticAnchorCapture,
    SemanticAnchorRuntime,
    find_target_token_indices,
    compute_anchor_measurements,
)
assert ATTENTION_COORDINATE_POLICY == "inverse_roll_image_after_latent_shift_r1"

sys.path.insert(0, str(BASELINE_SRC))
pipeline_path = BASELINE_SRC / "model" / "pipeline_semantic_draw.py"
spec = importlib.util.spec_from_file_location("pipeline_semantic_draw_original", pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline

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
num_batches = len(loader)
preview_batch = next(iter(loader))

print(f"[OK] Manifest records: {dataset_size}")
print(f"[OK] Dataloader batches: {num_batches} batch(es) x up to {BATCH_SIZE} sample(s)")
print(f"[OK] First batch size: {len(preview_batch['sample_ids'])}")
print(f"[OK] First batch masks shape: {tuple(preview_batch['masks'].shape)}  # (B, Pmax, C, H, W)")

# --- Source notebook cell 7 ---
# Helper functions, payload builder và attention layer selector cho Layer L04.
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
    import types
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
        print("[INFO] No HF_TOKEN found. Public/gated model access depends on your Hugging Face permissions.")

assert torch.cuda.is_available(), "Kaggle runtime chưa bật GPU. Hãy bật Accelerator = GPU rồi chạy lại."
device = torch.device("cuda:0")
dtype = torch.float16

maybe_login_to_huggingface()
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

# --- Source notebook cell 9 ---
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

# --- Source notebook cell 10 ---
# Load SemanticDraw SD1.5 pipeline, gỡ LCM và gắn Hyper-SD 4-step LoRA cho AnchorDraw.
from diffusers import DDIMScheduler
from huggingface_hub import hf_hub_download
import types

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

def prepare_hyper_sd_ddim_schedule(self, t_index_list=None, num_inference_steps=None) -> None:
    if num_inference_steps is None:
        if isinstance(t_index_list, int):
            num_inference_steps = t_index_list
        else:
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
smd.prepare_hyper_sd_ddim_schedule(num_inference_steps=HYPER_SD_NUM_INFERENCE_STEPS)

print("[OK] SemanticDrawPipeline is ready with SD1.5 + Hyper-SD for AnchorDraw.")

# --- Source notebook cell 11 ---
# Kiểm tra trực tiếp sampler/scheduler và layer L04 trong UNet.
print("Sampler:", "Hyper-SD")
print("Hyper-SD repo:", HYPER_SD_REPO_ID)
print("Hyper-SD LoRA:", HYPER_SD_WEIGHT_NAME)
print("Scheduler:", type(smd.scheduler).__name__)
print("Pipeline scheduler:", type(smd.pipe.scheduler).__name__)
print("Default inference steps:", smd.default_num_inference_steps)
print("SemanticDraw guidance scale:", smd.default_guidance_scale)
actual_timesteps = [int(t) for t in smd.timesteps.detach().cpu().tolist()]
print("Timesteps:", actual_timesteps)
assert actual_timesteps == EXPECTED_TIMESTEPS, (actual_timesteps, EXPECTED_TIMESTEPS)

assert type(smd.scheduler).__name__ == "DDIMScheduler", "Expected Hyper-SD sampler via DDIMScheduler trailing."
assert len(smd.timesteps) == HYPER_SD_NUM_INFERENCE_STEPS
assert smd.default_num_inference_steps == HYPER_SD_NUM_INFERENCE_STEPS
assert abs(float(smd.default_guidance_scale) - float(SEMANTICDRAW_GUIDANCE_SCALE)) < 1e-8

available_layer_names = set(smd.unet.attn_processors)
assert FIXED_LAYER["layer_name"] in available_layer_names, f"Missing {FIXED_LAYER['layer_name']} in UNet"
print(f"[OK] Fixed layer {FIXED_LAYER['layer_id']} verified: {FIXED_LAYER['layer_name']}")

# --- Source notebook cell 12 ---
# 8. Compose PLADIS with the *capture-installed* AnchorDraw processor.
# scale=0 delegates exactly; scale>0 sends mixed attention to both V and L04 map capture.
from contextlib import contextmanager
import importlib.metadata
from entmax import entmax15

class PLADISCaptureProcessor:
    def __init__(self, native_capture, scale):
        self.native = native_capture
        self.layer_name = native_capture.layer_name
        self.store = native_capture.store
        self.scale = float(scale)
        self.calls = 0
        self.modified_calls = 0
        self.delta_sum = 0.0
        self.delta_chunks = 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        self.calls += 1
        if self.scale == 0.0:
            return self.native(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)
        if encoder_hidden_states is None or args or kwargs:
            raise RuntimeError("PLADIS must receive only text cross-attention without extra arguments.")
        residual = hidden_states
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        head_batch, query_length, _ = query.shape
        key_length = key.shape[1]
        mixed_all = torch.empty((head_batch, query_length, key_length), device=query.device, dtype=query.dtype)
        delta_acc = torch.zeros((), device=query.device, dtype=torch.float32)
        finite_acc = torch.ones((), device=query.device, dtype=torch.bool)
        for start in range(0, head_batch, ATTENTION_CHUNK_SIZE):
            stop = min(start + ATTENTION_CHUNK_SIZE, head_batch)
            q = query[start:stop].float() if attn.upcast_attention else query[start:stop]
            k = key[start:stop].float() if attn.upcast_attention else key[start:stop]
            logits = torch.bmm(q, k.transpose(1, 2)) * attn.scale
            if attention_mask is not None:
                logits = logits + attention_mask[start:stop]
            if attn.upcast_softmax:
                logits = logits.float()
            dense = torch.softmax(logits, dim=-1)
            sparse = entmax15(logits, dim=-1)
            mixed = (self.scale * sparse + (1.0 - self.scale) * dense).to(query.dtype)
            finite_acc &= torch.isfinite(mixed).all()
            delta_acc += (mixed.float() - dense.float()).abs().mean()
            self.delta_chunks += 1
            mixed_all[start:stop] = mixed
        if not bool(finite_acc):
            raise FloatingPointError(f"Non-finite PLADIS attention at {self.layer_name}.")
        self.delta_sum += float(delta_acc.item())
        self.modified_calls += 1
        # L04 reads the same effective weights that produce the attended values.
        self.store.add(self.layer_name, mixed_all, batch_size, int(attn.heads))
        hidden_states = torch.bmm(mixed_all, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        return hidden_states / getattr(attn, "rescale_output_factor", 1.0)

@contextmanager
def composed_pladis(capture, scale):
    installed = dict(smd.unet.attn_processors)
    names = sorted(name for name in installed if name.endswith("attn2.processor"))
    assert len(names) == len(cross_names)
    assert all(getattr(installed[name], "store", None) is capture.store for name in names)
    wrappers = {name: PLADISCaptureProcessor(installed[name], scale) for name in names}
    smd.unet.set_attn_processor({name: wrappers.get(name, proc) for name, proc in installed.items()})
    try:
        yield wrappers
    finally:
        smd.unet.set_attn_processor(dict(installed))
        assert all(smd.unet.attn_processors[name] is proc for name, proc in installed.items())

cross_names = sorted(name for name in smd.unet.attn_processors if name.endswith("attn2.processor"))
assert FIXED_LAYER["layer_name"] in cross_names
assert len(cross_names) == 16, f"Unexpected SD1.5 cross-attention count: {len(cross_names)}"
print("[OK] PLADIS will compose with", len(cross_names), "capture layers, including L04.")

# --- Source notebook cell 13 ---
# Exact scale=0 gate against the original full1073 AnchorDraw generation path.
def restore_white(previous):
    if previous is None:
        if hasattr(smd, "white"):
            delattr(smd, "white")
    else:
        smd.white = previous

def parity_run(payload, scale):
    seed_everything(BASE_SEED)
    token_indices = [
        find_target_token_indices(smd.tokenizer, prompt, category)
        for prompt, category in zip(payload["foreground_prompts"], payload["category_names"])
    ]
    with SemanticAnchorCapture(smd.unet) as capture:
        capture.configure(token_indices)
        runtime = SemanticAnchorRuntime(smd, capture, image_size=TARGET_SIZE)
        install_runtime_patches(runtime, FIXED_LAYER["layer_name"])
        def call_original():
            return runtime.generate(
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
                denoised_refiner=None,
            )
        if scale is None:
            image, step_records = call_original()
            calls, modified = 0, 0
        else:
            with composed_pladis(capture, scale) as wrappers:
                image, step_records = call_original()
                calls = sum(wrapper.calls for wrapper in wrappers.values())
                modified = sum(wrapper.modified_calls for wrapper in wrappers.values())
        maps = {}
        for timestep in actual_timesteps:
            for region_index in range(len(token_indices)):
                found = [m for m in capture.maps[(timestep, region_index)]
                         if m.layer_name == FIXED_LAYER["layer_name"]]
                assert len(found) == 1
                maps[(timestep, region_index)] = found[0].values.clone()
        points = [(int(record.step_index), tuple((float(x), float(y)) for x, y in record.points_xy))
                  for record in step_records]
    return image, points, maps, calls, modified

parity_payload = make_semanticdraw_payload(preview_batch, 0)
white_before_parity = getattr(smd, "white", None)
try:
    native_image, native_points, native_maps, _, _ = parity_run(parity_payload, None)
    restore_white(white_before_parity)
    zero_image, zero_points, zero_maps, zero_calls, zero_modified = parity_run(parity_payload, 0.0)
    assert zero_calls == HYPER_SD_NUM_INFERENCE_STEPS * len(cross_names) == 64
    assert zero_modified == 0
    assert np.array_equal(np.asarray(native_image), np.asarray(zero_image)), "PLADIS scale=0 altered baseline pixels."
    assert native_points == zero_points, "PLADIS scale=0 altered anchor points."
    assert set(native_maps) == set(zero_maps)
    assert all(torch.equal(native_maps[key], zero_maps[key]) for key in native_maps), "PLADIS scale=0 altered L04 maps."
finally:
    restore_white(white_before_parity)
del parity_payload, native_image, zero_image, native_maps, zero_maps, native_points, zero_points
print("[OK] PLADIS scale=0 is exactly equivalent to the existing AnchorDraw full1073 path.")

# --- Source notebook cell 14 ---
# Record all fixed method and environment settings before full generation.
import hashlib
import importlib.metadata as package_metadata

def installed_version(package):
    try:
        return package_metadata.version(package)
    except package_metadata.PackageNotFoundError:
        return None

def git_output(*args):
    result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None

run_config = {
    "experiment_id": "OURS-SD15-HYPSD-W03-L04-SIGMA4-PLADIS-S05-F1073",
    "method_id": METHOD_ID,
    "model": MODEL_ID,
    "hyper_sd_repo": HYPER_SD_REPO_ID,
    "hyper_sd_weight": HYPER_SD_WEIGHT_NAME,
    "hyper_sd_lora_scale": HYPER_SD_LORA_SCALE,
    "resolution": list(TARGET_SIZE),
    "scheduler": "DDIMScheduler",
    "timestep_spacing": "trailing",
    "timesteps": actual_timesteps,
    "steps": HYPER_SD_NUM_INFERENCE_STEPS,
    "guidance_scale": SEMANTICDRAW_GUIDANCE_SCALE,
    "manifest": str(RUN_MANIFEST),
    "manifest_sha256": hashlib.sha256(RUN_MANIFEST.read_bytes()).hexdigest(),
    "manifest_records": dataset_size,
    "base_seed": BASE_SEED,
    "per_sample_seed": "BASE_SEED + global_index",
    "batch_size": BATCH_SIZE,
    "bootstrap_steps": BOOTSTRAP_STEPS,
    "mask_std": MASK_STD,
    "mask_strength": MASK_STRENGTH,
    "preprocess_mask_cover_alpha": PREPROCESS_MASK_COVER_ALPHA,
    "mask_type": MASK_TYPE,
    "negative_prompt": NEGATIVE_PROMPT,
    "anchor_mode": BEST_ANCHOR_MODE,
    "topk_attention_percent": TOPK_ATTENTION_PERCENT,
    "weight_policy": WEIGHT_POLICY,
    "fixed_layer": FIXED_LAYER,
    "sigma_d": SIGMA_D,
    "semantic_sigma_scale": SEMANTIC_SIGMA_SCALE,
    "distillation_pp": False,
    "denoised_refiner": None,
    "pladis_scale": PLADIS_SCALE,
    "pladis_formula": "scale * entmax15 + (1 - scale) * softmax",
    "pladis_layers": cross_names,
    "pladis_capture_map": "effective mixed attention",
    "pladis_attention_chunk_size": ATTENTION_CHUNK_SIZE,
    "scale_zero_parity": "exact pixels, anchor points, and L04 maps on first manifest sample",
    "metrics": list(METRIC_NAMES),
    "metric_batch_size": METRIC_BATCH_SIZE,
    "clip_batch_size": CLIP_BATCH_SIZE,
    "is_splits": IS_SPLITS,
    "repo_commit": git_output("rev-parse", "HEAD"),
    "source_sha256": {
        "semantic_anchor": hashlib.sha256((OURS_SRC / "experiments/semantic_anchor.py").read_bytes()).hexdigest(),
        "clip_metrics": hashlib.sha256((OURS_SRC / "metrics/clip_metrics.py").read_bytes()).hexdigest(),
        "semanticdraw_pipeline": hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
    },
    "package_versions": {p: installed_version(p) for p in [
        "torch", "diffusers", "transformers", "accelerate", "peft",
        "huggingface-hub", "entmax", "open-clip-torch", "torchmetrics"
    ]},
}
RUN_CONFIG_PATH.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
assert dataset_size == 1073 and actual_timesteps == EXPECTED_TIMESTEPS
print("[OK] Full1073 PLADIS method/provenance saved:", RUN_CONFIG_PATH)

# --- Source notebook cell 15 ---
failed_images = []
# Chạy generation cho mọi sample trong manifest bằng thuật toán AnchorDraw trên Hyper-SD 4-step.
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
                assert payload["sample_id"] == loader.dataset.records[global_index]["sample_id"]
                original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
                overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)

                # Seed được đồng bộ bit-exact với baseline: BASE_SEED (2024) + global_index
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

                    with composed_pladis(capture, PLADIS_SCALE) as pladis_wrappers:
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
                            denoised_refiner=None,
                        )
                    active_calls = sum(wrapper.calls for wrapper in pladis_wrappers.values())
                    modified_calls = sum(wrapper.modified_calls for wrapper in pladis_wrappers.values())
                    delta_chunks = sum(wrapper.delta_chunks for wrapper in pladis_wrappers.values())
                    attention_delta = (
                        sum(wrapper.delta_sum for wrapper in pladis_wrappers.values()) / delta_chunks
                        if delta_chunks else 0.0
                    )
                elapsed = time.perf_counter() - tic + retry_overhead_sec
                assert active_calls == modified_calls == HYPER_SD_NUM_INFERENCE_STEPS * len(cross_names) == 64
                assert attention_delta > 0, f'PLADIS had no measurable attention effect: {payload["sample_id"]}'
                assert len(runtime_records) == HYPER_SD_NUM_INFERENCE_STEPS
                assert len(runtime.weight_records) == HYPER_SD_NUM_INFERENCE_STEPS
                assert all(r.policy == WEIGHT_POLICY for r in runtime.weight_records[BOOTSTRAP_STEPS:])
                for timestep in actual_timesteps:
                    for region_index in range(len(token_indices)):
                        l04 = [m for m in capture.maps[(timestep, region_index)] if m.layer_name == FIXED_LAYER['layer_name']]
                        assert len(l04) == 1 and bool(torch.isfinite(l04[0].values).all()), (timestep, region_index)

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
                    "pladis_scale": PLADIS_SCALE,
                    "pladis_cross_attention_calls": active_calls,
                    "pladis_modified_calls": modified_calls,
                    "pladis_mean_attention_abs_delta": attention_delta,
                    "distillation_pp": False,
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

assert global_index == dataset_size == 1073
assert len(summary) == 1073 and len({row["sample_id"] for row in summary}) == 1073
assert all(row["pladis_modified_calls"] == 64 for row in summary)
summary_path = OUTPUT_DIR / "generation_summary.json"
with summary_path.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

display(Markdown(
    f"## Done\n"
    f"Generated `{len(summary)}` image(s) from `{dataset_size}` manifest record(s). "
    f"Summary saved to `{summary_path}`."
))
summary[:5]

# --- Source notebook cell 18 ---
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

# --- Source notebook cell 19 ---
# Đo FID, IS, CLIP(fg), CLIP(bg), Time(s) trên full output vừa generate.
import sys
import subprocess

# Guard chống lỗi thiếu open_clip hoặc thư viện metric nếu phiên làm việc có gián đoạn
try:
    import open_clip
except ImportError as exc:
    raise RuntimeError("open_clip is missing; install Ours/requirements.txt before starting this run.") from exc

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
assert metric_report["num_manifest_records"] == dataset_size == 1073
assert metric_report["num_evaluated"] == 1073
assert metric_report["num_missing_generated"] == 0
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
