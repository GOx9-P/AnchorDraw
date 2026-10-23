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

# Notebook: Ours/kaggle_semantic_anchor_experiments/pladis_semanticdraw_hypersd_smoke8_kaggle.ipynb
# Notebook SHA256: b6909390c4b6de39e29d2c66fe1362242db58add1148f8741b67eec53b99d8ae

# --- Source notebook cell 3 ---
# 3. Lock the same SemanticDraw + Hyper-SD generation protocol, but use smoke8 only.
import json, hashlib
from pathlib import Path

RUN_MANIFEST = REPO_ROOT / "Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl"
COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'COCO')))
RUN_DIR = Path(str(WORK_DIR / 'pladis_semanticdraw_hypersd_smoke8'))
MASK_CACHE_DIR = Path(str(WORK_DIR / 'semanticdraw_mask_cache'))
MODEL_ID = "runwayml/stable-diffusion-v1-5"
HYPER_SD_REPO_ID = "ByteDance/Hyper-SD"
HYPER_SD_WEIGHT_NAME = "Hyper-SD15-4steps-lora.safetensors"
HYPER_SD_NUM_INFERENCE_STEPS = 4
HYPER_SD_LORA_SCALE = 1.0
SEMANTICDRAW_GUIDANCE_SCALE = 1.0  # SemanticDraw CFG formula; equals conditional-only.
TARGET_SIZE = (512, 512)
BATCH_SIZE = 8
BASE_SEED = 2024
BOOTSTRAP_STEPS = 1
MASK_STD = 0.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = "discrete"
NEGATIVE_PROMPT = ""
PLADIS_SCALE = 1.5
REFERENCE_BASELINE_DIR = os.environ.get("SEMANTICDRAW_HYPSD_REFERENCE_DIR")
EXPECTED_TIMESTEPS = [999, 749, 499, 249]
assert RUN_MANIFEST.is_file(), RUN_MANIFEST
all_records = [json.loads(line) for line in RUN_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(all_records) == 1073 and len({r["sample_id"] for r in all_records}) == 1073
records = all_records[:8]
MANIFEST_SHA256 = hashlib.sha256(RUN_MANIFEST.read_bytes()).hexdigest()
RUN_DIR.mkdir(parents=True, exist_ok=True)
MASK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
print("[OK] First eight SemanticDraw samples:", [r["sample_id"] for r in records])
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 5 ---
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

# --- Source notebook cell 6 ---
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

# --- Source notebook cell 7 ---
# 5. Same COCORegionConfig and payload conversion as the full1073 SemanticDraw baseline.
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
assert len(loader.dataset) == 1073
first_batch = next(iter(loader))
assert first_batch["sample_ids"] == [r["sample_id"] for r in records]
payloads = [make_semanticdraw_payload(first_batch, i) for i in range(8)]
assert all(len(p["prompts"]) == len(p["all_masks"]) for p in payloads)
print("[OK] SemanticDraw payloads: first sample prompts =", payloads[0]["prompts"])

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

if hasattr(smd.pipe, "enable_attention_slicing"):
    smd.pipe.enable_attention_slicing()

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
actual_timesteps = [int(t) for t in smd.timesteps.detach().cpu().tolist()]
assert actual_timesteps == EXPECTED_TIMESTEPS, actual_timesteps
print("[OK] Baseline SemanticDraw + Hyper-SD model loaded.")

# --- Source notebook cell 9 ---
# 7. PLADIS's entmax15/dense extrapolation inside sliced SD1.5 cross-attention.
# Preserve the baseline's attention slicing; never replace self-attention.
# Source equation: https://github.com/cubeyoung/PLADIS/blob/main/pipeline/pipeline_sdxl.py
from contextlib import contextmanager
import importlib.metadata
import torch.nn.functional as F
from entmax import entmax15

class SlicedPLADISProcessor:
    def __init__(self, native, scale):
        self.native = native
        self.scale = float(scale)
        self.calls = 0
        self.modified_calls = 0
        self.delta_sum = 0.0
        self.delta_chunks = 0
        self.slice_size = max(1, int(getattr(native, "slice_size", 1)))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        self.calls += 1
        if self.scale == 0.0:
            return self.native(attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs)
        if args or kwargs or encoder_hidden_states is None:
            raise RuntimeError("Unexpected attention arguments or non-cross-attention input.")
        if attn.norm_q is not None or attn.norm_k is not None:
            raise RuntimeError("SD1.5 sliced-attention port expects no Q/K norm.")
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
        out = torch.zeros_like(query)
        delta_total = torch.zeros((), device=query.device, dtype=torch.float32)
        all_finite = torch.ones((), device=query.device, dtype=torch.bool)
        for start in range(0, query.shape[0], self.slice_size):
            stop = min(start + self.slice_size, query.shape[0])
            logits = torch.bmm(query[start:stop], key[start:stop].transpose(1, 2)) * attn.scale
            if attention_mask is not None:
                logits = logits + attention_mask[start:stop]
            dense = torch.softmax(logits, dim=-1)
            sparse = entmax15(logits, dim=-1)
            mixed = self.scale * sparse + (1.0 - self.scale) * dense
            all_finite &= torch.isfinite(mixed).all()
            delta_total += (mixed - dense).abs().mean().float()
            self.delta_chunks += 1
            out[start:stop] = torch.bmm(mixed, value[start:stop])
        if not bool(all_finite):
            raise FloatingPointError("Non-finite PLADIS attention.")
        self.delta_sum += float(delta_total.item())
        self.modified_calls += 1
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor

native_processors = dict(smd.unet.attn_processors)
cross_names = sorted(name for name in native_processors if ".attn2.processor" in name)
assert cross_names and all(name.startswith(("down_blocks.", "mid_block.", "up_blocks.")) for name in cross_names)
native_types = {type(native_processors[name]).__name__ for name in cross_names}
assert native_types == {"SlicedAttnProcessor"}, f"Expected unchanged SemanticDraw attention slicing, got {native_types}"
@contextmanager
def installed_pladis(scale):
    wrappers = {name: SlicedPLADISProcessor(native_processors[name], scale) for name in cross_names}
    smd.unet.set_attn_processor({name: wrappers.get(name, processor) for name, processor in native_processors.items()})
    try:
        yield wrappers
    finally:
        smd.unet.set_attn_processor(dict(native_processors))
        assert all(smd.unet.attn_processors[name] is processor for name, processor in native_processors.items())
print("[OK] PLADIS targets", len(cross_names), "cross-attention modules; baseline processors:", native_types)

# --- Source notebook cell 10 ---
# 8. Run the exact baseline call three times per sample, restoring white-latent state.
# Baseline branch keeps its original state progression across the eight manifest rows.
import numpy as np
import pandas as pd

def run_semanticdraw(payload, seed):
    seed_everything(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    image = smd(
        prompts=payload["prompts"],
        negative_prompts=payload["negative_prompts"],
        masks=payload["all_masks"].to(device=device, dtype=torch.float32),
        mask_stds=MASK_STD,
        mask_strengths=MASK_STRENGTH,
        height=payload["height"],
        width=payload["width"],
        bootstrap_steps=BOOTSTRAP_STEPS,
        preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
        guidance_scale=SEMANTICDRAW_GUIDANCE_SCALE,
        do_blend=False,
    )
    torch.cuda.synchronize()
    return image, time.perf_counter() - start

def restore_white(previous):
    if previous is None:
        if hasattr(smd, "white"):
            delattr(smd, "white")
    else:
        smd.white = previous

rows, pairs = [], []
for index, payload in enumerate(payloads):
    seed = BASE_SEED + index
    white_before = getattr(smd, "white", None)
    baseline, base_sec = run_semanticdraw(payload, seed)
    historical_equal = None
    if REFERENCE_BASELINE_DIR:
        reference_path = Path(REFERENCE_BASELINE_DIR) / f"{index:04d}_{payload['sample_id']}_generated.png"
        assert reference_path.is_file(), f"Missing historical baseline: {reference_path}"
        with Image.open(reference_path) as reference_image:
            historical_equal = bool(np.array_equal(np.asarray(reference_image.convert('RGB')), np.asarray(baseline)))
        assert historical_equal, f"Historical baseline pixel parity failed: {reference_path}"
    white_after_baseline = getattr(smd, "white", None)
    try:
        restore_white(white_before)
        with installed_pladis(0.0) as zero_wrappers:
            zero, zero_sec = run_semanticdraw(payload, seed)
            zero_calls = sum(w.calls for w in zero_wrappers.values())
            assert sum(w.modified_calls for w in zero_wrappers.values()) == 0
        assert np.array_equal(np.asarray(baseline), np.asarray(zero)), f"SemanticDraw parity failed: {payload['sample_id']}"
        restore_white(white_before)
        with installed_pladis(PLADIS_SCALE) as active_wrappers:
            pladis, pladis_sec = run_semanticdraw(payload, seed)
            active_calls = sum(w.modified_calls for w in active_wrappers.values())
            chunks = sum(w.delta_chunks for w in active_wrappers.values())
            attention_delta = sum(w.delta_sum for w in active_wrappers.values()) / chunks
        assert active_calls == zero_calls and active_calls > 0
        assert attention_delta > 0
    finally:
        restore_white(white_after_baseline)
    delta = np.abs(np.asarray(baseline, dtype=np.int16) - np.asarray(pladis, dtype=np.int16))
    row = dict(sample_id=payload["sample_id"], seed=seed, parity_exact=True, historical_baseline_equal=historical_equal,
               cross_attention_calls=active_calls, mean_attention_abs_delta=attention_delta,
               pixel_mae=float(delta.mean()), changed_pixel_fraction=float(np.any(delta != 0, axis=-1).mean()),
               baseline_sec=base_sec, scale0_sec=zero_sec, pladis_sec=pladis_sec)
    rows.append(row)
    pairs.append((baseline, pladis))
    print(f"[{index+1}/8] SemanticDraw parity=exact; PLADIS calls={active_calls}; pixel MAE={row['pixel_mae']:.3f}")
assert len(rows) == 8 and any(row["pixel_mae"] > 0 for row in rows)
print("[OK] 8/8 SemanticDraw baseline parity; PLADIS changed attention and image.")

# --- Source notebook cell 11 ---
# 9. Preview the baseline image at full size, then export a paired contact sheet and audit.
from PIL import ImageDraw
results = pd.DataFrame(rows)
display(results)
print("First SemanticDraw baseline image (sample 0):")
display(pairs[0][0])
sheet = Image.new("RGB", (1024, 8 * 540), "white")
draw = ImageDraw.Draw(sheet)
for index, (baseline, pladis) in enumerate(pairs):
    y = index * 540
    draw.text((8, y + 8), f"{index:02d} SemanticDraw + Hyper-SD | PLADIS=0", fill="black")
    draw.text((520, y + 8), f"{index:02d} SemanticDraw + Hyper-SD | PLADIS={PLADIS_SCALE}", fill="black")
    sheet.paste(baseline, (0, y + 28))
    sheet.paste(pladis, (512, y + 28))
sheet.save(RUN_DIR / "paired_contact_sheet.png")
results.to_csv(RUN_DIR / "smoke8_results.csv", index=False)
git_result = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, capture_output=True)
run_config = {
    "method": "SemanticDraw + Hyper-SD15 + PLADIS smoke8", "distillation_pp": False,
    "model": MODEL_ID, "lora": HYPER_SD_WEIGHT_NAME, "lora_scale": HYPER_SD_LORA_SCALE,
    "scheduler": "DDIMScheduler", "timestep_spacing": "trailing", "timesteps": actual_timesteps,
    "semanticdraw_guidance_scale": SEMANTICDRAW_GUIDANCE_SCALE, "bootstrap_steps": BOOTSTRAP_STEPS,
    "mask_std": MASK_STD, "mask_strength": MASK_STRENGTH,
    "preprocess_mask_cover_alpha": PREPROCESS_MASK_COVER_ALPHA, "mask_type": MASK_TYPE,
    "negative_prompt": NEGATIVE_PROMPT, "pladis_scale": PLADIS_SCALE, "sparse_activation": "entmax15",
    "pladis_layers": cross_names, "native_attention_processors": sorted(native_types),
    "manifest": str(RUN_MANIFEST), "manifest_sha256": MANIFEST_SHA256,
    "sample_ids": [r["sample_id"] for r in records], "base_seed": BASE_SEED,
    "baseline_white_latent_state": "preserved across samples; restored before each paired branch",
    "reference_baseline_dir": REFERENCE_BASELINE_DIR,
    "repo_commit": git_result.stdout.strip() if git_result.returncode == 0 else None,
    "versions": {"torch": torch.__version__, "diffusers": importlib.metadata.version("diffusers"),
                 "entmax": importlib.metadata.version("entmax")},
}
(RUN_DIR / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
display(sheet.resize((512, 2160)))
print("[OK] Saved:", RUN_DIR)
