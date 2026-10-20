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

# Notebook: Ours/kaggle_semantic_anchor_experiments/pladis_anchordraw_hypersd_w03_l04_sigmad4_scale_sweep128_kaggle.ipynb
# Notebook SHA256: 581a1996cbafc198b1de3a8c8e4dfdde1f4abbee1d1e3e86547bf71d666e3488

# --- Source notebook cell 3 ---
# 3. Freeze the full1073 protocol and select the first 128 manifest records.
import json, hashlib
from pathlib import Path
RUN_MANIFEST = REPO_ROOT / "Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl"
COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'COCO')))
RUN_DIR = Path(str(WORK_DIR / 'pladis_anchordraw_hypersd_w03_l04_sigmad4_scale_sweep128'))
RUN_SAMPLES = 128
PLADIS_SCALES = [0.0, 0.5, 1.0, 1.5]
MODEL_ID = "runwayml/stable-diffusion-v1-5"
HYPER_SD_REPO_ID = "ByteDance/Hyper-SD"
HYPER_SD_WEIGHT_NAME = "Hyper-SD15-4steps-lora.safetensors"
HYPER_SD_NUM_INFERENCE_STEPS = 4
HYPER_SD_LORA_SCALE = 1.0
SEMANTICDRAW_GUIDANCE_SCALE = 1.0
TARGET_SIZE = (512, 512)
BATCH_SIZE = 1
BASE_SEED = 2024
BOOTSTRAP_STEPS = 1
MASK_STD = 1.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = "discrete"
NEGATIVE_PROMPT = ""
METHOD_ID = "WM-03-TOPK-L04-SIGMA4"
BEST_ANCHOR_MODE = "semantic_topk_anchor"
TOPK_ATTENTION_PERCENT = 10.0
WEIGHT_POLICY = "adaptive_bilateral"
FIXED_LAYER = {"layer_id": "L04", "layer_name": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor"}
SIGMA_D = 4.0
SEMANTIC_SIGMA_SCALE = 1.0
ATTENTION_CHUNK_SIZE = 8
EXPECTED_TIMESTEPS = [999, 749, 499, 249]
REFERENCE_BASELINE_DIR = os.environ.get("ANCHORDRAW_HYPSD_REFERENCE_DIR")
assert RUN_MANIFEST.is_file(), RUN_MANIFEST
assert PLADIS_SCALES[0] == 0.0 and len(set(PLADIS_SCALES)) == len(PLADIS_SCALES)
all_records = [json.loads(line) for line in RUN_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(all_records) == 1073 and len({r["sample_id"] for r in all_records}) == 1073
records = all_records[:RUN_SAMPLES]
MANIFEST_SHA256 = hashlib.sha256(RUN_MANIFEST.read_bytes()).hexdigest()
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("[OK]", METHOD_ID, "| manifest:", len(all_records), "| paired subset:", len(records), "| scales:", PLADIS_SCALES)
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 5 ---
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
    SemanticAnchorCapture,
    SemanticAnchorRuntime,
    find_target_token_indices,
    compute_anchor_measurements,
)

sys.path.insert(0, str(BASELINE_SRC))
pipeline_path = BASELINE_SRC / "model" / "pipeline_semantic_draw.py"
spec = importlib.util.spec_from_file_location("pipeline_semantic_draw_original", pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline

# Reconstruct masks/prompts from the identical 1073-row manifest without decoding source images.
config = COCORegionConfig(
    coco_root=COCO_ROOT,
    split="val2017",
    instances_json=COCO_ROOT / "annotations/instances_val2017.json",
    captions_json=COCO_ROOT / "annotations/captions_val2017.json",
    manifest_path=RUN_MANIFEST,
    profile="multidiffusion_coco_all",
    model_family="sd15",
    target_size=TARGET_SIZE,
    return_image=False,
    cache_resized_masks=False,
    cache_dir=None,
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
)
loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
assert len(loader.dataset) == len(all_records) == 1073
first_batch = next(iter(loader))
assert first_batch["sample_ids"][0] == records[0]["sample_id"]
print("[OK] First 128 records will be read in the same manifest order.")

# --- Source notebook cell 6 ---
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

print("[OK] Original W03/L04 payload and anchor-selection helpers ready.")

# --- Source notebook cell 7 ---
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

# Kiểm tra trực tiếp sampler/scheduler và layer L04 trong UNet.
print("Sampler:", "Hyper-SD")
print("Hyper-SD repo:", HYPER_SD_REPO_ID)
print("Hyper-SD LoRA:", HYPER_SD_WEIGHT_NAME)
print("Scheduler:", type(smd.scheduler).__name__)
print("Pipeline scheduler:", type(smd.pipe.scheduler).__name__)
print("Default inference steps:", smd.default_num_inference_steps)
print("SemanticDraw guidance scale:", smd.default_guidance_scale)
print("Timesteps:", [int(t) for t in smd.timesteps.detach().cpu().tolist()])

assert type(smd.scheduler).__name__ == "DDIMScheduler", "Expected Hyper-SD sampler via DDIMScheduler trailing."
assert len(smd.timesteps) == HYPER_SD_NUM_INFERENCE_STEPS
assert smd.default_num_inference_steps == HYPER_SD_NUM_INFERENCE_STEPS
assert abs(float(smd.default_guidance_scale) - float(SEMANTICDRAW_GUIDANCE_SCALE)) < 1e-8

available_layer_names = set(smd.unet.attn_processors)
assert FIXED_LAYER["layer_name"] in available_layer_names, f"Missing {FIXED_LAYER['layer_name']} in UNet"
print(f"[OK] Fixed layer {FIXED_LAYER['layer_id']} verified: {FIXED_LAYER['layer_name']}")

actual_timesteps = [int(t) for t in smd.timesteps.detach().cpu().tolist()]
assert actual_timesteps == EXPECTED_TIMESTEPS, actual_timesteps
print("[OK] AnchorDraw Hyper-SD model and L04 are ready.")

# --- Source notebook cell 8 ---
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

# --- Source notebook cell 9 ---
# 9. Keep original AnchorDraw generation unchanged; compose PLADIS only at positive scales.
from itertools import islice
from contextlib import contextmanager
from PIL import ImageDraw
import gc

def restore_white(previous):
    if previous is None:
        if hasattr(smd, "white"):
            delattr(smd, "white")
    else:
        smd.white = previous

def anchor_points(step_records):
    return [(int(r.step_index), tuple((float(x), float(y)) for x, y in r.points_xy)) for r in step_records]

def mean_anchor_shift(first, second):
    shifts = []
    for (old_step, old_points), (new_step, new_points) in zip(first, second):
        assert old_step == new_step and len(old_points) == len(new_points)
        shifts.extend(
            float(((x1 - x0)**2 + (y1 - y0)**2)**0.5)
            for (x0, y0), (x1, y1) in zip(old_points, new_points)
        )
    return float(np.mean(shifts))

def run_one(payload, seed, scale=None, keep_maps=False):
    seed_everything(seed)
    token_indices = [
        find_target_token_indices(smd.tokenizer, prompt, category)
        for prompt, category in zip(payload["foreground_prompts"], payload["category_names"])
    ]
    torch.cuda.synchronize()
    tic = time.perf_counter()
    with SemanticAnchorCapture(smd.unet) as capture:
        capture.configure(token_indices)
        runtime = SemanticAnchorRuntime(smd, capture, image_size=TARGET_SIZE)
        install_runtime_patches(runtime, FIXED_LAYER["layer_name"])
        def run_original_method():
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
            image, step_records = run_original_method()
            calls, modified_calls, attention_delta = 0, 0, 0.0
        else:
            with composed_pladis(capture, scale) as wrappers:
                image, step_records = run_original_method()
                calls = sum(w.calls for w in wrappers.values())
                modified_calls = sum(w.modified_calls for w in wrappers.values())
                chunks = sum(w.delta_chunks for w in wrappers.values())
                attention_delta = sum(w.delta_sum for w in wrappers.values()) / chunks if chunks else 0.0
        l04_maps = {}
        for timestep in actual_timesteps:
            for region_index in range(len(token_indices)):
                matches = [m for m in capture.maps[(timestep, region_index)] if m.layer_name == FIXED_LAYER["layer_name"]]
                assert len(matches) == 1, f"Missing/duplicate L04 map at {(timestep, region_index)}"
                assert bool(torch.isfinite(matches[0].values).all())
                if keep_maps:
                    l04_maps[(timestep, region_index)] = matches[0].values.clone()
        assert len(step_records) == HYPER_SD_NUM_INFERENCE_STEPS
        assert len(runtime.weight_records) == HYPER_SD_NUM_INFERENCE_STEPS
        assert all(r.policy == WEIGHT_POLICY for r in runtime.weight_records[BOOTSTRAP_STEPS:])
        points = anchor_points(step_records)
    torch.cuda.synchronize()
    return image, time.perf_counter() - tic, calls, modified_calls, attention_delta, points, l04_maps

# Mandatory scale=0 parity gate before any expensive 128-sample sweep.
parity_payload = make_semanticdraw_payload(first_batch, 0)
white_preflight = getattr(smd, "white", None)
try:
    native, _, _, _, _, native_points, native_maps = run_one(parity_payload, BASE_SEED, keep_maps=True)
    restore_white(white_preflight)
    zero, _, zero_calls, zero_modified, _, zero_points, zero_maps = run_one(
        parity_payload, BASE_SEED, scale=0.0, keep_maps=True
    )
    assert zero_calls == HYPER_SD_NUM_INFERENCE_STEPS * len(cross_names) == 64
    assert zero_modified == 0
    assert np.array_equal(np.asarray(native), np.asarray(zero)), "PLADIS scale=0 changed original pixels."
    assert native_points == zero_points
    assert set(native_maps) == set(zero_maps)
    assert all(torch.equal(native_maps[key], zero_maps[key]) for key in native_maps)
finally:
    restore_white(white_preflight)
del native, zero, native_maps, zero_maps, parity_payload, first_batch
gc.collect(); torch.cuda.empty_cache()
print("[OK] Installed PLADIS scale=0 is pixel-, L04-map-, and anchor-identical to original AnchorDraw.")

# --- Source notebook cell 10 ---
# 10. Paired 128-sample sweep. Generated images stay in host RAM until CLIP scoring.
generated_in_memory = {scale: [] for scale in PLADIS_SCALES}
elapsed_by_scale = {scale: [] for scale in PLADIS_SCALES}
eval_samples, audit_rows = [], []
for sample_index, batch in enumerate(islice(loader, RUN_SAMPLES)):
    payload = make_semanticdraw_payload(batch, 0)
    sample_id, seed = payload["sample_id"], BASE_SEED + sample_index
    assert sample_id == records[sample_index]["sample_id"]
    eval_samples.append({
        "sample_index": sample_index, "sample_id": sample_id,
        "background_prompt": payload["prompts"][0],
        "foreground_prompts": list(payload["foreground_prompts"]),
        "foreground_masks": payload["foreground_masks"].bool().cpu().clone(),
    })
    white_before = getattr(smd, "white", None)
    baseline_points = None
    white_after_baseline = None
    try:
        for scale in PLADIS_SCALES:
            restore_white(white_before)
            image, elapsed, calls, modified, attention_delta, points, _ = run_one(
                payload, seed, scale=None if scale == 0.0 else scale
            )
            if scale == 0.0:
                assert calls == modified == 0 and attention_delta == 0.0
                baseline_points = points
                white_after_baseline = getattr(smd, "white", None)
                if REFERENCE_BASELINE_DIR:
                    reference_path = Path(REFERENCE_BASELINE_DIR) / f"{sample_index:04d}_{sample_id}_generated.png"
                    assert reference_path.is_file(), reference_path
                    with Image.open(reference_path) as reference:
                        assert np.array_equal(np.asarray(reference.convert("RGB")), np.asarray(image)), reference_path
            else:
                assert calls == modified == HYPER_SD_NUM_INFERENCE_STEPS * len(cross_names) == 64
                assert attention_delta > 0
            generated_in_memory[scale].append(image)
            elapsed_by_scale[scale].append(float(elapsed))
            audit_rows.append({
                "sample_index": sample_index, "sample_id": sample_id, "seed": seed, "pladis_scale": scale,
                "cross_attention_calls": calls, "modified_calls": modified,
                "mean_attention_abs_delta": attention_delta,
                "mean_anchor_shift_vs_scale0_px": 0.0 if scale == 0.0 else mean_anchor_shift(baseline_points, points),
                "generation_sec": float(elapsed),
            })
    finally:
        restore_white(white_after_baseline if white_after_baseline is not None else white_before)
    if (sample_index + 1) % 8 == 0:
        print(f"[RUN] {sample_index + 1}/{RUN_SAMPLES} paired samples complete.")
    if (sample_index + 1) % 16 == 0:
        gc.collect(); torch.cuda.empty_cache()
assert len(eval_samples) == RUN_SAMPLES
assert all(len(generated_in_memory[scale]) == RUN_SAMPLES for scale in PLADIS_SCALES)
audit_df = pd.DataFrame(audit_rows)
assert len(audit_df) == RUN_SAMPLES * len(PLADIS_SCALES)
audit_df.to_csv(RUN_DIR / "pladis_audit_per_sample.csv", index=False)
print("[OK] Paired generation complete:", len(audit_df), "images in RAM; no individual generated image persisted.")

# --- Source notebook cell 11 ---
# 11. Save a small preview and provenance; release generation model before loading OpenCLIP.
PREVIEW_COUNT = 8
sheet = Image.new("RGB", (512 * len(PLADIS_SCALES), 540 * PREVIEW_COUNT), "white")
draw = ImageDraw.Draw(sheet)
for index in range(PREVIEW_COUNT):
    for col, scale in enumerate(PLADIS_SCALES):
        draw.text((512 * col + 8, 540 * index + 8), f"{index:02d} AnchorDraw + Hyper-SD | PLADIS={scale:.1f}", fill="black")
        sheet.paste(generated_in_memory[scale][index], (512 * col, 540 * index + 28))
sheet.save(RUN_DIR / "preview_first8_four_scales.png")
display(sheet.resize((1024, 2160)))
del sheet, draw

def git_output(*args):
    result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None

def installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None

run_config = {
    "experiment_id": "OURS-SD15-HYPSD-PLADIS-SCALE-SWEEP128",
    "method_id": METHOD_ID,
    "purpose": "paired_pladis_scale_selection_128",
    "student": "SD1.5 + Hyper-SD15-4steps LoRA",
    "student_model": MODEL_ID, "hyper_sd_repo": HYPER_SD_REPO_ID,
    "hyper_sd_weight": HYPER_SD_WEIGHT_NAME, "hyper_sd_lora_scale": HYPER_SD_LORA_SCALE,
    "scheduler": "DDIMScheduler", "timestep_spacing": "trailing",
    "timesteps": actual_timesteps, "steps": HYPER_SD_NUM_INFERENCE_STEPS,
    "guidance_scale": SEMANTICDRAW_GUIDANCE_SCALE,
    "manifest": str(RUN_MANIFEST), "manifest_sha256": MANIFEST_SHA256,
    "manifest_records_total": len(all_records), "subset_rule": "first 128 records in manifest order",
    "sample_ids": [row["sample_id"] for row in records],
    "samples": RUN_SAMPLES, "resolution": list(TARGET_SIZE),
    "base_seed": BASE_SEED, "per_sample_seed": "BASE_SEED + sample_index",
    "bootstrap_steps": BOOTSTRAP_STEPS, "mask_std": MASK_STD,
    "mask_strength": MASK_STRENGTH, "mask_type": MASK_TYPE,
    "preprocess_mask_cover_alpha": PREPROCESS_MASK_COVER_ALPHA,
    "negative_prompt": NEGATIVE_PROMPT,
    "anchor_mode": BEST_ANCHOR_MODE, "weight_policy": WEIGHT_POLICY,
    "topk_attention_percent": TOPK_ATTENTION_PERCENT, "fixed_layer": FIXED_LAYER,
    "sigma_d": SIGMA_D, "semantic_sigma_scale": SEMANTIC_SIGMA_SCALE,
    "distillation_pp": False, "denoised_refiner": None,
    "pladis_scales": PLADIS_SCALES, "pladis_sparse_activation": "entmax15",
    "pladis_layers": cross_names, "pladis_capture_map": "effective mixed attention",
    "pladis_attention_chunk_size": ATTENTION_CHUNK_SIZE,
    "scale0_branch": "unmodified AnchorDraw; wrapper=0 exact parity checked on first sample",
    "metrics": ["clip_fg_x100", "clip_bg_x100", "generation_sec"],
    "clip_model": "ViT-B-32/openai", "clip_foreground_padding_ratio": 0.08,
    "selection_rule": "maximize arithmetic mean of aggregate CLIP(fg)x100 and CLIP(bg)x100; lower generation time breaks exact ties",
    "generated_individual_images_persisted": False,
    "preview_contact_sheet": "preview_first8_four_scales.png",
    "reference_baseline_dir": REFERENCE_BASELINE_DIR,
    "repo_commit": git_output("rev-parse", "HEAD"),
    "repo_status_short": git_output("status", "--short"),
    "source_sha256": {
        "semantic_anchor": hashlib.sha256((OURS_SRC / "experiments/semantic_anchor.py").read_bytes()).hexdigest(),
        "clip_metrics": hashlib.sha256((OURS_SRC / "metrics/clip_metrics.py").read_bytes()).hexdigest(),
        "semanticdraw_pipeline": hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
    },
    "versions": {name: installed_version(name) for name in [
        "torch", "diffusers", "transformers", "accelerate", "peft", "huggingface-hub",
        "open-clip-torch", "entmax", "pandas"
    ]},
}
(RUN_DIR / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
del smd, loader, batch, payload
gc.collect(); torch.cuda.empty_cache()
print("[OK] Provenance and 8-sample preview saved; generation model released.")

# --- Source notebook cell 12 ---
# 12. Score in-memory images with the same project CLIP(fg)/CLIP(bg) implementation.
from metrics.clip_metrics import CLIPScoreAccumulator
clip_accumulator = CLIPScoreAccumulator(device=device, model_name="ViT-B-32", pretrained="openai", batch_size=16)

def reset_clip_totals(accumulator):
    accumulator._pg_sum = accumulator._bg_sum = accumulator._fg_sum = 0.0
    accumulator._pg_count = accumulator._bg_count = accumulator._fg_count = 0

score_rows, metric_rows = [], []
for scale in PLADIS_SCALES:
    reset_clip_totals(clip_accumulator)
    for image, sample in zip(generated_in_memory[scale], eval_samples):
        bg_before, fg_before, fg_count_before = (
            clip_accumulator._bg_sum, clip_accumulator._fg_sum, clip_accumulator._fg_count
        )
        clip_accumulator.update_background_region(
            image, sample["foreground_masks"], sample["background_prompt"], apply_mask=True
        )
        clip_accumulator.update_foreground_regions(
            image, sample["foreground_masks"], sample["foreground_prompts"],
            padding_ratio=0.08, apply_mask=True
        )
        num_fg = clip_accumulator._fg_count - fg_count_before
        assert num_fg == len(sample["foreground_prompts"]) and num_fg > 0
        score_rows.append({
            "sample_index": sample["sample_index"], "sample_id": sample["sample_id"],
            "pladis_scale": scale,
            "clip_fg_x100": 100.0 * (clip_accumulator._fg_sum - fg_before) / num_fg,
            "clip_bg_x100": 100.0 * (clip_accumulator._bg_sum - bg_before),
            "num_foreground_regions": num_fg,
        })
    values = clip_accumulator.compute()
    assert int(values["clip_bg_count"]) == RUN_SAMPLES
    assert int(values["clip_fg_count"]) == sum(len(s["foreground_prompts"]) for s in eval_samples)
    elapsed_values = np.asarray(elapsed_by_scale[scale], dtype=np.float64)
    metric_rows.append({
        "pladis_scale": scale,
        "clip_fg_x100": values["clip_fg_x100"],
        "clip_bg_x100": values["clip_bg_x100"],
        "time_mean_sec": float(elapsed_values.mean()),
        "time_std_sec": float(elapsed_values.std()),
        "time_total_sec": float(elapsed_values.sum()),
        "num_evaluated": RUN_SAMPLES,
        "num_foreground_regions": int(values["clip_fg_count"]),
    })
    generated_in_memory[scale].clear()
    gc.collect(); torch.cuda.empty_cache()
    print(f"[CLIP] scale={scale:.1f} | FG={values['clip_fg_x100']:.4f} | BG={values['clip_bg_x100']:.4f}")

per_sample_df = pd.DataFrame(score_rows)
assert len(per_sample_df) == RUN_SAMPLES * len(PLADIS_SCALES)
base_scores = per_sample_df.loc[per_sample_df["pladis_scale"] == 0.0].set_index("sample_index")
for metric in ("clip_fg_x100", "clip_bg_x100"):
    per_sample_df[f"delta_{metric}"] = per_sample_df[metric] - per_sample_df["sample_index"].map(base_scores[metric])
per_sample_df.to_csv(RUN_DIR / "clip_per_sample_paired.csv", index=False)

metrics_df = pd.DataFrame(metric_rows).sort_values("pladis_scale")
metrics_df["selection_score"] = (metrics_df["clip_fg_x100"] + metrics_df["clip_bg_x100"]) / 2.0
baseline_row = metrics_df.loc[metrics_df["pladis_scale"] == 0.0].iloc[0]
for metric in ("clip_fg_x100", "clip_bg_x100", "selection_score"):
    metrics_df[f"delta_{metric}"] = metrics_df[metric] - baseline_row[metric]
metrics_df = metrics_df.sort_values(["selection_score", "time_mean_sec"], ascending=[False, True]).reset_index(drop=True)
metrics_df.to_csv(RUN_DIR / "pladis_scale_metrics_summary.csv", index=False)
(RUN_DIR / "pladis_scale_metrics_summary.json").write_text(
    metrics_df.to_json(orient="records", indent=2), encoding="utf-8"
)
display(Markdown("## Hyper-SD + AnchorDraw: paired PLADIS scale sweep on 128 samples"))
display(metrics_df)

# --- Source notebook cell 13 ---
# 13. Select the raw winner, assert output scope, and release image/CLIP memory.
assert set(metrics_df["pladis_scale"]) == set(PLADIS_SCALES)
assert set(metrics_df["num_evaluated"]) == {RUN_SAMPLES}
assert set(per_sample_df.groupby("pladis_scale")["sample_id"].nunique()) == {RUN_SAMPLES}
assert all(
    int(audit_df.loc[audit_df["pladis_scale"] == scale, "modified_calls"].sum())
    == (0 if scale == 0.0 else RUN_SAMPLES * 64)
    for scale in PLADIS_SCALES
)
best = metrics_df.iloc[0].to_dict()
selection = {
    "selected_pladis_scale": float(best["pladis_scale"]),
    "selection_score": float(best["selection_score"]),
    "delta_vs_scale0": float(best["delta_selection_score"]),
    "positive_scale_improved_score": bool(
        metrics_df.loc[metrics_df["pladis_scale"] > 0.0, "delta_selection_score"].max() > 0
    ),
    "selection_rule": run_config["selection_rule"],
    "num_samples": RUN_SAMPLES,
    "metrics": {key: (int(value) if key in ("num_evaluated", "num_foreground_regions") else float(value))
                for key, value in best.items()},
}
(RUN_DIR / "selected_pladis_scale.json").write_text(
    json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
)
del clip_accumulator, generated_in_memory, eval_samples
gc.collect(); torch.cuda.empty_cache()
expected_outputs = {
    "run_config.json", "pladis_audit_per_sample.csv", "clip_per_sample_paired.csv",
    "pladis_scale_metrics_summary.csv", "pladis_scale_metrics_summary.json",
    "selected_pladis_scale.json", "preview_first8_four_scales.png",
}
actual_outputs = {path.name for path in RUN_DIR.iterdir() if path.is_file()}
assert actual_outputs == expected_outputs, (actual_outputs, expected_outputs)
assert len(list(RUN_DIR.glob("*.png"))) == 1
print("[OK] Method/code and output-scope checks passed.")
print(
    f"[RESULT] Selected PLADIS scale = {selection['selected_pladis_scale']:.1f} | "
    f"score = {selection['selection_score']:.4f} | Δ vs 0 = {selection['delta_vs_scale0']:+.4f}"
)
print("[OK] Metrics + one preview sheet:", RUN_DIR)
