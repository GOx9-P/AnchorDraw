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

# Notebook: Ours/kaggle_semantic_anchor_experiments/anchordraw_sd15_hypersd_w03_l04_dpp_l001_full1073_kaggle.ipynb
# Notebook SHA256: 62474939ca90bae9d3e8bfa9ecb5d7cabe5f1fe5dfb01774bea84f6ccd5a5c36

# --- Source notebook cell 4 ---
# Frozen Hyper-SD full1073 benchmark configuration.
from pathlib import Path

EXPERIMENT_ID = "OURS-SD15-HYPSD-DPP-L001-F1073-COORD-R1"
RUN_ID = "semantic_anchor_sd15_hypersd_w03_l04_sigmad4_distillpp_l001_full1073_coord_aligned_r1"
METHOD_ID = "WM-03-TOPK-L04-SIGMA4-DISTILLPP-EQ9-L001"
RUN_MANIFEST = REPO_ROOT / "Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl"
COCO_ROOT = Path(os.environ.get("COCO_ROOT", str(WORK_DIR / 'COCO')))
RUN_ROOT = Path(str(WORK_DIR)) / RUN_ID
OUTPUT_DIR = RUN_ROOT / "generated_images"
METRICS_OUTPUT_DIR = RUN_ROOT / "metrics"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
HYPER_SD_REPO_ID = "ByteDance/Hyper-SD"
HYPER_SD_WEIGHT_NAME = "Hyper-SD15-4steps-lora.safetensors"
HYPER_SD_NUM_INFERENCE_STEPS = 4
HYPER_SD_LORA_SCALE = 1.0
SEMANTICDRAW_GUIDANCE_SCALE = 1.0

TARGET_SIZE = (512, 512)
EXPECTED_SAMPLES = 1073
BATCH_SIZE = 8
BASE_SEED = 2024
BOOTSTRAP_STEPS = 1
MASK_STD = 1.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = "discrete"
NEGATIVE_PROMPT = ""
BEST_ANCHOR_MODE = "semantic_topk_anchor"
WEIGHT_POLICY = "adaptive_bilateral"
TOPK_ATTENTION_PERCENT = 10.0
SIGMA_D = 4.0
SEMANTIC_SIGMA_SCALE = 1.0
FIXED_LAYER = {
    "layer_id": "L04",
    "layer_index": 4,
    "native_spatial_size": 16,
    "layer_name": "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor",
}
DISTILL_LAMBDA = 0.01
TEACHER_CFG_SCALE = 7.5
GUIDE_STEPS = 1
TEACHER_SEED_OFFSET = 1_000_003
EXPECTED_TIMESTEPS = [999, 749, 499, 249]
METRIC_NAMES = ("fid", "is", "clip_fg", "clip_bg", "time")
METRIC_BATCH_SIZE = 8
CLIP_BATCH_SIZE = 16
IS_SPLITS = 10
METRICS_REPORT_PREFIX = RUN_ID + "_metrics"

assert RUN_MANIFEST.exists(), RUN_MANIFEST
assert DISTILL_LAMBDA == 0.01 and SIGMA_D == 4.0
RUN_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("[OK]", EXPERIMENT_ID, "|", METHOD_ID)
print("[OK] Hyper-SD:", HYPER_SD_REPO_ID, HYPER_SD_WEIGHT_NAME)
print("[OK] Output:", RUN_ROOT)
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 6 ---
# Import Ours ahead of Baseline to avoid the two top-level data modules colliding.
import sys
import gc
import json
import time
import types
import hashlib
import importlib
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
# terminal display provided above

OURS_SRC = REPO_ROOT / "Ours/src"
BASELINE_SRC = REPO_ROOT / "Baseline/semantic-draw-main/src"
for module_name in list(sys.modules):
    if module_name == "data" or module_name.startswith("data.") or module_name == "experiments" or module_name.startswith("experiments."):
        del sys.modules[module_name]
sys.path.insert(0, str(BASELINE_SRC))
sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from experiments.semantic_anchor import ATTENTION_COORDINATE_POLICY, SemanticAnchorCapture, SemanticAnchorRuntime, compute_anchor_measurements, find_target_token_indices
assert ATTENTION_COORDINATE_POLICY == "inverse_roll_image_after_latent_shift_r1"
from experiments.distillation_pp import DistillationPPRefiner
from diffusers import UNet2DConditionModel

pipeline_path = BASELINE_SRC / "model/pipeline_semantic_draw.py"
spec = importlib.util.spec_from_file_location("pipeline_semantic_draw_distillpp_full1073", pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline

manifest_records = [json.loads(line) for line in RUN_MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(manifest_records) == EXPECTED_SAMPLES
assert len({row["sample_id"] for row in manifest_records}) == EXPECTED_SAMPLES
print(f"[OK] Imports and {len(manifest_records)} unique manifest records ready.")

# --- Source notebook cell 7 ---
# Build the unchanged full1073 loader. Originals are read later by the metric evaluator.
loader_config = COCORegionConfig(
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
loader = build_coco_region_dataloader(loader_config, shuffle=False, drop_last=False)
assert len(loader.dataset) == EXPECTED_SAMPLES
assert len(loader) == (EXPECTED_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
preview_batch = next(iter(loader))
print(f"[OK] Loader: {len(loader.dataset)} samples, {len(loader)} batches, batch_size={BATCH_SIZE}.")

# --- Source notebook cell 9 ---
# Exact payload, fixed L04 selector, and deterministic timing helpers.
def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def make_payload(batch: dict, index: int) -> dict:
    item = batch_item_to_semanticdraw_inputs(batch, index)
    metadata = item["metadata"]
    foreground_masks = item["masks"].float().cpu()
    foreground_union = foreground_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    background_mask = (1.0 - foreground_union).clamp(0, 1)
    prompts = [item["background_prompt"], *item["prompts"]]
    return {
        "sample_id": metadata["sample_id"],
        "image_id": metadata["image_id"],
        "file_name": metadata["file_name"],
        "prompts": prompts,
        "negative_prompts": [NEGATIVE_PROMPT] * len(prompts),
        "foreground_prompts": list(item["prompts"]),
        "foreground_masks": foreground_masks,
        "all_masks": torch.cat([background_mask, foreground_masks], dim=0),
        "category_names": list(metadata["category_names"]),
    }

def select_layer_attention_maps(captured_maps, output_size, layer_name):
    selected = {}
    for key, layer_maps in captured_maps.items():
        matches = [layer_map for layer_map in layer_maps if layer_map.layer_name == layer_name]
        if len(matches) != 1:
            available = [layer_map.layer_name for layer_map in layer_maps]
            raise RuntimeError(f"Expected one {layer_name} map at {key}; got {len(matches)}. Available: {available}")
        values = matches[0].values.detach().float().cpu()[None, None]
        values = F.interpolate(values, size=output_size, mode="bilinear", align_corners=False)[0, 0]
        selected[key] = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)
    return selected

def install_l04(runtime) -> None:
    def anchors(self, timestep, foreground_masks, *, strategy, topk_percent, layer_index=None, layer_name=None, **kwargs):
        maps = select_layer_attention_maps(self.attention_capture.maps, self.image_size, FIXED_LAYER["layer_name"])
        result = []
        for region_index, mask in enumerate(foreground_masks):
            measurement = compute_anchor_measurements(maps[(int(timestep), region_index)], mask.cpu(), topk_percent=topk_percent)
            prefix = "" if strategy == "argmax" else "topk_"
            result.append((float(measurement[prefix + "anchor_x"]), float(measurement[prefix + "anchor_y"])))
        return result
    runtime._anchors_from_current_step = types.MethodType(anchors, runtime)

print("[OK] Exact W03/L04 helpers ready.")

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
# Load the unchanged Hyper-SD student and a separate unfused base SD1.5 teacher.
from diffusers import DDIMScheduler
from huggingface_hub import hf_hub_download

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
    # SemanticDrawPipeline loads LCM-LoRA by default; remove it before Hyper-SD.
    if hasattr(pipe, "unfuse_lora"):
        try:
            pipe.unfuse_lora()
        except Exception as error:
            print("[WARN] unfuse_lora skipped:", error)
    if hasattr(pipe, "unload_lora_weights"):
        pipe.unload_lora_weights()
        print("[OK] Existing LoRA weights unloaded.")
        return
    if hasattr(pipe, "delete_adapters"):
        try:
            pipe.delete_adapters(["lcm"])
            print("[OK] Existing LCM adapter deleted.")
            return
        except Exception as error:
            print("[WARN] delete_adapters skipped:", error)
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

def hyper_sd_ddim_predict_x0_step(self, noise_pred, idx, latent):
    alpha = self.alpha_prod_t_sqrt[idx].to(dtype=latent.dtype, device=latent.device).clamp_min(1e-6)
    beta = self.beta_prod_t_sqrt[idx].to(dtype=latent.dtype, device=latent.device)
    return (latent - beta * noise_pred) / alpha

clear_existing_lora_adapters(smd.pipe)
hyper_sd_lora_path = hf_hub_download(repo_id=HYPER_SD_REPO_ID, filename=HYPER_SD_WEIGHT_NAME)
smd.pipe.load_lora_weights(hyper_sd_lora_path, adapter_name="hyper_sd")
if hasattr(smd.pipe, "set_adapters"):
    try:
        smd.pipe.set_adapters(["hyper_sd"], adapter_weights=[HYPER_SD_LORA_SCALE])
    except TypeError:
        smd.pipe.set_adapters(["hyper_sd"])
if hasattr(smd.pipe, "fuse_lora"):
    try:
        smd.pipe.fuse_lora(lora_scale=HYPER_SD_LORA_SCALE, safe_fusing=False)
    except TypeError:
        smd.pipe.fuse_lora(lora_scale=HYPER_SD_LORA_SCALE)
    except Exception as error:
        print("[WARN] fuse_lora skipped; running with active adapter instead:", error)

try:
    smd.pipe.scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", timestep_spacing="trailing")
except Exception as error:
    print("[WARN] DDIMScheduler.from_pretrained fallback:", error)
    smd.pipe.scheduler = DDIMScheduler.from_config(smd.pipe.scheduler.config, timestep_spacing="trailing")
smd.scheduler = smd.pipe.scheduler
smd.unet = smd.pipe.unet
smd.default_num_inference_steps = HYPER_SD_NUM_INFERENCE_STEPS
smd.default_guidance_scale = SEMANTICDRAW_GUIDANCE_SCALE
smd.prepare_hyper_sd_ddim_schedule = types.MethodType(prepare_hyper_sd_ddim_schedule, smd)
smd.prepare_lcm_schedule = smd.prepare_hyper_sd_ddim_schedule
smd.scheduler_step = types.MethodType(hyper_sd_ddim_predict_x0_step, smd)
smd.prepare_hyper_sd_ddim_schedule(num_inference_steps=HYPER_SD_NUM_INFERENCE_STEPS)

# The teacher is loaded independently from the base checkpoint and never receives Hyper-SD LoRA.
teacher_unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=dtype).to(device).eval()
for parameter in teacher_unet.parameters():
    parameter.requires_grad_(False)
actual_timesteps = [int(value) for value in smd.timesteps.detach().cpu().tolist()]
assert type(smd.scheduler).__name__ == "DDIMScheduler"
assert actual_timesteps == EXPECTED_TIMESTEPS, (actual_timesteps, EXPECTED_TIMESTEPS)
assert smd.default_num_inference_steps == HYPER_SD_NUM_INFERENCE_STEPS
assert abs(float(smd.default_guidance_scale) - SEMANTICDRAW_GUIDANCE_SCALE) < 1e-12
assert FIXED_LAYER["layer_name"] in set(smd.unet.attn_processors)
assert teacher_unet is not smd.unet
assert teacher_unet.config.in_channels == smd.unet.config.in_channels
full_refiner = DistillationPPRefiner(
    teacher_unet=teacher_unet,
    pipeline=smd,
    teacher_guidance_scale=DISTILL_LAMBDA,
    guide_steps=GUIDE_STEPS,
    teacher_cfg_scale=TEACHER_CFG_SCALE,
    seed_offset=TEACHER_SEED_OFFSET,
)
print("[OK] Hyper-SD student, separate base teacher, timesteps, guidance and L04 gates passed.")

# --- Source notebook cell 13 ---
# Generation helper and mandatory lambda=0 pixel-parity gate.
def generate_one(payload, seed, refiner):
    token_indices = [find_target_token_indices(smd.tokenizer, prompt, category) for prompt, category in zip(payload["foreground_prompts"], payload["category_names"])]
    if refiner is not None:
        refiner.configure_sample(seed)
    seed_everything(seed)
    sync_cuda()
    started = time.perf_counter()
    with SemanticAnchorCapture(smd.unet) as capture:
        capture.configure(token_indices)
        runtime = SemanticAnchorRuntime(smd, capture, image_size=TARGET_SIZE)
        install_l04(runtime)
        image, runtime_records = runtime.generate(
            prompts=payload["prompts"],
            negative_prompts=payload["negative_prompts"],
            masks=payload["all_masks"].to(device=device, dtype=torch.float32),
            foreground_masks=payload["foreground_masks"],
            mode=BEST_ANCHOR_MODE,
            weight_policy=WEIGHT_POLICY,
            bootstrap_steps=BOOTSTRAP_STEPS,
            topk_percent=TOPK_ATTENTION_PERCENT,
            spatial_sigma_latent=SIGMA_D,
            semantic_sigma_scale=SEMANTIC_SIGMA_SCALE,
            mask_stds=MASK_STD,
            mask_strengths=MASK_STRENGTH,
            preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
            attention_layer_name=FIXED_LAYER["layer_name"],
            denoised_refiner=refiner,
        )
    sync_cuda()
    elapsed = time.perf_counter() - started
    records = [] if refiner is None else [record.to_dict() for record in refiner.records]
    return image, elapsed, records

parity_payload = make_payload(preview_batch, 0)
zero_refiner = DistillationPPRefiner(
    teacher_unet=teacher_unet, pipeline=smd, teacher_guidance_scale=0.0,
    guide_steps=GUIDE_STEPS, teacher_cfg_scale=TEACHER_CFG_SCALE, seed_offset=TEACHER_SEED_OFFSET,
)
baseline_image, _, _ = generate_one(parity_payload, BASE_SEED, None)
zero_image, _, zero_records = generate_one(parity_payload, BASE_SEED, zero_refiner)
assert np.array_equal(np.asarray(baseline_image), np.asarray(zero_image)), "lambda=0 changed W03 pixels"
assert len(zero_records) == len(EXPECTED_TIMESTEPS)
assert sum(int(record["applied"]) for record in zero_records) == 0
del parity_payload, zero_refiner, baseline_image, zero_image, zero_records
gc.collect()
torch.cuda.empty_cache()
print("[OK] lambda=0 is pixel-identical to the original W03/L04 path.")

# --- Source notebook cell 14 ---
# Generate all 1073 samples with fixed lambda=0.01 and audit every student step.
existing_generated = sorted(OUTPUT_DIR.glob("*_generated.png"))
assert not existing_generated, f"{len(existing_generated)} stale images already exist in {OUTPUT_DIR}"
summary = []
audit_rows = []
global_index = 0
for batch_index, batch in enumerate(loader):
    print(f"[BATCH] {batch_index + 1}/{len(loader)} | completed={global_index}/{EXPECTED_SAMPLES}")
    for local_index in range(len(batch["sample_ids"])):
        payload = make_payload(batch, local_index)
        assert payload["sample_id"] == manifest_records[global_index]["sample_id"]
        seed = BASE_SEED + global_index
        generated, elapsed, records = generate_one(payload, seed, full_refiner)
        applied = [record for record in records if record["applied"]]
        assert len(records) == len(EXPECTED_TIMESTEPS)
        assert len(applied) == GUIDE_STEPS
        first = applied[0]
        assert first["step_index"] == 0 and first["timestep"] == EXPECTED_TIMESTEPS[0]
        assert first["teacher_timestep"] == EXPECTED_TIMESTEPS[1]
        assert abs(float(first["teacher_guidance_scale"]) - DISTILL_LAMBDA) < 1e-12
        assert abs(float(first["teacher_cfg_scale"]) - TEACHER_CFG_SCALE) < 1e-12
        expected_l2 = DISTILL_LAMBDA * float(first["student_teacher_l2"])
        actual_l2 = float(first["refinement_l2"])
        assert abs(actual_l2 - expected_l2) <= max(2e-4, 0.01 * expected_l2)

        stem = f"{global_index:04d}_{payload['sample_id']}"
        generated_path = OUTPUT_DIR / f"{stem}_generated.png"
        generated.save(generated_path)
        with Image.open(generated_path) as check_image:
            assert check_image.convert("RGB").size == TARGET_SIZE
        summary.append({
            "index": global_index,
            "batch_index": batch_index,
            "local_index": local_index,
            "sample_id": payload["sample_id"],
            "image_id": int(payload["image_id"]),
            "file_name": payload["file_name"],
            "seed": seed,
            "experiment_id": EXPERIMENT_ID,
            "method_id": METHOD_ID,
            "attention_coordinate_policy": ATTENTION_COORDINATE_POLICY,
            "model_family": "sd15",
            "sampler": "DDIMScheduler trailing + Hyper-SD15-4steps LoRA",
            "anchor_mode": BEST_ANCHOR_MODE,
            "weight_policy": WEIGHT_POLICY,
            "fixed_layer": FIXED_LAYER["layer_id"],
            "spatial_sigma_latent": SIGMA_D,
            "semantic_sigma_scale": SEMANTIC_SIGMA_SCALE,
            "bootstrap_steps": BOOTSTRAP_STEPS,
            "distillationpp_lambda": DISTILL_LAMBDA,
            "distillationpp_guide_steps": GUIDE_STEPS,
            "teacher_cfg_scale": TEACHER_CFG_SCALE,
            "num_regions_including_background": len(payload["prompts"]),
            "elapsed_sec": float(elapsed),
            "generated_path": str(generated_path),
        })
        for record in records:
            audit_rows.append({"sample_index": global_index, "sample_id": payload["sample_id"], "seed": seed, **record})
        global_index += 1
        del payload, generated, records, applied, first
        torch.cuda.empty_cache()

assert global_index == EXPECTED_SAMPLES
assert len(summary) == EXPECTED_SAMPLES
assert len(list(OUTPUT_DIR.glob("*_generated.png"))) == EXPECTED_SAMPLES
SUMMARY_PATH = RUN_ROOT / "generation_summary.json"
AUDIT_CSV_PATH = RUN_ROOT / "distillationpp_step_audit.csv"
AUDIT_JSONL_PATH = RUN_ROOT / "distillationpp_step_audit.jsonl"
SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
pd.DataFrame(audit_rows).to_csv(AUDIT_CSV_PATH, index=False)
with AUDIT_JSONL_PATH.open("w", encoding="utf-8") as handle:
    for row in audit_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"[OK] Generated and audited {global_index} images.")

# --- Source notebook cell 15 ---
# Persist provenance and reject a silent Distillation++ no-op.
def git_output(*args):
    result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None

audit_df = pd.DataFrame(audit_rows)
applied_df = audit_df.loc[audit_df["applied"] == True].copy()
assert len(audit_df) == EXPECTED_SAMPLES * len(EXPECTED_TIMESTEPS)
assert len(applied_df) == EXPECTED_SAMPLES
assert applied_df["sample_id"].nunique() == EXPECTED_SAMPLES
assert set(applied_df["step_index"].astype(int)) == {0}
assert set(applied_df["timestep"].astype(int)) == {EXPECTED_TIMESTEPS[0]}
assert set(applied_df["teacher_timestep"].astype(int)) == {EXPECTED_TIMESTEPS[1]}
assert np.allclose(applied_df["teacher_guidance_scale"], DISTILL_LAMBDA, rtol=0, atol=1e-12)
assert np.allclose(applied_df["teacher_cfg_scale"], TEACHER_CFG_SCALE, rtol=0, atol=1e-12)
assert (applied_df["student_teacher_l2"] > 0).all() and (applied_df["refinement_l2"] > 0).all()

run_config = {
    "experiment_id": EXPERIMENT_ID,
    "run_id": RUN_ID,
    "method_id": METHOD_ID,
    "attention_coordinate_policy": ATTENTION_COORDINATE_POLICY,
    "purpose": "Hyper-SD full1073 metric with fixed lambda=0.01 selected on paired LCM validation",
    "lambda_selection": {"selected_lambda": DISTILL_LAMBDA, "validation_samples": 128, "selection_score": 27.0184, "runner_up_lambda": 0.02, "runner_up_score": 27.0175},
    "paper": "arXiv:2412.08871v1",
    "official_code": "https://github.com/geonyeong-park/inference_distillation",
    "official_code_audited_commit": "59dd61e6b2393dd16b8cadadef8897dad22e468a",
    "formulation": "paper Eq.9 clean-estimate interpolation",
    "student": "SD1.5 + Hyper-SD15-4steps LoRA",
    "student_guidance_scale": float(smd.default_guidance_scale),
    "teacher": "separate unfused base SD1.5 UNet",
    "teacher_cfg_scale": TEACHER_CFG_SCALE,
    "distillationpp_lambda": DISTILL_LAMBDA,
    "guide_steps": GUIDE_STEPS,
    "renoise": "fresh Gaussian noise at next scheduled student timestep",
    "teacher_rng": f"private seed = sample seed + {TEACHER_SEED_OFFSET}",
    "manifest": str(RUN_MANIFEST),
    "manifest_records": EXPECTED_SAMPLES,
    "sample_order": "complete manifest order",
    "resolution": list(TARGET_SIZE),
    "batch_size": BATCH_SIZE,
    "base_seed": BASE_SEED,
    "per_sample_seed": "BASE_SEED + global_index",
    "timesteps": actual_timesteps,
    "bootstrap_steps": BOOTSTRAP_STEPS,
    "mask_std": MASK_STD,
    "mask_strength": MASK_STRENGTH,
    "preprocess_mask_cover_alpha": PREPROCESS_MASK_COVER_ALPHA,
    "anchor_mode": BEST_ANCHOR_MODE,
    "weight_policy": WEIGHT_POLICY,
    "topk_attention_percent": TOPK_ATTENTION_PERCENT,
    "fixed_layer": FIXED_LAYER,
    "spatial_sigma_latent": SIGMA_D,
    "semantic_sigma_scale": SEMANTIC_SIGMA_SCALE,
    "metrics": list(METRIC_NAMES),
    "metric_batch_size": METRIC_BATCH_SIZE,
    "clip_batch_size": CLIP_BATCH_SIZE,
    "is_splits": IS_SPLITS,
    "generated_images_persisted": True,
    "overlay_images_persisted": False,
    "git_commit": git_output("rev-parse", "HEAD"),
    "git_status_short": git_output("status", "--short"),
    "semantic_anchor_sha256": hashlib.sha256((OURS_SRC / "experiments/semantic_anchor.py").read_bytes()).hexdigest(),
    "distillation_pp_sha256": hashlib.sha256((OURS_SRC / "experiments/distillation_pp.py").read_bytes()).hexdigest(),
}
RUN_CONFIG_PATH = RUN_ROOT / "run_config.json"
RUN_CONFIG_PATH.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
audit_summary = {
    "num_samples": EXPECTED_SAMPLES,
    "num_student_steps": int(len(audit_df)),
    "num_applied_teacher_steps": int(len(applied_df)),
    "teacher_timestep": int(applied_df["teacher_timestep"].iloc[0]),
    "teacher_elapsed_mean_sec": float(applied_df["teacher_elapsed_sec"].mean()),
    "student_teacher_l2_mean": float(applied_df["student_teacher_l2"].mean()),
    "refinement_l2_mean": float(applied_df["refinement_l2"].mean()),
}
AUDIT_SUMMARY_PATH = RUN_ROOT / "distillationpp_audit_summary.json"
AUDIT_SUMMARY_PATH.write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("[OK] Method/code gates and provenance passed.")
print(json.dumps(audit_summary, indent=2))

# --- Source notebook cell 17 ---
# Release both diffusion models before loading Inception and CLIP metric models.
for variable_name in ("full_refiner", "teacher_unet", "smd", "runtime", "capture", "generated", "payload", "batch", "loader", "preview_batch", "summary", "audit_rows", "audit_df", "applied_df"):
    if variable_name in globals():
        del globals()[variable_name]
gc.collect()
torch.cuda.empty_cache()
try:
    torch.cuda.ipc_collect()
except Exception:
    pass
print("[OK] Generation models released before metric evaluation.")

# --- Source notebook cell 18 ---
# Evaluate the same full1073 metric implementation as the existing experiments.
from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report

assert SUMMARY_PATH.exists()
assert len(list(OUTPUT_DIR.glob("*_generated.png"))) == EXPECTED_SAMPLES
metric_config = MetricEvaluationConfig(
    manifest_path=RUN_MANIFEST,
    coco_root=COCO_ROOT,
    generated_dir=OUTPUT_DIR,
    generation_summary=SUMMARY_PATH,
    output_dir=METRICS_OUTPUT_DIR,
    model_family="sd15",
    target_size=TARGET_SIZE,
    metrics=METRIC_NAMES,
    batch_size=METRIC_BATCH_SIZE,
    num_workers=0,
    pin_memory=False,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    clip_batch_size=CLIP_BATCH_SIZE,
    is_splits=IS_SPLITS,
)
metric_report = run_evaluation(metric_config)
metrics_json, metrics_csv = write_metrics_report(metric_report, METRICS_OUTPUT_DIR, prefix=METRICS_REPORT_PREFIX)
assert int(metric_report["num_manifest_records"]) == EXPECTED_SAMPLES
assert int(metric_report["num_evaluated"]) == EXPECTED_SAMPLES
assert int(metric_report["num_missing_generated"]) == 0
values = metric_report["metrics"]
result_summary = {
    "experiment_id": EXPERIMENT_ID,
    "run_id": RUN_ID,
    "method_id": METHOD_ID,
    "num_evaluated": int(metric_report["num_evaluated"]),
    "fid": values.get("fid"),
    "is_mean": values.get("is_mean"),
    "is_std": values.get("is_std"),
    "clip_fg_x100": values.get("clip_fg_x100"),
    "clip_bg_x100": values.get("clip_bg_x100"),
    "time_mean_sec": values.get("time_mean_sec"),
    "time_total_sec": values.get("time_total_sec"),
    "metrics_json": str(metrics_json),
    "metrics_csv": str(metrics_csv),
}
RESULT_SUMMARY_PATH = RUN_ROOT / "result_summary.json"
RESULT_SUMMARY_PATH.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2), encoding="utf-8")
display(Markdown("## Full1073 metric evaluation passed"))
display(pd.DataFrame([result_summary]))
print(json.dumps(result_summary, ensure_ascii=False, indent=2))

# --- Source notebook cell 19 ---
# Final artifact contract.
required_files = [RUN_CONFIG_PATH, SUMMARY_PATH, AUDIT_CSV_PATH, AUDIT_JSONL_PATH, AUDIT_SUMMARY_PATH, RESULT_SUMMARY_PATH, Path(metrics_json), Path(metrics_csv)]
missing = [str(path) for path in required_files if not path.exists()]
assert not missing, missing
assert len(list(OUTPUT_DIR.glob("*_generated.png"))) == EXPECTED_SAMPLES
print(RUN_ROOT.name + "/")
print("  run_config.json")
print("  generation_summary.json")
print("  distillationpp_step_audit.csv")
print("  distillationpp_step_audit.jsonl")
print("  distillationpp_audit_summary.json")
print("  result_summary.json")
print(f"  generated_images/  ({EXPECTED_SAMPLES} PNG)")
print("  metrics/")
print("    " + Path(metrics_json).name)
print("    " + Path(metrics_csv).name)
print("[DONE] Full1073 Distillation++ benchmark is comparison-ready.")
