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

# Notebook: Ours/kaggle_semantic_anchor_experiments/anchordraw_sdxl_euler_w03_l04_sigmad4_full1073_kaggle.ipynb
# Notebook SHA256: 8f58a78048518bde30c363fe503619e49ea7350298970f667458b679704a1d07

# --- Source notebook cell 2 ---
from pathlib import Path
import os, json, importlib.util, types, zipfile
import torch

RUN_MANIFEST = REPO_ROOT / 'Ours/data_manifests/coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_all.jsonl'
COCO_ROOT = Path(os.environ.get('COCO_ROOT', str(WORK_DIR / 'COCO')))
OUTPUT_DIR = WORK_DIR / 'anchordraw_sdxl_euler_w03_l04_sigmad4_full1073_coord_aligned_r1'
RUN_SAMPLES, BASE_SEED = 1073, 2024
TARGET_SIZE, T_INDEX_LIST = (1024, 1024), [0, 4, 12, 25, 37]
MASK_STD, MASK_STRENGTH, PREPROCESS_MASK_COVER_ALPHA = 0.0, 1.0, 0.0
BOOTSTRAP_STEPS, GUIDANCE_SCALE = 1, 0.0
NEGATIVE_PROMPT = ''
manifest_rows = [json.loads(s) for s in RUN_MANIFEST.read_text(encoding='utf-8').splitlines() if s.strip()]
assert len(manifest_rows) == 1073 and RUN_SAMPLES <= len(manifest_rows)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('[OK] Repo:', REPO_ROOT, '| Manifest:', len(manifest_rows), '| Full:', RUN_SAMPLES)

GENERATED_DIR = OUTPUT_DIR / 'generated_images'
METRICS_DIR = OUTPUT_DIR / 'metrics'
RUN_SUMMARY_PATH = OUTPUT_DIR / 'generation_summary.json'
AUDIT_PATH = OUTPUT_DIR / 'anchordraw_audit.jsonl'
PROTOCOL_PATH = OUTPUT_DIR / 'protocol.json'
for folder in (GENERATED_DIR, METRICS_DIR):
    folder.mkdir(parents=True, exist_ok=True)
existing_images = list(GENERATED_DIR.glob('*_generated.png'))
if existing_images or RUN_SUMMARY_PATH.exists() or AUDIT_PATH.exists() or PROTOCOL_PATH.exists():
    raise RuntimeError(
        f'Output folder already contains a run: {OUTPUT_DIR}. Use a fresh OUTPUT_DIR; do not mix or overwrite experiments.'
    )
METRIC_NAMES = ('fid', 'is', 'clip_fg', 'clip_bg', 'time')
METRIC_BATCH_SIZE, CLIP_BATCH_SIZE, IS_SPLITS = 1, 4, 10
print('[OK] Generated:', GENERATED_DIR, '| Metrics:', METRICS_DIR)
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 4 ---
# Load hai phiên bản class từ CÙNG source: baseline đã vá Kaggle và baseline + identity hook sites.
# Không tạo model thứ hai: gọi trực tiếp __call__ của class hook trên cùng instance.
if 'torchao' in sys.modules or importlib.util.find_spec('torchao') is not None:
    raise RuntimeError('torchao vẫn import được; restart Kaggle session và Run All.')
OURS_SRC = REPO_ROOT / 'Ours/src'
BASELINE_SRC = REPO_ROOT / 'Baseline/semantic-draw-main/src'
sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
import data as _ours_data
assert Path(_ours_data.__file__).resolve().is_relative_to(OURS_SRC.resolve()), 'Wrong data module; restart Kaggle session.'
sys.path.insert(0, str(BASELINE_SRC))
from safetensors.torch import load_file as _safetensors_load_file
pipeline_path = BASELINE_SRC / 'model/pipeline_semantic_draw_sdxl.py'
source = pipeline_path.read_text(encoding='utf-8')
def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'Source drift: expected one occurrence, found {count}: {old[:90]!r}')
    return text.replace(old, new, 1)

# Giữ đúng ba patch trong notebook baseline full1073.
source = replace_once(source,
    '                latent = torch.where(count_all > 0, value / count_all, value)',
    '                denom = count_all.clamp_min(torch.finfo(count_all.dtype).tiny)\n                latent = torch.where(count_all > 0, value / denom, latent)')
source = replace_once(source,
    '                has_background = False # Regard that background does not exist.',
    '                has_background = False # Regard that background does not exist.\n                num_masks = len(prompts)\n                num_prompts = len(prompts)\n                num_nprompts = len(negative_prompts)')
source = replace_once(source,
    '        else:\n            image = latent\n\n        # Return PIL Image.',
    '        else:\n            image = latent\n\n        if output_type == "latent":\n            return image\n\n        # Return PIL Image.')

hooked_source = replace_once(source,
    '                fg_mask = fg_masks[:, i]',
    '                fg_mask = fg_masks[:, i]\n                fg_mask = self._anchor_mask_hook(fg_mask, i, t)')
hooked_source = replace_once(hooked_source,
    '                    latent_model_input = torch.cat([latent_] * 2) if do_classifier_free_guidance else latent_',
    '                    latent_ = self._anchor_pre_unet_hook(latent_, fg_mask_, i, t)\n                    latent_model_input = torch.cat([latent_] * 2) if do_classifier_free_guidance else latent_')
hooked_source = replace_once(hooked_source,
    '                    latent_ = self.scheduler_step(noise_pred, i, latent_)',
    '                    student_noisy_latent = latent_\n                    latent_ = self.scheduler_step(noise_pred, i, latent_)\n                    latent_ = self._anchor_post_euler_hook(latent_, student_noisy_latent, noise_pred, i, t)')

hooked_source = replace_once(hooked_source,
    '                    fg_mask_ = fg_mask_ * tile_masks[:, j:j + 1, h_start:h_end, w_start:w_end]',
    '                    latent_ = self._anchor_post_region_hook(latent_, i, t)\n                    fg_mask_ = fg_mask_ * tile_masks[:, j:j + 1, h_start:h_end, w_start:w_end]')

def load_pipeline_module(name, text):
    module = types.ModuleType(name)
    module.__file__ = str(pipeline_path)
    exec(compile(text, str(pipeline_path), 'exec'), module.__dict__)
    module.load_file = lambda filename, device=None: _safetensors_load_file(filename, device='cpu')
    return module
baseline_module = load_pipeline_module('sdxl_euler_baseline_patched', source)
hooked_module = load_pipeline_module('sdxl_euler_anchor_w03_hooks', hooked_source)
BaselineClass = baseline_module.SemanticDrawSDXLPipeline
HookedClass = hooked_module.SemanticDrawSDXLPipeline
assert source != hooked_source
print('[OK] Baseline and AnchorDraw-hook classes compiled from the same source + same compatibility patches.')

# --- Source notebook cell 5 ---
# Load một student UNet duy nhất; giữ cách xử lý SDXL VAE/white bootstrap của baseline notebook.
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

token = os.environ.get('HF_TOKEN')
if not token:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        token = None
if token:
    from huggingface_hub import login
    login(token=token)

# Phát hiện model còn sống từ notebook trước hoặc do chạy lại cell load trong cùng session.
# empty_cache không giải phóng tensor/model còn được tham chiếu; khi guard này fail, Restart Session.
import gc
gc.collect()
torch.cuda.empty_cache()
free_bytes, total_bytes = torch.cuda.mem_get_info()
free_gib, total_gib = free_bytes / 2**30, total_bytes / 2**30
allocated_gib = torch.cuda.memory_allocated() / 2**30
print(f'[VRAM before load] free={free_gib:.2f}/{total_gib:.2f} GiB; PyTorch allocated={allocated_gib:.2f} GiB')
if 'smd' in globals() or (total_gib < 18 and free_gib < 12):
    raise RuntimeError('GPU còn giữ model/tensor từ session trước hoặc VRAM T4 không đủ. Restart Kaggle Session rồi Run All notebook này; không chạy tiếp từ cell lỗi.')
seed_everything(BASE_SEED)
smd = BaselineClass(device=torch.device('cuda:0'), dtype=torch.float16, hf_key=None,
                    has_i2t=False, t_index_list=T_INDEX_LIST, default_mask_std=MASK_STD,
                    default_mask_strength=MASK_STRENGTH,
                    default_preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA, mask_type='discrete')
if hasattr(smd.pipe, 'enable_attention_slicing'): smd.pipe.enable_attention_slicing()
if hasattr(smd.pipe, 'enable_vae_slicing'): smd.pipe.enable_vae_slicing()
if hasattr(smd.pipe, 'enable_vae_tiling'): smd.pipe.enable_vae_tiling()
def _upcast_full_vae_for_kaggle():
    smd.vae.to(dtype=torch.float32)
smd.upcast_vae = _upcast_full_vae_for_kaggle
def _safe_get_white_background(height, width):
    latent_h, latent_w = height // smd.vae_scale_factor, width // smd.vae_scale_factor
    cache = getattr(smd, '_safe_white_bootstrap_latent', None)
    if cache is None or cache.shape[-2] < latent_h or cache.shape[-1] < latent_w:
        old_dtype = smd.vae.dtype
        try:
            smd.vae.to(dtype=torch.float32)
            white_img = torch.ones(1, 3, height, width, dtype=torch.float32, device=smd.device)
            with torch.random.fork_rng(devices=[smd.device.index or 0] if smd.device.type == 'cuda' else []):
                torch.manual_seed(BASE_SEED + 500000)
                white_latent = smd.encode_imgs(white_img).to(dtype=smd.dtype)
        finally:
            smd.vae.to(dtype=old_dtype)
        if not bool(torch.isfinite(white_latent).all().item()):
            raise RuntimeError('Safe white-bootstrap latent contains NaN/Inf.')
        smd._safe_white_bootstrap_latent = white_latent
    return smd._safe_white_bootstrap_latent[..., :latent_h, :latent_w]
smd.get_white_background = _safe_get_white_background
_ = smd.get_white_background(*TARGET_SIZE)
assert type(smd.scheduler).__name__ == 'EulerDiscreteScheduler'
assert smd.scheduler.config.timestep_spacing == 'trailing'
assert smd.scheduler.config.prediction_type == 'epsilon'
assert smd.default_num_inference_steps == 4 and len(smd.timesteps) == 5
assert float(smd.gammas.abs().max()) == 0.0
assert MASK_STD == 0.0 and BOOTSTRAP_STEPS == 1 and GUIDANCE_SCALE == 0.0
print('[OK] Euler schedule:', [int(t) for t in smd.timesteps.cpu().tolist()])
print('[OK] Sigmas:', [float(s) for s in smd.sigmas.cpu().tolist()])

# --- Source notebook cell 6 ---
# Cùng loader/mask/manifest với baseline; return_image=False cho generation; evaluator baseline sẽ đọc ảnh COCO gốc riêng.
config = COCORegionConfig(coco_root=COCO_ROOT, split='val2017',
    instances_json=COCO_ROOT / 'annotations/instances_val2017.json',
    captions_json=COCO_ROOT / 'annotations/captions_val2017.json',
    manifest_path=RUN_MANIFEST, profile='multidiffusion_coco_all', model_family='sdxl',
    target_size=TARGET_SIZE, return_image=False, cache_resized_masks=True,
    cache_dir=WORK_DIR / 'semanticdraw_mask_cache', batch_size=1, num_workers=0,
    pin_memory=False, persistent_workers=False)
loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
assert len(loader.dataset) == RUN_SAMPLES
print('[OK] Loader ready. First:', loader.dataset.records[0]['sample_id'])

# --- Source notebook cell 7 ---
# Capture chỉ một attn2. Output student vẫn do processor gốc trả về.
import time
import numpy as np
from PIL import Image, ImageDraw
from itertools import islice
import inspect
from experiments.semantic_anchor import (
    SemanticAnchorCapture, CapturedLayerMap,
    find_target_token_indices, aggregate_attention_maps,
    compute_anchor_measurements, build_weighted_overlap_masks,
    _shift_to_reference_points,
)
from util import shift_to_mask_bbox_center as baseline_shift_to_mask_bbox_center
from math import isqrt

LAYER_NAME = 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor'
SPATIAL_SIGMA_LATENT = 4.0
TOPK_PERCENT = 10.0
SEMANTIC_SIGMA_SCALE = 1.0
assert LAYER_NAME in smd.unet.attn_processors, (
    'Structural L04 not found in this SDXL UNet. Do not silently choose another layer.'
)
capture = SemanticAnchorCapture(smd.unet)
ATTENTION_QUERY_CHUNK = 64  # giới hạn đỉnh VRAM của phép đọc xác suất

def bootstrap_foreground_shifts(mask, background_count):
    # Match util.shift_to_mask_bbox_center(..., reverse=True) on the actual
    # quantized latent masks used by this step, including empty-mask behavior.
    h, w = mask.shape[-2:]
    flat = mask.reshape(-1, h, w)
    columns = flat.sum(dim=-2) > 0
    rows = flat.sum(dim=-1) > 0
    left = torch.argmax(columns * torch.arange(w, 0, -1, device=mask.device), dim=1)
    right = torch.argmax(columns * torch.arange(w, device=mask.device), dim=1)
    top = torch.argmax(rows * torch.arange(h, 0, -1, device=mask.device), dim=1)
    bottom = torch.argmax(rows * torch.arange(h, device=mask.device), dim=1)
    dy = h // 2 - (top + bottom + 1) // 2
    dx = w // 2 - (left + right + 1) // 2
    return list(zip(dy[background_count:].cpu().tolist(), dx[background_count:].cpu().tolist()))

def semantic_foreground_shifts(points, latent_size, image_size):
    # Exact integer shifts used by _shift_to_reference_points(reverse=False).
    latent_h, latent_w = latent_size
    image_h, image_w = image_size
    return [(latent_h // 2 - round(float(y) * latent_h / image_h),
             latent_w // 2 - round(float(x) * latent_w / image_w)) for x, y in points]

def attention_in_mask_frame(attention_map, shift_yx, latent_size, image_size):
    # Attention was measured after a roll in latent space. Undo that roll
    # on the image-resolution map before testing pixels inside the COCO mask.
    # This is geometric alignment, not an inverse of U-Net downsampling.
    assert tuple(attention_map.shape) == tuple(image_size)
    latent_h, latent_w = latent_size
    image_h, image_w = image_size
    dy, dx = shift_yx
    inverse_image_shift = (-round(dy * image_h / latent_h),
                           -round(dx * image_w / latent_w))
    return attention_map.roll(shifts=inverse_image_shift, dims=(-2, -1))

# Geometry gates run before loading 1,073 images. They check the actual
# bootstrap utility and semantic shift helper, not merely inverse-roll math.
_geometry_masks = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
_geometry_masks[0, 0, 1:3, 2:4] = 1
_geometry_masks[1, 0, 5:8, 4:7] = 1
_geometry_latents = torch.arange(128, dtype=torch.float32).reshape(2, 1, 8, 8)
_bootstrap_shifts = bootstrap_foreground_shifts(_geometry_masks, 0)
_bootstrap_expected = baseline_shift_to_mask_bbox_center(
    _geometry_latents, _geometry_masks, reverse=True)
assert torch.equal(_bootstrap_expected, torch.stack([
    value.roll(shifts=shift, dims=(-2, -1))
    for value, shift in zip(_geometry_latents, _bootstrap_shifts)]))
_geometry_with_background = torch.cat([
    torch.zeros_like(_geometry_masks[:1]), _geometry_masks], dim=0)
assert bootstrap_foreground_shifts(_geometry_with_background, 1) == _bootstrap_shifts
_geometry_empty = torch.zeros_like(_geometry_masks[:1])
_empty_shift = bootstrap_foreground_shifts(_geometry_empty, 0)[0]
assert torch.equal(baseline_shift_to_mask_bbox_center(
    _geometry_latents[:1], _geometry_empty, reverse=True),
    _geometry_latents[:1].roll(shifts=_empty_shift, dims=(-2, -1)))
_geometry_points = [(256.0, 512.0), (896.0, 128.0)]
_semantic_shifts = semantic_foreground_shifts(_geometry_points, (8, 8), TARGET_SIZE)
_semantic_expected = _shift_to_reference_points(
    _geometry_latents, _geometry_points, image_size=TARGET_SIZE, reverse=False)
assert torch.equal(_semantic_expected, torch.stack([
    value.roll(shifts=shift, dims=(-2, -1))
    for value, shift in zip(_geometry_latents, _semantic_shifts)]))
_attention_reference = torch.zeros((64, 64), dtype=torch.float32)
_attention_reference[15, 23] = 1.0
_attention_shift = (2, -3)
_attention_centered = _attention_reference.roll(
    shifts=(_attention_shift[0] * 8, _attention_shift[1] * 8), dims=(-2, -1))
assert torch.equal(attention_in_mask_frame(
    _attention_centered, _attention_shift, (8, 8), (64, 64)), _attention_reference)
del _geometry_masks, _geometry_latents, _bootstrap_shifts, _bootstrap_expected
del _geometry_with_background, _geometry_empty, _empty_shift
del _geometry_points, _semantic_shifts, _semantic_expected
del _attention_reference, _attention_shift, _attention_centered
print('[PASS] Bootstrap/semantic attention coordinate geometry.')

class OutputPreservingCapture:
    def __init__(self, original, layer_name, store):
        self.original = original
        self.layer_name = layer_name
        self.store = store
        self.original_accepts_temb = 'temb' in inspect.signature(original.__call__).parameters
    def _capture_map(self, attn, hidden_states, encoder_hidden_states, attention_mask, temb):
        if not self.store.enabled:
            return
        if attention_mask is not None:
            raise RuntimeError('L04 streaming capture expects no attention_mask in this SDXL protocol.')
        if encoder_hidden_states is None:
            raise RuntimeError('Selected L04 must be cross-attention.')
        if getattr(attn, 'spatial_norm', None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if hidden_states.ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h*w).transpose(1, 2)
        batch_size, query_length, _ = hidden_states.shape
        side = isqrt(query_length)
        assert side*side == query_length, (query_length, self.layer_name)
        if getattr(attn, 'group_norm', None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        if getattr(attn, 'norm_cross', False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        heads = int(attn.heads)
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        query = query.reshape(batch_size, heads, query_length, -1)
        key = key.reshape(batch_size, heads, key.shape[1], -1)
        region_count = self.store.region_count
        assert batch_size >= region_count > 0
        key_length = key.shape[2]
        for region in range(region_count):
            batch_index = batch_size - region_count + region
            selected = [i for i in self.store.token_indices[region] if 0 <= i < key_length]
            assert selected, (region, key_length)
            selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=query.device)
            token_values = torch.empty(query_length, device=query.device, dtype=torch.float32)
            for start in range(0, query_length, ATTENTION_QUERY_CHUNK):
                q_chunk = query[batch_index, :, start:start+ATTENTION_QUERY_CHUNK, :]
                probs = attn.get_attention_scores(q_chunk, key[batch_index], None)
                token_values[start:start+q_chunk.shape[1]] = probs.index_select(-1, selected_tensor).mean(dim=(0, 2)).float()
                del probs
            token_map = token_values.reshape(side, side).cpu()
            token_map = (token_map - token_map.min()) / (token_map.max() - token_map.min()).clamp_min(1e-8)
            timestep = self.store.current_timestep
            assert timestep is not None
            self.store.maps[(timestep, region)].append(
                CapturedLayerMap(layer_name=self.layer_name, spatial_size=side, values=token_map))
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        if args:
            raise RuntimeError(f'Unexpected positional attention arguments: {len(args)}')
        original_kwargs = dict(kwargs)
        if self.original_accepts_temb:
            original_kwargs['temb'] = temb
        result = self.original(attn, hidden_states,
                               encoder_hidden_states=encoder_hidden_states,
                               attention_mask=attention_mask, **original_kwargs)
        # Tính QK từng khối nhỏ, bỏ V/to_out; không trả output tính lại.
        self._capture_map(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        return result

_original_processors = dict(smd.unet.attn_processors)
def attach_capture():
    processors = dict(_original_processors)
    processors[LAYER_NAME] = OutputPreservingCapture(
        _original_processors[LAYER_NAME], LAYER_NAME, capture.store
    )
    smd.unet.set_attn_processor(processors)

def detach_capture():
    capture.store.disable()
    smd.unet.set_attn_processor(dict(_original_processors))

def token_indices_for(item):
    names = item['metadata']['category_names']
    assert len(names) == len(item['prompts'])
    indices, pairs = [], []
    for prompt, category in zip(item['prompts'], names):
        first = find_target_token_indices(smd.tokenizer, prompt, category)
        second = find_target_token_indices(smd.tokenizer_2, prompt, category)
        indices.append(sorted(set(first) | set(second)))
        pairs.append({'tokenizer_1': first, 'tokenizer_2': second})
    return indices, pairs

def call_args_for(item, output_type='pil'):
    return dict(prompts=item['prompts'],
        negative_prompts=[NEGATIVE_PROMPT] * len(item['prompts']),
        background_prompt=item['background_prompt'],
        background_negative_prompt=NEGATIVE_PROMPT,
        mask_stds=MASK_STD, mask_strengths=MASK_STRENGTH,
        height=item['height'], width=item['width'],
        num_inference_steps=None, guidance_scale=GUIDANCE_SCALE,
        bootstrap_steps=BOOTSTRAP_STEPS,
        preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
        guidance_rescale=0.0, do_blend=False, output_type=output_type)

def call_baseline(item, output_type='pil'):
    return smd(masks=item['masks'].clone().to('cuda:0', dtype=torch.float32),
               **call_args_for(item, output_type))

def call_hooked(item, output_type='pil'):
    return HookedClass.__call__(smd,
        masks=item['masks'].clone().to('cuda:0', dtype=torch.float32),
        **call_args_for(item, output_type))

print('[OK] Selected SDXL structural L04:', LAYER_NAME)
print('[INFO] This layer must still be evaluated as an SDXL candidate, not inherited as optimum.')

# --- Source notebook cell 8 ---
# Gate: processor đọc L04 + 4 hook identity không được đổi latent baseline.
first_item = batch_item_to_semanticdraw_inputs(next(iter(loader)), 0)
seed_everything(BASE_SEED)
control_latent = call_baseline(first_item, 'latent').detach().cpu()
indices, tokenizer_positions = token_indices_for(first_item)
capture.configure(indices)
smd._anchor_mask_hook = lambda mask, step, timestep: mask
def capture_only_pre_unet(latent, mask, step, timestep):
    capture.store.set_timestep(timestep)
    return latent
smd._anchor_pre_unet_hook = capture_only_pre_unet
smd._anchor_post_euler_hook = lambda result, noisy, eps, step, timestep: result
smd._anchor_post_region_hook = lambda latent, step, timestep: latent
try:
    attach_capture()
    seed_everything(BASE_SEED)
    captured_latent = call_hooked(first_item, 'latent').detach().cpu()
finally:
    detach_capture()
capture_parity_exact = bool(torch.equal(control_latent, captured_latent))
capture_parity_max_abs = float((control_latent.float() - captured_latent.float()).abs().max())
expected_keys = len(T_INDEX_LIST) * len(first_item['prompts'])
assert len(capture.maps) == expected_keys, (len(capture.maps), expected_keys)
native_sizes = sorted({m.spatial_size for maps in capture.maps.values() for m in maps})
assert len(native_sizes) == 1, native_sizes
assert capture_parity_exact, f'Capture itself changed student output: max_abs={capture_parity_max_abs}'
print('[PASS] Capture-only latent parity:', capture_parity_exact,
      '| maps:', len(capture.maps), '| native map:', native_sizes[0],
      '| tokenizer positions:', tokenizer_positions)
del control_latent, captured_latent

# --- Source notebook cell 10 ---
# Full1073: cùng dataloader, cùng seed và cùng API SemanticDraw của baseline.
import hashlib, gc, subprocess, importlib.metadata, math
# terminal display provided above
from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report
import pandas as pd

assert len(loader.dataset) == RUN_SAMPLES == len(manifest_rows) == 1073
assert MASK_STD == 0.0 and BOOTSTRAP_STEPS == 1 and GUIDANCE_SCALE == 0.0
assert T_INDEX_LIST == [0, 4, 12, 25, 37]
assert SPATIAL_SIGMA_LATENT == 4.0 and TOPK_PERCENT == 10.0
assert len(T_INDEX_LIST) == len(smd.timesteps) == 5
assert capture_parity_exact and capture_parity_max_abs == 0.0
git_revision = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'],
    capture_output=True, text=True).stdout.strip()
protocol = {
    'method_id': 'ANCHORDRAW-SDXL-EULER-W03-L04-SIGMAD4-COORDALIGNED',
    'baseline_notebook': 'semanticdraw_sdxl_euler_full1073_kaggle.ipynb',
    'repo_commit': git_revision,
    'baseline_pipeline_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
    'manifest_sha256': hashlib.sha256(RUN_MANIFEST.read_bytes()).hexdigest(),
    'manifest_path': str(RUN_MANIFEST),
    'num_samples': RUN_SAMPLES, 'seed_rule': '2024 + global_index',
    'model': 'stabilityai/stable-diffusion-xl-base-1.0',
    'student_checkpoint': 'ByteDance/SDXL-Lightning/sdxl_lightning_4step_unet.safetensors',
    'scheduler': type(smd.scheduler).__name__,
    'timestep_spacing': str(smd.scheduler.config.timestep_spacing),
    'prediction_type': str(smd.scheduler.config.prediction_type),
    'semanticdraw_t_index_list': T_INDEX_LIST,
    'student_nfe': len(T_INDEX_LIST),
    'guidance_scale': GUIDANCE_SCALE, 'bootstrap_steps': BOOTSTRAP_STEPS,
    'mask_std': MASK_STD, 'mask_strength': MASK_STRENGTH,
    'preprocess_mask_cover_alpha': PREPROCESS_MASK_COVER_ALPHA,
    'target_size': list(TARGET_SIZE), 'mask_type': 'discrete',
    'attention_layer': LAYER_NAME, 'topk_percent': TOPK_PERCENT,
    'attention_coordinate_policy': 'geometrically inverse-align image-resized attention using actual bootstrap/semantic latent roll before original-mask top-k; native attention downsampling is not inverted',
    'weight_policy': 'adaptive_bilateral',
    'no_background_policy': 'shift all foreground regions; virtual zero background for W03; preserve foreground mask mass',
    'pixel_quality_policy': 'save every generated PNG; record advisory flags; exclude none',
    'spatial_sigma_latent': SPATIAL_SIGMA_LATENT,
    'semantic_sigma_scale': SEMANTIC_SIGMA_SCALE,
    'distillation_pp': False,
    'capture_only_parity_exact': capture_parity_exact,
    'capture_only_parity_max_abs': capture_parity_max_abs,
    'metric_names': list(METRIC_NAMES),
    'metric_batch_size': METRIC_BATCH_SIZE,
    'clip_batch_size': CLIP_BATCH_SIZE, 'is_splits': IS_SPLITS,
    'torch_version': torch.__version__,
    'package_versions': {name: importlib.metadata.version(name)
                         for name in ('diffusers', 'transformers', 'open-clip-torch',
                                      'torch-fidelity', 'torchmetrics')},
}
PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
print('[OK] Protocol frozen:', PROTOCOL_PATH)

# --- Source notebook cell 11 ---
# Sinh đúng 1073 ảnh AnchorDraw; checkpoint summary/audit theo từng mẫu.
def weight_masks_for_available_regions(mask, features, anchors, has_background):
    # The SDXL baseline omits its background region when foreground masks cover the canvas.
    # W03 expects a background slot; add a zero-valued one only for weighting, then
    # discard it and preserve the original foreground mixing mass pixel by pixel.
    foreground_count = len(anchors)
    assert features.shape[0] == foreground_count
    assert mask.shape[0] == foreground_count + int(has_background)
    helper_mask = mask if has_background else torch.cat([torch.zeros_like(mask[:1]), mask], dim=0)
    weighted, record = build_weighted_overlap_masks(
        helper_mask, features, anchors, image_size=TARGET_SIZE,
        policy='adaptive_bilateral', spatial_sigma_latent=SPATIAL_SIGMA_LATENT,
        semantic_sigma_scale=SEMANTIC_SIGMA_SCALE)
    if not has_background:
        foreground = weighted[1:].float()
        original_mass = mask.float().sum(dim=0, keepdim=True)
        weighted_mass = foreground.sum(dim=0, keepdim=True)
        weighted = torch.where(
            weighted_mass > 1e-8,
            foreground * (original_mass / weighted_mass.clamp_min(1e-8)),
            mask.float(),
        ).to(dtype=mask.dtype)
        assert weighted.shape == mask.shape
        assert torch.allclose(weighted.float().sum(dim=0), original_mass[0], atol=2e-3, rtol=2e-3)
    return weighted, record

# Fast regression gate for the no-background case that stopped the previous run at sample 190.
test_masks = torch.ones((2, 1, 8, 8), dtype=torch.float32)
test_features = torch.randn((2, 4, 8, 8), dtype=torch.float32)
test_anchors = [(256.0, 256.0), (768.0, 768.0)]
test_no_bg, _ = weight_masks_for_available_regions(test_masks, test_features, test_anchors, False)
assert test_no_bg.shape == test_masks.shape
assert torch.allclose(test_no_bg.sum(dim=0), test_masks.sum(dim=0), atol=2e-3, rtol=2e-3)
test_with_bg = torch.cat([torch.zeros_like(test_masks[:1]), test_masks], dim=0)
test_existing, _ = build_weighted_overlap_masks(
    test_with_bg, test_features, test_anchors, image_size=TARGET_SIZE,
    policy='adaptive_bilateral', spatial_sigma_latent=SPATIAL_SIGMA_LATENT,
    semantic_sigma_scale=SEMANTIC_SIGMA_SCALE)
test_preserved, _ = weight_masks_for_available_regions(test_with_bg, test_features, test_anchors, True)
assert torch.equal(test_existing, test_preserved)
del test_masks, test_features, test_anchors, test_no_bg, test_with_bg, test_existing, test_preserved
print('[PASS] W03 background/no-background weighting gates.')

summary_rows = []
audit_rows = []
for index, batch in enumerate(loader):
    item = batch_item_to_semanticdraw_inputs(batch, 0)
    assert item['metadata']['sample_id'] == manifest_rows[index]['sample_id']
    seed = BASE_SEED + index
    seed_everything(seed)
    method_tic = time.perf_counter()
    indices, tokenizer_positions = token_indices_for(item)
    capture.configure(indices)
    state = {'previous_anchors': None, 'previous_features': None, 'has_background': None,
             'current_attention_shifts_latent': None, 'current_attention_latent_size': None,
             'alignment_applied_steps': 0, 'step_rows': [], 'mask_rows': [],
             'shift_count': 0, 'nonzero_shift_steps': 0}
    foreground_masks = item['masks'].float().cpu()

    def mask_hook(mask, step, timestep):
        if step == 0:
            state['has_background'] = (mask.shape[0] == len(item['prompts']) + 1)
            assert mask.shape[0] == len(item['prompts']) + int(state['has_background'])
            return mask
        assert state['has_background'] is not None
        assert state['previous_anchors'] is not None and state['previous_features'] is not None
        weighted, record = weight_masks_for_available_regions(
            mask, state['previous_features'], state['previous_anchors'], state['has_background'])
        state['mask_rows'].append({
            'step': int(step), 'timestep': int(timestep.item()),
            'mean_abs_delta': float((weighted.float() - mask.float()).abs().mean().item()),
            'overlap_pixels': int(record.overlap_pixel_count),
            'semantic_sigmas': [float(v) for v in record.semantic_sigma_by_region]})
        return weighted

    def pre_unet_hook(latent, mask, step, timestep):
        capture.store.set_timestep(timestep)
        background_count = int(state['has_background'])
        state['current_attention_latent_size'] = tuple(latent.shape[-2:])
        if step == 0:
            state['current_attention_shifts_latent'] = bootstrap_foreground_shifts(
                mask, background_count)
            assert len(state['current_attention_shifts_latent']) == len(item['prompts'])
            return latent
        points = state['previous_anchors']
        assert points is not None and len(points) == latent.shape[0] - background_count
        h, w = latent.shape[-2:]
        state['current_attention_shifts_latent'] = semantic_foreground_shifts(
            points, (h, w), TARGET_SIZE)
        if any(round(x*w/TARGET_SIZE[1]) != w//2 or round(y*h/TARGET_SIZE[0]) != h//2 for x, y in points):
            state['nonzero_shift_steps'] += 1
        shifted = _shift_to_reference_points(
            latent[background_count:], points, image_size=TARGET_SIZE, reverse=False)
        state['shift_count'] += 1
        return torch.cat([latent[:background_count], shifted], dim=0)

    def post_euler_hook(student_next, student_noisy, noise_pred, step, timestep):
        assert student_next.shape == student_noisy.shape == noise_pred.shape
        if step == 0:
            return student_next
        background_count = int(state['has_background'])
        restored = _shift_to_reference_points(
            student_next[background_count:], state['previous_anchors'],
            image_size=TARGET_SIZE, reverse=True)
        return torch.cat([student_next[:background_count], restored], dim=0)

    def post_region_hook(latent, step, timestep):
        # Sau baseline bootstrap-unshift (step 0) hoặc semantic-unshift (step >=1).
        t = int(timestep.item())
        current = {}
        for region in range(len(item['prompts'])):
            key = (t, region)
            maps = capture.maps.get(key, [])
            assert len(maps) == 1, (key, len(maps))
            current[key] = maps
        attention = aggregate_attention_maps(
            current, TARGET_SIZE, layer_name=LAYER_NAME)
        shifts = state['current_attention_shifts_latent']
        latent_size = state['current_attention_latent_size']
        assert shifts is not None and len(shifts) == len(item['prompts'])
        assert latent_size is not None
        anchors, native = [], []
        for region, mask in enumerate(foreground_masks):
            aligned_attention = attention_in_mask_frame(
                attention[(t, region)], shifts[region], latent_size, TARGET_SIZE)
            measurement = compute_anchor_measurements(
                aligned_attention, mask, topk_percent=TOPK_PERCENT)
            x, y = measurement['topk_anchor_x'], measurement['topk_anchor_y']
            assert measurement['topk_anchor_inside_mask']
            anchors.append((float(x), float(y)))
            native.append(int(current[(t, region)][0].spatial_size))
        state['previous_anchors'] = anchors
        state['previous_features'] = latent[int(state['has_background']):].detach().clone()
        assert state['previous_features'].shape[0] == len(anchors)
        state['alignment_applied_steps'] += 1
        state['step_rows'].append({
            'step': int(step), 'timestep': t,
            'attention_shifts_latent_yx': shifts,
            'anchors_xy_image': anchors, 'native_attention_sizes': native})
        return latent


    smd._anchor_mask_hook = mask_hook
    smd._anchor_pre_unet_hook = pre_unet_hook
    smd._anchor_post_euler_hook = post_euler_hook
    smd._anchor_post_region_hook = post_region_hook
    try:
        attach_capture()
        seed_everything(seed)
        tic = time.perf_counter()
        generated = call_hooked(item)
        elapsed = time.perf_counter() - tic
    finally:
        detach_capture()
    generated = generated.convert('RGB')
    method_wall_sec = time.perf_counter() - method_tic
    assert generated.size == (TARGET_SIZE[1], TARGET_SIZE[0])
    assert len(state['step_rows']) == len(T_INDEX_LIST)
    assert state['alignment_applied_steps'] == len(T_INDEX_LIST)
    assert len(state['mask_rows']) == len(T_INDEX_LIST) - BOOTSTRAP_STEPS
    assert state['shift_count'] == len(T_INDEX_LIST) - BOOTSTRAP_STEPS

    # Save first. Pixel statistics are advisory and must never discard a measured sample.
    generated_path = GENERATED_DIR / f'{index:04d}_{item["metadata"]["sample_id"]}_generated.png'
    generated.save(generated_path)
    pixels = np.asarray(generated)
    image_min, image_max = int(pixels.min()), int(pixels.max())
    image_mean, image_std = float(pixels.mean()), float(pixels.std())
    quality_flags = []
    if image_max <= 5 and image_std <= 1.0:
        quality_flags.append('near_black_candidate')
    if image_min == 0 and image_max == 255 and 95 <= image_mean <= 160 and image_std >= 90.0:
        quality_flags.append('high_contrast_candidate')
    mean_mask_delta = sum(m['mean_abs_delta'] for m in state['mask_rows']) / len(state['mask_rows'])
    row = {
        'index': index, 'sample_id': item['metadata']['sample_id'],
        'image_id': int(item['metadata']['image_id']),
        'file_name': item['metadata']['file_name'],
        'seed': seed, 'model_family': 'sdxl', 'sampler': 'euler_discrete',
        'scheduler': type(smd.scheduler).__name__, 'timestep_spacing': 'trailing',
        'lightning_repo': 'ByteDance/SDXL-Lightning',
        'lightning_weight': 'sdxl_lightning_4step_unet.safetensors',
        'plain_num_inference_steps': 4,
        'semanticdraw_num_inference_steps_arg': None,
        'semanticdraw_t_index_list': T_INDEX_LIST,
        'guidance_scale': GUIDANCE_SCALE, 'bootstrap_steps': BOOTSTRAP_STEPS,
        'num_regions_including_background': len(item['prompts']) + int(state['has_background']),
        'has_background_region': bool(state['has_background']),
        'elapsed_sec': elapsed, 'method_wall_sec': method_wall_sec,
        'generated_path': str(generated_path),
        'image_min': image_min, 'image_max': image_max,
        'image_mean': image_mean, 'image_std': image_std,
        'quality_flags': quality_flags,
        'method_id': protocol['method_id'], 'distillation_pp': False,
    }
    summary_rows.append(row)
    audit_rows.append({
        'index': index, 'sample_id': row['sample_id'], 'seed': seed,
        'tokenizer_positions': tokenizer_positions,
        'nonzero_shift_steps': state['nonzero_shift_steps'],
        'has_background_region': bool(state['has_background']),
        'attention_alignment_applied_steps': state['alignment_applied_steps'],
        'attention_shifts_latent_yx_by_step': [
            step_row['attention_shifts_latent_yx'] for step_row in state['step_rows']],
        'anchors_xy_image_by_step': [
            step_row['anchors_xy_image'] for step_row in state['step_rows']],
        'mean_mask_abs_delta': mean_mask_delta,
        'quality_flags': quality_flags,
        'native_attention_sizes': sorted(set(
            size for step in state['step_rows'] for size in step['native_attention_sizes'])),
    })
    RUN_SUMMARY_PATH.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    AUDIT_PATH.write_text(
        '\n'.join(json.dumps(v, ensure_ascii=False) for v in audit_rows) + '\n',
        encoding='utf-8')
    print(f'[{index+1}/{RUN_SAMPLES}] {row["sample_id"]} | mask_delta={mean_mask_delta:.6f}'
          f' | call={elapsed:.2f}s | method={method_wall_sec:.2f}s')
    if quality_flags:
        print(f'[AUDIT] Image {index} flagged for review (not excluded): {quality_flags}')
    if index < 8:
        display(generated.resize((384, 384)))
    del generated, pixels, batch, item
    torch.cuda.empty_cache()

assert len(summary_rows) == len(audit_rows) == RUN_SAMPLES
assert [r['sample_id'] for r in summary_rows] == [r['sample_id'] for r in manifest_rows]
assert [r['seed'] for r in summary_rows] == [BASE_SEED+i for i in range(RUN_SAMPLES)]
assert len({r['generated_path'] for r in summary_rows}) == RUN_SAMPLES
assert all(Path(r['generated_path']).is_file() for r in summary_rows)
assert sum(r['mean_mask_abs_delta'] for r in audit_rows) > 0, 'W03 was a no-op.'
assert sum(r['nonzero_shift_steps'] for r in audit_rows) > 0, 'Anchor shift was a no-op.'
print('[PASS] Full1073 generation and manifest/seed/output integrity.')
print('[AUDIT] Pixel-statistic candidates retained in metric set:',
      [(row['index'], row['quality_flags']) for row in summary_rows if row['quality_flags']])
print('[AUDIT] Samples without a separate background region:',
      sum(not row['has_background_region'] for row in summary_rows))

# --- Source notebook cell 13 ---
# Giải phóng model trước khi đánh giá metric, tương tự baseline.
for var_name in ('smd', 'capture', 'loader', 'first_item', 'foreground_masks',
                 'state', 'indices', 'tokenizer_positions'):
    if var_name in globals():
        del globals()[var_name]
gc.collect()
torch.cuda.empty_cache()
try:
    torch.cuda.ipc_collect()
except Exception:
    pass
print('[OK] Released generation model before metric evaluation.')

# --- Source notebook cell 14 ---
# Cùng MetricEvaluationConfig và evaluator với SDXL-Euler baseline.
assert len(json.loads(RUN_SUMMARY_PATH.read_text(encoding='utf-8'))) == RUN_SAMPLES
metric_config = MetricEvaluationConfig(
    manifest_path=RUN_MANIFEST,
    coco_root=COCO_ROOT,
    generated_dir=GENERATED_DIR,
    generation_summary=RUN_SUMMARY_PATH,
    output_dir=METRICS_DIR,
    model_family='sdxl',
    target_size=TARGET_SIZE,
    metrics=METRIC_NAMES,
    batch_size=METRIC_BATCH_SIZE,
    num_workers=0,
    pin_memory=False,
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    clip_batch_size=CLIP_BATCH_SIZE,
    is_splits=IS_SPLITS,
)
report = run_evaluation(metric_config)
assert report['num_manifest_records'] == report['num_evaluated'] == RUN_SAMPLES
assert report['num_missing_generated'] == 0
assert [row['sample_id'] for row in report['evaluated']] == [row['sample_id'] for row in manifest_rows]
metrics_json, metrics_csv = write_metrics_report(
    report, METRICS_DIR, prefix=OUTPUT_DIR.name + '_metrics')
v = report['metrics']
assert all(math.isfinite(float(v[key])) for key in
           ('fid', 'is_mean', 'is_std', 'clip_fg_x100', 'clip_bg_x100',
            'time_mean_sec', 'time_total_sec')), v
display(pd.DataFrame([{
    'method_id': protocol['method_id'], 'num_evaluated': report['num_evaluated'],
    'fid': v.get('fid'), 'is_mean': v.get('is_mean'), 'is_std': v.get('is_std'),
    'clip_fg_x100': v.get('clip_fg_x100'), 'clip_bg_x100': v.get('clip_bg_x100'),
    'time_mean_sec': v.get('time_mean_sec'), 'time_total_sec': v.get('time_total_sec'),
    'method_wall_mean_sec': sum(r['method_wall_sec'] for r in summary_rows) / RUN_SAMPLES,
}]))
print('[PASS] Full1073 metrics:', metrics_json, metrics_csv)
