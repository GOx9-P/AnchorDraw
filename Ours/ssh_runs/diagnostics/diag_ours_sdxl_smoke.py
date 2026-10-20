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

# Notebook: Ours/kaggle_semantic_anchor_experiments/anchordraw_sdxl_euler_w03_l04_sigmad4_smoke8_kaggle.ipynb
# Notebook SHA256: f03608dc98e9d2820cdb47350952e93e3dd65b430e2b2a433362ed8528b4ab0d

# --- Source notebook cell 2 ---
from pathlib import Path
import os, json, importlib.util, types, zipfile
import torch

RUN_MANIFEST = REPO_ROOT / 'Ours/data_manifests/coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_all.jsonl'
COCO_ROOT = Path(os.environ.get('COCO_ROOT', str(WORK_DIR / 'COCO')))
OUTPUT_DIR = WORK_DIR / 'anchordraw_sdxl_euler_w03_l04_sigmad4_smoke8'
RUN_SAMPLES, BASE_SEED = 8, 2024
TARGET_SIZE, T_INDEX_LIST = (1024, 1024), [0, 4, 12, 25, 37]
MASK_STD, MASK_STRENGTH, PREPROCESS_MASK_COVER_ALPHA = 0.0, 1.0, 0.0
BOOTSTRAP_STEPS, GUIDANCE_SCALE = 1, 0.0
NEGATIVE_PROMPT = ''
manifest_rows = [json.loads(s) for s in RUN_MANIFEST.read_text(encoding='utf-8').splitlines() if s.strip()]
assert len(manifest_rows) == 1073 and RUN_SAMPLES <= len(manifest_rows)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print('[OK] Repo:', REPO_ROOT, '| Manifest:', len(manifest_rows), '| Smoke:', RUN_SAMPLES)
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
# Cùng loader/mask/manifest với baseline; return_image=False vì parity không cần ảnh gốc.
config = COCORegionConfig(coco_root=COCO_ROOT, split='val2017',
    instances_json=COCO_ROOT / 'annotations/instances_val2017.json',
    captions_json=COCO_ROOT / 'annotations/captions_val2017.json',
    manifest_path=RUN_MANIFEST, profile='multidiffusion_coco_all', model_family='sdxl',
    target_size=TARGET_SIZE, return_image=False, cache_resized_masks=True,
    cache_dir=WORK_DIR / 'semanticdraw_mask_cache', batch_size=1, num_workers=0,
    pin_memory=False, persistent_workers=False)
loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
assert len(loader.dataset) == 1073
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

# --- Source notebook cell 9 ---
# Paired smoke8: baseline vs semantic top-k anchor + adaptive bilateral mask.
audit_rows, image_pairs = [], []
for index, batch in enumerate(islice(loader, RUN_SAMPLES)):
    item = batch_item_to_semanticdraw_inputs(batch, 0)
    assert item['metadata']['sample_id'] == manifest_rows[index]['sample_id']
    seed = BASE_SEED + index
    seed_everything(seed)
    torch.cuda.synchronize()
    tic = time.perf_counter()
    baseline_image = call_baseline(item).convert('RGB')
    torch.cuda.synchronize()
    baseline_sec = time.perf_counter() - tic

    indices, tokenizer_positions = token_indices_for(item)
    capture.configure(indices)
    state = {'previous_anchors': None, 'previous_features': None,
             'step_rows': [], 'mask_rows': [], 'shift_count': 0, 'nonzero_shift_steps': 0}
    foreground_masks = item['masks'].float().cpu()

    def mask_hook(mask, step, timestep):
        if step == 0:
            return mask
        assert state['previous_anchors'] is not None and state['previous_features'] is not None
        weighted, record = build_weighted_overlap_masks(
            mask, state['previous_features'], state['previous_anchors'],
            image_size=TARGET_SIZE, policy='adaptive_bilateral',
            spatial_sigma_latent=SPATIAL_SIGMA_LATENT,
            semantic_sigma_scale=SEMANTIC_SIGMA_SCALE)
        state['mask_rows'].append({
            'step': int(step), 'timestep': int(timestep.item()),
            'mean_abs_delta': float((weighted.float() - mask.float()).abs().mean().item()),
            'overlap_pixels': int(record.overlap_pixel_count),
            'semantic_sigmas': [float(v) for v in record.semantic_sigma_by_region]})
        return weighted

    def pre_unet_hook(latent, mask, step, timestep):
        capture.store.set_timestep(timestep)
        if step == 0:
            return latent
        points = state['previous_anchors']
        assert points is not None and len(points) == latent.shape[0] - 1
        h, w = latent.shape[-2:]
        if any(round(x*w/TARGET_SIZE[1]) != w//2 or round(y*h/TARGET_SIZE[0]) != h//2 for x, y in points):
            state['nonzero_shift_steps'] += 1
        shifted = _shift_to_reference_points(
            latent[1:], points, image_size=TARGET_SIZE, reverse=False)
        state['shift_count'] += 1
        return torch.cat([latent[:1], shifted], dim=0)

    def post_euler_hook(student_next, student_noisy, noise_pred, step, timestep):
        assert student_next.shape == student_noisy.shape == noise_pred.shape
        if step == 0:
            return student_next
        restored = _shift_to_reference_points(
            student_next[1:], state['previous_anchors'],
            image_size=TARGET_SIZE, reverse=True)
        return torch.cat([student_next[:1], restored], dim=0)

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
        anchors, native = [], []
        for region, mask in enumerate(foreground_masks):
            measurement = compute_anchor_measurements(
                attention[(t, region)], mask, topk_percent=TOPK_PERCENT)
            x, y = measurement['topk_anchor_x'], measurement['topk_anchor_y']
            assert measurement['topk_anchor_inside_mask']
            anchors.append((float(x), float(y)))
            native.append(int(current[(t, region)][0].spatial_size))
        state['previous_anchors'] = anchors
        state['previous_features'] = latent[1:].detach().clone()
        state['step_rows'].append({
            'step': int(step), 'timestep': t,
            'anchors_xy_image': anchors, 'native_attention_sizes': native})
        return latent

    smd._anchor_mask_hook = mask_hook
    smd._anchor_pre_unet_hook = pre_unet_hook
    smd._anchor_post_euler_hook = post_euler_hook
    smd._anchor_post_region_hook = post_region_hook
    try:
        attach_capture()
        seed_everything(seed)
        torch.cuda.synchronize()
        tic = time.perf_counter()
        anchor_image = call_hooked(item).convert('RGB')
        torch.cuda.synchronize()
        anchor_sec = time.perf_counter() - tic
    finally:
        detach_capture()
    assert len(state['step_rows']) == len(T_INDEX_LIST)
    assert len(state['mask_rows']) == len(T_INDEX_LIST) - BOOTSTRAP_STEPS
    assert state['shift_count'] == len(T_INDEX_LIST) - BOOTSTRAP_STEPS
    baseline_pixels = np.asarray(baseline_image, dtype=np.int16)
    anchor_pixels = np.asarray(anchor_image, dtype=np.int16)
    pixel_mae = float(np.abs(baseline_pixels - anchor_pixels).mean())
    changed_fraction = float(np.any(baseline_pixels != anchor_pixels, axis=2).mean())
    row = {
        'index': index, 'sample_id': item['metadata']['sample_id'],
        'seed': seed, 'foreground_regions': len(item['prompts']),
        'tokenizer_positions': tokenizer_positions,
        'baseline_sec': baseline_sec, 'anchor_w03_sec': anchor_sec,
        'pixel_mae': pixel_mae, 'changed_pixel_fraction': changed_fraction,
        'nonzero_shift_steps': state['nonzero_shift_steps'],
        'attention_layer': LAYER_NAME, 'spatial_sigma_latent': SPATIAL_SIGMA_LATENT,
        'step_rows': state['step_rows'], 'mask_rows': state['mask_rows']}
    audit_rows.append(row)
    image_pairs.append((baseline_image.resize((512,512)),
                        anchor_image.resize((512,512))))
    mean_mask_delta = sum(m['mean_abs_delta'] for m in state['mask_rows']) / len(state['mask_rows'])
    print(f"[{index+1}/{RUN_SAMPLES}] {row['sample_id']} | pixel MAE={pixel_mae:.3f}"
          f" | changed={changed_fraction:.3%} | mean mask delta={mean_mask_delta:.6f}"
          f" | t={baseline_sec:.2f}/{anchor_sec:.2f}s")

assert len(audit_rows) == RUN_SAMPLES
assert sum(r['pixel_mae'] for r in audit_rows) > 0, 'AnchorDraw had no visible effect.'
assert sum(r['nonzero_shift_steps'] for r in audit_rows) > 0, 'Anchor shifts were all zero.'
assert sum(m['mean_abs_delta'] for r in audit_rows for m in r['mask_rows']) > 0, 'W03 was a no-op.'
print('[PASS] Anchor + W03 changed output and W03 masks on the paired smoke8.')

# --- Source notebook cell 10 ---
# Audit + contact sheet; no full1073 metrics or individual generated PNG archive.
audit = {
    'status': 'PASS', 'scope': 'paired smoke8, no full1073 metric claim',
    'baseline_reference': 'semanticdraw_sdxl_euler_full1073_kaggle.ipynb',
    'manifest': str(RUN_MANIFEST), 'samples': RUN_SAMPLES,
    'student': 'SDXL-Lightning 4-step UNet', 'scheduler': 'EulerDiscreteScheduler/trailing',
    'student_nfe': len(T_INDEX_LIST), 't_index_list': T_INDEX_LIST,
    'guidance_scale': GUIDANCE_SCALE, 'bootstrap_steps': BOOTSTRAP_STEPS,
    'mask_std': MASK_STD, 'attention_layer': LAYER_NAME,
    'layer_claim': 'SDXL structural path candidate, not proven equivalent to SD1.5 L04',
    'anchor_strategy': 'masked top-k projected centroid', 'topk_percent': TOPK_PERCENT,
    'weight_policy': 'adaptive_bilateral', 'spatial_sigma_latent': SPATIAL_SIGMA_LATENT,
    'distillation_pp': False, 'capture_only_parity_exact': capture_parity_exact,
    'capture_only_parity_max_abs': capture_parity_max_abs,
    'rows': audit_rows}
audit_path = OUTPUT_DIR / 'anchor_w03_smoke8_audit.json'
audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
sheet = Image.new('RGB', (1024, 512 * RUN_SAMPLES), 'white')
draw = ImageDraw.Draw(sheet)
for i, (left, right) in enumerate(image_pairs):
    sheet.paste(left, (0, i*512))
    sheet.paste(right, (512, i*512))
    draw.text((8, i*512+8), f'{i}: baseline', fill='black')
    draw.text((520, i*512+8), f'{i}: AnchorDraw W03', fill='black')
sheet_path = OUTPUT_DIR / 'baseline_vs_anchordraw_w03_smoke8.png'
sheet.save(sheet_path)
display(sheet.resize((512, 256 * RUN_SAMPLES)))
print('[OK] Audit:', audit_path)
print('[OK] Contact sheet:', sheet_path)
