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

# Notebook: Ours/kaggle_semantic_anchor_experiments/anchordraw_sdxl_euler_noop_parity_smoke8_kaggle.ipynb
# Notebook SHA256: 6d4cc84709334f350beb49b9adf4de03aaaf39310abbd7e6dbc88f4119620057

# --- Source notebook cell 2 ---
from pathlib import Path
import os, json, importlib.util, types, zipfile
import torch

RUN_MANIFEST = REPO_ROOT / 'Ours/data_manifests/coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_all.jsonl'
COCO_ROOT = Path(os.environ.get('COCO_ROOT', str(WORK_DIR / 'COCO')))
OUTPUT_DIR = WORK_DIR / 'anchordraw_sdxl_euler_noop_parity_smoke8'
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

def load_pipeline_module(name, text):
    module = types.ModuleType(name)
    module.__file__ = str(pipeline_path)
    exec(compile(text, str(pipeline_path), 'exec'), module.__dict__)
    module.load_file = lambda filename, device=None: _safetensors_load_file(filename, device='cpu')
    return module
baseline_module = load_pipeline_module('sdxl_euler_baseline_patched', source)
hooked_module = load_pipeline_module('sdxl_euler_anchor_noop_hooks', hooked_source)
BaselineClass = baseline_module.SemanticDrawSDXLPipeline
HookedClass = hooked_module.SemanticDrawSDXLPipeline
assert source != hooked_source
print('[OK] Baseline and no-op-hook classes compiled from the same source + same compatibility patches.')

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
# Hai lượt trên CÙNG instance/model. Hooks identity phải thực sự được gọi và latent phải khớp bit.
from itertools import islice
audit_rows = []
for index, batch in enumerate(islice(loader, RUN_SAMPLES)):
    item = batch_item_to_semanticdraw_inputs(batch, 0)
    assert item['metadata']['sample_id'] == manifest_rows[index]['sample_id']
    seed = BASE_SEED + index
    call_args = dict(prompts=item['prompts'], negative_prompts=[NEGATIVE_PROMPT] * len(item['prompts']),
        background_prompt=item['background_prompt'], background_negative_prompt=NEGATIVE_PROMPT,
        mask_stds=MASK_STD, mask_strengths=MASK_STRENGTH, height=item['height'], width=item['width'],
        num_inference_steps=None, guidance_scale=GUIDANCE_SCALE, bootstrap_steps=BOOTSTRAP_STEPS,
        preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA, guidance_rescale=0.0,
        do_blend=False, output_type='latent')
    assert (item['height'], item['width']) == TARGET_SIZE
    seed_everything(seed)
    baseline_latent = smd(masks=item['masks'].clone().to('cuda:0', dtype=torch.float32),
                          **call_args).detach().cpu()
    counts = {'mask': 0, 'pre_unet': 0, 'post_euler': 0}
    def mask_hook(mask, step, timestep):
        counts['mask'] += 1
        return mask
    def pre_unet_hook(latent, mask, step, timestep):
        counts['pre_unet'] += 1
        return latent
    def post_euler_hook(student_next, student_noisy, noise_pred, step, timestep):
        counts['post_euler'] += 1
        assert student_next.shape == student_noisy.shape == noise_pred.shape
        return student_next
    smd._anchor_mask_hook = mask_hook
    smd._anchor_pre_unet_hook = pre_unet_hook
    smd._anchor_post_euler_hook = post_euler_hook
    seed_everything(seed)
    hooked_latent = HookedClass.__call__(smd,
        masks=item['masks'].clone().to('cuda:0', dtype=torch.float32),
        **call_args).detach().cpu()
    max_abs = float((baseline_latent.float() - hooked_latent.float()).abs().max().item())
    exact = bool(torch.equal(baseline_latent, hooked_latent))
    row = {'index': index, 'sample_id': item['metadata']['sample_id'], 'seed': seed,
           'num_foreground_regions': len(item['prompts']), 'exact': exact, 'max_abs_latent': max_abs,
           'hook_calls': dict(counts)}
    audit_rows.append(row)
    print(f"[{index+1}/{RUN_SAMPLES}] exact={exact} max_abs={max_abs:.8g} hooks={counts} {row['sample_id']}")
    assert all(value == len(T_INDEX_LIST) for value in counts.values()), row
    assert exact, f'No-op AnchorDraw path changed baseline latent: {row}'
    del baseline_latent, hooked_latent
print('[PASS] No-op AnchorDraw Euler path matches baseline latent exactly on all 8 samples.')

# --- Source notebook cell 8 ---
# Chỉ lưu audit JSON; không lưu ảnh hay metric FID/IS/CLIP.
audit = {'status': 'PASS' if len(audit_rows) == RUN_SAMPLES and all(r['exact'] for r in audit_rows) else 'FAIL',
         'scope': 'same-runtime baseline vs identity-hook parity; not historical-session parity',
         'manifest': str(RUN_MANIFEST), 'samples': RUN_SAMPLES, 'seed_start': BASE_SEED,
         'model': 'SDXL base + SDXL-Lightning 4-step UNet',
         'scheduler': 'EulerDiscreteScheduler/trailing', 't_index_list': T_INDEX_LIST,
         'student_nfe': len(T_INDEX_LIST), 'guidance_scale': GUIDANCE_SCALE,
         'bootstrap_steps': BOOTSTRAP_STEPS, 'mask_std': MASK_STD,
         'anchor_enabled': False, 'w03_enabled': False, 'distillation_pp_enabled': False,
         'rows': audit_rows}
assert audit['status'] == 'PASS'
audit_path = OUTPUT_DIR / 'noop_parity_audit.json'
audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print('[OK] Saved:', audit_path)
