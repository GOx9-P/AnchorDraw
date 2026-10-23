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

# Notebook: Ours/semantic_anchor_experiments/semantic_anchor_sd15_lcm_spatial_sigma_sweep_colab.ipynb
# Notebook SHA256: 8ce001f3a2ef7aeab9962ba69a757332c591a9bb8e7bbac5927d0581f9b5ab02

# --- Source notebook cell 3 ---
# 2. Run configuration: WM-03 (Top-k + Adaptive Bilateral) + Bootstrap=1 + Layer L04.
import json, hashlib

RUN_PROFILE = os.environ.get('RUN_PROFILE', 'smoke8')
PROFILE = {
    'run_id': 'semantic_anchor_sd15_lcm_spatial_sigma_sweep_wm03_l04_bt1_smoke8',
    'manifest': 'Ours/test_sets/manifests/smoke/coco_val2017_multidiffusion_coco_all_512x512_smoke_bs8.jsonl',
    'expected_samples': 8,
}
RUN_SAMPLES = 8
MODEL_ID = 'runwayml/stable-diffusion-v1-5'
TARGET_SIZE = (512, 512)
BASE_SEED = 2024
BATCH_SIZE = 1
BOOTSTRAP_STEPS = 1  # 1 step baseline bootstrap (standard per checklist)
MASK_STD = 1.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = 'discrete'
NEGATIVE_PROMPT = ''

# Method configuration: WM-03 (Top-k + Adaptive Bilateral)
METHOD_ID = 'WM-03'
METHOD_LABEL = 'Top-k + Adaptive Bilateral'
BEST_ANCHOR_MODE = 'semantic_topk_anchor'  # WM-03 uses semantic_topk_anchor (topk centroid)
WEIGHT_POLICY = 'adaptive_bilateral'
TOPK_ATTENTION_PERCENT = 10.0
SEMANTIC_SIGMA_SCALE = 1.0

# Fixed layer: L04 (Rank 2 Down-block 16x16)
FIXED_LAYER = {
    'layer_id': 'L04',
    'layer_index': 4,
    'native_spatial_size': 16,
    'layer_name': 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
}

# 10 spatial sigma_D values to sweep across the 64x64 latent grid
SIGMA_D_SWEEP = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 64.0]

COCO_ROOT = Path(os.environ.get('COCO_ROOT', str(WORK_DIR / 'COCO')))
RUN_MANIFEST = REPO_ROOT / PROFILE['manifest']
BASE_OUTPUT_DIR = Path(str(WORK_DIR / 'anchordraw_runs'))
RUN_ID = PROFILE['run_id']
RUN_ROOT = BASE_OUTPUT_DIR / RUN_ID
MASK_CACHE_DIR = RUN_ROOT / 'mask_cache'
GENERATED_DIR = RUN_ROOT / 'generated_images'
BY_SIGMA_DIR = GENERATED_DIR / 'by_sigma'
BY_SAMPLE_DIR = GENERATED_DIR / 'by_sample'
COMPARISON_DIR = RUN_ROOT / 'comparison_grids'

for path in [MASK_CACHE_DIR, GENERATED_DIR, BY_SIGMA_DIR, BY_SAMPLE_DIR, COMPARISON_DIR]:
    path.mkdir(parents=True, exist_ok=True)
for sigma in SIGMA_D_SWEEP:
    (BY_SIGMA_DIR / f'sigma_{sigma:04.1f}').mkdir(parents=True, exist_ok=True)

assert RUN_MANIFEST.exists(), f'Missing manifest: {RUN_MANIFEST}'
assert RUN_SAMPLES <= PROFILE['expected_samples'], f'Requested {RUN_SAMPLES} but manifest only has {PROFILE["expected_samples"]}'
print(f'[OK] Method: {METHOD_ID} ({METHOD_LABEL}) | anchor_mode: {BEST_ANCHOR_MODE}')
print(f'[OK] Bootstrap steps: {BOOTSTRAP_STEPS}')
print(f'[OK] Fixed layer: {FIXED_LAYER["layer_id"]} ({FIXED_LAYER["layer_name"]})')
print(f'[OK] Run samples: {RUN_SAMPLES} | sigma_D values ({len(SIGMA_D_SWEEP)}): {SIGMA_D_SWEEP}')
print(f'[OK] Output directory: {RUN_ROOT}')
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 4 ---
# 3. Imports and source modules with disk-patch for bootstrap_steps compatibility.
import sys, importlib.util, time, gc, inspect, shutil, types, re
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
# terminal display provided above

OURS_SRC = REPO_ROOT / 'Ours/src'
BASELINE_SRC = REPO_ROOT / 'Baseline/semantic-draw-main/src'

# Auto-patch semantic_anchor.py on disk if cloned from remote with bootstrap_steps != 1 assertion
sa_file = OURS_SRC / 'experiments/semantic_anchor.py'
if sa_file.exists():
    sa_text = sa_file.read_text(encoding='utf-8').replace('\r\n', '\n')
    if 'if bootstrap_steps != 1:' in sa_text:
        print('[INFO] Patching semantic_anchor.py on disk to support bootstrap_steps >= 1...')
        sa_text = re.sub(
            r'if\s+bootstrap_steps\s*!=\s*1:\s*\n\s*raise\s+ValueError\([^\)]+\)',
            'if False and bootstrap_steps != 1:\n            pass',
            sa_text
        )
        sa_text = sa_text.replace(
            'if step_index > 0 and weight_policy != "quantized_baseline":',
            'if step_index >= bootstrap_steps and weight_policy != "quantized_baseline":',
        )
        sa_text = sa_text.replace(
            'policy=("bootstrap_baseline" if step_index == 0 else weight_policy),',
            'policy=("bootstrap_baseline" if step_index < bootstrap_steps else weight_policy),',
        )
        sa_text = sa_text.replace(
            'if step_index == 0:\n                    selection_source = "baseline_bbox_bootstrap"',
            'if step_index < bootstrap_steps:\n                    selection_source = "baseline_bbox_bootstrap"',
        )
        sa_text = sa_text.replace(
            'if step_index == 0:\n                        # Exact baseline bootstrap path',
            'if step_index < bootstrap_steps:\n                        # Exact baseline bootstrap path',
        )
        sa_text = sa_text.replace(
            'if step_index == 0:\n                        from util import shift_to_mask_bbox_center',
            'if step_index < bootstrap_steps:\n                        from util import shift_to_mask_bbox_center',
        )
        sa_file.write_text(sa_text, encoding='utf-8')
        print('[OK] semantic_anchor.py patched successfully.')
    if 'bootstrap_steps >= len(pipeline.timesteps):' in sa_text and sa_text.find('bootstrap_steps >= len(pipeline.timesteps):') < sa_text.find('pipeline = self.pipeline'):
        print('[INFO] Fixing pipeline assignment order in semantic_anchor.py...')
        sa_text = sa_text.replace(
            '        if bootstrap_steps < 1 or bootstrap_steps >= len(pipeline.timesteps):\n            raise ValueError(f"bootstrap_steps must be in [1, {len(pipeline.timesteps) - 1}].")\n\n        pipeline = self.pipeline',
            '        pipeline = self.pipeline\n        if bootstrap_steps < 1 or bootstrap_steps >= len(pipeline.timesteps):\n            raise ValueError(f"bootstrap_steps must be in [1, {len(pipeline.timesteps) - 1}].")'
        )
        sa_file.write_text(sa_text, encoding='utf-8')
        print('[OK] pipeline assignment order fixed.')
    if 'layer_name=attention_layer_name' in sa_text:
        print('[INFO] Patching anchor_kwargs in semantic_anchor.py...')
        sa_text = sa_text.replace(
            '                current_anchors = self._anchors_from_current_step(\n                    int(timestep.item()),\n                    foreground_masks,\n                    strategy=("topk_projected_centroid" if mode == "semantic_topk_anchor" else "argmax"),\n                    topk_percent=topk_percent,\n                    layer_index=attention_layer_index,\n                    layer_name=attention_layer_name,\n                )',
            '                anchor_kwargs = {}\n                if attention_layer_name is not None:\n                    anchor_kwargs["layer_name"] = attention_layer_name\n                if attention_layer_index is not None:\n                    anchor_kwargs["layer_index"] = attention_layer_index\n                current_anchors = self._anchors_from_current_step(\n                    int(timestep.item()),\n                    foreground_masks,\n                    strategy=("topk_projected_centroid" if mode == "semantic_topk_anchor" else "argmax"),\n                    topk_percent=topk_percent,\n                    **anchor_kwargs,\n                )'
        )
        sa_file.write_text(sa_text, encoding='utf-8')
        print('[OK] anchor_kwargs patched successfully.')

for module_name in list(sys.modules):
    if module_name == 'data' or module_name.startswith('data.') or module_name == 'experiments' or module_name.startswith('experiments.'):
        del sys.modules[module_name]
sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from experiments.semantic_anchor import (
    SemanticAnchorCapture, SemanticAnchorRuntime, aggregate_attention_maps,
    compute_anchor_measurements, find_target_token_indices,
)
sys.path.insert(0, str(BASELINE_SRC))

pipeline_path = BASELINE_SRC / 'model/pipeline_semantic_draw.py'
spec = importlib.util.spec_from_file_location('pipeline_semantic_draw_original', pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline
print('[OK] Modules loaded.')

# --- Source notebook cell 6 ---
# 5. Dataloader.
config = COCORegionConfig(
    coco_root=COCO_ROOT, split='val2017',
    instances_json=COCO_ROOT / 'annotations/instances_val2017.json',
    captions_json=COCO_ROOT / 'annotations/captions_val2017.json',
    manifest_path=RUN_MANIFEST, profile='multidiffusion_coco_all',
    model_family='sd15', target_size=TARGET_SIZE, return_image=True,
    cache_resized_masks=True, cache_dir=MASK_CACHE_DIR,
    batch_size=BATCH_SIZE, num_workers=0, pin_memory=False, persistent_workers=False,
)
loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
assert len(loader.dataset) == PROFILE['expected_samples'], f'Expected {PROFILE["expected_samples"]} manifest rows, got {len(loader.dataset)}'
print('[OK] Dataset samples:', len(loader.dataset), '| batches:', len(loader))

# --- Source notebook cell 7 ---
# 6. Load SemanticDraw SD1.5 + LCM.
def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def maybe_login_hf():
    token = os.environ.get('HF_TOKEN')
    if token is None:
        try:
            from google.colab import userdata
            token = userdata.get('HF_TOKEN')
        except Exception:
            token = None
    if token:
        from huggingface_hub import login
        login(token=token)

assert torch.cuda.is_available(), 'Enable GPU in Runtime > Change runtime type.'
device = torch.device('cuda:0')
dtype = torch.float16
maybe_login_hf()
seed_everything(BASE_SEED)
smd = SemanticDrawPipeline(
    device=device, dtype=dtype, sd_version='1.5', hf_key=MODEL_ID, has_i2t=False,
    default_mask_std=MASK_STD, default_mask_strength=MASK_STRENGTH,
    default_preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA, mask_type=MASK_TYPE,
)
assert type(smd.scheduler).__name__ == 'LCMScheduler'
available_layer_names = set(smd.unet.attn_processors)
assert FIXED_LAYER['layer_name'] in available_layer_names, f"Missing {FIXED_LAYER['layer_name']} in UNet"
print('[OK] GPU:', torch.cuda.get_device_name(0))
print('[OK] Fixed layer name:', FIXED_LAYER['layer_name'])
print('[OK] Timesteps:', [int(t) for t in smd.timesteps.cpu().tolist()])

# --- Source notebook cell 8 ---
# 7. Helpers: payload builder, contact sheet visualizer, and layer selection.
def make_payload(batch, index):
    item = batch_item_to_semanticdraw_inputs(batch, index)
    metadata = item['metadata']
    foreground_masks = item['masks'].float().cpu()
    background_mask = (1.0 - foreground_masks.sum(dim=0, keepdim=True).clamp(0, 1)).clamp(0, 1)
    all_masks = torch.cat([background_mask, foreground_masks], dim=0)
    prompts = [item['background_prompt'], *item['prompts']]
    assert len(prompts) == len(all_masks)
    return {
        'sample_id': metadata['sample_id'], 'image_id': metadata['image_id'],
        'background_prompt': item['background_prompt'], 'prompts': prompts,
        'negative_prompts': [NEGATIVE_PROMPT] * len(prompts),
        'foreground_prompts': item['prompts'], 'foreground_masks': foreground_masks,
        'all_masks': all_masks, 'category_names': metadata['category_names'],
        'annotation_ids': metadata['annotation_ids'], 'area_ratios': metadata['area_ratios'],
    }

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def select_layer_attention_maps(captured_maps, output_size, layer_name):
    selected_maps = {}
    for key, layer_maps in captured_maps.items():
        matches = [layer_map for layer_map in layer_maps if layer_map.layer_name == layer_name]
        if len(matches) != 1:
            available = [layer_map.layer_name for layer_map in layer_maps]
            raise RuntimeError(f'Expected exactly one map for {layer_name} at {key}; got {len(matches)}. Available: {available}')
        values = matches[0].values.detach().float().cpu()[None, None]
        values = F.interpolate(values, size=output_size, mode='bilinear', align_corners=False)[0, 0]
        selected_maps[key] = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)
    return selected_maps

def install_runtime_patches(runtime, target_layer_name):
    """Install Layer L04 selection."""
    def anchors_from_current_step(self, timestep, foreground_masks, *, strategy, topk_percent, layer_index=None, layer_name=None, **kwargs):
        eff_layer = layer_name or target_layer_name
        maps = select_layer_attention_maps(self.attention_capture.maps, self.image_size, eff_layer)
        anchors = []
        for region_index, mask in enumerate(foreground_masks):
            measurement = compute_anchor_measurements(
                maps[(int(timestep), region_index)], mask.cpu(), topk_percent=topk_percent
            )
            if strategy == 'argmax':
                anchors.append((float(measurement['anchor_x']), float(measurement['anchor_y'])))
            else:
                anchors.append((float(measurement['topk_anchor_x']), float(measurement['topk_anchor_y'])))
        return anchors
    runtime._anchors_from_current_step = types.MethodType(anchors_from_current_step, runtime)

def create_comparison_grid(image_dict, sample_title, destination_path):
    """Create a 2x5 grid contact sheet comparing all 10 sigma_D values for a sample."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8.5))
    axes = axes.flatten()
    for idx, (sigma_val, img) in enumerate(image_dict.items()):
        axes[idx].imshow(img)
        axes[idx].set_title(rf'$\sigma_D = {sigma_val:.1f}$', fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    fig.suptitle(sample_title, fontsize=14, y=0.98)
    fig.tight_layout()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

print('[OK] Helpers ready.')

# --- Source notebook cell 9 ---
# 8. Hyperparameter Sweep: Generate all 8 samples x 10 sigma_D values for WM-03 + Bootstrap=1 + Layer L04.
generation_rows = []
actual_timesteps = [int(t) for t in smd.timesteps.detach().cpu().tolist()]
global_index = 0

for batch_index, batch in enumerate(loader):
    print(f'[BATCH] {batch_index + 1}/{len(loader)}')
    for local_index in range(len(batch['sample_ids'])):
        if global_index >= RUN_SAMPLES:
            break
        payload = make_payload(batch, local_index)
        seed = BASE_SEED + global_index
        sample_id = payload['sample_id']
        sample_folder = BY_SAMPLE_DIR / f'{global_index:04d}_{sample_id}'
        sample_folder.mkdir(parents=True, exist_ok=True)

        token_indices = [
            find_target_token_indices(smd.tokenizer, prompt, category)
            for prompt, category in zip(payload['foreground_prompts'], payload['category_names'])
        ]

        sample_generated_images = {}
        print(f'\n[SAMPLE {global_index + 1}/{RUN_SAMPLES}] {sample_id} | prompts: {payload["foreground_prompts"]}')

        for sigma_idx, sigma_d in enumerate(SIGMA_D_SWEEP):
            print(f'  [SWEEP {sigma_idx + 1}/10] sigma_D = {sigma_d:4.1f} | sample = {global_index}')
            with SemanticAnchorCapture(smd.unet) as capture:
                capture.configure(token_indices)
                runtime = SemanticAnchorRuntime(smd, capture, image_size=TARGET_SIZE)
                install_runtime_patches(runtime, FIXED_LAYER['layer_name'])

                seed_everything(seed)
                sync_cuda(); tic = time.perf_counter()
                generated, runtime_records = runtime.generate(
                    prompts=payload['prompts'], negative_prompts=payload['negative_prompts'],
                    masks=payload['all_masks'].to(device=device, dtype=torch.float32),
                    foreground_masks=payload['foreground_masks'], mode=BEST_ANCHOR_MODE,
                    weight_policy=WEIGHT_POLICY, bootstrap_steps=BOOTSTRAP_STEPS,
                    topk_percent=TOPK_ATTENTION_PERCENT,
                    spatial_sigma_latent=float(sigma_d),
                    semantic_sigma_scale=SEMANTIC_SIGMA_SCALE,
                    mask_stds=MASK_STD, mask_strengths=MASK_STRENGTH,
                    preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
                    attention_layer_name=FIXED_LAYER['layer_name'],
                )
                sync_cuda(); elapsed = time.perf_counter() - tic

                # 1. Save to by_sigma directory
                sigma_path = BY_SIGMA_DIR / f'sigma_{sigma_d:04.1f}' / f'{global_index:04d}_{sample_id}.png'
                generated.save(sigma_path)

                # 2. Save to by_sample directory
                sample_path = sample_folder / f'sigma_{sigma_d:04.1f}.png'
                generated.save(sample_path)

                sample_generated_images[sigma_d] = generated

                generation_rows.append({
                    'sample_index': global_index, 'sample_id': sample_id,
                    'image_id': payload['image_id'], 'seed': seed,
                    'method_id': METHOD_ID,
                    'method_label': METHOD_LABEL,
                    'anchor_mode': BEST_ANCHOR_MODE,
                    'weight_policy': WEIGHT_POLICY,
                    'anchor_strategy': ('topk_projected_centroid' if BEST_ANCHOR_MODE == 'semantic_topk_anchor' else 'argmax'),
                    'bootstrap_steps': BOOTSTRAP_STEPS,
                    'layer_id': FIXED_LAYER['layer_id'],
                    'layer_name': FIXED_LAYER['layer_name'],
                    'spatial_sigma_latent': float(sigma_d),
                    'semantic_sigma_scale': SEMANTIC_SIGMA_SCALE,
                    'elapsed_sec': elapsed,
                    'sigma_path': str(sigma_path),
                    'sample_path': str(sample_path),
                })

                capture.maps.clear()
                del generated, runtime_records, runtime, capture
                gc.collect(); torch.cuda.empty_cache()

        # Generate side-by-side comparison grid for this sample
        sample_title = rf'Sample {global_index}: {sample_id} | {METHOD_ID} (Top-k) | Bootstrap={BOOTSTRAP_STEPS} | Layer {FIXED_LAYER["layer_id"]} | $\sigma_D$ Sweep (1.0 $\rightarrow$ 64.0)'
        grid_path = COMPARISON_DIR / f'{global_index:04d}_{sample_id}_grid.png'
        create_comparison_grid(sample_generated_images, sample_title, grid_path)
        print(f'  [OK] Saved comparison grid: {grid_path.name}')

        del payload, token_indices, sample_generated_images
        gc.collect(); torch.cuda.empty_cache()
        global_index += 1

    del batch
    gc.collect(); torch.cuda.empty_cache()
    if global_index >= RUN_SAMPLES:
        break

print(f'\n[COMPLETED] Generated {len(generation_rows)} images across {global_index} samples.')

# --- Source notebook cell 10 ---
# 9. Save image manifest and sweep summary.
generation_df = pd.DataFrame(generation_rows)
expected_images = RUN_SAMPLES * len(SIGMA_D_SWEEP)
assert len(generation_df) == expected_images, (len(generation_df), expected_images)

GENERATION_CSV = RUN_ROOT / 'generation_summary.csv'
GENERATION_JSON = RUN_ROOT / 'generation_summary.json'
CONFIG_JSON = RUN_ROOT / 'run_config.json'

generation_df.to_csv(GENERATION_CSV, index=False, encoding='utf-8-sig')
GENERATION_JSON.write_text(json.dumps(generation_rows, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
CONFIG_JSON.write_text(json.dumps({
    'run_id': RUN_ID, 'profile': RUN_PROFILE, 'run_samples': RUN_SAMPLES,
    'model': MODEL_ID, 'sampler': 'LCM', 'resolution': list(TARGET_SIZE),
    'seed_rule': 'BASE_SEED + sample_index', 'base_seed': BASE_SEED,
    'bootstrap_steps': BOOTSTRAP_STEPS,
    'fixed_layer': FIXED_LAYER,
    'method_id': METHOD_ID,
    'method_label': METHOD_LABEL,
    'anchor_mode': BEST_ANCHOR_MODE,
    'weight_policy': WEIGHT_POLICY,
    'anchor_strategy': ('topk_projected_centroid' if BEST_ANCHOR_MODE == 'semantic_topk_anchor' else 'argmax'),
    'topk_attention_percent': TOPK_ATTENTION_PERCENT,
    'semantic_sigma_scale': SEMANTIC_SIGMA_SCALE,
    'sigma_d_sweep': SIGMA_D_SWEEP,
    'expected_images': expected_images,
    'generation_csv': str(GENERATION_CSV),
    'semantic_anchor_runtime_sha256': hashlib.sha256((OURS_SRC / 'experiments/semantic_anchor.py').read_bytes()).hexdigest(),
}, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

summary = generation_df.groupby('spatial_sigma_latent', as_index=False).agg(
    samples=('sample_index', 'nunique'),
    mean_elapsed_sec=('elapsed_sec', 'mean'),
    total_elapsed_sec=('elapsed_sec', 'sum'),
)
display(Markdown(rf'## Spatial Sigma ($\sigma_D$) Sweep Summary: {METHOD_ID} (Bootstrap={BOOTSTRAP_STEPS}, Layer {FIXED_LAYER["layer_id"]})'))
display(summary.style.format({
    'mean_elapsed_sec': '{:.3f}s',
    'total_elapsed_sec': '{:.1f}s',
}))
print('[OK] Summary saved to:', GENERATION_CSV)
