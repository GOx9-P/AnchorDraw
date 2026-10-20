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

# Notebook: Ours/semantic_anchor_experiments/semantic_anchor_sd15_lcm_top3_layers_smoke4_colab.ipynb
# Notebook SHA256: ec5a8e1cd6ec59a489d860c93cdf388cf69e81863b608c88e23698db6159e569

# --- Source notebook cell 3 ---
# 2. Run configuration. Four samples, three layer ablations.
import json, hashlib

RUN_PROFILE = os.environ.get('RUN_PROFILE', 'smoke8')
PROFILE = {
    'run_id': 'semantic_anchor_weighted_mask_sd15_lcm_smoke2_all_artifacts_shortpaths',
    'manifest': 'Ours/test_sets/manifests/smoke/coco_val2017_multidiffusion_coco_all_512x512_smoke_bs8.jsonl',
    'expected_samples': 8,
}
RUN_SAMPLES = 4
MODEL_ID = 'runwayml/stable-diffusion-v1-5'
TARGET_SIZE = (512, 512)
BASE_SEED = 2024
BATCH_SIZE = 1
BOOTSTRAP_STEPS = 1
MASK_STD = 1.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = 'discrete'
NEGATIVE_PROMPT = ''
BEST_ANCHOR_MODE = 'semantic_topk_anchor'
TOPK_ATTENTION_PERCENT = 10.0
SPATIAL_SIGMA_LATENT = 8.0
SEMANTIC_SIGMA_SCALE = 1.0

# layer_index is retained for ranking metadata; runtime selection uses layer_name.
LAYER_EXPERIMENTS = {
    'L06': {
        'rank': 1, 'layer_index': 6, 'native_spatial_size': 8,
        'layer_name': 'mid_block.attentions.0.transformer_blocks.0.attn2.processor',
    },
    'L04': {
        'rank': 2, 'layer_index': 4, 'native_spatial_size': 16,
        'layer_name': 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    },
    'L02': {
        'rank': 3, 'layer_index': 2, 'native_spatial_size': 32,
        'layer_name': 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    },
}
LAYER_IDS = tuple(LAYER_EXPERIMENTS)
EXPERIMENTS = {
    'WM-00': {'label': 'Quantized baseline', 'anchor_mode': 'baseline', 'weight_policy': 'quantized_baseline'},
    'WM-01': {'label': 'BBox + adaptive bilateral', 'anchor_mode': 'bbox_control', 'weight_policy': 'adaptive_bilateral'},
    'WM-02': {'label': 'Argmax + adaptive bilateral', 'anchor_mode': 'semantic_anchor', 'weight_policy': 'adaptive_bilateral'},
    'WM-03': {'label': 'Top-k + adaptive bilateral', 'anchor_mode': BEST_ANCHOR_MODE, 'weight_policy': 'adaptive_bilateral'},
    'WM-04': {'label': 'Top-k + spatial only', 'anchor_mode': BEST_ANCHOR_MODE, 'weight_policy': 'spatial_only'},
    'WM-05': {'label': 'Top-k + semantic only', 'anchor_mode': BEST_ANCHOR_MODE, 'weight_policy': 'semantic_only'},
}
EXPERIMENT_IDS = tuple(EXPERIMENTS)

COCO_ROOT = Path(os.environ.get('COCO_ROOT', str(WORK_DIR / 'COCO')))
RUN_MANIFEST = REPO_ROOT / PROFILE['manifest']
BASE_OUTPUT_DIR = Path(str(WORK_DIR / 'anchordraw_runs'))
RUN_ID = 'semantic_anchor_sd15_lcm_all_methods_top3_layers_smoke4'
RUN_ROOT = BASE_OUTPUT_DIR / RUN_ID
MASK_CACHE_DIR = RUN_ROOT / 'mask_cache'
GENERATED_DIR = RUN_ROOT / 'generated_images'
ATTENTION_DIR = RUN_ROOT / 'selected_attention_maps'
for path in [MASK_CACHE_DIR, GENERATED_DIR, ATTENTION_DIR]:
    path.mkdir(parents=True, exist_ok=True)
for experiment_id in EXPERIMENT_IDS:
    for path in [GENERATED_DIR, ATTENTION_DIR]:
        for layer_id in LAYER_IDS:
            (path / experiment_id / layer_id).mkdir(parents=True, exist_ok=True)

assert RUN_MANIFEST.exists(), f'Missing manifest: {RUN_MANIFEST}'
assert RUN_SAMPLES <= PROFILE['expected_samples']
print('[OK] Run samples:', RUN_SAMPLES, '| methods:', EXPERIMENT_IDS, '| layers:', LAYER_IDS, '| output:', RUN_ROOT)
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 4 ---
# 3. Imports and source modules.
import sys, importlib.util, time, gc, inspect, shutil, types
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
# A failed rerun can leave the wrong `data` module cached from Baseline.
for module_name in list(sys.modules):
    if module_name == 'data' or module_name.startswith('data.') or module_name == 'experiments' or module_name.startswith('experiments.'):
        del sys.modules[module_name]
sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from experiments.semantic_anchor import (
    SemanticAnchorCapture, SemanticAnchorRuntime, aggregate_attention_maps,
    compute_anchor_measurements, find_target_token_indices,
)
# Add Baseline only after importing Ours' data package; both trees contain a data.py.
sys.path.insert(0, str(BASELINE_SRC))

pipeline_path = BASELINE_SRC / 'model/pipeline_semantic_draw.py'
spec = importlib.util.spec_from_file_location('pipeline_semantic_draw_original', pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline
runtime_parameters = inspect.signature(SemanticAnchorRuntime.generate).parameters
RUNTIME_SUPPORTS_LAYER_NAME = 'attention_layer_name' in runtime_parameters
assert RUNTIME_SUPPORTS_LAYER_NAME or 'attention_layer_index' in runtime_parameters, (
    'The checked-out repo is missing the per-layer runtime update.'
)
print('[OK] Runtime selector:', 'layer_name' if RUNTIME_SUPPORTS_LAYER_NAME else 'name-compatibility fallback')

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
for layer_id, layer_config in LAYER_EXPERIMENTS.items():
    assert layer_config['layer_name'] in available_layer_names, layer_config
print('[OK] GPU:', torch.cuda.get_device_name(0))
print('[OK] Selected layer names:', {k: v['layer_name'] for k, v in LAYER_EXPERIMENTS.items()})
print('[OK] Timesteps:', [int(t) for t in smd.timesteps.cpu().tolist()])

# --- Source notebook cell 8 ---
# 7. Payload and result helpers.
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

def append_jsonl(path, records):
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def select_layer_attention_maps(captured_maps, output_size, layer_name):
    """Resize one named capture map per timestep/region; no layer-order assumption."""
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

def install_name_selector_fallback(runtime, layer_name):
    """Use layer names even with an older runtime that only exposes layer indices."""
    def anchors_from_current_step(self, timestep, foreground_masks, *, strategy, topk_percent, layer_index=None):
        maps = select_layer_attention_maps(self.attention_capture.maps, self.image_size, layer_name)
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

print('[OK] Helpers ready.')

# --- Source notebook cell 9 ---
# 8. Generate every WM-00...WM-05 method for every selected layer.
generation_rows, anchor_rows = [], []
actual_timesteps = [int(t) for t in smd.timesteps.detach().cpu().tolist()]
global_index = 0

for batch_index, batch in enumerate(loader):
    print(f'[BATCH] {batch_index + 1}/{len(loader)}')
    for local_index in range(len(batch['sample_ids'])):
        if global_index >= RUN_SAMPLES:
            break
        payload = make_payload(batch, local_index)
        seed = BASE_SEED + global_index
        token_indices = [
            find_target_token_indices(smd.tokenizer, prompt, category)
            for prompt, category in zip(payload['foreground_prompts'], payload['category_names'])
        ]

        for experiment_id, experiment in EXPERIMENTS.items():
            for layer_id, layer_config in LAYER_EXPERIMENTS.items():
                layer_index = layer_config['layer_index']
                print(f'  [RUN] sample={global_index} | {experiment_id} | {layer_id} | {payload["sample_id"]}')
                with SemanticAnchorCapture(smd.unet) as capture:
                    capture.configure(token_indices)
                    runtime = SemanticAnchorRuntime(smd, capture, image_size=TARGET_SIZE)
                    if RUNTIME_SUPPORTS_LAYER_NAME:
                        layer_selector_kwargs = {'attention_layer_name': layer_config['layer_name']}
                        runtime_selector = 'native_layer_name'
                    else:
                        install_name_selector_fallback(runtime, layer_config['layer_name'])
                        layer_selector_kwargs = {'attention_layer_index': 0}
                        runtime_selector = 'name_compatibility_fallback'
                    seed_everything(seed)
                    sync_cuda(); tic = time.perf_counter()
                    generated, runtime_records = runtime.generate(
                        prompts=payload['prompts'], negative_prompts=payload['negative_prompts'],
                        masks=payload['all_masks'].to(device=device, dtype=torch.float32),
                        foreground_masks=payload['foreground_masks'], mode=experiment['anchor_mode'],
                        weight_policy=experiment['weight_policy'], bootstrap_steps=BOOTSTRAP_STEPS,
                        topk_percent=TOPK_ATTENTION_PERCENT,
                        spatial_sigma_latent=SPATIAL_SIGMA_LATENT,
                        semantic_sigma_scale=SEMANTIC_SIGMA_SCALE,
                        mask_stds=MASK_STD, mask_strengths=MASK_STRENGTH,
                        preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
                        **layer_selector_kwargs,
                    )
                    sync_cuda(); elapsed = time.perf_counter() - tic

                    generated_path = GENERATED_DIR / experiment_id / layer_id / f'{global_index:04d}_{payload["sample_id"]}_generated.png'
                    generated.save(generated_path)
                    selected_maps = select_layer_attention_maps(
                        runtime.aligned_attention_maps(), TARGET_SIZE, layer_config['layer_name']
                    )
                    for step_index, timestep in enumerate(actual_timesteps):
                        for region_index, (prompt, category, ann_id) in enumerate(zip(
                            payload['foreground_prompts'], payload['category_names'], payload['annotation_ids']
                        )):
                            heatmap = selected_maps[(timestep, region_index)]
                            measurement = compute_anchor_measurements(
                                heatmap, payload['foreground_masks'][region_index],
                                topk_percent=TOPK_ATTENTION_PERCENT,
                            )
                            runtime_record = runtime_records[step_index]
                            measurement.update({
                                'experiment_id': experiment_id, 'experiment_label': experiment['label'],
                                'layer_id': layer_id, 'layer_rank': layer_config['rank'],
                                'layer_index': layer_index, 'layer_name': layer_config['layer_name'],
                                'native_spatial_size': layer_config['native_spatial_size'],
                                'sample_index': global_index, 'sample_id': payload['sample_id'],
                                'image_id': payload['image_id'], 'region_index': region_index,
                                'annotation_id': int(ann_id), 'category': category, 'prompt': prompt,
                                'step_index': step_index, 'timestep': timestep, 'seed': seed,
                                'anchor_mode': experiment['anchor_mode'], 'weight_policy': experiment['weight_policy'],
                                'runtime_selector': runtime_selector,
                                'runtime_selection_source': runtime_record.selection_source,
                                'runtime_anchor_source_step_index': runtime_record.anchor_source_step_index,
                                'generated_path': str(generated_path),
                            })
                            anchor_rows.append(measurement)
                            debug_path = ATTENTION_DIR / experiment_id / layer_id / f'{global_index:04d}_{payload["sample_id"]}' / f'r{region_index:02d}_s{step_index:02d}_t{timestep}.png'
                            debug_path.parent.mkdir(parents=True, exist_ok=True)
                            plt.imsave(debug_path, heatmap.numpy(), cmap='magma', vmin=0, vmax=1)

                    generation_rows.append({
                        'experiment_id': experiment_id, 'experiment_label': experiment['label'],
                        'layer_id': layer_id, 'layer_rank': layer_config['rank'],
                        'layer_index': layer_index, 'layer_name': layer_config['layer_name'],
                        'native_spatial_size': layer_config['native_spatial_size'],
                        'sample_index': global_index, 'sample_id': payload['sample_id'],
                        'image_id': payload['image_id'], 'seed': seed, 'elapsed_sec': elapsed,
                        'anchor_mode': experiment['anchor_mode'], 'weight_policy': experiment['weight_policy'],
                        'runtime_selector': runtime_selector,
                        'generated_path': str(generated_path), 'timesteps': actual_timesteps,
                    })
                    capture.maps.clear()
                    del selected_maps, generated, runtime_records, runtime, capture
                    gc.collect(); torch.cuda.empty_cache()

        del payload, token_indices
        gc.collect(); torch.cuda.empty_cache()
        global_index += 1
    del batch
    gc.collect(); torch.cuda.empty_cache()
    if global_index >= RUN_SAMPLES:
        break

print('[OK] Generated', len(generation_rows), 'images for', global_index, 'samples.')

# --- Source notebook cell 10 ---
# 9. Save image manifest, per-step anchor metrics and summary.
generation_df = pd.DataFrame(generation_rows)
anchor_df = pd.DataFrame(anchor_rows)
assert len(generation_df) == RUN_SAMPLES * len(EXPERIMENT_IDS) * len(LAYER_IDS)
assert set(generation_df['layer_id']) == set(LAYER_IDS)
assert set(generation_df['experiment_id']) == set(EXPERIMENT_IDS)

GENERATION_CSV = RUN_ROOT / 'generation_summary.csv'
GENERATION_JSON = RUN_ROOT / 'generation_summary.json'
ANCHOR_CSV = RUN_ROOT / 'per_layer_anchor_metrics.csv'
ANCHOR_JSONL = RUN_ROOT / 'per_layer_anchor_metrics.jsonl'
SUMMARY_CSV = RUN_ROOT / 'per_layer_anchor_summary.csv'
CONFIG_JSON = RUN_ROOT / 'run_config.json'
generation_df.to_csv(GENERATION_CSV, index=False, encoding='utf-8-sig')
GENERATION_JSON.write_text(json.dumps(generation_rows, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
anchor_df.to_csv(ANCHOR_CSV, index=False, encoding='utf-8-sig')
append_jsonl(ANCHOR_JSONL, anchor_rows)

temporal_rows = []
for keys, group in anchor_df.groupby(['experiment_id', 'layer_id', 'sample_index', 'region_index']):
    points = group.sort_values('step_index')[['anchor_x', 'anchor_y']].to_numpy(float)
    center = points.mean(axis=0, keepdims=True)
    jumps = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.asarray([0.0])
    temporal_rows.append({
        'experiment_id': keys[0], 'layer_id': keys[1], 'sample_index': keys[2], 'region_index': keys[3],
        'temporal_anchor_std_px': float(np.linalg.norm(points - center, axis=1).mean()),
        'mean_adjacent_anchor_jump_px': float(jumps.mean()),
    })
temporal_df = pd.DataFrame(temporal_rows)
temporal_summary = temporal_df.groupby(['experiment_id', 'layer_id'], as_index=False).agg(
    temporal_anchor_std_px_mean=('temporal_anchor_std_px', 'mean'),
    mean_adjacent_anchor_jump_px=('mean_adjacent_anchor_jump_px', 'mean'),
)
summary = anchor_df.groupby(['experiment_id', 'experiment_label', 'anchor_mode', 'weight_policy', 'layer_id', 'layer_index', 'layer_name', 'native_spatial_size'], as_index=False).agg(
    samples=('sample_index', 'nunique'), measurements=('anchor_attention', 'size'),
    anchor_attention_mean=('anchor_attention', 'mean'),
    topk_anchor_attention_mean=('topk_anchor_attention_mean', 'mean'),
    global_peak_inside_mask_mean=('global_peak_inside_mask', 'mean'),
    distance_to_bbox_center_norm_mean=('distance_to_bbox_center_norm', 'mean'),
)
summary = summary.merge(temporal_summary, on=['experiment_id', 'layer_id'], how='left')
summary.to_csv(SUMMARY_CSV, index=False, encoding='utf-8-sig')
CONFIG_JSON.write_text(json.dumps({
    'run_id': RUN_ID, 'profile': RUN_PROFILE, 'run_samples': RUN_SAMPLES,
    'model': MODEL_ID, 'sampler': 'LCM', 'resolution': list(TARGET_SIZE),
    'seed_rule': 'BASE_SEED + sample_index', 'base_seed': BASE_SEED,
    'topk_attention_percent': TOPK_ATTENTION_PERCENT,
    'layers': LAYER_EXPERIMENTS, 'experiments': EXPERIMENTS, 'generation_csv': str(GENERATION_CSV),
    'anchor_metrics_csv': str(ANCHOR_CSV),
    'semantic_anchor_runtime_sha256': hashlib.sha256((OURS_SRC / 'experiments/semantic_anchor.py').read_bytes()).hexdigest(),
}, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

display(Markdown('## WM-00...WM-05 × top-3 layer generation summary'))
display(summary.style.format({
    'anchor_attention_mean': '{:.4f}', 'topk_anchor_attention_mean': '{:.4f}',
    'global_peak_inside_mask_mean': '{:.4f}',
    'distance_to_bbox_center_norm_mean': '{:.4f}',
    'temporal_anchor_std_px_mean': '{:.2f}',
}))
print('[OK] Images:', GENERATED_DIR)
print('[OK] Generation CSV:', GENERATION_CSV)
print('[OK] Anchor metrics:', ANCHOR_CSV)
