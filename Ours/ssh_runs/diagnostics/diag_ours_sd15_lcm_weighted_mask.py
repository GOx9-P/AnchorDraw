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

# Notebook: Ours/semantic_anchor_experiments/semantic_anchor_sd15_lcm_weighted_mask_all_ids_colab.ipynb
# Notebook SHA256: 4f648c696b6515ccdaa9631a76ff8a9a348604afa2ad395bab3ab3483634c367

# --- Source notebook cell 3 ---
# 2. Cáº¥u hĂ¬nh sĂ¡u thĂ­ nghiá»‡m Weighted Mask.
import json
import hashlib

RUN_PROFILE = os.environ.get('RUN_PROFILE', 'smoke8')  # 'smoke8' hoáº·c 'full1073'
PROFILE_CONFIGS = {
    'smoke8': {
        'run_id': 'semantic_anchor_weighted_mask_sd15_lcm_smoke2_all_artifacts_shortpaths',
        'manifest': 'Ours/test_sets/manifests/smoke/coco_val2017_multidiffusion_coco_all_512x512_smoke_bs8.jsonl',
        'expected_samples': 8,
    },
    'full1073': {
        'run_id': 'semantic_anchor_weighted_mask_sd15_lcm_full1073',
        'manifest': 'Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl',
        'expected_samples': 1073,
    },
}
assert RUN_PROFILE in PROFILE_CONFIGS
PROFILE = PROFILE_CONFIGS[RUN_PROFILE]
RUN_ID, EXPECTED_SAMPLES = PROFILE['run_id'], PROFILE['expected_samples']
# Chi xu ly 2 sample dau tien cua manifest smoke8 de giam RAM, nhung van chay du 6 ID.
MAX_RUN_SAMPLES = 2
RUN_SAMPLES = min(EXPECTED_SAMPLES, MAX_RUN_SAMPLES)

MODEL_ID = 'runwayml/stable-diffusion-v1-5'
TARGET_SIZE = (512, 512)
BASE_SEED = 2024
# Process one sample at a time to lower CPU RAM while running all 6 ablations.
BATCH_SIZE = 1
BOOTSTRAP_STEPS = 1
# Mask blur pháº£i khĂ¡c 0 Ä‘á»ƒ cĂ¡c biĂªn mask cĂ³ vĂ¹ng overlap cho Weighted Mask xá»­ lĂ½.
MASK_STD = 1.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = 'discrete'
NEGATIVE_PROMPT = ''

BEST_ANCHOR_MODE = 'semantic_topk_anchor'  # CĂ³ thá»ƒ Ä‘á»•i thĂ nh 'semantic_anchor'.
TOPK_ATTENTION_PERCENT = 10.0
SPATIAL_SIGMA_LATENT = 8.0
SEMANTIC_SIGMA_SCALE = 1.0
EXPERIMENTS = {
    'WM-00': {'label': 'Quantized baseline', 'anchor_mode': 'baseline', 'weight_policy': 'quantized_baseline'},
    'WM-01': {'label': 'BBox + adaptive bilateral', 'anchor_mode': 'bbox_control', 'weight_policy': 'adaptive_bilateral'},
    'WM-02': {'label': 'Argmax + adaptive bilateral', 'anchor_mode': 'semantic_anchor', 'weight_policy': 'adaptive_bilateral'},
    'WM-03': {'label': 'Top-k + adaptive bilateral', 'anchor_mode': BEST_ANCHOR_MODE, 'weight_policy': 'adaptive_bilateral'},
    'WM-04': {'label': 'Best anchor + spatial only', 'anchor_mode': BEST_ANCHOR_MODE, 'weight_policy': 'spatial_only'},
    'WM-05': {'label': 'Best anchor + semantic only', 'anchor_mode': BEST_ANCHOR_MODE, 'weight_policy': 'semantic_only'},
}
EXPERIMENT_IDS = tuple(EXPERIMENTS)  # CĂ³ thá»ƒ chá»n má»™t pháº§n, vĂ­ dá»¥ ('WM-00', 'WM-03').

# Run-safe artifact policy.
# Final generated images and scalar metrics are always saved for every selected ID.
# Heavy artifacts contain decoded latents, full-resolution maps, matplotlib figures, and optional arrays.
# Chi luu artifact cho 2 sample dau tien; moi ID duoc giai phong bo nho sau khi xong.
SAVE_DIAGNOSTIC_ARTIFACTS = True
MAX_ARTIFACT_SAMPLES = 2
ARTIFACT_EXPERIMENT_IDS = EXPERIMENT_IDS
ARTIFACT_STEP_INDICES = (0, 1, 2, 3, 4)
SAVE_INTERMEDIATE_STEP_IMAGES = True
SAVE_WEIGHTED_FIGURES = True
SAVE_NUMERIC_ARRAYS = True
# Xuất riêng từng cross-attention layer; map aggregate chỉ giữ để chọn anchor/runtime.
SAVE_PER_LAYER_ATTENTION = True
LAYER_ANALYSIS_MAX_LAYERS = None  # None = giữ toàn bộ layer attn2 đã capture.
LAYER_ANALYSIS_STEP_INDICES = None  # None = xuất mọi denoising step trong artifact.
SHOW_NOTEBOOK_PREVIEWS = False
MAX_DISPLAY_SAMPLES = 0
DISPLAY_EXPERIMENT_IDS = ()
AUTO_DOWNLOAD_ZIP = True
RUN_PROCESSOR_PARITY_CHECK = True
RUN_WM00_PARITY_CHECK = True

COCO_ROOT = Path(os.environ.get('COCO_ROOT', str(WORK_DIR / 'COCO')))
RUN_MANIFEST = REPO_ROOT / PROFILE['manifest']
BASE_OUTPUT_DIR = Path(str(WORK_DIR / 'anchordraw_runs'))
RUN_ROOT = BASE_OUTPUT_DIR / RUN_ID
MASK_CACHE_DIR = RUN_ROOT / 'mask_cache'
GENERATED_DIR = RUN_ROOT / 'generated_images'
OVERLAY_DIR = RUN_ROOT / 'mask_overlays'
ATTENTION_DIR = RUN_ROOT / 'attention_maps'
WEIGHT_DIR = RUN_ROOT / 'effective_weight_maps'
VISUALIZATION_DIR = RUN_ROOT / 'visualizations'
INTERMEDIATE_DIR = RUN_ROOT / 'intermediate_step_images'
NUMERIC_DIR = RUN_ROOT / 'attention_and_weight_arrays'
ZIP_PATH = BASE_OUTPUT_DIR / f'{RUN_ID}__export.zip'
for path in [RUN_ROOT, MASK_CACHE_DIR, GENERATED_DIR, OVERLAY_DIR, ATTENTION_DIR, WEIGHT_DIR, VISUALIZATION_DIR, INTERMEDIATE_DIR, NUMERIC_DIR]:
    path.mkdir(parents=True, exist_ok=True)
for experiment_id in EXPERIMENT_IDS:
    for parent in [GENERATED_DIR, ATTENTION_DIR, WEIGHT_DIR, VISUALIZATION_DIR, INTERMEDIATE_DIR, NUMERIC_DIR]:
        (parent / experiment_id).mkdir(parents=True, exist_ok=True)

assert RUN_MANIFEST.exists(), f'Thiáº¿u manifest: {RUN_MANIFEST}'
assert BOOTSTRAP_STEPS == 1, 'Thiáº¿t káº¿ ablation nĂ y giá»¯ nguyĂªn step 0 cá»§a baseline.'
assert 0 < TOPK_ATTENTION_PERCENT <= 100
assert set(EXPERIMENT_IDS).issubset(EXPERIMENTS)
print('[OK] Profile:', RUN_PROFILE, '| manifest samples:', EXPECTED_SAMPLES, '| run samples:', RUN_SAMPLES)
print('[OK] IDs:', EXPERIMENT_IDS)
print('[OK] Output:', RUN_ROOT)
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 5 ---
# 4. Import dataloader, attention capture, runtime Weighted Mask vĂ  baseline gá»‘c.
import sys, importlib, importlib.util, time, csv, shutil, gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
# terminal display provided above

OURS_SRC = REPO_ROOT / 'Ours/src'
BASELINE_SRC = REPO_ROOT / 'Baseline/semantic-draw-main/src'
sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from data.visualize import make_mask_overlay
from experiments.semantic_anchor import (
    SemanticAnchorCapture, SemanticLatentStepCapture, SemanticAnchorRuntime,
    aggregate_attention_maps, resize_layer_attention_maps, compute_anchor_measurements,
    find_target_token_indices,
)
sys.path.insert(0, str(BASELINE_SRC))
pipeline_path = BASELINE_SRC / 'model/pipeline_semantic_draw.py'
spec = importlib.util.spec_from_file_location('pipeline_semantic_draw_original', pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline
print('[OK] Baseline file:', pipeline_path)
print('[OK] Runtime:', SemanticAnchorRuntime.__name__)

# --- Source notebook cell 6 ---
# 5. Dataloader dĂ¹ng manifest theo RUN_PROFILE.
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
assert len(loader.dataset) == EXPECTED_SAMPLES, f'Manifest pháº£i cĂ³ {EXPECTED_SAMPLES} record, hiá»‡n cĂ³ {len(loader.dataset)}'
preview_batch = next(iter(loader))
print('[OK] Samples:', len(loader.dataset))
print('[OK] Batches:', len(loader))
print('[OK] Mask tensor:', tuple(preview_batch['masks'].shape))
del preview_batch
gc.collect()

# --- Source notebook cell 7 ---
# 6. Load SemanticDraw SD1.5 + LCM theo Ä‘Ăºng constructor baseline.
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

assert torch.cuda.is_available(), 'HĂ£y báº­t GPU trong Runtime > Change runtime type.'
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
print('[OK] GPU:', torch.cuda.get_device_name(0))
print('[OK] Scheduler:', type(smd.scheduler).__name__)
print('[OK] Actual timesteps:', [int(t) for t in smd.timesteps.cpu().tolist()])

# --- Source notebook cell 8 ---
# 7. Kiá»ƒm tra processor thu attention cĂ³ Ä‘áº§u ra gáº§n tÆ°Æ¡ng Ä‘Æ°Æ¡ng processor hiá»‡n táº¡i.
# ÄĂ¢y lĂ  phĂ©p kiá»ƒm tra cá»¥c bá»™ trĂªn má»™t lá»›p attn2, khĂ´ng cháº¡y thĂªm má»™t áº£nh diffusion.
if RUN_PROCESSOR_PARITY_CHECK:
    capture_test = SemanticAnchorCapture(smd.unet)
    attn_name, attn_module = next((name, module) for name, module in smd.unet.named_modules() if name.endswith('attn2'))
    original_processor = attn_module.processor
    query_dim = attn_module.to_q.in_features
    cross_dim = attn_module.to_k.in_features
    hidden = torch.randn(2, 16, query_dim, device=device, dtype=dtype)
    encoder = torch.randn(2, 77, cross_dim, device=device, dtype=dtype)
    with torch.no_grad():
        original_output = attn_module(hidden, encoder_hidden_states=encoder)
    capture_test.install()
    capture_test.store.disable()
    with torch.no_grad():
        captured_output = attn_module(hidden, encoder_hidden_states=encoder)
    capture_test.restore()
    parity_max = float((original_output.float() - captured_output.float()).abs().max())
    parity_mean = float((original_output.float() - captured_output.float()).abs().mean())
    print({'layer': attn_name, 'max_abs_diff': parity_max, 'mean_abs_diff': parity_mean})
    assert parity_max < 0.02, 'Processor thu attention lá»‡ch quĂ¡ lá»›n so vá»›i processor ban Ä‘áº§u.'
    del hidden, encoder, original_output, captured_output
    torch.cuda.empty_cache()
else:
    parity_max = None
    parity_mean = None
    print('[INFO] Bá» qua processor parity check.')

# --- Source notebook cell 9 ---
# 8. Chuáº©n bá»‹ protocol input, artifact vĂ  helper ghi káº¿t quáº£.
def make_payload(batch, index):
    item = batch_item_to_semanticdraw_inputs(batch, index)
    metadata = item['metadata']
    foreground_masks = item['masks'].float().cpu()
    background_mask = (1.0 - foreground_masks.sum(dim=0, keepdim=True).clamp(0, 1)).clamp(0, 1)
    all_masks = torch.cat([background_mask, foreground_masks], dim=0)
    prompts = [item['background_prompt'], *item['prompts']]
    assert len(prompts) == len(all_masks)
    return {
        'sample_id': metadata['sample_id'], 'image_id': metadata['image_id'], 'file_name': metadata['file_name'],
        'background_prompt': item['background_prompt'], 'prompts': prompts,
        'negative_prompts': [NEGATIVE_PROMPT] * len(prompts),
        'foreground_prompts': item['prompts'], 'foreground_masks': foreground_masks,
        'all_masks': all_masks, 'category_names': metadata['category_names'],
        'annotation_ids': metadata['annotation_ids'], 'area_ratios': metadata['area_ratios'],
    }

def decode_captured_step(record):
    latent = record.latent.to(device=device, dtype=smd.dtype)
    image = smd.decode_latents(latent)[0].detach().float().cpu().clamp(0, 1)
    return Image.fromarray((image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))

def append_jsonl(path, records):
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

def save_weighted_figure(original, step_image, mask, heatmap, effective_weight, measurement, runtime_record, weight_record, title, destination):
    fig, axes = plt.subplots(1, 6, figsize=(27, 4.5))
    axes[0].imshow(original); axes[0].set_title('áº¢nh COCO gá»‘c')
    axes[1].imshow(mask.squeeze().cpu().numpy(), cmap='gray', vmin=0, vmax=1); axes[1].set_title('Object mask')
    axes[2].imshow(heatmap.cpu().numpy(), cmap='magma', vmin=0, vmax=1); axes[2].set_title('Cross-attention heatmap')
    axes[3].imshow(effective_weight.squeeze().cpu().numpy(), cmap='viridis', vmin=0, vmax=1); axes[3].set_title(f'Effective weight ({weight_record.policy})')
    axes[4].imshow(original); axes[4].imshow(heatmap.cpu().numpy(), cmap='magma', alpha=0.45, vmin=0, vmax=1)
    # Current-step candidates are shown separately from the point actually consumed by this step.
    axes[4].scatter(measurement['anchor_x'], measurement['anchor_y'], c='lime', marker='x', s=80, linewidths=3, label='Masked argmax (current)')
    axes[4].scatter(measurement['topk_center_x'], measurement['topk_center_y'], facecolors='none', edgecolors='white', marker='o', s=58, linewidths=1.5, label=f"Top-{measurement['topk_percent']:g}% centroid (current)")
    axes[4].scatter(measurement['topk_anchor_x'], measurement['topk_anchor_y'], c='orange', marker='D', s=46, label='Top-k projected anchor (current)')
    axes[4].scatter(measurement['bbox_center_x'], measurement['bbox_center_y'], c='yellow', marker='+', s=65, label='BBox center')
    if runtime_record.points_xy:
        x, y = runtime_record.points_xy[measurement['region_index']]
        source_step = runtime_record.anchor_source_step_index
        runtime_label = (f'Runtime reference (from step {source_step})' if source_step is not None else 'Runtime reference (bootstrap)')
        axes[4].scatter(x, y, c='cyan', marker='o', s=45, label=runtime_label)
    source_title = runtime_record.selection_source
    if runtime_record.anchor_source_step_index is not None:
        source_title += f' | uses attention from step {runtime_record.anchor_source_step_index}'
    axes[4].set_title(source_title); axes[4].legend(loc='lower right', fontsize=6)
    axes[5].imshow(step_image); axes[5].set_title('áº¢nh trung gian sau step')
    for axis in axes: axis.axis('off')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title); plt.tight_layout(); fig.savefig(destination, dpi=150, bbox_inches='tight'); plt.close(fig)

print('[OK] Helper functions sáºµn sĂ ng.')

# --- Source notebook cell 10 ---
# 9. Cháº¡y WM-00 Ä‘áº¿n WM-05 theo tá»«ng sample, cĂ¹ng seed giá»¯a cĂ¡c ID.
metrics_rows, layer_metrics_rows, generation_rows, debug_rows = [], [], [], []
global_index = 0
actual_timesteps = [int(t) for t in smd.timesteps.detach().cpu().tolist()]

for batch_index, batch in enumerate(loader):
    print(f'[BATCH] {batch_index + 1}/{len(loader)} - {len(batch["sample_ids"])} sample')
    for local_index in range(len(batch['sample_ids'])):
        payload = make_payload(batch, local_index)
        seed = BASE_SEED + global_index
        save_sample_artifacts = SAVE_DIAGNOSTIC_ARTIFACTS and global_index < MAX_ARTIFACT_SAMPLES
        original = batch['images'][local_index].resize(TARGET_SIZE[::-1], Image.Resampling.BILINEAR) if save_sample_artifacts else None
        if save_sample_artifacts:
            make_mask_overlay(original, payload['foreground_masks'], payload['category_names'], alpha=0.45).save(OVERLAY_DIR / f'{global_index:04d}_{payload["sample_id"]}_overlay.png')
        token_indices = [find_target_token_indices(smd.tokenizer, prompt, category) for prompt, category in zip(payload['foreground_prompts'], payload['category_names'])]

        for experiment_id in EXPERIMENT_IDS:
            experiment = EXPERIMENTS[experiment_id]
            save_artifacts = save_sample_artifacts and experiment_id in ARTIFACT_EXPERIMENT_IDS
            capture_step_latents = save_artifacts and (SAVE_INTERMEDIATE_STEP_IMAGES or SAVE_WEIGHTED_FIGURES)
            with SemanticAnchorCapture(smd.unet) as capture, SemanticLatentStepCapture(smd) as latent_capture:
                capture.configure(token_indices)
                runtime = SemanticAnchorRuntime(smd, capture, image_size=TARGET_SIZE)
                latent_capture.configure(actual_timesteps, enabled=capture_step_latents)
                seed_everything(seed)
                torch.cuda.synchronize(); tic = time.perf_counter()
                generated, runtime_records = runtime.generate(
                    prompts=payload['prompts'], negative_prompts=payload['negative_prompts'],
                    masks=payload['all_masks'].to(device=device, dtype=torch.float32),
                    foreground_masks=payload['foreground_masks'], mode=experiment['anchor_mode'],
                    weight_policy=experiment['weight_policy'], bootstrap_steps=BOOTSTRAP_STEPS,
                    topk_percent=TOPK_ATTENTION_PERCENT, spatial_sigma_latent=SPATIAL_SIGMA_LATENT,
                    semantic_sigma_scale=SEMANTIC_SIGMA_SCALE, capture_weight_masks=save_artifacts,
                    mask_stds=MASK_STD, mask_strengths=MASK_STRENGTH,
                    preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
                )
                torch.cuda.synchronize(); elapsed = time.perf_counter() - tic
                latent_capture.disable()
                generated_path = GENERATED_DIR / experiment_id / f'{global_index:04d}_{payload["sample_id"]}_generated.png'
                generated_path.parent.mkdir(parents=True, exist_ok=True)
                generated.save(generated_path)
                captured_steps = {record.step_index: decode_captured_step(record) for record in latent_capture.records} if capture_step_latents else {}
                if SAVE_INTERMEDIATE_STEP_IMAGES and capture_step_latents:
                    assert set(captured_steps) == set(range(len(actual_timesteps)))
                    step_dir = INTERMEDIATE_DIR / experiment_id / f'{global_index:04d}_{payload["sample_id"]}'
                    step_dir.mkdir(parents=True, exist_ok=True)
                    for step_index, image in captured_steps.items():
                        if step_index in ARTIFACT_STEP_INDICES:
                            image.save(step_dir / f'step{step_index:02d}_t{actual_timesteps[step_index]}_generated.png')

                # Runtime vẫn dùng map trung bình để giữ nguyên kết quả ablation.
                # Phân tích/export giữ lại từng map attn2 riêng, không làm mất layer.
                aggregated = runtime.align_attention_heatmaps(aggregate_attention_maps(capture.maps, TARGET_SIZE))
                resized_layer_maps = (
                    resize_layer_attention_maps(runtime.aligned_attention_maps(), TARGET_SIZE)
                    if save_artifacts and SAVE_PER_LAYER_ATTENTION else {}
                )
                arrays = {'attention_coordinate_policy': np.array('inverse_roll_image_after_latent_shift_r1')}
                for step_index, timestep in enumerate(actual_timesteps):
                    runtime_record, weight_record = runtime_records[step_index], runtime.weight_records[step_index]
                    effective_masks = runtime.weight_masks[step_index] if save_artifacts else None
                    for region_index, (prompt, category, ann_id) in enumerate(zip(payload['foreground_prompts'], payload['category_names'], payload['annotation_ids'])):
                        key = (timestep, region_index)
                        heatmap = aggregated[key]
                        layer_maps = resized_layer_maps.get(key, [])
                        if LAYER_ANALYSIS_MAX_LAYERS is not None:
                            layer_maps = layer_maps[:LAYER_ANALYSIS_MAX_LAYERS]
                        if save_artifacts and SAVE_PER_LAYER_ATTENTION:
                            assert layer_maps, f'Khong co map rieng theo layer cho key={key}.'
                        measurement = compute_anchor_measurements(heatmap, payload['foreground_masks'][region_index], topk_percent=TOPK_ATTENTION_PERCENT)
                        reference = runtime_record.points_xy[region_index] if runtime_record.points_xy else (measurement['bbox_center_x'], measurement['bbox_center_y'])
                        measurement.update({
                            'experiment_id': experiment_id, 'experiment_label': experiment['label'],
                            'anchor_mode': experiment['anchor_mode'], 'weight_policy': experiment['weight_policy'],
                            'sample_index': global_index, 'sample_id': payload['sample_id'], 'image_id': payload['image_id'],
                            'region_index': region_index, 'annotation_id': int(ann_id), 'category': category, 'prompt': prompt,
                            'step_index': step_index, 'timestep': timestep, 'selection_source': runtime_record.selection_source,
                            'runtime_reference_x': float(reference[0]), 'runtime_reference_y': float(reference[1]),
                            'overlap_pixel_count': weight_record.overlap_pixel_count, 'overlap_ratio': weight_record.overlap_ratio,
                            'spatial_sigma_latent': weight_record.spatial_sigma_latent,
                            'semantic_sigma': weight_record.semantic_sigma_by_region[region_index] if region_index < len(weight_record.semantic_sigma_by_region) else None,
                            'raw_weight_mean': weight_record.raw_weight_mean, 'raw_weight_min': weight_record.raw_weight_min,
                            'raw_weight_max': weight_record.raw_weight_max, 'generated_path': str(generated_path),
                        })
                        measurement['runtime_reference_to_anchor_px'] = float(((reference[0]-measurement['anchor_x'])**2 + (reference[1]-measurement['anchor_y'])**2)**0.5)
                        metrics_rows.append(measurement)
                        if save_artifacts and SAVE_PER_LAYER_ATTENTION and (
                            LAYER_ANALYSIS_STEP_INDICES is None
                            or step_index in LAYER_ANALYSIS_STEP_INDICES
                        ):
                            for layer_index, layer_map in enumerate(layer_maps):
                                layer_measurement = compute_anchor_measurements(
                                    layer_map.values, payload['foreground_masks'][region_index],
                                    topk_percent=TOPK_ATTENTION_PERCENT,
                                )
                                layer_measurement.update({
                                    'experiment_id': experiment_id, 'experiment_label': experiment['label'],
                                    'anchor_mode': experiment['anchor_mode'], 'weight_policy': experiment['weight_policy'],
                                    'sample_index': global_index, 'sample_id': payload['sample_id'],
                                    'image_id': payload['image_id'], 'region_index': region_index,
                                    'annotation_id': int(ann_id), 'category': category, 'prompt': prompt,
                                    'step_index': step_index, 'timestep': timestep,
                                    'layer_index': layer_index, 'layer_name': layer_map.layer_name,
                                    'native_spatial_size': layer_map.spatial_size,
                                })
                                layer_metrics_rows.append(layer_measurement)
                        if save_artifacts and step_index in ARTIFACT_STEP_INDICES:
                            # Keep extracted Windows paths short; sample_id remains in CSV/JSON metadata.
                            sample_key = f'{global_index:04d}'
                            heatmap_path = ATTENTION_DIR / experiment_id / sample_key / 'aggregate' / f'r{region_index:02d}_s{step_index:02d}_t{timestep}_mean_layers_heatmap.png'
                            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
                            plt.imsave(heatmap_path, heatmap.numpy(), cmap='magma', vmin=0, vmax=1)
                            if SAVE_PER_LAYER_ATTENTION and (
                                LAYER_ANALYSIS_STEP_INDICES is None
                                or step_index in LAYER_ANALYSIS_STEP_INDICES
                            ):
                                layer_dir = ATTENTION_DIR / experiment_id / sample_key / 'per_layer'
                                layer_dir.mkdir(parents=True, exist_ok=True)
                                for layer_index, layer_map in enumerate(layer_maps):
                                    # Full layer_name is retained in per-layer metrics; use a short Lxx filename.
                                    layer_path = layer_dir / f'r{region_index:02d}_s{step_index:02d}_t{timestep}_L{layer_index:02d}.png'
                                    plt.imsave(layer_path, layer_map.values.numpy(), cmap='magma', vmin=0, vmax=1)
                            effective_weight = effective_masks[region_index] if effective_masks is not None else payload['foreground_masks'][region_index]
                            weight_path = WEIGHT_DIR / experiment_id / sample_key / f'r{region_index:02d}_s{step_index:02d}_t{timestep}_weight.png'
                            weight_path.parent.mkdir(parents=True, exist_ok=True)
                            plt.imsave(weight_path, effective_weight.squeeze().numpy(), cmap='viridis', vmin=0, vmax=1)
                            if SAVE_WEIGHTED_FIGURES:
                                figure_path = VISUALIZATION_DIR / experiment_id / sample_key / f'r{region_index:02d}_s{step_index:02d}_t{timestep}_weighted_mask.png'
                                save_weighted_figure(original, captured_steps[step_index], payload['foreground_masks'][region_index], heatmap, effective_weight, measurement, runtime_record, weight_record, f'{experiment_id} | {category}', figure_path)
                            if SAVE_NUMERIC_ARRAYS:
                                arrays[f'r{region_index:02d}_s{step_index:02d}_t{timestep}_mean_layers_attention'] = heatmap.numpy().astype(np.float16)
                                for layer_index, layer_map in enumerate(layer_maps):
                                    layer_id = layer_map.layer_name.replace('.', '_').replace('/', '_')
                                    arrays[f'r{region_index:02d}_s{step_index:02d}_t{timestep}_layer{layer_index:02d}_{layer_id}_attention'] = layer_map.values.numpy().astype(np.float16)
                                arrays[f'r{region_index:02d}_s{step_index:02d}_t{timestep}_weight'] = effective_weight.squeeze().numpy().astype(np.float16)
                if SAVE_NUMERIC_ARRAYS and save_artifacts and arrays:
                    numeric_path = NUMERIC_DIR / experiment_id / f'{global_index:04d}_{payload["sample_id"]}_maps.npz'
                    numeric_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(numeric_path, **arrays)
                generation_rows.append({'experiment_id': experiment_id, 'experiment_label': experiment['label'], 'sample_index': global_index, 'sample_id': payload['sample_id'], 'image_id': payload['image_id'], 'seed': seed, 'elapsed_sec': elapsed, 'generated_path': str(generated_path), 'timesteps': actual_timesteps, 'weight_records': [record.__dict__ for record in runtime.weight_records]})
                if SHOW_NOTEBOOK_PREVIEWS and save_artifacts and global_index < MAX_DISPLAY_SAMPLES and experiment_id in DISPLAY_EXPERIMENT_IDS:
                    display(Markdown(f'### `{experiment_id}` | sample `{global_index}` | {elapsed:.2f}s'))
                    display(generated.resize((320, 320)))
            # Release large CPU/GPU objects only after both capture contexts are closed.
            capture.maps.clear()
            latent_capture.records.clear()
            runtime.weight_masks.clear()
            del aggregated, resized_layer_maps, generated, captured_steps, arrays, runtime_records, runtime, capture, latent_capture
            gc.collect()
            torch.cuda.empty_cache()
        del payload, original, token_indices
        gc.collect()
        global_index += 1
        if global_index >= RUN_SAMPLES:
            break
    # Drop the previous dataloader batch before requesting the next one.
    del batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if global_index >= RUN_SAMPLES:
        break

print('[OK] HoĂ n táº¥t', global_index, 'sample vá»›i', len(EXPERIMENT_IDS), 'ID.')

# --- Source notebook cell 11 ---
# 10. LÆ°u CSV/JSONL, generation summary vĂ  cáº¥u hĂ¬nh tĂ¡i láº­p.
metrics_df = pd.DataFrame(metrics_rows)
layer_metrics_df = pd.DataFrame(layer_metrics_rows)
generation_df = pd.DataFrame(generation_rows)
assert generation_df.shape[0] == RUN_SAMPLES * len(EXPERIMENT_IDS)
assert set(generation_df['experiment_id']) == set(EXPERIMENT_IDS)
assert metrics_df['experiment_id'].nunique() == len(EXPERIMENT_IDS)
METRICS_CSV = RUN_ROOT / f'weighted_mask_metrics_{RUN_PROFILE}.csv'
METRICS_JSONL = RUN_ROOT / f'weighted_mask_metrics_{RUN_PROFILE}.jsonl'
LAYER_METRICS_CSV = RUN_ROOT / f'weighted_mask_per_layer_metrics_{RUN_PROFILE}.csv'
LAYER_METRICS_JSONL = RUN_ROOT / f'weighted_mask_per_layer_metrics_{RUN_PROFILE}.jsonl'
GENERATION_JSON = RUN_ROOT / 'generation_summary.json'
CONFIG_JSON = RUN_ROOT / 'run_config.json'
metrics_df.to_csv(METRICS_CSV, index=False, encoding='utf-8-sig')
append_jsonl(METRICS_JSONL, metrics_rows)
if not layer_metrics_df.empty:
    layer_metrics_df.to_csv(LAYER_METRICS_CSV, index=False, encoding='utf-8-sig')
    append_jsonl(LAYER_METRICS_JSONL, layer_metrics_rows)
GENERATION_JSON.write_text(json.dumps(generation_rows, ensure_ascii=False, indent=2), encoding='utf-8')
CONFIG_JSON.write_text(json.dumps({
    'experiment': 'semantic_anchor_weighted_mask_ablation', 'profile': RUN_PROFILE,
    'manifest_samples': EXPECTED_SAMPLES, 'run_samples': RUN_SAMPLES,
    'model': MODEL_ID, 'sampler': 'LCM', 'resolution': list(TARGET_SIZE), 'seed_rule': 'BASE_SEED + sample_index',
    'bootstrap_steps': BOOTSTRAP_STEPS, 'mask_std': MASK_STD, 'mask_strength': MASK_STRENGTH,
    'spatial_sigma_latent': SPATIAL_SIGMA_LATENT, 'semantic_sigma_scale': SEMANTIC_SIGMA_SCALE,
    'topk_attention_percent': TOPK_ATTENTION_PERCENT,
    'per_layer_attention': {'enabled': SAVE_PER_LAYER_ATTENTION, 'max_layers': LAYER_ANALYSIS_MAX_LAYERS, 'step_indices': LAYER_ANALYSIS_STEP_INDICES, 'layer_metrics_csv': str(LAYER_METRICS_CSV)},
    'experiments': {key: EXPERIMENTS[key] for key in EXPERIMENT_IDS},
    'baseline_pipeline_sha256': hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
    'weighted_runtime_sha256': hashlib.sha256((OURS_SRC / 'experiments/semantic_anchor.py').read_bytes()).hexdigest(),
}, ensure_ascii=False, indent=2), encoding='utf-8')
print('[OK] Metrics:', METRICS_CSV)
print('[OK] Per-layer attention metrics:', LAYER_METRICS_CSV, '| rows:', len(layer_metrics_df))

# --- Source notebook cell 12 ---
# 11. Báº£ng metric vĂ  nháº­n Ä‘á»‹nh tá»± Ä‘á»™ng cho tá»«ng ID, chá»‰ xĂ©t step sau bootstrap.
post_df = metrics_df[metrics_df['step_index'] >= BOOTSTRAP_STEPS].copy()
summary = post_df.groupby(['experiment_id', 'experiment_label', 'anchor_mode', 'weight_policy'], as_index=False).agg(
    measurements=('anchor_attention', 'size'), regions=('annotation_id', 'nunique'),
    attention_at_anchor=('anchor_attention', 'mean'), peak_inside_mask_percent=('global_peak_inside_mask', lambda v: 100.0 * float(v.mean())),
    reference_to_raw_anchor_px=('runtime_reference_to_anchor_px', 'mean'), overlap_ratio=('overlap_ratio', 'mean'),
    overlap_pixels=('overlap_pixel_count', 'mean'), generation_sec=('sample_index', 'size'),
)
timing = generation_df.groupby('experiment_id', as_index=False).agg(mean_generation_sec=('elapsed_sec', 'mean'), total_generation_sec=('elapsed_sec', 'sum'))
summary = summary.merge(timing, on='experiment_id', how='left')
step_summary = post_df.groupby(['experiment_id', 'step_index', 'timestep', 'weight_policy'], as_index=False).agg(
    measurements=('anchor_attention', 'size'), attention_at_anchor=('anchor_attention', 'mean'),
    overlap_ratio=('overlap_ratio', 'mean'), overlap_pixels=('overlap_pixel_count', 'mean'),
    reference_to_raw_anchor_px=('runtime_reference_to_anchor_px', 'mean'),
)
SUMMARY_CSV, STEP_CSV = RUN_ROOT / 'weighted_mask_summary.csv', RUN_ROOT / 'weighted_mask_by_step.csv'
summary.to_csv(SUMMARY_CSV, index=False, encoding='utf-8-sig'); step_summary.to_csv(STEP_CSV, index=False, encoding='utf-8-sig')
display(Markdown('## Weighted Mask: so sĂ¡nh sau bootstrap'))
display(summary.style.format({'attention_at_anchor':'{:.4f}', 'peak_inside_mask_percent':'{:.2f}', 'reference_to_raw_anchor_px':'{:.2f}', 'overlap_ratio':'{:.5f}', 'overlap_pixels':'{:.1f}', 'mean_generation_sec':'{:.3f}', 'total_generation_sec':'{:.2f}'}))
display(Markdown('## Theo tá»«ng timestep'))
display(step_summary.style.format({'attention_at_anchor':'{:.4f}', 'overlap_ratio':'{:.5f}', 'overlap_pixels':'{:.1f}', 'reference_to_raw_anchor_px':'{:.2f}'}))
overlap_mean = float(post_df['overlap_ratio'].mean()) if len(post_df) else 0.0
if overlap_mean == 0.0:
    verdict = 'KhĂ´ng cĂ³ overlap sau quantization; WM-01 Ä‘áº¿n WM-05 sáº½ gáº§n nhÆ° trĂ¹ng WM-00. TÄƒng MASK_STD trÆ°á»›c khi káº¿t luáº­n.'
else:
    verdict = 'ÄĂ£ cĂ³ vĂ¹ng overlap; so sĂ¡nh cháº¥t lÆ°á»£ng áº£nh cuá»‘i vĂ  metric FID/IS/CLIP á»Ÿ thĂ­ nghiá»‡m káº¿ tiáº¿p Ä‘á»ƒ káº¿t luáº­n hiá»‡u quáº£ cá»§a tá»«ng ID.'
display(Markdown(f'**Nháº­n Ä‘á»‹nh:** overlap trung bĂ¬nh sau bootstrap = `{overlap_mean:.6f}`. {verdict}'))

# --- Source notebook cell 13 ---
# 12. Kiá»ƒm tra export trÆ°á»›c khi nĂ©n.
expected_images = RUN_SAMPLES * len(EXPERIMENT_IDS)
assert len(generation_rows) == expected_images
assert set(metrics_df['step_index']) == set(range(len(actual_timesteps)))
artifact_counts = {
    'mask_overlays': len(list(OVERLAY_DIR.rglob('*.png'))),
    'attention_maps': len(list(ATTENTION_DIR.rglob('*.png'))),
    'per_layer_attention_maps': len(list(ATTENTION_DIR.rglob('*_layer*_heatmap.png'))),
    'effective_weight_maps': len(list(WEIGHT_DIR.rglob('*.png'))),
    'visualizations': len(list(VISUALIZATION_DIR.rglob('*.png'))),
    'intermediate_step_images': len(list(INTERMEDIATE_DIR.rglob('*.png'))),
    'attention_and_weight_arrays': len(list(NUMERIC_DIR.rglob('*.npz'))),
}
assert all(count > 0 for count in artifact_counts.values()), f'Artifact export thieu: {artifact_counts}'
check = pd.DataFrame([
    ('áº¢nh cuá»‘i', len(list(GENERATED_DIR.rglob('*.png'))), expected_images),
    ('DĂ²ng metric', len(metrics_df), 'foreground regions x 5 x sá»‘ ID'),
    ('CSV tá»•ng há»£p', SUMMARY_CSV.exists(), True),
    ('Cáº¥u hĂ¬nh tĂ¡i láº­p', CONFIG_JSON.exists(), True),
    *[(f'Artifact {name}', count, '> 0') for name, count in artifact_counts.items()],
], columns=['Háº¡ng má»¥c', 'Thá»±c táº¿', 'Ká»³ vá»ng'])
display(Markdown('## Kiá»ƒm tra export'))
display(check)

# --- Source notebook cell 14 ---
# 13. Đánh giá offline anchor của từng cross-attention layer; không sinh ảnh mới.
from collections import defaultdict
import re
import torch.nn.functional as F

LAYER_EVAL_ROOT = RUN_ROOT / 'per_layer_anchor_evaluation'
LAYER_EVAL_OVERLAY_DIR = LAYER_EVAL_ROOT / 'overlays'
LAYER_EVAL_SHEET_DIR = LAYER_EVAL_ROOT / 'contact_sheets'
LAYER_EVAL_CSV = RUN_ROOT / f'per_layer_anchor_evaluation_{RUN_PROFILE}.csv'
LAYER_EVAL_JSONL = RUN_ROOT / f'per_layer_anchor_evaluation_{RUN_PROFILE}.jsonl'
LAYER_RANKING_CSV = RUN_ROOT / f'per_layer_anchor_ranking_{RUN_PROFILE}.csv'
LAYER_EVAL_SAMPLE_COUNT = min(RUN_SAMPLES, MAX_ARTIFACT_SAMPLES)
for path in [LAYER_EVAL_ROOT, LAYER_EVAL_OVERLAY_DIR, LAYER_EVAL_SHEET_DIR]:
    path.mkdir(parents=True, exist_ok=True)

def _load_eval_payloads(sample_count):
    payloads, originals = {}, {}
    sample_index = 0
    for batch in loader:
        for local_index in range(len(batch['sample_ids'])):
            if sample_index >= sample_count:
                return payloads, originals
            payloads[sample_index] = make_payload(batch, local_index)
            originals[sample_index] = batch['images'][local_index].resize(TARGET_SIZE[::-1], Image.Resampling.BILINEAR)
            sample_index += 1
        if sample_index >= sample_count:
            break
    return payloads, originals

payloads_for_eval, originals_for_eval = _load_eval_payloads(LAYER_EVAL_SAMPLE_COUNT)
assert len(payloads_for_eval) == LAYER_EVAL_SAMPLE_COUNT, (len(payloads_for_eval), LAYER_EVAL_SAMPLE_COUNT)

layer_metric_lookup = {}
if not layer_metrics_df.empty:
    for item in layer_metrics_df.itertuples(index=False):
        lookup_key = (str(item.experiment_id), int(item.sample_index), int(item.region_index), int(item.step_index), int(item.timestep), int(item.layer_index))
        layer_metric_lookup[lookup_key] = (str(item.layer_name), int(item.native_spatial_size))

layer_key_pattern = re.compile(r'^r(?P<region>\d+)_s(?P<step>\d+)_t(?P<timestep>-?\d+)_layer(?P<layer>\d+)_')
evaluation_rows = []

def _global_topk_metrics(attention, mask, topk_percent):
    attention = attention.detach().float().cpu()
    mask = mask.detach().float().cpu().squeeze().bool()
    if attention.shape != mask.shape:
        mask = F.interpolate(mask[None, None].float(), size=attention.shape, mode='nearest')[0, 0].bool()
    flat = attention.flatten()
    topk_count = max(1, int(np.ceil(flat.numel() * topk_percent / 100.0)))
    topk_indices = torch.topk(flat, k=topk_count).indices
    topk_mask = torch.zeros_like(flat, dtype=torch.bool)
    topk_mask[topk_indices] = True
    topk_mask = topk_mask.reshape(attention.shape)
    intersection = (topk_mask & mask).sum().float()
    union = (topk_mask | mask).sum().float().clamp_min(1)
    total = attention.sum().clamp_min(1e-8)
    inside_mean = attention[mask].mean() if mask.any() else torch.tensor(0.0)
    outside_mean = attention[~mask].mean() if (~mask).any() else torch.tensor(0.0)
    return {
        'global_topk_iou': float(intersection / union),
        'global_topk_precision': float(intersection / topk_count),
        'global_topk_recall': float(intersection / mask.sum().float().clamp_min(1)),
        'attention_mass_inside': float(attention[mask].sum() / total),
        'attention_inside_mean': float(inside_mean),
        'attention_outside_mean': float(outside_mean),
        'attention_inside_outside_gap': float(inside_mean - outside_mean),
    }

def _draw_anchor_overlay(original, mask, attention, measurement, title, destination):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(original)
    ax.imshow(attention, cmap='magma', alpha=0.48, vmin=0, vmax=1)
    ax.contour(mask, levels=[0.5], colors='cyan', linewidths=0.8)
    ax.scatter(measurement['anchor_x'], measurement['anchor_y'], c='lime', marker='x', s=100, linewidths=2.5, label='masked argmax')
    ax.scatter(measurement['topk_center_x'], measurement['topk_center_y'], facecolors='none', edgecolors='white', marker='o', s=70, linewidths=1.5, label='top-k centroid')
    ax.scatter(measurement['topk_anchor_x'], measurement['topk_anchor_y'], c='orange', marker='D', s=48, label='top-k projected')
    ax.scatter(measurement['bbox_center_x'], measurement['bbox_center_y'], c='yellow', marker='+', s=90, linewidths=2, label='bbox center')
    ax.set_title(title, fontsize=9)
    ax.axis('off')
    ax.legend(loc='lower right', fontsize=6)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=130, bbox_inches='tight')
    plt.close(fig)

for experiment_id in EXPERIMENT_IDS:
    for sample_index in range(LAYER_EVAL_SAMPLE_COUNT):
        npz_candidates = sorted((NUMERIC_DIR / experiment_id).glob(f'{sample_index:04d}_*_maps.npz'))
        if not npz_candidates:
            print('[WARN] Missing layer NPZ:', experiment_id, sample_index)
            continue
        payload = payloads_for_eval[sample_index]
        original = originals_for_eval[sample_index]
        with np.load(npz_candidates[0], allow_pickle=False) as arrays:
            grouped_for_sheet = defaultdict(list)
            for array_key in arrays.files:
                match = layer_key_pattern.match(array_key)
                if match is None or not array_key.endswith('_attention'):
                    continue
                region_index = int(match.group('region'))
                step_index = int(match.group('step'))
                timestep = int(match.group('timestep'))
                layer_index = int(match.group('layer'))
                attention = torch.from_numpy(np.asarray(arrays[array_key])).float()
                mask = payload['foreground_masks'][region_index]
                measurement = compute_anchor_measurements(attention, mask, topk_percent=TOPK_ATTENTION_PERCENT)
                extra_metrics = _global_topk_metrics(attention, mask, TOPK_ATTENTION_PERCENT)
                lookup_key = (experiment_id, sample_index, region_index, step_index, timestep, layer_index)
                layer_name, native_size = layer_metric_lookup.get(lookup_key, (f'layer_{layer_index:02d}', 0))
                row = {**measurement, **extra_metrics, 'experiment_id': experiment_id, 'sample_index': sample_index, 'sample_id': payload['sample_id'], 'image_id': payload['image_id'], 'region_index': region_index, 'category': payload['category_names'][region_index], 'step_index': step_index, 'timestep': timestep, 'layer_index': layer_index, 'layer_name': layer_name, 'native_spatial_size': native_size, 'array_key': array_key}
                evaluation_rows.append(row)
                overlay_path = LAYER_EVAL_OVERLAY_DIR / experiment_id / f'{sample_index:04d}' / f'r{region_index:02d}_s{step_index:02d}_t{timestep}_L{layer_index:02d}.png'
                _draw_anchor_overlay(original, mask.squeeze().cpu().numpy(), attention.numpy(), measurement, f'{experiment_id} | sample {sample_index} | region {region_index} | step {step_index} | L{layer_index:02d} ({native_size}x{native_size})', overlay_path)
                grouped_for_sheet[(region_index, step_index, timestep)].append((layer_index, native_size, attention.numpy(), measurement))
            for (region_index, step_index, timestep), items in grouped_for_sheet.items():
                items.sort(key=lambda item: item[0])
                columns = 4
                rows = int(np.ceil(len(items) / columns))
                fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.5 * rows))
                axes = np.atleast_1d(axes).ravel()
                mask_np = payload['foreground_masks'][region_index].squeeze().cpu().numpy()
                for axis, (layer_index, native_size, attention_np, measurement) in zip(axes, items):
                    axis.imshow(original)
                    axis.imshow(attention_np, cmap='magma', alpha=0.48, vmin=0, vmax=1)
                    axis.contour(mask_np, levels=[0.5], colors='cyan', linewidths=0.6)
                    axis.scatter(measurement['anchor_x'], measurement['anchor_y'], c='lime', marker='x', s=65, linewidths=2)
                    axis.scatter(measurement['topk_anchor_x'], measurement['topk_anchor_y'], c='orange', marker='D', s=28)
                    axis.set_title(f'L{layer_index:02d} | {native_size}x{native_size}', fontsize=8)
                    axis.axis('off')
                for axis in axes[len(items):]:
                    axis.axis('off')
                sheet_path = LAYER_EVAL_SHEET_DIR / experiment_id / f'{sample_index:04d}' / f'r{region_index:02d}_s{step_index:02d}_t{timestep}_layers.png'
                sheet_path.parent.mkdir(parents=True, exist_ok=True)
                fig.suptitle(f'{experiment_id} | sample {sample_index} | region {region_index} | step {step_index} | từng layer', fontsize=12)
                fig.tight_layout()
                fig.savefig(sheet_path, dpi=120, bbox_inches='tight')
                plt.close(fig)

evaluation_df = pd.DataFrame(evaluation_rows)
assert not evaluation_df.empty, 'Không tìm thấy per-layer attention NPZ để đánh giá.'
evaluation_df['temporal_anchor_std_px'] = np.nan
evaluation_df['mean_adjacent_anchor_jump_px'] = np.nan
evaluation_df['max_adjacent_anchor_jump_px'] = np.nan
for _, group in evaluation_df.groupby(['experiment_id', 'sample_index', 'region_index', 'layer_index']):
    indices = group.sort_values('step_index').index
    points = evaluation_df.loc[indices, ['anchor_x', 'anchor_y']].to_numpy(dtype=float)
    center = points.mean(axis=0, keepdims=True)
    jumps = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.asarray([0.0])
    evaluation_df.loc[indices, 'temporal_anchor_std_px'] = float(np.linalg.norm(points - center, axis=1).mean())
    evaluation_df.loc[indices, 'mean_adjacent_anchor_jump_px'] = float(jumps.mean())
    evaluation_df.loc[indices, 'max_adjacent_anchor_jump_px'] = float(jumps.max())

ranking_df = evaluation_df.groupby(['experiment_id', 'layer_index', 'layer_name', 'native_spatial_size'], as_index=False).agg(
    measurements=('array_key', 'size'), global_topk_iou=('global_topk_iou', 'mean'),
    global_topk_precision=('global_topk_precision', 'mean'), global_topk_recall=('global_topk_recall', 'mean'),
    attention_mass_inside=('attention_mass_inside', 'mean'), attention_inside_outside_gap=('attention_inside_outside_gap', 'mean'),
    distance_to_bbox_center_norm=('distance_to_bbox_center_norm', 'mean'), temporal_anchor_std_px=('temporal_anchor_std_px', 'mean'),
    mean_adjacent_anchor_jump_px=('mean_adjacent_anchor_jump_px', 'mean'),
)

def _rank_component(series, higher_is_better=True):
    low, high = float(series.min()), float(series.max())
    if abs(high - low) < 1e-8:
        return pd.Series(0.5, index=series.index)
    scaled = (series - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled

ranking_df['score_topk_iou'] = _rank_component(ranking_df['global_topk_iou'])
ranking_df['score_inside_mass'] = _rank_component(ranking_df['attention_mass_inside'])
ranking_df['score_inside_gap'] = _rank_component(ranking_df['attention_inside_outside_gap'])
ranking_df['score_stability'] = _rank_component(ranking_df['temporal_anchor_std_px'], higher_is_better=False)
ranking_df['score_geometry'] = _rank_component(ranking_df['distance_to_bbox_center_norm'], higher_is_better=False)
ranking_df['layer_score'] = (0.30 * ranking_df['score_topk_iou'] + 0.25 * ranking_df['score_inside_mass'] + 0.20 * ranking_df['score_inside_gap'] + 0.15 * ranking_df['score_stability'] + 0.10 * ranking_df['score_geometry'])
ranking_df = ranking_df.sort_values(['experiment_id', 'layer_score'], ascending=[True, False]).reset_index(drop=True)
evaluation_df.to_csv(LAYER_EVAL_CSV, index=False, encoding='utf-8-sig')
append_jsonl(LAYER_EVAL_JSONL, evaluation_rows)
ranking_df.to_csv(LAYER_RANKING_CSV, index=False, encoding='utf-8-sig')
(RUN_ROOT / 'per_layer_anchor_evaluation_config.json').write_text(json.dumps({
    'sample_count': LAYER_EVAL_SAMPLE_COUNT, 'experiments': list(EXPERIMENT_IDS),
    'topk_attention_percent': TOPK_ATTENTION_PERCENT, 'generation_rerun': False,
    'score_weights': {'global_topk_iou': 0.30, 'attention_mass_inside': 0.25, 'attention_inside_outside_gap': 0.20, 'temporal_stability': 0.15, 'bbox_geometry': 0.10},
}, ensure_ascii=False, indent=2), encoding='utf-8')
display(Markdown('## Offline per-layer anchor evaluation'))
display(ranking_df.groupby('experiment_id', as_index=False).head(5))
print('[OK] Per-layer evaluation rows:', len(evaluation_df))
print('[OK] Ranking:', LAYER_RANKING_CSV)
