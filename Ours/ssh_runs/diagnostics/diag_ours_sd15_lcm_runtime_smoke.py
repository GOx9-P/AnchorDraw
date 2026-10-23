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

# Notebook: Ours/semantic_anchor_experiments/semantic_anchor_sd15_lcm_anchor_runtime_smoke8_colab.ipynb
# Notebook SHA256: ab81d9470a2c78cb1c8e004a8149d79549d59b2e4b6bad8d21aa0e26c5ec9932

# --- Source notebook cell 3 ---
# 2. Cấu hình ablation runtime. Mặc định smoke8 để kiểm tra ba nhánh cùng seed.
import json
RUN_PROFILE = os.environ.get('RUN_PROFILE', 'smoke8')       # Chọn 'smoke8' hoặc 'full1073'.
PROFILE_CONFIGS = {
    'smoke8': {
        'run_id': 'semantic_anchor_runtime_sem_sd15_lcm_smoke8',
        'manifest': 'Ours/test_sets/manifests/smoke/coco_val2017_multidiffusion_coco_all_512x512_smoke_bs8.jsonl',
        'expected_samples': 8,
    },
    'full1073': {
        'run_id': 'semantic_anchor_runtime_sem_sd15_lcm_full1073',
        'manifest': 'Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl',
        'expected_samples': 1073,
    },
}
assert RUN_PROFILE in PROFILE_CONFIGS, f'RUN_PROFILE không hợp lệ: {RUN_PROFILE}'
PROFILE = PROFILE_CONFIGS[RUN_PROFILE]
RUN_ID = PROFILE['run_id']
EXPECTED_SAMPLES = PROFILE['expected_samples']

MODEL_ID = 'runwayml/stable-diffusion-v1-5'
TARGET_SIZE = (512, 512)
BASE_SEED = 2024
BATCH_SIZE = 8
BOOTSTRAP_STEPS = 1          # Bắt buộc: chỉ step 0 giữ bbox-centering baseline.
RUNTIME_MODES = ('baseline', 'bbox_control', 'semantic_anchor', 'semantic_topk_anchor')
TOPK_ATTENTION_PERCENT = 10.0  # Top k% attention trong foreground mask cho nhánh semantic_topk_anchor.
MASK_STD = 0.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = 'discrete'
NEGATIVE_PROMPT = ''
MAX_DISPLAY_SAMPLES = 4
MAX_ARTIFACT_SAMPLES = 8

# Phân tích chỉ-đọc theo từng cross-attention layer `attn2`.
# Các lựa chọn này KHÔNG thay đổi latent, scheduler hay logic baseline.
SAVE_PER_LAYER_ANALYSIS = True
MAX_LAYER_ANALYSIS_SAMPLES = 2  # Giới hạn artifact để Colab không đầy RAM/đĩa.
LAYER_ANALYSIS_STEP_INDICES = None  # None = mọi denoising step đã capture.
LAYER_ANALYSIS_MAX_LAYERS = None  # None = mọi layer attn2 có map.
DISPLAY_PER_LAYER_CONTACT_SHEETS = True
MAX_PER_LAYER_DISPLAY_SHEETS = 12  # None = hiển thị tất cả sheet đã export.

# Phân tích chỉ-đọc theo từng cross-attention layer `attn2`.
# Các lựa chọn này KHÔNG thay đổi latent, scheduler hay logic baseline.
# Phân tích từng lớp attn2 chỉ đọc: không làm thay đổi latent hay ảnh sinh.
AUTO_DOWNLOAD_ZIP = True
RUN_PROCESSOR_PARITY_CHECK = True
RUN_BASELINE_PARITY_CHECK = True  # So nhánh baseline runtime với pipeline baseline ở sample đầu.

COCO_ROOT = Path(os.environ.get('COCO_ROOT', str(WORK_DIR / 'COCO')))
RUN_MANIFEST = REPO_ROOT / PROFILE['manifest']
BASE_OUTPUT_DIR = Path(str(WORK_DIR / 'anchordraw_runs'))
RUN_ROOT = BASE_OUTPUT_DIR / RUN_ID
MASK_CACHE_DIR = RUN_ROOT / 'mask_cache'
GENERATED_DIR = RUN_ROOT / 'generated_images'
OVERLAY_DIR = RUN_ROOT / 'mask_overlays'
ATTENTION_DIR = RUN_ROOT / 'attention_maps'
VISUALIZATION_DIR = RUN_ROOT / 'visualizations'
INTERMEDIATE_DIR = RUN_ROOT / 'intermediate_step_images'
NUMERIC_DIR = RUN_ROOT / 'attention_arrays'
LAYER_CONTACT_SHEET_DIR = RUN_ROOT / 'per_layer_attn2_contact_sheets'
ZIP_PATH = BASE_OUTPUT_DIR / f'{RUN_ID}__export.zip'
for path in [RUN_ROOT, MASK_CACHE_DIR, GENERATED_DIR, OVERLAY_DIR, ATTENTION_DIR, VISUALIZATION_DIR, INTERMEDIATE_DIR, NUMERIC_DIR, LAYER_CONTACT_SHEET_DIR]:
    path.mkdir(parents=True, exist_ok=True)
for mode in RUNTIME_MODES:
    for parent in [GENERATED_DIR, ATTENTION_DIR, VISUALIZATION_DIR, INTERMEDIATE_DIR, NUMERIC_DIR, LAYER_CONTACT_SHEET_DIR]:
        (parent / mode).mkdir(parents=True, exist_ok=True)
assert RUN_MANIFEST.exists(), f'Thiếu manifest: {RUN_MANIFEST}'
assert MAX_DISPLAY_SAMPLES <= MAX_ARTIFACT_SAMPLES
assert BOOTSTRAP_STEPS == 1, 'Ablation này cố định bbox baseline tại step 0.'
assert 0.0 < TOPK_ATTENTION_PERCENT <= 100.0, 'TOPK_ATTENTION_PERCENT phải thuộc (0, 100].'
print('[OK] Profile:', RUN_PROFILE, '| expected samples:', EXPECTED_SAMPLES)
print('[OK] Modes:', RUNTIME_MODES)
print('[OK] Top-k attention percent:', TOPK_ATTENTION_PERCENT)
print('[OK] Output:', RUN_ROOT)
print('[OK] Per-layer attn2 analysis:', SAVE_PER_LAYER_ANALYSIS,
      '| samples:', MAX_LAYER_ANALYSIS_SAMPLES,
      '| display sheets:', MAX_PER_LAYER_DISPLAY_SHEETS)

assert MAX_LAYER_ANALYSIS_SAMPLES <= MAX_ARTIFACT_SAMPLES
print('[OK] Per-layer attn2 analysis:', SAVE_PER_LAYER_ANALYSIS,
      '| samples:', MAX_LAYER_ANALYSIS_SAMPLES,
      '| display sheets:', MAX_PER_LAYER_DISPLAY_SHEETS)
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 5 ---
# 4. Import dataloader, bộ thu attention, runtime ablation và pipeline baseline nguyên bản.
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
    aggregate_attention_maps, resize_layer_attention_maps, compute_anchor_measurements, find_target_token_indices,
)

sys.path.insert(0, str(BASELINE_SRC))
pipeline_path = BASELINE_SRC / 'model/pipeline_semantic_draw.py'
spec = importlib.util.spec_from_file_location('pipeline_semantic_draw_original', pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline
print('[OK] Baseline file:', pipeline_path)
print('[OK] Runtime class: SemanticAnchorRuntime')

# --- Source notebook cell 6 ---
# 5. Dataloader dùng manifest theo RUN_PROFILE.
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
assert len(loader.dataset) == EXPECTED_SAMPLES, f'Manifest phải có {EXPECTED_SAMPLES} record, hiện có {len(loader.dataset)}'
preview_batch = next(iter(loader))
print('[OK] Samples:', len(loader.dataset))
print('[OK] Batches:', len(loader))
print('[OK] Mask tensor:', tuple(preview_batch['masks'].shape))

# --- Source notebook cell 7 ---
# 6. Load SemanticDraw SD1.5 + LCM theo đúng constructor baseline.
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

assert torch.cuda.is_available(), 'Hãy bật GPU trong Runtime > Change runtime type.'
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
# 7. Kiểm tra processor thu attention có đầu ra gần tương đương processor hiện tại.
# Đây là phép kiểm tra cục bộ trên một lớp attn2, không chạy thêm một ảnh diffusion.
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
    assert parity_max < 0.02, 'Processor thu attention lệch quá lớn so với processor ban đầu.'
    del hidden, encoder, original_output, captured_output
    torch.cuda.empty_cache()
else:
    parity_max = None
    parity_mean = None
    print('[INFO] Bỏ qua processor parity check.')

# --- Source notebook cell 9 ---
# 8. Hàm chuẩn bị input và artifact cho ablation runtime.
def make_payload(batch, index):
    item = batch_item_to_semanticdraw_inputs(batch, index)
    metadata = item['metadata']
    foreground_masks = item['masks'].float().cpu()
    foreground_union = foreground_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    background_mask = (1.0 - foreground_union).clamp(0, 1)
    all_masks = torch.cat([background_mask, foreground_masks], dim=0)
    all_prompts = [item['background_prompt'], *item['prompts']]
    all_negative_prompts = [NEGATIVE_PROMPT for _ in all_prompts]
    assert len(all_prompts) == len(all_masks)
    return {
        'sample_id': metadata['sample_id'], 'image_id': metadata['image_id'],
        'file_name': metadata['file_name'], 'height': item['height'], 'width': item['width'],
        'background_prompt': item['background_prompt'],
        'prompts': all_prompts, 'negative_prompts': all_negative_prompts,
        'foreground_prompts': item['prompts'], 'foreground_masks': foreground_masks,
        'background_mask': background_mask, 'all_masks': all_masks,
        'category_names': metadata['category_names'], 'annotation_ids': metadata['annotation_ids'],
        'area_ratios': metadata['area_ratios'],
    }

def decode_captured_step(captured_step):
    latent = captured_step.latent.to(device=device, dtype=smd.dtype)
    decoded = smd.decode_latents(latent)[0].detach().float().cpu().clamp(0, 1)
    return Image.fromarray((decoded.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))

def save_runtime_figure(original, step_image, mask, heatmap, measurement, runtime_record, title, destination):
    mask_np = mask.squeeze().cpu().numpy()
    heat_np = heatmap.cpu().numpy()
    fig, axes = plt.subplots(1, 5, figsize=(22.5, 4.5))
    axes[0].imshow(original); axes[0].set_title('Ảnh COCO gốc')
    axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1); axes[1].set_title('Object mask')
    axes[2].imshow(heat_np, cmap='magma', vmin=0, vmax=1); axes[2].set_title('Cross-attention heatmap')
    axes[3].imshow(original); axes[3].imshow(heat_np, cmap='magma', alpha=0.5, vmin=0, vmax=1)
    axes[3].scatter(measurement['anchor_x'], measurement['anchor_y'], c='lime', s=90, marker='x', linewidths=3, label='Argmax anchor')
    axes[3].scatter(measurement['topk_center_x'], measurement['topk_center_y'], c='white', edgecolors='black', s=35, marker='o', label='Top-k centroid')
    axes[3].scatter(measurement['topk_anchor_x'], measurement['topk_anchor_y'], c='orange', s=45, marker='D', label=f"Top {measurement['topk_percent']:.0f}% projected anchor")
    axes[3].scatter(measurement['bbox_center_x'], measurement['bbox_center_y'], c='yellow', s=45, marker='+', label='BBox center')
    if runtime_record.points_xy:
        x, y = runtime_record.points_xy[measurement['region_index']]
        axes[3].scatter(x, y, c='cyan', s=45, marker='o', label='Runtime reference')
    axes[3].set_title(runtime_record.selection_source); axes[3].legend(loc='lower right', fontsize=7)
    axes[4].imshow(step_image); axes[4].set_title('Ảnh trung gian sau step')
    axes[4].text(0.02, 0.98, f"step={measurement['step_index']}, t={measurement['timestep']}", transform=axes[4].transAxes, va='top', color='white', fontsize=8, bbox={'facecolor': 'black', 'alpha': 0.6, 'pad': 2})
    for axis in axes: axis.axis('off')
    fig.suptitle(title); plt.tight_layout(); fig.savefig(destination, dpi=150, bbox_inches='tight'); plt.close(fig)

def append_jsonl(path, records):
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

print('[OK] Helper functions đã sẵn sàng.')



def _scatter_anchor_markers(ax, measurement):
    """Draw candidate points from one heatmap in image pixel coordinates."""
    anchor_x, anchor_y = measurement['anchor_xy']
    topk_x, topk_y = measurement['topk_centroid_xy']
    projected_x, projected_y = measurement['topk_projected_anchor_xy']
    bbox_x, bbox_y = measurement['bbox_center_xy']
    ax.scatter(anchor_x, anchor_y, marker='x', c='#75ff40', s=120, linewidths=3, label='Masked argmax')
    ax.scatter(topk_x, topk_y, marker='o', facecolors='white', edgecolors='black', s=76, linewidths=1.5, label='Top-k centroid')
    ax.scatter(projected_x, projected_y, marker='D', c='#f9ac25', s=70, label='Top-k projected anchor')
    ax.scatter(bbox_x, bbox_y, marker='+', c='#fff04b', s=105, linewidths=2.5, label='BBox center')
    if 'runtime_reference_x' in measurement and 'runtime_reference_y' in measurement:
        ref_x, ref_y = measurement['runtime_reference_x'], measurement['runtime_reference_y']
        ax.scatter(ref_x, ref_y, marker='o', c='#3ee7f2', s=70, label='Runtime reference')


def save_per_layer_contact_sheet(
    original, step_image, mask, aggregate_heatmap, aggregate_measurement,
    layer_maps, layer_measurements, title, destination,
):
    """Save one panel per captured U-Net ``attn2`` layer for a region/timestep.

    Every individual map is independently normalized. Brightness therefore
    compares spatial preference inside one layer only, not absolute attention
    strength across layers.
    """
    panels = 4 + len(layer_maps)
    columns = min(4, panels)
    rows = (panels + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 5.0 * rows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(original); axes[0].set_title('Ảnh COCO gốc')
    axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1); axes[1].set_title('Object mask')
    axes[2].imshow(aggregate_heatmap, cmap='magma', vmin=0, vmax=1)
    _scatter_anchor_markers(axes[2], aggregate_measurement)
    axes[2].set_title('Trung bình toàn bộ attn2')
    axes[3].imshow(step_image); axes[3].set_title('Ảnh trung gian sau step')

    for offset, layer_map in enumerate(layer_maps, start=4):
        ax = axes[offset]
        metric = layer_measurements[layer_map.layer_name]
        ax.imshow(layer_map.values, cmap='magma', vmin=0, vmax=1)
        _scatter_anchor_markers(ax, metric)
        ax.set_title(f"{layer_map.layer_name}\nlatent attention: {layer_map.spatial_size}x{layer_map.spatial_size}")

    for ax in axes:
        ax.axis('off')
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=min(5, len(labels)), fontsize=9)
    fig.suptitle(title + '\nMỗi map theo layer được chuẩn hóa độc lập; dùng để so vị trí anchor, không so độ sáng giữa layer.', fontsize=14)
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140, bbox_inches='tight')
    plt.close(fig)


def build_layer_stability_summary(layer_metrics):
    """Summarize spatial stability of the anchor from each U-Net cross-attention layer."""
    if layer_metrics.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ['branch', 'layer_name', 'native_spatial_size']
    for key, group in layer_metrics.groupby(group_columns, dropna=False):
        tracks = []
        for _, track in group[group['step_index'] >= BOOTSTRAP_STEPS].groupby(['sample_id', 'region_index']):
            track = track.sort_values('step_index')
            points = track[['anchor_x', 'anchor_y']].to_numpy(dtype=float)
            if len(points) < 2:
                continue
            center = points.mean(axis=0, keepdims=True)
            spread = np.linalg.norm(points - center, axis=1).mean()
            jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
            tracks.append((spread, jumps.mean(), jumps.max()))
        rows.append({
            'branch': key[0],
            'layer_name': key[1],
            'native_spatial_size': key[2],
            'measurement_count': len(group),
            'mean_anchor_attention': group['anchor_attention'].mean(),
            'global_peak_inside_mask_percent': 100.0 * group['global_peak_inside_mask'].mean(),
            'mean_anchor_to_bbox_px': group['distance_to_bbox_center_px'].mean(),
            'tracked_regions_after_bootstrap': len(tracks),
            'mean_temporal_anchor_std_px': float(np.mean([item[0] for item in tracks])) if tracks else np.nan,
            'mean_adjacent_anchor_jump_px': float(np.mean([item[1] for item in tracks])) if tracks else np.nan,
            'mean_max_adjacent_anchor_jump_px': float(np.mean([item[2] for item in tracks])) if tracks else np.nan,
        })
    return pd.DataFrame(rows).sort_values(['branch', 'native_spatial_size', 'layer_name']).reset_index(drop=True)

# --- Source notebook cell 10 ---
# 9. Sinh ba nhánh runtime với cùng input và cùng seed.
metrics_rows, layer_metrics_rows, generation_rows, debug_rows = [], [], [], []
expected_metric_rows = 0
global_index = 0
layer_contact_sheet_display_count = 0
actual_timesteps = [int(t) for t in smd.timesteps.detach().cpu().tolist()]

for batch_index, batch in enumerate(loader):
    print(f'[BATCH] {batch_index + 1}/{len(loader)} - {len(batch["sample_ids"])} sample')
    for local_index in range(len(batch['sample_ids'])):
        payload = make_payload(batch, local_index)
        expected_metric_rows += len(payload['foreground_prompts']) * len(actual_timesteps) * len(RUNTIME_MODES)
        save_artifacts = global_index < MAX_ARTIFACT_SAMPLES
        analyze_layers = (
            SAVE_PER_LAYER_ANALYSIS
            and save_artifacts
            and global_index < MAX_LAYER_ANALYSIS_SAMPLES
        )
        original = batch['images'][local_index].resize(TARGET_SIZE[::-1], Image.Resampling.BILINEAR) if save_artifacts else None
        overlay = make_mask_overlay(original, payload['foreground_masks'], payload['category_names'], alpha=0.45) if save_artifacts else None
        overlay_path = OVERLAY_DIR / f'{global_index:04d}_{payload["sample_id"]}_overlay.png' if save_artifacts else None
        if overlay_path: overlay.save(overlay_path)

        token_indices = [find_target_token_indices(smd.tokenizer, prompt, category) for prompt, category in zip(payload['foreground_prompts'], payload['category_names'])]
        seed = BASE_SEED + global_index

        # Parity bắt buộc một lần: mode baseline của runtime phải tái lập pipeline gốc.
        if RUN_BASELINE_PARITY_CHECK and global_index == 0:
            # Xóa cache để cả baseline và runtime cùng encode white latent sau cùng seed.
            if hasattr(smd, 'white'):
                delattr(smd, 'white')
            seed_everything(seed); torch.cuda.synchronize()
            expected_baseline = smd(
                prompts=payload['foreground_prompts'], negative_prompts=payload['negative_prompts'][1:],
                background_prompt=payload['background_prompt'], background_negative_prompt=payload['negative_prompts'][0],
                height=TARGET_SIZE[0], width=TARGET_SIZE[1], masks=payload['foreground_masks'].to(device=device, dtype=torch.float32),
                bootstrap_steps=BOOTSTRAP_STEPS, mask_stds=MASK_STD, mask_strengths=MASK_STRENGTH,
                preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA, do_blend=False,
            )
            torch.cuda.synchronize()
            if hasattr(smd, 'white'):
                delattr(smd, 'white')
            with SemanticAnchorCapture(smd.unet) as parity_capture:
                parity_capture.configure(token_indices)
                runtime_parity = SemanticAnchorRuntime(smd, parity_capture, image_size=TARGET_SIZE)
                seed_everything(seed); torch.cuda.synchronize()
                runtime_baseline, _ = runtime_parity.generate(
                    prompts=payload['prompts'], negative_prompts=payload['negative_prompts'],
                    masks=payload['all_masks'].to(device=device, dtype=torch.float32), foreground_masks=payload['foreground_masks'],
                    mode='baseline', bootstrap_steps=BOOTSTRAP_STEPS, topk_percent=TOPK_ATTENTION_PERCENT, mask_stds=MASK_STD,
                    mask_strengths=MASK_STRENGTH, preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
                )
            parity_delta = np.abs(np.asarray(expected_baseline, dtype=np.int16) - np.asarray(runtime_baseline, dtype=np.int16))
            parity_error = {'max_rgb_abs': float(parity_delta.max()), 'mean_rgb_abs': float(parity_delta.mean())}
            print('[PARITY] baseline pipeline vs runtime RGB error:', parity_error)
            assert parity_error['max_rgb_abs'] <= 2.0 and parity_error['mean_rgb_abs'] <= 0.25, f'Baseline parity failed: {parity_error}'
            del expected_baseline, runtime_baseline, runtime_parity
            torch.cuda.empty_cache()

        for mode in RUNTIME_MODES:
            with SemanticAnchorCapture(smd.unet) as capture, SemanticLatentStepCapture(smd) as latent_capture:
                runtime = SemanticAnchorRuntime(smd, capture, image_size=TARGET_SIZE)
                capture.configure(token_indices)
                latent_capture.configure(actual_timesteps, enabled=save_artifacts)
                seed_everything(seed); torch.cuda.synchronize()
                tic = time.perf_counter()
                generated, runtime_records = runtime.generate(
                    prompts=payload['prompts'], negative_prompts=payload['negative_prompts'],
                    masks=payload['all_masks'].to(device=device, dtype=torch.float32),
                    foreground_masks=payload['foreground_masks'], mode=mode,
                    bootstrap_steps=BOOTSTRAP_STEPS, topk_percent=TOPK_ATTENTION_PERCENT, mask_stds=MASK_STD,
                    mask_strengths=MASK_STRENGTH,
                    preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
                )
                torch.cuda.synchronize(); elapsed = time.perf_counter() - tic
                latent_capture.disable()
                generated_path = GENERATED_DIR / mode / f'{global_index:04d}_{payload["sample_id"]}_generated.png'
                generated.save(generated_path)

                captured_steps = sorted(latent_capture.records, key=lambda record: record.step_index)
                step_images, step_paths = {}, {}
                if save_artifacts:
                    assert [record.step_index for record in captured_steps] == list(range(len(actual_timesteps)))
                    step_dir = INTERMEDIATE_DIR / mode / f'{global_index:04d}_{payload["sample_id"]}'
                    step_dir.mkdir(parents=True, exist_ok=True)
                    for record in captured_steps:
                        image = decode_captured_step(record)
                        path = step_dir / f'step{record.step_index:02d}_t{record.timestep}_generated.png'
                        image.save(path); step_images[record.step_index] = image; step_paths[record.step_index] = path

                aligned_maps = runtime.aligned_attention_maps()
                aggregated = aggregate_attention_maps(aligned_maps, TARGET_SIZE)
                resized_layer_maps = (
                    resize_layer_attention_maps(aligned_maps, TARGET_SIZE)
                    if analyze_layers else {}
                )
                maps_to_save = {} if save_artifacts else None
                for step_index, timestep in enumerate(actual_timesteps):
                    runtime_record = runtime_records[step_index]
                    for region_index, (prompt, category, ann_id) in enumerate(zip(payload['foreground_prompts'], payload['category_names'], payload['annotation_ids'])):
                        key = (timestep, region_index)
                        assert key in aggregated, f'Missing map: {key}'
                        heatmap = aggregated[key]
                        measurement = compute_anchor_measurements(heatmap, payload['foreground_masks'][region_index], topk_percent=TOPK_ATTENTION_PERCENT)
                        assert measurement['topk_anchor_inside_mask'], 'Projected top-k anchor must remain inside its foreground mask.'
                        measurement.update({
                            'sample_index': global_index, 'sample_id': payload['sample_id'], 'image_id': payload['image_id'],
                            'branch': mode, 'region_index': region_index, 'annotation_id': int(ann_id),
                            'category': category, 'prompt': prompt, 'step_index': step_index, 'timestep': timestep,
                            'selection_source': runtime_record.selection_source,
                            'anchor_source_step_index': runtime_record.anchor_source_step_index,
                            'anchor_source_timestep': runtime_record.anchor_source_timestep,
                            'generated_path': str(generated_path),
                            'intermediate_image_path': str(step_paths[step_index]) if step_index in step_paths else None,
                        })
                        if runtime_record.points_xy:
                            point_x, point_y = runtime_record.points_xy[region_index]
                            measurement['runtime_reference_x'] = float(point_x); measurement['runtime_reference_y'] = float(point_y)
                        else:
                            measurement['runtime_reference_x'] = measurement['bbox_center_x']; measurement['runtime_reference_y'] = measurement['bbox_center_y']
                        selected_anchor_prefix = 'topk_anchor' if mode == 'semantic_topk_anchor' else 'anchor'
                        measurement['runtime_reference_to_anchor_px'] = float(((measurement['runtime_reference_x'] - measurement['anchor_x']) ** 2 + (measurement['runtime_reference_y'] - measurement['anchor_y']) ** 2) ** 0.5)
                        measurement['runtime_reference_to_selected_anchor_px'] = float(((measurement['runtime_reference_x'] - measurement[f'{selected_anchor_prefix}_x']) ** 2 + (measurement['runtime_reference_y'] - measurement[f'{selected_anchor_prefix}_y']) ** 2) ** 0.5)
                        metrics_rows.append(measurement)

                        # Phân tích theo layer chỉ đọc các map CPU đã capture; không đổi latent hay runtime.generate().
                        if analyze_layers and (LAYER_ANALYSIS_STEP_INDICES is None or step_index in LAYER_ANALYSIS_STEP_INDICES):
                            selected_layers = resized_layer_maps.get(key, [])
                            if LAYER_ANALYSIS_MAX_LAYERS is not None:
                                selected_layers = selected_layers[:LAYER_ANALYSIS_MAX_LAYERS]
                            per_layer_measurements = {}
                            for layer_map in selected_layers:
                                layer_measurement = compute_anchor_measurements(
                                    attention_map=layer_map.values,
                                    mask=payload['foreground_masks'][region_index],
                                    topk_percent=TOPK_ATTENTION_PERCENT,
                                )
                                layer_measurement.update({
                                    'branch': mode, 'sample_id': payload['sample_id'], 'image_id': payload['image_id'],
                                    'region_index': region_index, 'annotation_id': payload['annotation_ids'][region_index],
                                    'category_name': payload['category_names'][region_index],
                                    'foreground_prompt': payload['foreground_prompts'][region_index],
                                    'step_index': step_index, 'timestep': timestep,
                                    'layer_name': layer_map.layer_name, 'native_spatial_size': layer_map.spatial_size,
                                })
                                if runtime_record.points_xy:
                                    point_x, point_y = runtime_record.points_xy[region_index]
                                    layer_measurement['runtime_reference_x'] = float(point_x)
                                    layer_measurement['runtime_reference_y'] = float(point_y)
                                else:
                                    layer_measurement['runtime_reference_x'] = layer_measurement['bbox_center_x']
                                    layer_measurement['runtime_reference_y'] = layer_measurement['bbox_center_y']
                                layer_metrics_rows.append(layer_measurement)
                                per_layer_measurements[layer_map.layer_name] = layer_measurement

                            if selected_layers:
                                layer_dir = LAYER_CONTACT_SHEET_DIR / mode / f'{global_index:04d}_{payload["sample_id"]}'
                                layer_path = layer_dir / f'r{region_index:02d}_step{step_index:02d}_t{timestep}_attn2_layers.png'
                                save_per_layer_contact_sheet(
                                    original=original, step_image=step_images[step_index], mask=payload['foreground_masks'][region_index],
                                    aggregate_heatmap=heatmap, aggregate_measurement=measurement,
                                    layer_maps=selected_layers, layer_measurements=per_layer_measurements,
                                                    title=f'{mode} | {payload["category_names"][region_index]} | step={step_index}, t={timestep}',
                                    destination=layer_path,
                                )
                                for item in layer_metrics_rows[-len(selected_layers):]:
                                    item['contact_sheet_path'] = str(layer_path)

                                if DISPLAY_PER_LAYER_CONTACT_SHEETS and (
                                    MAX_PER_LAYER_DISPLAY_SHEETS is None
                                    or layer_contact_sheet_display_count < MAX_PER_LAYER_DISPLAY_SHEETS
                                ):
                                    with Image.open(layer_path) as contact_sheet:
                                        display(contact_sheet.copy())
                                    layer_contact_sheet_display_count += 1

                        if save_artifacts:
                            sample_dir = ATTENTION_DIR / mode / f'{global_index:04d}_{payload["sample_id"]}'
                            vis_dir = VISUALIZATION_DIR / mode / f'{global_index:04d}_{payload["sample_id"]}'
                            sample_dir.mkdir(parents=True, exist_ok=True); vis_dir.mkdir(parents=True, exist_ok=True)
                            heatmap_path = sample_dir / f'r{region_index:02d}_step{step_index:02d}_t{timestep}_heatmap.png'
                            plt.imsave(heatmap_path, heatmap.numpy(), cmap='magma', vmin=0, vmax=1)
                            figure_path = vis_dir / f'r{region_index:02d}_step{step_index:02d}_t{timestep}_runtime.png'
                            save_runtime_figure(original, step_images[step_index], payload['foreground_masks'][region_index], heatmap, measurement, runtime_record, f'{mode} | {category}', figure_path)
                            maps_to_save[f'r{region_index:02d}_step{step_index:02d}_t{timestep}'] = heatmap.numpy().astype(np.float16)

                if save_artifacts and maps_to_save:
                    np.savez_compressed(NUMERIC_DIR / mode / f'{global_index:04d}_{payload["sample_id"]}_attention_maps.npz', **maps_to_save)
                generation_rows.append({
                    'sample_index': global_index, 'sample_id': payload['sample_id'], 'image_id': payload['image_id'],
                    'branch': mode, 'seed': seed, 'elapsed_sec': elapsed, 'generated_path': str(generated_path),
                    'actual_timesteps': actual_timesteps,
                    'runtime_step_records': [record.__dict__ for record in runtime_records],
                    'intermediate_image_paths': [str(step_paths[i]) for i in sorted(step_paths)],
                })
                if save_artifacts and global_index < MAX_DISPLAY_SAMPLES:
                    display(Markdown(f'### `{mode}` | {global_index}: `{payload["sample_id"]}` | {elapsed:.2f}s'))
                    display(overlay.resize((320, 320)), generated.resize((320, 320)))
                del aggregated, generated, captured_steps, step_images, step_paths
                torch.cuda.empty_cache()
        global_index += 1

print('[OK] Đã xử lý', global_index, 'sample ở', len(RUNTIME_MODES), 'nhánh.')

# --- Source notebook cell 11 ---
# 10. Lưu kết quả: aggregate metrics, per-layer metrics, cấu hình và manifest của run.
METRICS_CSV = RUN_ROOT / f'anchor_runtime_metrics_{RUN_PROFILE}.csv'
METRICS_JSONL = RUN_ROOT / f'anchor_runtime_metrics_{RUN_PROFILE}.jsonl'
LAYER_METRICS_CSV = RUN_ROOT / f'per_layer_attn2_metrics_{RUN_PROFILE}.csv'
LAYER_METRICS_JSONL = RUN_ROOT / f'per_layer_attn2_metrics_{RUN_PROFILE}.jsonl'
LAYER_STABILITY_CSV = RUN_ROOT / f'per_layer_attn2_stability_{RUN_PROFILE}.csv'
GENERATION_JSON = RUN_ROOT / 'generation_summary.json'
RUN_CONFIG_JSON = RUN_ROOT / 'run_config.json'

metrics_df = pd.DataFrame(metrics_rows)
layer_metrics_df = pd.DataFrame(layer_metrics_rows)
layer_stability_df = build_layer_stability_summary(layer_metrics_df)
generation_df = pd.DataFrame(generation_rows)

assert len(generation_rows) == EXPECTED_SAMPLES, (len(generation_rows), EXPECTED_SAMPLES)

metrics_df.to_csv(METRICS_CSV, index=False, encoding='utf-8-sig')
append_jsonl(METRICS_JSONL, metrics_rows)

if not layer_metrics_df.empty:
    layer_metrics_df.to_csv(LAYER_METRICS_CSV, index=False, encoding='utf-8-sig')
    append_jsonl(LAYER_METRICS_JSONL, layer_metrics_rows)
    layer_stability_df.to_csv(LAYER_STABILITY_CSV, index=False, encoding='utf-8-sig')

with GENERATION_JSON.open('w', encoding='utf-8') as handle:
    json.dump(generation_rows, handle, ensure_ascii=False, indent=2)

run_config = {
    'run_profile': RUN_PROFILE,
    'manifest': str(RUN_MANIFEST),
    'expected_samples': EXPECTED_SAMPLES,
    'target_size': list(TARGET_SIZE),
    'batch_size': BATCH_SIZE,
    'bootstrap_steps': BOOTSTRAP_STEPS,
    'runtime_modes': RUNTIME_MODES,
    'topk_attention_percent': TOPK_ATTENTION_PERCENT,
    'pipeline_class': type(smd).__name__,
    'scheduler_class': type(smd.scheduler).__name__,
    'per_layer_attn2_analysis': {
        'enabled': SAVE_PER_LAYER_ANALYSIS,
        'max_samples': MAX_LAYER_ANALYSIS_SAMPLES,
        'step_indices': LAYER_ANALYSIS_STEP_INDICES,
        'max_layers': LAYER_ANALYSIS_MAX_LAYERS,
        'contact_sheets_dir': str(LAYER_CONTACT_SHEET_DIR),
    },
}
with RUN_CONFIG_JSON.open('w', encoding='utf-8') as handle:
    json.dump(run_config, handle, ensure_ascii=False, indent=2)

display(Markdown('## Đã lưu artifact'))
display(pd.DataFrame([
    {'artifact': 'Aggregate anchor metrics', 'path': str(METRICS_CSV), 'rows': len(metrics_df)},
    {'artifact': 'Per-layer attn2 metrics', 'path': str(LAYER_METRICS_CSV), 'rows': len(layer_metrics_df)},
    {'artifact': 'Per-layer stability summary', 'path': str(LAYER_STABILITY_CSV), 'rows': len(layer_stability_df)},
    {'artifact': 'Per-layer contact sheets', 'path': str(LAYER_CONTACT_SHEET_DIR), 'rows': ''},
]))

# --- Source notebook cell 12 ---
# 11. Metric và nhận định runtime theo nhánh, đặc biệt step 1-4.
post_df = metrics_df[metrics_df['step_index'] >= BOOTSTRAP_STEPS].copy()
branch_summary = post_df.groupby('branch', as_index=False).agg(
    measurements=('anchor_attention', 'size'),
    regions=('annotation_id', 'nunique'),
    mean_anchor_attention=('anchor_attention', 'mean'),
    global_peak_inside_mask_percent=('global_peak_inside_mask', lambda values: 100.0 * float(values.mean())),
    mean_anchor_to_bbox_px=('distance_to_bbox_center_px', 'mean'),
    mean_runtime_reference_to_anchor_px=('runtime_reference_to_anchor_px', 'mean'),
    mean_runtime_reference_to_selected_anchor_px=('runtime_reference_to_selected_anchor_px', 'mean'),
)
time_summary = generation_df.groupby('branch', as_index=False).agg(
    images=('sample_id', 'size'), mean_generation_sec=('elapsed_sec', 'mean'), total_generation_sec=('elapsed_sec', 'sum')
)
branch_summary = branch_summary.merge(time_summary, on='branch', how='left')
STEP_METRICS_CSV = RUN_ROOT / 'runtime_anchor_metrics_by_branch_and_step.csv'
BRANCH_SUMMARY_CSV = RUN_ROOT / 'runtime_anchor_branch_summary.csv'
step_summary = post_df.groupby(['branch', 'step_index', 'timestep', 'selection_source'], as_index=False).agg(
    measurements=('anchor_attention', 'size'),
    mean_anchor_attention=('anchor_attention', 'mean'),
    mean_runtime_reference_to_anchor_px=('runtime_reference_to_anchor_px', 'mean'),
    mean_runtime_reference_to_selected_anchor_px=('runtime_reference_to_selected_anchor_px', 'mean'),
    global_peak_inside_mask_percent=('global_peak_inside_mask', lambda values: 100.0 * float(values.mean())),
)
step_summary.to_csv(STEP_METRICS_CSV, index=False, encoding='utf-8-sig')
branch_summary.to_csv(BRANCH_SUMMARY_CSV, index=False, encoding='utf-8-sig')
display(Markdown('## So sánh runtime từ step 1 đến step 4'))
display(branch_summary.style.format({
    'mean_anchor_attention': '{:.4f}', 'global_peak_inside_mask_percent': '{:.2f}',
    'mean_anchor_to_bbox_px': '{:.2f}', 'mean_runtime_reference_to_anchor_px': '{:.2f}', 'mean_runtime_reference_to_selected_anchor_px': '{:.2f}',
    'mean_generation_sec': '{:.3f}', 'total_generation_sec': '{:.2f}',
}))
display(Markdown('## Theo từng timestep'))
display(step_summary.style.format({
    'mean_anchor_attention': '{:.4f}', 'mean_runtime_reference_to_anchor_px': '{:.2f}', 'mean_runtime_reference_to_selected_anchor_px': '{:.2f}',
    'global_peak_inside_mask_percent': '{:.2f}',
}))
display(Markdown(
    '- `baseline` không tái-centering sau step 0.\n'
    '- `bbox_control` tái-centering theo bbox ở step 1-4.\n'
    '- `semantic_anchor` dùng masked argmax của step trước, nên reference ở step 1 là attention tại step 0.\n'
    '- `semantic_topk_anchor` lấy centroid của top `TOPK_ATTENTION_PERCENT`% pixel attention cao nhất trong mask ở step trước, rồi chiếu về pixel top-k gần nhất để anchor luôn thuộc mask.\n'
    '- Các bảng này đo sự khác nhau của reference; chất lượng ảnh cuối cần đối chiếu bằng ảnh saved cùng seed, rồi mới mở rộng full1073 để tính FID/IS/CLIP.'
))



if not layer_stability_df.empty:
    display(Markdown('## So sánh từng lớp cross-attention `attn2`'))
    display(Markdown(
        'Mỗi hàng là một lớp U-Net được quan sát trên tối đa '
        f'`{MAX_LAYER_ANALYSIS_SAMPLES}` sample đầu. Ưu tiên lớp có '
        '`global_peak_inside_mask_percent` cao, khoảng cách/jump theo thời gian thấp, '
        'rồi mới kiểm tra trực quan bằng contact sheet. Đây là bằng chứng chọn layer; '
        'không phải metric chất lượng ảnh sinh.'
    ))
    display(layer_stability_df)
else:
    print('[INFO] Chưa có dữ liệu từng layer. Kiểm tra SAVE_PER_LAYER_ANALYSIS và MAX_LAYER_ANALYSIS_SAMPLES.')

# --- Source notebook cell 13 ---
# 12. Kiểm tra xuất file trước khi nén.
expected_images = EXPECTED_SAMPLES * len(RUNTIME_MODES)
assert len(generation_rows) == expected_images, (len(generation_rows), expected_images)
assert metrics_df['branch'].nunique() == len(RUNTIME_MODES)
assert set(metrics_df['step_index'].unique()) == set(range(len(actual_timesteps)))
check = pd.DataFrame([
    ('Ảnh cuối đã sinh', len(generation_rows), expected_images),
    ('Phép đo attention', len(metrics_df), '>= số region x 5 x 3 nhánh'),
    ('Ảnh trung gian được giữ', len(list(INTERMEDIATE_DIR.rglob('*.png'))), 'artifact samples x 5 x 3 nhánh'),
], columns=['Hạng mục', 'Thực tế', 'Kỳ vọng'])
display(Markdown('## Kiểm tra export'))
display(check)
print('[OK] Runtime ablation export hoàn tất:', RUN_ROOT)



layer_sheet_count = len(list(LAYER_CONTACT_SHEET_DIR.rglob('*.png')))
print('[CHECK] Per-layer attn2 contact sheets:', layer_sheet_count)
if SAVE_PER_LAYER_ANALYSIS:
    assert layer_sheet_count > 0, 'Đã bật phân tích từng layer nhưng chưa có contact sheet nào.'
