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

# Notebook: Ours/semantic_anchor_experiments/semantic_anchor_sd15_hypersd_smoke8_colab.ipynb
# Notebook SHA256: 1d67ab5f4dceb16da6ed0ffd99faa987cd1fd01efdf8d2223eb5090c4c4e9f2f

# --- Source notebook cell 3 ---
# 2. Cấu hình thí nghiệm. Mặc định chạy smoke8 để so sánh với LCM.
import json
RUN_PROFILE = os.environ.get('RUN_PROFILE', 'smoke8')       # Chọn 'smoke8' hoặc 'full1073'.
PROFILE_CONFIGS = {
    'smoke8': {
        'run_id': 'semantic_anchor_sem_sd15_hypersd_smoke8',
        'manifest': 'Ours/test_sets/manifests/smoke/coco_val2017_multidiffusion_coco_all_512x512_smoke_bs8.jsonl',
        'expected_samples': 8,
    },
    'full1073': {
        'run_id': 'semantic_anchor_sem_sd15_hypersd_full1073',
        'manifest': 'Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl',
        'expected_samples': 1073,
    },
}
assert RUN_PROFILE in PROFILE_CONFIGS, f'RUN_PROFILE không hợp lệ: {RUN_PROFILE}'
PROFILE = PROFILE_CONFIGS[RUN_PROFILE]
RUN_ID = PROFILE['run_id']
EXPECTED_SAMPLES = PROFILE['expected_samples']
MODEL_ID = 'runwayml/stable-diffusion-v1-5'
HYPER_SD_REPO_ID = 'ByteDance/Hyper-SD'
HYPER_SD_WEIGHT_NAME = 'Hyper-SD15-4steps-lora.safetensors'
HYPER_SD_NUM_INFERENCE_STEPS = 4
HYPER_SD_LORA_SCALE = 1.0
SEMANTICDRAW_GUIDANCE_SCALE = 1.0
TARGET_SIZE = (512, 512)
BASE_SEED = 2024
BATCH_SIZE = 8                 # Batch của dataloader; baseline vẫn sinh tuần tự từng ảnh.
BOOTSTRAP_STEPS = 1
HYBRID_BOOTSTRAP_POINT = 'bbox_center'  # Khớp centering của bootstrap baseline; chỉ dùng để phân tích offline.
MASK_STD = 0.0
MASK_STRENGTH = 1.0
PREPROCESS_MASK_COVER_ALPHA = 0.0
MASK_TYPE = 'discrete'
NEGATIVE_PROMPT = ''
MAX_DISPLAY_SAMPLES = 4       # Chỉ hiển thị trực tiếp bấy nhiêu sample đầu.
MAX_ARTIFACT_SAMPLES = 8      # Chỉ lưu heatmap/NPZ/ảnh latent từng step/visualization cho bấy nhiêu sample đầu.
AUTO_DOWNLOAD_ZIP = True      # True: tự mở hộp thoại tải ZIP sau khi nén.
RUN_PROCESSOR_PARITY_CHECK = True

COCO_ROOT = Path(os.environ.get('COCO_ROOT', str(WORK_DIR / 'COCO')))
RUN_MANIFEST = REPO_ROOT / PROFILE['manifest']
BASE_OUTPUT_DIR = Path(str(WORK_DIR / 'anchordraw_runs'))
RUN_ROOT = BASE_OUTPUT_DIR / RUN_ID
GENERATED_DIR = RUN_ROOT / 'generated_images'
OVERLAY_DIR = RUN_ROOT / 'mask_overlays'
ATTENTION_DIR = RUN_ROOT / 'attention_maps'
VISUALIZATION_DIR = RUN_ROOT / 'visualizations'
INTERMEDIATE_DIR = RUN_ROOT / 'intermediate_step_images'
NUMERIC_DIR = RUN_ROOT / 'attention_arrays'
MASK_CACHE_DIR = RUN_ROOT / 'mask_cache'
ZIP_PATH = BASE_OUTPUT_DIR / f'{RUN_ID}__export.zip'
for path in [RUN_ROOT, GENERATED_DIR, OVERLAY_DIR, ATTENTION_DIR, VISUALIZATION_DIR, INTERMEDIATE_DIR, NUMERIC_DIR, MASK_CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)
assert RUN_MANIFEST.exists(), f'Thiếu manifest: {RUN_MANIFEST}'
assert MAX_DISPLAY_SAMPLES <= MAX_ARTIFACT_SAMPLES, 'Sample hiển thị phải nằm trong sample có artifact.'
print('[OK] Profile:', RUN_PROFILE, '| expected samples:', EXPECTED_SAMPLES)
print('[OK] Output:', RUN_ROOT)
print('[OK] Manifest:', RUN_MANIFEST)
print('[OK] Hyper-SD:', f'{HYPER_SD_REPO_ID}/{HYPER_SD_WEIGHT_NAME}')
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 5 ---
# 4. Import dataloader, bộ thu attention và pipeline baseline nguyên bản.
import sys, importlib, importlib.util, time, csv, shutil, gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
# terminal display provided above

# Hyper-SD LoRA không cần torchao. Nếu package này đã bị import trước khi gỡ,
# PEFT có thể chọn nhầm torchao dispatcher và báo lỗi version không tương thích.
importlib.invalidate_caches()
if 'torchao' in sys.modules or importlib.util.find_spec('torchao') is not None:
    raise RuntimeError(
        'torchao vẫn còn trong runtime. Hãy chọn Runtime > Restart session, '
        'sau đó chạy lại notebook từ cell cài dependency.'
    )

OURS_SRC = REPO_ROOT / 'Ours/src'
BASELINE_SRC = REPO_ROOT / 'Baseline/semantic-draw-main/src'
sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from data.visualize import make_mask_overlay
from experiments.semantic_anchor import SemanticAnchorCapture, SemanticLatentStepCapture, aggregate_attention_maps, align_captured_attention_maps, _mask_bbox_center_shifts, compute_anchor_measurements, find_target_token_indices

sys.path.insert(0, str(BASELINE_SRC))
pipeline_path = BASELINE_SRC / 'model/pipeline_semantic_draw.py'
spec = importlib.util.spec_from_file_location('pipeline_semantic_draw_original', pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline
print('[OK] Baseline file:', pipeline_path)

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
# 6. Load baseline SD1.5, gỡ LCM mà constructor tự nạp, rồi gắn Hyper-SD 4-step.
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

def clear_existing_lora_adapters(pipe):
    # Constructor baseline tự nạp LCM LoRA; phải bỏ LCM trước khi gắn Hyper-SD.
    if hasattr(pipe, 'unfuse_lora'):
        try:
            pipe.unfuse_lora()
        except Exception as exc:
            print('[WARN] Bỏ qua unfuse_lora:', exc)
    if hasattr(pipe, 'unload_lora_weights'):
        pipe.unload_lora_weights()
        print('[OK] Đã gỡ LoRA do constructor baseline nạp.')
        return
    if hasattr(pipe, 'delete_adapters'):
        try:
            pipe.delete_adapters(['lcm'])
            print('[OK] Đã xóa adapter LCM.')
            return
        except Exception as exc:
            print('[WARN] Không xóa được adapter LCM bằng delete_adapters:', exc)
    raise RuntimeError('Không thể xác nhận đã gỡ LCM LoRA trước khi nạp Hyper-SD.')

def prepare_hyper_sd_ddim_schedule(self, t_index_list=None, num_inference_steps=None):
    # Giữ đúng chữ ký prepare_lcm_schedule của baseline. __call__ truyền cả
    # list(range(num_inference_steps)) và num_inference_steps vào method này.
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
    # Baseline cần dự đoán x0 để trộn các region rồi thêm nhiễu cho timestep kế tiếp.
    alpha = self.alpha_prod_t_sqrt[idx].to(dtype=latent.dtype, device=latent.device).clamp_min(1e-6)
    beta = self.beta_prod_t_sqrt[idx].to(dtype=latent.dtype, device=latent.device)
    return (latent - beta * noise_pred) / alpha

assert torch.cuda.is_available(), 'Hãy bật GPU trong Runtime > Change runtime type.'
from diffusers import DDIMScheduler
from huggingface_hub import hf_hub_download
import types

device = torch.device('cuda:0')
dtype = torch.float16
maybe_login_hf()
seed_everything(BASE_SEED)
smd = SemanticDrawPipeline(
    device=device, dtype=dtype, sd_version='1.5', hf_key=MODEL_ID, has_i2t=False,
    default_mask_std=MASK_STD, default_mask_strength=MASK_STRENGTH,
    default_preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA, mask_type=MASK_TYPE,
)
clear_existing_lora_adapters(smd.pipe)

hyper_sd_lora_path = hf_hub_download(repo_id=HYPER_SD_REPO_ID, filename=HYPER_SD_WEIGHT_NAME)
smd.pipe.load_lora_weights(hyper_sd_lora_path, adapter_name='hyper_sd')
if hasattr(smd.pipe, 'set_adapters'):
    try:
        smd.pipe.set_adapters(['hyper_sd'], adapter_weights=[HYPER_SD_LORA_SCALE])
    except TypeError:
        smd.pipe.set_adapters(['hyper_sd'])
if hasattr(smd.pipe, 'fuse_lora'):
    try:
        smd.pipe.fuse_lora(lora_scale=HYPER_SD_LORA_SCALE, safe_fusing=False)
    except TypeError:
        smd.pipe.fuse_lora(lora_scale=HYPER_SD_LORA_SCALE)

try:
    smd.pipe.scheduler = DDIMScheduler.from_pretrained(
        MODEL_ID, subfolder='scheduler', timestep_spacing='trailing'
    )
except Exception as exc:
    print('[WARN] Không tải được scheduler riêng, dùng config đang có:', exc)
    smd.pipe.scheduler = DDIMScheduler.from_config(
        smd.pipe.scheduler.config, timestep_spacing='trailing'
    )

smd.scheduler = smd.pipe.scheduler
smd.unet = smd.pipe.unet
smd.default_num_inference_steps = HYPER_SD_NUM_INFERENCE_STEPS
smd.default_guidance_scale = SEMANTICDRAW_GUIDANCE_SCALE
smd.prepare_hyper_sd_ddim_schedule = types.MethodType(prepare_hyper_sd_ddim_schedule, smd)
# __call__ baseline gọi tên method này; chỉ đổi implementation trên instance runtime.
smd.prepare_lcm_schedule = smd.prepare_hyper_sd_ddim_schedule
smd.scheduler_step = types.MethodType(hyper_sd_ddim_predict_x0_step, smd)
smd.prepare_hyper_sd_ddim_schedule(num_inference_steps=HYPER_SD_NUM_INFERENCE_STEPS)

try:
    actual_adapter = smd.pipe.get_active_adapters() if hasattr(smd.pipe, 'get_active_adapters') else ['hyper_sd']
except Exception:
    actual_adapter = ['hyper_sd (fused)']
assert type(smd.scheduler).__name__ == 'DDIMScheduler'
assert len(smd.timesteps) == HYPER_SD_NUM_INFERENCE_STEPS
print('[OK] GPU:', torch.cuda.get_device_name(0))
print('[OK] Accelerator:', f'{HYPER_SD_REPO_ID}/{HYPER_SD_WEIGHT_NAME}')
print('[OK] Active adapters:', actual_adapter)
print('[OK] Scheduler:', type(smd.scheduler).__name__, '| spacing:', getattr(smd.scheduler.config, 'timestep_spacing', None))
print('[OK] Guidance:', smd.default_guidance_scale)
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
# 8. Hàm chuẩn bị payload và lưu visualization.
def make_payload(batch, index):
    item = batch_item_to_semanticdraw_inputs(batch, index)
    metadata = item['metadata']
    foreground_masks = item['masks'].float().cpu()
    foreground_union = foreground_masks.sum(dim=0, keepdim=True).clamp(0, 1)
    background_mask = (1.0 - foreground_union).clamp(0, 1)
    all_masks = torch.cat([background_mask, foreground_masks], dim=0)
    all_prompts = [item['background_prompt'], *item['prompts']]
    all_negative_prompts = [NEGATIVE_PROMPT for _ in all_prompts]
    assert len(all_prompts) == len(all_masks), (len(all_prompts), len(all_masks))
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

def save_anchor_figure(original, step_image, mask, heatmap, measurement, title, destination):
    mask_np = mask.squeeze().cpu().numpy()
    heat_np = heatmap.cpu().numpy()
    fig, axes = plt.subplots(1, 5, figsize=(22.5, 4.5))
    axes[0].imshow(original); axes[0].set_title('Ảnh COCO gốc')
    axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1); axes[1].set_title('Object mask')
    axes[2].imshow(heat_np, cmap='magma', vmin=0, vmax=1); axes[2].set_title('Cross-attention heatmap')
    axes[3].imshow(original); axes[3].imshow(heat_np, cmap='magma', alpha=0.5, vmin=0, vmax=1)
    axes[3].scatter(measurement['anchor_x'], measurement['anchor_y'], c='lime', s=90, marker='x', linewidths=3, label='Anchor I')
    axes[3].scatter(measurement['centroid_x'], measurement['centroid_y'], c='cyan', s=45, marker='o', label='Centroid')
    axes[3].scatter(measurement['bbox_center_x'], measurement['bbox_center_y'], c='yellow', s=45, marker='+', label='BBox center')
    axes[3].set_title('Heatmap + anchor'); axes[3].legend(loc='lower right', fontsize=7)
    axes[4].imshow(step_image); axes[4].set_title('Ảnh trung gian sau step')
    axes[4].text(0.02, 0.98, f"step={measurement['step_index']}, t={measurement['timestep']}", transform=axes[4].transAxes, va='top', color='white', fontsize=8, bbox={'facecolor': 'black', 'alpha': 0.6, 'pad': 2})
    for axis in axes: axis.axis('off')
    fig.suptitle(title); plt.tight_layout(); fig.savefig(destination, dpi=150, bbox_inches='tight'); plt.close(fig)

def decode_captured_step(captured_step):
    latent = captured_step.latent.to(device=device, dtype=smd.dtype)
    decoded = smd.decode_latents(latent)[0].detach().float().cpu().clamp(0, 1)
    array = (decoded.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)

def append_jsonl(path, records):
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

print('[OK] Helper functions đã sẵn sàng.')

# --- Source notebook cell 10 ---
# 9. Sinh ảnh và tính anchor trên toàn bộ profile; artifact nặng chỉ lưu cho các sample đầu.
metrics_rows, debug_rows, generation_rows = [], [], []
global_index = 0
actual_timesteps = [int(t) for t in smd.timesteps.detach().cpu().tolist()]

with SemanticAnchorCapture(smd.unet) as capture, SemanticLatentStepCapture(smd) as latent_capture:
    for batch_index, batch in enumerate(loader):
        print(f'[BATCH] {batch_index + 1}/{len(loader)} - {len(batch["sample_ids"])} sample')
        for local_index in range(len(batch['sample_ids'])):
            payload = make_payload(batch, local_index)
            save_artifacts = global_index < MAX_ARTIFACT_SAMPLES
            sample_dir = ATTENTION_DIR / f'{global_index:04d}_{payload["sample_id"]}' if save_artifacts else None
            vis_dir = VISUALIZATION_DIR / f'{global_index:04d}_{payload["sample_id"]}' if save_artifacts else None
            if save_artifacts:
                sample_dir.mkdir(parents=True, exist_ok=True); vis_dir.mkdir(parents=True, exist_ok=True)
                original = batch['images'][local_index].resize(TARGET_SIZE[::-1], Image.Resampling.BILINEAR)
                overlay = make_mask_overlay(original, payload['foreground_masks'], payload['category_names'], alpha=0.45)
                overlay_path = OVERLAY_DIR / f'{global_index:04d}_{payload["sample_id"]}_overlay.png'
                overlay.save(overlay_path)
            else:
                original = overlay = overlay_path = None

            token_indices = [find_target_token_indices(smd.tokenizer, prompt, category) for prompt, category in zip(payload['foreground_prompts'], payload['category_names'])]
            token_debug = []
            for prompt, category, indices in zip(payload['foreground_prompts'], payload['category_names'], token_indices):
                ids = smd.tokenizer(prompt, add_special_tokens=True)['input_ids']
                token_debug.append({'prompt': prompt, 'target': category, 'indices': indices, 'tokens': smd.tokenizer.convert_ids_to_tokens(ids)})
            capture.configure(token_indices)
            latent_capture.configure(actual_timesteps, enabled=save_artifacts)
            seed = BASE_SEED + global_index; seed_everything(seed); torch.cuda.synchronize()
            tic = time.perf_counter()
            generated = smd(
                prompts=payload['prompts'],
                negative_prompts=payload['negative_prompts'],
                masks=payload['all_masks'].to(device=device, dtype=torch.float32),
                height=payload['height'], width=payload['width'],
                num_inference_steps=HYPER_SD_NUM_INFERENCE_STEPS,
                guidance_scale=SEMANTICDRAW_GUIDANCE_SCALE,
                bootstrap_steps=BOOTSTRAP_STEPS,
                mask_stds=MASK_STD, mask_strengths=MASK_STRENGTH,
                preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA, do_blend=False,
            )
            torch.cuda.synchronize(); elapsed = time.perf_counter() - tic
            generated_path = GENERATED_DIR / f'{global_index:04d}_{payload["sample_id"]}_generated.png'
            generated.save(generated_path)
            latent_capture.disable()
            captured_steps = sorted(latent_capture.records, key=lambda record: record.step_index)
            step_images, step_image_paths = {}, {}
            if save_artifacts:
                captured_indices = [record.step_index for record in captured_steps]
                expected_indices = list(range(len(actual_timesteps)))
                assert captured_indices == expected_indices, (captured_indices, expected_indices)
                intermediate_sample_dir = INTERMEDIATE_DIR / f'{global_index:04d}_{payload["sample_id"]}'
                intermediate_sample_dir.mkdir(parents=True, exist_ok=True)
                for captured_step in captured_steps:
                    step_image = decode_captured_step(captured_step)
                    step_image_path = intermediate_sample_dir / f'step{captured_step.step_index:02d}_t{captured_step.timestep}_generated.png'
                    step_image.save(step_image_path)
                    step_images[captured_step.step_index] = step_image
                    step_image_paths[captured_step.step_index] = step_image_path

            processed_masks, _, _ = smd.process_mask(
                payload['all_masks'].to(device=device, dtype=torch.float32), MASK_STRENGTH, MASK_STD,
                height=payload['height'], width=payload['width'], timesteps=smd.timesteps,
                preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
            )
            shifts = {t: [(0, 0)] * len(payload['foreground_prompts']) for t in actual_timesteps}
            for step_index, t in enumerate(actual_timesteps[:BOOTSTRAP_STEPS]):
                shifts[t] = _mask_bbox_center_shifts(processed_masks[1:, step_index])
            latent_size = (TARGET_SIZE[0] // smd.vae_scale_factor, TARGET_SIZE[1] // smd.vae_scale_factor)
            aligned_maps = align_captured_attention_maps(capture.maps, shifts, latent_size=latent_size, image_size=TARGET_SIZE)
            aggregated = aggregate_attention_maps(aligned_maps, TARGET_SIZE)
            numeric_maps = {} if save_artifacts else None
            sample_metric_start = len(metrics_rows)
            for step_index, timestep in enumerate(actual_timesteps):
                for region_index, (prompt, category, ann_id) in enumerate(zip(payload['foreground_prompts'], payload['category_names'], payload['annotation_ids'])):
                    key = (timestep, region_index)
                    if key not in aggregated:
                        continue
                    heatmap = aggregated[key]; mask = payload['foreground_masks'][region_index]
                    measurement = compute_anchor_measurements(heatmap, mask)
                    heatmap_path = figure_path = None
                    if save_artifacts:
                        heatmap_path = sample_dir / f'r{region_index:02d}_step{step_index:02d}_t{timestep}_heatmap.png'
                        plt.imsave(heatmap_path, heatmap.numpy(), cmap='magma', vmin=0, vmax=1)
                        figure_path = vis_dir / f'r{region_index:02d}_step{step_index:02d}_t{timestep}_anchor.png'
                        plot_measurement = {**measurement, 'step_index': step_index, 'timestep': timestep}
                        save_anchor_figure(original, step_images[step_index], mask, heatmap, plot_measurement, f'{category} | step={step_index}, t={timestep}', figure_path)
                        numeric_maps[f'r{region_index:02d}_step{step_index:02d}_t{timestep}'] = heatmap.numpy().astype(np.float16)
                    metrics_rows.append({
                        'sample_index': global_index, 'sample_id': payload['sample_id'], 'image_id': payload['image_id'],
                        'region_index': region_index, 'annotation_id': int(ann_id), 'category': category, 'prompt': prompt,
                        'step_index': step_index, 'timestep': timestep, 'num_captured_layers': len(capture.maps[key]),
                        'heatmap_path': str(heatmap_path) if heatmap_path else None,
                        'visualization_path': str(figure_path) if figure_path else None,
                        'intermediate_image_path': str(step_image_paths.get(step_index)) if step_index in step_image_paths else None, **measurement,
                    })

            npz_path = None
            if save_artifacts and numeric_maps:
                npz_path = NUMERIC_DIR / f'{global_index:04d}_{payload["sample_id"]}_attention_maps.npz'
                np.savez_compressed(npz_path, **numeric_maps)
            sample_rows = metrics_rows[sample_metric_start:]
            for region_index in range(len(payload['foreground_prompts'])):
                region_rows = [row for row in sample_rows if row['region_index'] == region_index]
                if not region_rows: continue
                xs = np.asarray([row['anchor_x'] for row in region_rows]); ys = np.asarray([row['anchor_y'] for row in region_rows])
                jumps = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2) if len(xs) > 1 else np.asarray([0.0])
                for row in region_rows:
                    row['temporal_anchor_std_px'] = float(np.sqrt(xs.var() + ys.var()))
                    row['temporal_mean_step_jump_px'] = float(jumps.mean())
                    row['temporal_max_step_jump_px'] = float(jumps.max())

            if save_artifacts:
                debug_rows.append({
                    'sample_index': global_index, 'sample_id': payload['sample_id'], 'image_id': payload['image_id'],
                    'background_prompt': payload['background_prompt'], 'token_mapping': token_debug,
                    'actual_timesteps': actual_timesteps,
                    'captured_keys': [{'timestep': k[0], 'region_index': k[1], 'layers': len(v), 'spatial_sizes': [m.spatial_size for m in v]} for k, v in capture.maps.items()],
                    'numeric_attention_path': str(npz_path) if npz_path else None,
                    'captured_intermediate_steps': [{'step_index': record.step_index, 'timestep': record.timestep} for record in captured_steps],
                })
            generation_rows.append({
                'index': global_index, 'sample_id': payload['sample_id'], 'image_id': payload['image_id'], 'file_name': payload['file_name'],
                'seed': seed, 'elapsed_sec': elapsed, 'generated_path': str(generated_path),
                'overlay_path': str(overlay_path) if overlay_path else None,
                'background_prompt': payload['background_prompt'], 'foreground_prompts': payload['foreground_prompts'],
                'annotation_ids': payload['annotation_ids'], 'actual_timesteps': actual_timesteps,
                'intermediate_image_paths': [str(step_image_paths[index]) for index in sorted(step_image_paths)],
            })
            if global_index < MAX_DISPLAY_SAMPLES and save_artifacts:
                display(Markdown(f'### {global_index}: `{payload["sample_id"]}` | {elapsed:.2f}s'))
                display(overlay.resize((384, 384)), generated.resize((384, 384)))
                first_vis = next(iter(sorted(vis_dir.glob('*.png'))), None)
                if first_vis: display(Image.open(first_vis))
            global_index += 1
            del aggregated, numeric_maps, generated, overlay, original, captured_steps, step_images, step_image_paths
            torch.cuda.empty_cache()

print('[OK] Đã xử lý', global_index, 'sample.')

# --- Source notebook cell 11 ---
# 10. Lưu CSV/JSONL, summary và run config.
OUTPUT_TAG = RUN_PROFILE
METRICS_CSV = RUN_ROOT / f'anchor_metrics_{OUTPUT_TAG}.csv'
METRICS_JSONL = RUN_ROOT / f'anchor_metrics_{OUTPUT_TAG}.jsonl'
DEBUG_JSONL = RUN_ROOT / f'anchor_debug_{OUTPUT_TAG}.jsonl'
GENERATION_JSON = RUN_ROOT / 'generation_summary.json'
SUMMARY_JSON = RUN_ROOT / 'summary.json'
RUN_CONFIG_JSON = RUN_ROOT / 'run_config.json'
metrics_df = pd.DataFrame(metrics_rows)
generation_df = pd.DataFrame(generation_rows)
if metrics_df.empty:
    raise RuntimeError('Không thu được phép đo anchor nào. Không thể tổng hợp kết quả.')
region_df = metrics_df.drop_duplicates(['sample_id', 'region_index']).copy()
peak_by_region = metrics_df.groupby(['sample_id', 'region_index'])['global_peak_inside_mask'].agg(['any', 'all'])
num_regions = int(region_df.shape[0])
expected_measurements = num_regions * len(actual_timesteps)
numeric_check_columns = [
    'anchor_attention', 'distance_to_centroid_px', 'distance_to_bbox_center_px',
    'distance_to_centroid_norm', 'distance_to_bbox_center_norm',
    'temporal_anchor_std_px', 'temporal_mean_step_jump_px', 'temporal_max_step_jump_px',
]
all_metrics_finite = bool(np.isfinite(metrics_df[numeric_check_columns].to_numpy(dtype=float)).all())
num_intermediate_images = int(sum(len(row['intermediate_image_paths']) for row in generation_rows))
expected_intermediate_images = min(len(generation_rows), MAX_ARTIFACT_SAMPLES) * len(actual_timesteps)
intermediate_capture_complete = num_intermediate_images == expected_intermediate_images
complete_run = bool(
    len(generation_rows) == EXPECTED_SAMPLES
    and len(metrics_df) == expected_measurements
    and all_metrics_finite
    and intermediate_capture_complete
)
metrics_df.to_csv(METRICS_CSV, index=False, encoding='utf-8-sig')
append_jsonl(METRICS_JSONL, metrics_rows); append_jsonl(DEBUG_JSONL, debug_rows)
GENERATION_JSON.write_text(json.dumps(generation_rows, ensure_ascii=False, indent=2), encoding='utf-8')
summary = {
    'run_id': RUN_ID, 'run_profile': RUN_PROFILE, 'expected_samples': EXPECTED_SAMPLES,
    'num_samples': len(generation_rows), 'num_regions': num_regions,
    'num_anchor_measurements': len(metrics_df), 'expected_anchor_measurements': expected_measurements,
    'actual_timesteps': actual_timesteps, 'all_metrics_finite': all_metrics_finite, 'complete_run': complete_run,
    'num_intermediate_images': num_intermediate_images, 'expected_intermediate_images': expected_intermediate_images,
    'intermediate_capture_complete': intermediate_capture_complete,
    'mean_anchor_attention': float(metrics_df['anchor_attention'].mean()),
    'global_peak_inside_mask_rate': float(metrics_df['global_peak_inside_mask'].mean()),
    'regions_with_peak_inside_at_least_once_rate': float(peak_by_region['any'].mean()),
    'regions_with_peak_inside_all_timesteps_rate': float(peak_by_region['all'].mean()),
    'mean_distance_to_centroid_px': float(metrics_df['distance_to_centroid_px'].mean()),
    'mean_distance_to_bbox_center_px': float(metrics_df['distance_to_bbox_center_px'].mean()),
    'mean_distance_to_centroid_norm': float(metrics_df['distance_to_centroid_norm'].mean()),
    'mean_distance_to_bbox_center_norm': float(metrics_df['distance_to_bbox_center_norm'].mean()),
    'mean_temporal_anchor_std_px': float(metrics_df.drop_duplicates(['sample_id', 'region_index'])['temporal_anchor_std_px'].mean()),
    'mean_temporal_step_jump_px': float(metrics_df.drop_duplicates(['sample_id', 'region_index'])['temporal_mean_step_jump_px'].mean()),
    'mean_temporal_max_step_jump_px': float(region_df['temporal_max_step_jump_px'].mean()),
    'mean_generation_time_sec': float(generation_df['elapsed_sec'].mean()),
    'total_generation_time_sec': float(generation_df['elapsed_sec'].sum()),
}
run_config = {
    'model_id': MODEL_ID,
    'accelerator_repo': HYPER_SD_REPO_ID, 'accelerator_weight': HYPER_SD_WEIGHT_NAME,
    'sampler': type(smd.scheduler).__name__, 'timestep_spacing': getattr(smd.scheduler.config, 'timestep_spacing', None),
    'num_inference_steps': HYPER_SD_NUM_INFERENCE_STEPS, 'actual_timesteps': actual_timesteps,
    'guidance_scale': SEMANTICDRAW_GUIDANCE_SCALE, 'lora_scale': HYPER_SD_LORA_SCALE,
    'run_profile': RUN_PROFILE, 'expected_samples': EXPECTED_SAMPLES,
    'manifest': str(RUN_MANIFEST), 'target_size': list(TARGET_SIZE), 'batch_size': BATCH_SIZE,
    'max_artifact_samples': MAX_ARTIFACT_SAMPLES, 'max_display_samples': MAX_DISPLAY_SAMPLES,
    'intermediate_capture_point': 'merged global canvas latent after scheduler step and before next-step noise',
    'bootstrap_steps': BOOTSTRAP_STEPS, 'mask_std': MASK_STD, 'mask_strength': MASK_STRENGTH,
    'semanticdraw_input_protocol': 'explicit_background_mask_and_caption_prepended_to_foreground_regions',
    'anchor_definition': 'argmax attention restricted to object mask',
    'processor_parity_max_abs_diff': parity_max, 'processor_parity_mean_abs_diff': parity_mean,
}
SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
RUN_CONFIG_JSON.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
display(pd.DataFrame([summary]).T.rename(columns={0: 'Giá trị'}))
display(metrics_df.head(10))

# --- Source notebook cell 12 ---
# 11. Phân tích từng timestep và mô phỏng lịch hybrid: bbox center khi bootstrap, attention anchor sau bootstrap.
STEP_METRICS_CSV = RUN_ROOT / 'anchor_metrics_by_step.csv'
HYBRID_SCHEDULE_CSV = RUN_ROOT / 'hybrid_anchor_schedule_metrics.csv'
POST_BOOTSTRAP_SUMMARY_CSV = RUN_ROOT / 'semantic_anchor_post_bootstrap_summary.csv'
schedule_df = metrics_df.sort_values(['sample_id', 'region_index', 'step_index']).copy()
assert HYBRID_BOOTSTRAP_POINT in {'bbox_center', 'centroid'}
bootstrap_x_column = 'bbox_center_x' if HYBRID_BOOTSTRAP_POINT == 'bbox_center' else 'centroid_x'
bootstrap_y_column = 'bbox_center_y' if HYBRID_BOOTSTRAP_POINT == 'bbox_center' else 'centroid_y'
is_bootstrap_step = schedule_df['step_index'] < BOOTSTRAP_STEPS
schedule_df['selection_source'] = np.where(is_bootstrap_step, HYBRID_BOOTSTRAP_POINT, 'attention_anchor')
schedule_df['selected_x'] = np.where(is_bootstrap_step, schedule_df[bootstrap_x_column], schedule_df['anchor_x'])
schedule_df['selected_y'] = np.where(is_bootstrap_step, schedule_df[bootstrap_y_column], schedule_df['anchor_y'])
groups = schedule_df.groupby(['sample_id', 'region_index'], sort=False)
raw_dx = groups['anchor_x'].diff(); raw_dy = groups['anchor_y'].diff()
hybrid_dx = groups['selected_x'].diff(); hybrid_dy = groups['selected_y'].diff()
schedule_df['raw_anchor_jump_from_previous_px'] = np.sqrt(raw_dx ** 2 + raw_dy ** 2)
schedule_df['hybrid_jump_from_previous_px'] = np.sqrt(hybrid_dx ** 2 + hybrid_dy ** 2)
schedule_df.to_csv(HYBRID_SCHEDULE_CSV, index=False, encoding='utf-8-sig')

step_metrics = schedule_df.groupby(['step_index', 'timestep'], as_index=False).agg(
    num_measurements=('anchor_attention', 'size'),
    mean_anchor_attention=('anchor_attention', 'mean'),
    global_peak_inside_mask_rate=('global_peak_inside_mask', 'mean'),
    mean_anchor_to_centroid_px=('distance_to_centroid_px', 'mean'),
    mean_anchor_to_bbox_center_px=('distance_to_bbox_center_px', 'mean'),
    mean_anchor_to_global_peak_px=('distance_to_global_peak_px', 'mean'),
    mean_raw_jump_from_previous_px=('raw_anchor_jump_from_previous_px', 'mean'),
    mean_hybrid_jump_from_previous_px=('hybrid_jump_from_previous_px', 'mean'),
)
step_metrics['phase'] = np.where(step_metrics['step_index'] < BOOTSTRAP_STEPS, 'bootstrap', 'post-bootstrap')
step_metrics['selection_source'] = np.where(step_metrics['step_index'] < BOOTSTRAP_STEPS, HYBRID_BOOTSTRAP_POINT, 'attention_anchor')
step_metrics['global_peak_inside_mask_percent'] = 100.0 * step_metrics.pop('global_peak_inside_mask_rate')
step_metrics.to_csv(STEP_METRICS_CSV, index=False, encoding='utf-8-sig')

hybrid_track = schedule_df.groupby(['sample_id', 'region_index']).agg(
    selected_x_var=('selected_x', lambda values: float(np.var(values.to_numpy(dtype=float)))),
    selected_y_var=('selected_y', lambda values: float(np.var(values.to_numpy(dtype=float)))),
)
hybrid_track['temporal_std_px'] = np.sqrt(hybrid_track['selected_x_var'] + hybrid_track['selected_y_var'])
summary['hybrid_bootstrap_point'] = HYBRID_BOOTSTRAP_POINT
summary['mean_hybrid_temporal_anchor_std_px'] = float(hybrid_track['temporal_std_px'].mean())
summary['mean_hybrid_step_jump_px'] = float(schedule_df['hybrid_jump_from_previous_px'].dropna().mean())

# Độ ổn định nội tại của Semantic Anchor chỉ được tính sau bootstrap.
# Bước chuyển bbox/centroid -> attention anchor được báo cáo riêng, không trộn vào step jump.
semantic_schedule_df = schedule_df[schedule_df['step_index'] >= BOOTSTRAP_STEPS].copy()
semantic_groups = semantic_schedule_df.groupby(['sample_id', 'region_index'], sort=False)
semantic_dx = semantic_groups['anchor_x'].diff()
semantic_dy = semantic_groups['anchor_y'].diff()
semantic_schedule_df['semantic_anchor_jump_from_previous_px'] = np.sqrt(semantic_dx ** 2 + semantic_dy ** 2)
semantic_track = semantic_groups.agg(
    anchor_x_var=('anchor_x', lambda values: float(np.var(values.to_numpy(dtype=float)))),
    anchor_y_var=('anchor_y', lambda values: float(np.var(values.to_numpy(dtype=float)))),
)
semantic_track['temporal_std_px'] = np.sqrt(semantic_track['anchor_x_var'] + semantic_track['anchor_y_var'])
transition_rows = schedule_df[schedule_df['step_index'] == BOOTSTRAP_STEPS]
summary['semantic_anchor_first_step_index'] = int(BOOTSTRAP_STEPS)
summary['num_semantic_anchor_timesteps'] = int(semantic_schedule_df['step_index'].nunique())
summary['post_bootstrap_mean_anchor_attention'] = float(semantic_schedule_df['anchor_attention'].mean())
summary['post_bootstrap_global_peak_inside_mask_rate'] = float(semantic_schedule_df['global_peak_inside_mask'].mean())
summary['post_bootstrap_mean_distance_to_centroid_px'] = float(semantic_schedule_df['distance_to_centroid_px'].mean())
summary['post_bootstrap_mean_distance_to_bbox_center_px'] = float(semantic_schedule_df['distance_to_bbox_center_px'].mean())
summary['post_bootstrap_mean_temporal_anchor_std_px'] = float(semantic_track['temporal_std_px'].mean())
summary['post_bootstrap_mean_step_jump_px'] = float(semantic_schedule_df['semantic_anchor_jump_from_previous_px'].dropna().mean())
summary['bootstrap_to_semantic_transition_jump_px'] = float(transition_rows['hybrid_jump_from_previous_px'].dropna().mean())
post_bootstrap_summary = pd.DataFrame([
    ('Timestep Semantic Anchor được đo', summary['num_semantic_anchor_timesteps'], 'timestep'),
    ('Attention tại anchor trung bình', summary['post_bootstrap_mean_anchor_attention'], 'normalized score'),
    ('Global peak nằm trong mask', 100.0 * summary['post_bootstrap_global_peak_inside_mask_rate'], '% measurement'),
    ('Khoảng cách anchor-centroid', summary['post_bootstrap_mean_distance_to_centroid_px'], 'pixel'),
    ('Khoảng cách anchor-bbox center', summary['post_bootstrap_mean_distance_to_bbox_center_px'], 'pixel'),
    ('Độ lệch Semantic Anchor qua timestep', summary['post_bootstrap_mean_temporal_anchor_std_px'], 'pixel'),
    ('Bước nhảy Semantic Anchor trung bình', summary['post_bootstrap_mean_step_jump_px'], 'pixel'),
    ('Bước chuyển baseline sang Semantic Anchor', summary['bootstrap_to_semantic_transition_jump_px'], 'pixel'),
], columns=['Metric từ step >= BOOTSTRAP_STEPS', 'Giá trị', 'Đơn vị'])
post_bootstrap_summary.to_csv(POST_BOOTSTRAP_SUMMARY_CSV, index=False, encoding='utf-8-sig')
run_config['hybrid_anchor_schedule'] = f'{HYBRID_BOOTSTRAP_POINT} for step_index < {BOOTSTRAP_STEPS}; masked attention argmax otherwise'
SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
RUN_CONFIG_JSON.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')

display(Markdown('## Metric theo từng timestep'))
display(step_metrics.style.format({
    'mean_anchor_attention': '{:.4f}',
    'global_peak_inside_mask_percent': '{:.2f}',
    'mean_anchor_to_centroid_px': '{:.2f}',
    'mean_anchor_to_bbox_center_px': '{:.2f}',
    'mean_anchor_to_global_peak_px': '{:.2f}',
    'mean_raw_jump_from_previous_px': '{:.2f}',
    'mean_hybrid_jump_from_previous_px': '{:.2f}',
}))
display(Markdown(f'## Metric Semantic Anchor từ step {BOOTSTRAP_STEPS} trở đi'))
display(post_bootstrap_summary.style.format({'Giá trị': lambda value: f'{value:.4f}' if isinstance(value, (float, np.floating)) else str(value)}))
post_bootstrap = step_metrics[step_metrics['phase'] == 'post-bootstrap']
if not post_bootstrap.empty:
    best_peak_row = post_bootstrap.loc[post_bootstrap['global_peak_inside_mask_percent'].idxmax()]
    display(Markdown(
        f"**Đọc nhanh:** sau bootstrap, step `{int(best_peak_row['step_index'])}` "
        f"(t={int(best_peak_row['timestep'])}) có tỷ lệ global peak trong mask cao nhất: "
        f"{best_peak_row['global_peak_inside_mask_percent']:.2f}%. "
        f"Riêng Semantic Anchor từ step {BOOTSTRAP_STEPS} có độ lệch trung bình "
        f"{summary['post_bootstrap_mean_temporal_anchor_std_px']:.2f} px và bước nhảy nội tại trung bình "
        f"{summary['post_bootstrap_mean_step_jump_px']:.2f} px. Bước chuyển từ {HYBRID_BOOTSTRAP_POINT} sang anchor "
        f"là {summary['bootstrap_to_semantic_transition_jump_px']:.2f} px và được báo cáo riêng. "
        f"Đây là phân tích offline, chưa phải kết quả chất lượng ảnh khi dùng lịch này để điều khiển generation."
    ))
print('[OK] Đã lưu:', STEP_METRICS_CSV, HYBRID_SCHEDULE_CSV, 'và', POST_BOOTSTRAP_SUMMARY_CSV)

# --- Source notebook cell 13 ---
# 12. Hiển thị bảng metric và đưa ra nhận định tự động, có kiểm soát.
METRIC_SUMMARY_CSV = RUN_ROOT / 'metric_summary.csv'
ASSESSMENT_MD = RUN_ROOT / 'metric_assessment.md'
metric_table = pd.DataFrame([
    ('Số sample hoàn tất', summary['num_samples'], 'sample'),
    ('Số object region', summary['num_regions'], 'region'),
    ('Số phép đo anchor', summary['num_anchor_measurements'], 'measurement'),
    ('Số timestep được đo', len(actual_timesteps), 'timestep'),
    ('Attention tại anchor trung bình', summary['mean_anchor_attention'], 'normalized score'),
    ('Global peak nằm trong mask', 100.0 * summary['global_peak_inside_mask_rate'], '% measurement'),
    ('Region có peak trong mask ít nhất một lần', 100.0 * summary['regions_with_peak_inside_at_least_once_rate'], '% region'),
    ('Region có peak trong mask ở mọi timestep', 100.0 * summary['regions_with_peak_inside_all_timesteps_rate'], '% region'),
    ('Khoảng cách anchor-centroid', summary['mean_distance_to_centroid_px'], 'pixel'),
    ('Khoảng cách anchor-bbox center', summary['mean_distance_to_bbox_center_px'], 'pixel'),
    ('Khoảng cách anchor-centroid chuẩn hóa', summary['mean_distance_to_centroid_norm'], 'image diagonal'),
    ('Độ lệch anchor qua timestep', summary['mean_temporal_anchor_std_px'], 'pixel'),
    ('Bước nhảy anchor trung bình', summary['mean_temporal_step_jump_px'], 'pixel'),
    ('Bước nhảy anchor cực đại trung bình', summary['mean_temporal_max_step_jump_px'], 'pixel'),
    (f'Độ lệch lịch {HYBRID_BOOTSTRAP_POINT} -> anchor', summary['mean_hybrid_temporal_anchor_std_px'], 'pixel'),
    (f'Bước nhảy lịch {HYBRID_BOOTSTRAP_POINT} -> anchor', summary['mean_hybrid_step_jump_px'], 'pixel'),
    (f'Độ lệch Semantic Anchor từ step {BOOTSTRAP_STEPS}', summary['post_bootstrap_mean_temporal_anchor_std_px'], 'pixel'),
    (f'Bước nhảy Semantic Anchor từ step {BOOTSTRAP_STEPS}', summary['post_bootstrap_mean_step_jump_px'], 'pixel'),
    (f'Bước chuyển {HYBRID_BOOTSTRAP_POINT} -> Semantic Anchor', summary['bootstrap_to_semantic_transition_jump_px'], 'pixel'),
    ('Thời gian sinh trung bình', summary['mean_generation_time_sec'], 'second/sample'),
    ('Tổng thời gian sinh', summary['total_generation_time_sec'], 'second'),
], columns=['Metric', 'Giá trị', 'Đơn vị'])
metric_table.to_csv(METRIC_SUMMARY_CSV, index=False, encoding='utf-8-sig')
display(Markdown('## Kết quả tổng hợp'))
display(metric_table.style.format({'Giá trị': lambda value: f'{value:.4f}' if isinstance(value, (float, np.floating)) else str(value)}))

stability_ratio = summary['post_bootstrap_mean_temporal_anchor_std_px'] / max(TARGET_SIZE)
if stability_ratio <= 0.05:
    stability_text = 'ổn định cao (độ lệch trung bình không quá 5% cạnh ảnh)'
elif stability_ratio <= 0.10:
    stability_text = 'ổn định trung bình (độ lệch trung bình từ 5% đến 10% cạnh ảnh)'
else:
    stability_text = 'chưa ổn định (độ lệch trung bình lớn hơn 10% cạnh ảnh)'

assessment = [
    '# Nhận định Semantic Anchor',
    '',
    f"- **Tính toàn vẹn:** {'ĐẠT' if summary['complete_run'] else 'CHƯA ĐẠT'}. "
    f"Đã xử lý {summary['num_samples']}/{summary['expected_samples']} sample và thu "
    f"{summary['num_anchor_measurements']}/{summary['expected_anchor_measurements']} phép đo; "
    f"giá trị hữu hạn: {summary['all_metrics_finite']}.",
    f"- **Độ bám ngữ nghĩa thô:** global cross-attention peak nằm trong object mask ở "
    f"{100.0 * summary['global_peak_inside_mask_rate']:.2f}% phép đo. "
    'Đây mới là chỉ báo attention tự nhiên có tập trung vào object hay không.',
    f"- **Độ ổn định theo timestep:** anchor được đánh giá là **{stability_text}**, "
    f"với độ lệch trung bình {summary['post_bootstrap_mean_temporal_anchor_std_px']:.2f} px, chỉ tính từ step {BOOTSTRAP_STEPS}.",
    f"- **Lịch hybrid được mô phỏng:** dùng {HYBRID_BOOTSTRAP_POINT} trong {BOOTSTRAP_STEPS} bước bootstrap, sau đó dùng attention anchor. "
    f"Độ lệch trung bình của lịch này là {summary['mean_hybrid_temporal_anchor_std_px']:.2f} px và bước nhảy trung bình là "
    f"{summary['mean_hybrid_step_jump_px']:.2f} px. Đây mới là phân tích offline, chưa chứng minh chất lượng generation tăng.",
    f"- **Tách outlier chuyển cơ chế:** bước chuyển từ {HYBRID_BOOTSTRAP_POINT} ở step {BOOTSTRAP_STEPS - 1} sang Semantic Anchor ở step {BOOTSTRAP_STEPS} "
    f"có độ dài trung bình {summary['bootstrap_to_semantic_transition_jump_px']:.2f} px. Giá trị này được báo cáo riêng và không nằm trong "
    f"bước nhảy nội tại Semantic Anchor {summary['post_bootstrap_mean_step_jump_px']:.2f} px.",
    f"- **So với tâm hình học:** khoảng cách trung bình tới centroid là "
    f"{summary['mean_distance_to_centroid_px']:.2f} px và tới bbox center là "
    f"{summary['mean_distance_to_bbox_center_px']:.2f} px. Khoảng cách khác 0 chỉ chứng minh "
    'anchor không đồng nhất với tâm hình học; chưa tự nó chứng minh anchor tốt hơn.',
    '- **Lưu ý phương pháp:** Anchor I luôn nằm trong mask do định nghĩa là argmax của attention sau khi giới hạn bởi mask. Vì vậy, không dùng tỷ lệ anchor-in-mask để tuyên bố độ chính xác.',
]
if RUN_PROFILE == 'smoke8':
    assessment.append('- **Phạm vi kết luận:** smoke8 chỉ xác nhận pipeline và định dạng output hoạt động; không đủ để kết luận robust.')
else:
    assessment.append('- **Phạm vi kết luận:** full1073 cung cấp bằng chứng thống kê cho tập đã lọc, nhưng vẫn cần đối chứng centroid/bbox/random và ablation để chứng minh lợi ích nhân quả của Semantic Anchor.')
ASSESSMENT_MD.write_text('\n'.join(assessment) + '\n', encoding='utf-8')
display(Markdown('\n'.join(assessment)))
print('[OK] Đã lưu:', METRIC_SUMMARY_CSV, 'và', ASSESSMENT_MD)
