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

# Notebook: Ours/kaggle_semantic_anchor_experiments/semantic_anchor_sd15_lcm_w03_l04_distillationpp_lambda_sweep_kaggle.ipynb
# Notebook SHA256: 6b7cbdd2926a0b79a0063689375a4410590a868ef47c064ba15d4335e2edb846

# --- Source notebook cell 2 ---
# 2. Locate or clone the repository and freeze the paired-sweep protocol.
import os, json, time, gc, types, hashlib, zipfile, importlib.util
from pathlib import Path
from itertools import islice
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# terminal display provided above

RUN_ID = 'semantic_anchor_sd15_lcm_w03_l04_sigmad4_distillpp_lambda_validation_128'
RUN_MANIFEST = REPO_ROOT / 'Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl'
COCO_ROOT = Path(os.environ.get('COCO_ROOT', str(WORK_DIR / 'COCO')))
RUN_ROOT = WORK_DIR / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_ID = 'runwayml/stable-diffusion-v1-5'
TARGET_SIZE = (512, 512)
RUN_SAMPLES = 128
BATCH_SIZE = 1
BASE_SEED = 2024
BOOTSTRAP_STEPS = 1
MASK_STD, MASK_STRENGTH, PREPROCESS_MASK_COVER_ALPHA = 1.0, 1.0, 0.0
MASK_TYPE, NEGATIVE_PROMPT = 'discrete', ''
ANCHOR_MODE, WEIGHT_POLICY = 'semantic_topk_anchor', 'adaptive_bilateral'
TOPK_ATTENTION_PERCENT, SIGMA_D, SEMANTIC_SIGMA_SCALE = 10.0, 4.0, 1.0
FIXED_LAYER = {'layer_id': 'L04', 'layer_index': 4, 'native_spatial_size': 16, 'layer_name': 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor'}
LAMBDA_SWEEP = [0.0, 0.01, 0.02]
TEACHER_CFG_SCALE = 7.5
GUIDE_STEPS = 1
EXPECTED_TIMESTEPS = [999, 919, 759, 499, 259]
assert RUN_MANIFEST.exists()
print('[OK] Repo:', REPO_ROOT)
print('[OK] Sweep:', LAMBDA_SWEEP, '| paper LCM/LCM-LoRA value: 0.02')
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 3 ---
# 3. Import project code and prepare COCO annotations (original images are not needed).
OURS_SRC = REPO_ROOT / 'Ours/src'
BASELINE_SRC = REPO_ROOT / 'Baseline/semantic-draw-main/src'
# Put Ours ahead of Baseline: both expose a top-level module named `data`.
# Purging the cache also makes this cell safe to rerun after a failed import.
for module_name in list(sys.modules):
    if module_name == 'data' or module_name.startswith('data.') or module_name == 'experiments' or module_name.startswith('experiments.'):
        del sys.modules[module_name]
sys.path.insert(0, str(BASELINE_SRC)); sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from experiments.semantic_anchor import SemanticAnchorCapture, SemanticAnchorRuntime, compute_anchor_measurements, find_target_token_indices
from experiments.distillation_pp import DistillationPPRefiner
from diffusers import UNet2DConditionModel
pipeline_path = BASELINE_SRC / 'model/pipeline_semantic_draw.py'
spec = importlib.util.spec_from_file_location('pipeline_semantic_draw_distillpp', pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline

all_manifest_records = [json.loads(line) for line in RUN_MANIFEST.read_text(encoding='utf-8').splitlines() if line.strip()]
assert len(all_manifest_records) >= RUN_SAMPLES, (len(all_manifest_records), RUN_SAMPLES)
manifest_records = all_manifest_records[:RUN_SAMPLES]
annotations = [COCO_ROOT / 'annotations/instances_val2017.json', COCO_ROOT / 'annotations/captions_val2017.json']
def download(urls, destination):
    if destination.exists() and destination.stat().st_size: return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        result = subprocess.run(['wget', '-q', '-c', '--no-check-certificate', '-O', str(destination), url])
        if result.returncode == 0 and destination.exists() and destination.stat().st_size: return
    raise RuntimeError(f'Download failed: {destination}')
if not all(p.exists() for p in annotations):
    archive = COCO_ROOT / 'annotations_trainval2017.zip'
    download(['https://images.cocodataset.org/annotations/annotations_trainval2017.zip', 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'], archive)
    with zipfile.ZipFile(archive) as zf: zf.extractall(COCO_ROOT)
assert all(p.exists() for p in annotations)
print(f'[OK] Source imports and COCO assets ready for {RUN_SAMPLES}/{len(all_manifest_records)} manifest records.')

# --- Source notebook cell 4 ---
# 4. Load the unchanged student and a separate unfused SD1.5 teacher.
def maybe_login_hf():
    token = os.environ.get('HF_TOKEN')
    if token is None:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret('HF_TOKEN')
        except Exception: token = None
    if token:
        from huggingface_hub import login
        login(token=token)
def seed_everything(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def sync_cuda(): torch.cuda.synchronize()
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator.'
maybe_login_hf(); device, dtype = torch.device('cuda:0'), torch.float16
seed_everything(BASE_SEED)
smd = SemanticDrawPipeline(device=device, dtype=dtype, sd_version='1.5', hf_key=MODEL_ID, has_i2t=False, default_mask_std=MASK_STD, default_mask_strength=MASK_STRENGTH, default_preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA, mask_type=MASK_TYPE)
teacher_unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder='unet', torch_dtype=dtype).to(device).eval()
for p in teacher_unet.parameters(): p.requires_grad_(False)
actual_timesteps = [int(t) for t in smd.timesteps.detach().cpu().tolist()]
assert type(smd.scheduler).__name__ == 'LCMScheduler'
assert actual_timesteps == EXPECTED_TIMESTEPS, (actual_timesteps, EXPECTED_TIMESTEPS)
assert FIXED_LAYER['layer_name'] in set(smd.unet.attn_processors)
assert teacher_unet is not smd.unet and teacher_unet.config.in_channels == smd.unet.config.in_channels
print('[OK] Student=SD1.5+LCM-LoRA; teacher=separate base SD1.5; timesteps=', actual_timesteps)

# --- Source notebook cell 5 ---
# 5. Build the full-manifest loader and keep the established W03/L04 generation path.
config = COCORegionConfig(coco_root=COCO_ROOT, split='val2017', instances_json=annotations[0], captions_json=annotations[1], manifest_path=RUN_MANIFEST, profile='multidiffusion_coco_all', model_family='sd15', target_size=TARGET_SIZE, return_image=False, cache_resized_masks=False, cache_dir=None, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False, persistent_workers=False)
loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
assert len(loader.dataset) == len(all_manifest_records) and len(loader.dataset) >= RUN_SAMPLES
def make_payload(batch, index):
    item = batch_item_to_semanticdraw_inputs(batch, index); meta = item['metadata']
    fg = item['masks'].float().cpu(); bg = (1.0 - fg.sum(dim=0, keepdim=True).clamp(0, 1)).clamp(0, 1)
    prompts = [item['background_prompt'], *item['prompts']]
    return {'sample_id': meta['sample_id'], 'image_id': meta['image_id'], 'prompts': prompts, 'negative_prompts': [NEGATIVE_PROMPT] * len(prompts), 'foreground_prompts': item['prompts'], 'foreground_masks': fg, 'all_masks': torch.cat([bg, fg], dim=0), 'category_names': meta['category_names']}
def select_layer_attention_maps(captured_maps, output_size, layer_name):
    selected = {}
    for key, layer_maps in captured_maps.items():
        matches = [m for m in layer_maps if m.layer_name == layer_name]
        if len(matches) != 1: raise RuntimeError(f'Expected one {layer_name} map at {key}, got {len(matches)}')
        values = F.interpolate(matches[0].values.detach().float().cpu()[None, None], size=output_size, mode='bilinear', align_corners=False)[0, 0]
        selected[key] = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)
    return selected
def install_l04(runtime):
    def anchors(self, timestep, foreground_masks, *, strategy, topk_percent, layer_index=None, layer_name=None, **kwargs):
        maps = select_layer_attention_maps(self.attention_capture.maps, self.image_size, FIXED_LAYER['layer_name']); result = []
        for region_index, mask in enumerate(foreground_masks):
            m = compute_anchor_measurements(maps[(int(timestep), region_index)], mask.cpu(), topk_percent=topk_percent)
            prefix = '' if strategy == 'argmax' else 'topk_'
            result.append((float(m[prefix + 'anchor_x']), float(m[prefix + 'anchor_y'])))
        return result
    runtime._anchors_from_current_step = types.MethodType(anchors, runtime)
def run_one(payload, seed, lambda_value, use_hook=True):
    token_indices = [find_target_token_indices(smd.tokenizer, p, c) for p, c in zip(payload['foreground_prompts'], payload['category_names'])]
    refiner = DistillationPPRefiner(teacher_unet=teacher_unet, pipeline=smd, teacher_guidance_scale=lambda_value, guide_steps=GUIDE_STEPS, teacher_cfg_scale=TEACHER_CFG_SCALE) if use_hook else None
    if refiner: refiner.configure_sample(seed)
    seed_everything(seed); sync_cuda(); started = time.perf_counter()
    with SemanticAnchorCapture(smd.unet) as capture:
        capture.configure(token_indices); runtime = SemanticAnchorRuntime(smd, capture, image_size=TARGET_SIZE); install_l04(runtime)
        image, _ = runtime.generate(prompts=payload['prompts'], negative_prompts=payload['negative_prompts'], masks=payload['all_masks'].to(device=device, dtype=torch.float32), foreground_masks=payload['foreground_masks'], mode=ANCHOR_MODE, weight_policy=WEIGHT_POLICY, bootstrap_steps=BOOTSTRAP_STEPS, topk_percent=TOPK_ATTENTION_PERCENT, spatial_sigma_latent=SIGMA_D, semantic_sigma_scale=SEMANTIC_SIGMA_SCALE, mask_stds=MASK_STD, mask_strengths=MASK_STRENGTH, preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA, attention_layer_name=FIXED_LAYER['layer_name'], denoised_refiner=refiner)
    sync_cuda(); elapsed = time.perf_counter() - started
    records = [] if refiner is None else [r.to_dict() for r in refiner.records]
    return image, elapsed, records
print('[OK] Loader and generation helper ready.')

# --- Source notebook cell 6 ---
# 6. Mandatory parity gate: lambda=0 must be pixel-identical to the original W03 path.
first_batch = next(iter(loader)); parity_payload = make_payload(first_batch, 0); parity_seed = BASE_SEED
baseline_image, _, _ = run_one(parity_payload, parity_seed, 0.0, use_hook=False)
zero_image, _, zero_records = run_one(parity_payload, parity_seed, 0.0, use_hook=True)
assert np.array_equal(np.asarray(baseline_image), np.asarray(zero_image)), 'lambda=0 changed W03 pixels'
assert sum(r['applied'] for r in zero_records) == 0
del baseline_image, zero_image, first_batch, parity_payload; gc.collect(); torch.cuda.empty_cache()
print('[OK] lambda=0 is pixel-identical to the unmodified W03/L04 path.')

# --- Source notebook cell 7 ---
# 7. Paired lambda validation. Images stay in RAM only until CLIP scoring.
generated_in_memory = {value: [] for value in LAMBDA_SWEEP}
elapsed_by_lambda = {value: [] for value in LAMBDA_SWEEP}
eval_samples, audit_rows = [], []
for sample_index, batch in enumerate(islice(loader, RUN_SAMPLES)):
    payload = make_payload(batch, 0); sample_id = payload['sample_id']; seed = BASE_SEED + sample_index
    eval_samples.append({'sample_id': sample_id, 'background_prompt': payload['prompts'][0], 'foreground_prompts': list(payload['foreground_prompts']), 'foreground_masks': payload['foreground_masks'].bool().cpu().clone()})
    for lambda_value in LAMBDA_SWEEP:
        print(f'[RUN] sample={sample_index + 1}/{RUN_SAMPLES} lambda={lambda_value:.3f}')
        image, elapsed, records = run_one(payload, seed, lambda_value, use_hook=True)
        applied = [record for record in records if record['applied']]
        assert len(applied) == (0 if lambda_value == 0 else GUIDE_STEPS)
        if applied:
            assert applied[0]['step_index'] == 0 and applied[0]['teacher_timestep'] == EXPECTED_TIMESTEPS[1]
            lhs, rhs = applied[0]['refinement_l2'], lambda_value * applied[0]['student_teacher_l2']
            assert abs(lhs - rhs) <= max(2e-4, 0.01 * rhs), (lhs, rhs)
        generated_in_memory[lambda_value].append(image)
        elapsed_by_lambda[lambda_value].append(float(elapsed))
        audit_rows.append({'sample_index': sample_index, 'sample_id': sample_id, 'lambda': lambda_value, 'applied_steps': len(applied), 'teacher_timestep': (applied[0]['teacher_timestep'] if applied else None), 'teacher_elapsed_sec': (applied[0]['teacher_elapsed_sec'] if applied else 0.0), 'student_teacher_l2': (applied[0]['student_teacher_l2'] if applied else None), 'refinement_l2': (applied[0]['refinement_l2'] if applied else None)})
    del payload, batch; gc.collect(); torch.cuda.empty_cache()
assert len(eval_samples) == RUN_SAMPLES
assert all(len(images) == RUN_SAMPLES for images in generated_in_memory.values())
audit_df = pd.DataFrame(audit_rows)
audit_summary = audit_df.groupby('lambda', as_index=False).agg(samples=('sample_index', 'nunique'), applied_steps=('applied_steps', 'sum'), teacher_timestep=('teacher_timestep', 'first'), teacher_elapsed_mean_sec=('teacher_elapsed_sec', 'mean'), student_teacher_l2_mean=('student_teacher_l2', 'mean'), refinement_l2_mean=('refinement_l2', 'mean'))
audit_summary.to_csv(RUN_ROOT / 'distillationpp_audit_summary.csv', index=False)
print(f'[OK] Generated {RUN_SAMPLES * len(LAMBDA_SWEEP)} images in RAM; no generated image was written to disk.')

# --- Source notebook cell 8 ---
# 8. Save method provenance before releasing generation models.
def git_output(*args):
    result = subprocess.run(['git', '-C', str(REPO_ROOT), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None
run_config = {'run_id': RUN_ID, 'purpose': 'paired_lambda_validation_128', 'paper': 'arXiv:2412.08871v1', 'official_code': 'https://github.com/geonyeong-park/inference_distillation', 'official_code_audited_commit': '59dd61e6b2393dd16b8cadadef8897dad22e468a', 'formulation': 'paper Eq.9 clean-estimate interpolation (not the repository noise-direction approximation)', 'student': 'SD1.5 + LCM-LoRA', 'teacher': 'unfused base SD1.5 UNet', 'manifest': str(RUN_MANIFEST), 'manifest_records_total': len(all_manifest_records), 'subset_rule': 'first 128 records in manifest order', 'samples': RUN_SAMPLES, 'resolution': list(TARGET_SIZE), 'base_seed': BASE_SEED, 'timesteps': actual_timesteps, 'bootstrap_steps': BOOTSTRAP_STEPS, 'anchor_mode': ANCHOR_MODE, 'weight_policy': WEIGHT_POLICY, 'topk_attention_percent': TOPK_ATTENTION_PERCENT, 'fixed_layer': FIXED_LAYER, 'spatial_sigma_latent': SIGMA_D, 'lambda_sweep': LAMBDA_SWEEP, 'paper_lcm_lambda': 0.02, 'teacher_cfg_scale': TEACHER_CFG_SCALE, 'guide_steps': GUIDE_STEPS, 'renoise': 'random at next scheduled timestep', 'teacher_rng': 'private seed = sample seed + 1000003', 'metrics': ['clip_fg', 'clip_bg', 'time'], 'selection_rule': 'maximize arithmetic mean of CLIP(fg)x100 and CLIP(bg)x100; lower time breaks exact ties', 'generated_images_persisted': False, 'git_commit': git_output('rev-parse', 'HEAD'), 'git_status_short': git_output('status', '--short'), 'semantic_anchor_sha256': hashlib.sha256((OURS_SRC / 'experiments/semantic_anchor.py').read_bytes()).hexdigest(), 'distillation_pp_sha256': hashlib.sha256((OURS_SRC / 'experiments/distillation_pp.py').read_bytes()).hexdigest()}
(RUN_ROOT / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
del teacher_unet, smd, loader; gc.collect(); torch.cuda.empty_cache()
print('[OK] Provenance saved and generation models released.')

# --- Source notebook cell 9 ---
# 9. Score the in-memory images with the same project CLIP(fg)/CLIP(bg) implementation.
from metrics.clip_metrics import CLIPScoreAccumulator
clip_accumulator = CLIPScoreAccumulator(device=device, model_name='ViT-B-32', pretrained='openai', batch_size=16)
def reset_clip_totals(accumulator):
    accumulator._pg_sum = accumulator._bg_sum = accumulator._fg_sum = 0.0
    accumulator._pg_count = accumulator._bg_count = accumulator._fg_count = 0
metric_rows = []
for lambda_value in LAMBDA_SWEEP:
    reset_clip_totals(clip_accumulator)
    for image, sample in zip(generated_in_memory[lambda_value], eval_samples):
        clip_accumulator.update_background_region(image, sample['foreground_masks'], sample['background_prompt'], apply_mask=True)
        clip_accumulator.update_foreground_regions(image, sample['foreground_masks'], sample['foreground_prompts'], padding_ratio=0.08, apply_mask=True)
    values = clip_accumulator.compute(); elapsed_values = np.asarray(elapsed_by_lambda[lambda_value], dtype=np.float64)
    metric_rows.append({'lambda': lambda_value, 'clip_fg_x100': values['clip_fg_x100'], 'clip_bg_x100': values['clip_bg_x100'], 'time_mean_sec': float(elapsed_values.mean()), 'time_std_sec': float(elapsed_values.std()), 'time_total_sec': float(elapsed_values.sum()), 'num_evaluated': RUN_SAMPLES})
    generated_in_memory[lambda_value].clear(); gc.collect(); torch.cuda.empty_cache()
metrics_df = pd.DataFrame(metric_rows).sort_values('lambda')
metrics_df['selection_score'] = (metrics_df['clip_fg_x100'] + metrics_df['clip_bg_x100']) / 2.0
baseline_row = metrics_df.loc[metrics_df['lambda'] == 0.0].iloc[0]
metrics_df['delta_clip_fg_x100'] = metrics_df['clip_fg_x100'] - baseline_row['clip_fg_x100']
metrics_df['delta_clip_bg_x100'] = metrics_df['clip_bg_x100'] - baseline_row['clip_bg_x100']
metrics_df['delta_selection_score'] = metrics_df['selection_score'] - baseline_row['selection_score']
metrics_df = metrics_df.sort_values(['selection_score', 'time_mean_sec'], ascending=[False, True]).reset_index(drop=True)
metrics_df.to_csv(RUN_ROOT / 'lambda_metrics_summary.csv', index=False)
(RUN_ROOT / 'lambda_metrics_summary.json').write_text(metrics_df.to_json(orient='records', indent=2), encoding='utf-8')
display(Markdown('## 128-sample paired lambda validation'))
display(metrics_df.style.format({'lambda': '{:.3f}', 'clip_fg_x100': '{:.4f}', 'clip_bg_x100': '{:.4f}', 'selection_score': '{:.4f}', 'delta_clip_fg_x100': '{:+.4f}', 'delta_clip_bg_x100': '{:+.4f}', 'delta_selection_score': '{:+.4f}', 'time_mean_sec': '{:.3f}', 'time_std_sec': '{:.3f}', 'time_total_sec': '{:.1f}'}))

# --- Source notebook cell 10 ---
# 10. Select the best lambda and persist metrics only.
assert set(metrics_df['num_evaluated']) == {RUN_SAMPLES}
assert set(metrics_df['lambda']) == set(LAMBDA_SWEEP)
assert int(audit_summary.loc[audit_summary['lambda'] == 0.0, 'applied_steps'].iloc[0]) == 0
assert all(int(audit_summary.loc[audit_summary['lambda'] == value, 'applied_steps'].iloc[0]) == RUN_SAMPLES for value in LAMBDA_SWEEP if value > 0)
best = metrics_df.iloc[0].to_dict()
selection = {'selected_lambda': float(best['lambda']), 'selection_score': float(best['selection_score']), 'selection_rule': run_config['selection_rule'], 'num_samples': RUN_SAMPLES, 'metrics': {key: (int(value) if key == 'num_evaluated' else float(value)) for key, value in best.items()}}
(RUN_ROOT / 'selected_lambda.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
del clip_accumulator, generated_in_memory, eval_samples; gc.collect(); torch.cuda.empty_cache()
assert not list(RUN_ROOT.rglob('*.png')), 'Generated PNG persistence is disabled for this validation.'
expected_outputs = {'run_config.json', 'distillationpp_audit_summary.csv', 'lambda_metrics_summary.csv', 'lambda_metrics_summary.json', 'selected_lambda.json'}
actual_outputs = {path.name for path in RUN_ROOT.iterdir() if path.is_file()}
assert actual_outputs == expected_outputs, (actual_outputs, expected_outputs)
print('[OK] Method/code gates passed.')
print(f'[RESULT] Selected lambda = {selection["selected_lambda"]:.3f} | score = {selection["selection_score"]:.4f}')
print('[OK] Metrics-only output:', RUN_ROOT)
