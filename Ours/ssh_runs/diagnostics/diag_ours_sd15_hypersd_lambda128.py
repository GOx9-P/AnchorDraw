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

# Notebook: Ours/kaggle_semantic_anchor_experiments/semantic_anchor_sd15_hypersd_w03_l04_distillationpp_lambda_sweep128_kaggle.ipynb
# Notebook SHA256: 75c20ddcc3d0dbeac4887276066faeae88a168711cc5b21f347920f82138746d

# --- Source notebook cell 2 ---
# 2. Locate or clone the repository and freeze the paired Hyper-SD sweep protocol.
import os, json, time, gc, types, hashlib, zipfile, importlib, importlib.util
import importlib.metadata as package_metadata
from pathlib import Path
from itertools import islice
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# terminal display provided above

EXPERIMENT_ID = 'OURS-SD15-HYPSD-DPP-LAMBDA-SWEEP128'
RUN_ID = 'semantic_anchor_sd15_hypersd_w03_l04_sigmad4_distillpp_lambda_sweep128'
RUN_MANIFEST = REPO_ROOT / 'Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl'
COCO_ROOT = Path(os.environ.get('COCO_ROOT', str(WORK_DIR / 'COCO')))
RUN_ROOT = WORK_DIR / RUN_ID
RUN_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_ID = 'runwayml/stable-diffusion-v1-5'
HYPER_SD_REPO_ID = 'ByteDance/Hyper-SD'
HYPER_SD_WEIGHT_NAME = 'Hyper-SD15-4steps-lora.safetensors'
HYPER_SD_NUM_INFERENCE_STEPS = 4
HYPER_SD_LORA_SCALE = 1.0
SEMANTICDRAW_GUIDANCE_SCALE = 1.0
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
LAMBDA_SWEEP = [0.0, 0.005, 0.01, 0.02]
TEACHER_CFG_SCALE = 7.5
GUIDE_STEPS = 1
TEACHER_SEED_OFFSET = 1_000_003
EXPECTED_TIMESTEPS = [999, 749, 499, 249]
assert RUN_MANIFEST.exists()
print('[OK]', EXPERIMENT_ID)
print('[OK] Repo:', REPO_ROOT)
print('[OK] Sweep:', LAMBDA_SWEEP)
COCO_ROOT = SSH_COCO_ROOT

# --- Source notebook cell 3 ---
# 3. Import project code and prepare COCO annotations; source images are unnecessary for CLIP selection.
importlib.invalidate_caches()
if 'torchao' in sys.modules or importlib.util.find_spec('torchao') is not None:
    raise RuntimeError('torchao remains importable. Restart the Kaggle session, then Run All from cell 1.')
OURS_SRC = REPO_ROOT / 'Ours/src'
BASELINE_SRC = REPO_ROOT / 'Baseline/semantic-draw-main/src'
for module_name in list(sys.modules):
    if module_name == 'data' or module_name.startswith('data.') or module_name == 'experiments' or module_name.startswith('experiments.'):
        del sys.modules[module_name]
sys.path.insert(0, str(BASELINE_SRC)); sys.path.insert(0, str(OURS_SRC))
from data import COCORegionConfig, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from experiments.semantic_anchor import SemanticAnchorCapture, SemanticAnchorRuntime, compute_anchor_measurements, find_target_token_indices
from experiments.distillation_pp import DistillationPPRefiner
from diffusers import DDIMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
pipeline_path = BASELINE_SRC / 'model/pipeline_semantic_draw.py'
spec = importlib.util.spec_from_file_location('pipeline_semantic_draw_hypersd_distillpp_sweep', pipeline_path)
pipeline_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pipeline_module)
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline

all_manifest_records = [json.loads(line) for line in RUN_MANIFEST.read_text(encoding='utf-8').splitlines() if line.strip()]
assert len(all_manifest_records) == 1073 and len({row['sample_id'] for row in all_manifest_records}) == 1073
manifest_records = all_manifest_records[:RUN_SAMPLES]
annotations = [COCO_ROOT / 'annotations/instances_val2017.json', COCO_ROOT / 'annotations/captions_val2017.json']
def download_annotations():
    if all(path.exists() for path in annotations):
        return
    archive = COCO_ROOT / 'annotations_trainval2017.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists() and not zipfile.is_zipfile(archive):
        archive.unlink()
    if not archive.exists():
        urls = ['https://images.cocodataset.org/annotations/annotations_trainval2017.zip', 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip']
        for url in urls:
            result = subprocess.run(['wget', '-q', '-c', '--no-check-certificate', '-O', str(archive), url])
            if result.returncode == 0 and zipfile.is_zipfile(archive):
                break
            if archive.exists():
                archive.unlink()
    assert zipfile.is_zipfile(archive), f'Invalid COCO annotation archive: {archive}'
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(COCO_ROOT)
download_annotations()
assert all(path.exists() for path in annotations), annotations
print(f'[OK] Source imports and COCO annotations ready for paired subset {RUN_SAMPLES}/1073.')

# --- Source notebook cell 4 ---
# 4. Load the exact Hyper-SD student used by the full1073 baseline and a separate base SD1.5 teacher.
def maybe_login_hf():
    token = os.environ.get('HF_TOKEN')
    if token is None:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret('HF_TOKEN')
        except Exception:
            token = None
    if token:
        from huggingface_hub import login
        login(token=token)
def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
def clear_existing_lora_adapters(pipe):
    if hasattr(pipe, 'unfuse_lora'):
        try:
            pipe.unfuse_lora()
        except Exception as error:
            print('[WARN] unfuse_lora skipped:', error)
    if hasattr(pipe, 'unload_lora_weights'):
        pipe.unload_lora_weights()
        return
    if hasattr(pipe, 'delete_adapters'):
        pipe.delete_adapters(['lcm'])
        return
    raise RuntimeError('Could not remove the default LCM-LoRA before Hyper-SD loading.')
def prepare_hyper_sd_ddim_schedule(self, t_index_list=None, num_inference_steps=None):
    if num_inference_steps is None:
        num_inference_steps = t_index_list if isinstance(t_index_list, int) else self.default_num_inference_steps
    try:
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
    except TypeError:
        self.scheduler.set_timesteps(num_inference_steps)
    self.timesteps = self.scheduler.timesteps.to(dtype=torch.long).cpu()
    shape = (len(self.timesteps), 1, 1, 1)
    alphas = self.scheduler.alphas_cumprod.to(dtype=torch.float32)
    alpha_prod = alphas[self.timesteps].view(*shape).clamp(1e-8, 1.0).to(self.device)
    beta_prod = (1.0 - alpha_prod).clamp(0.0, 1.0)
    self.alpha_prod_t_sqrt = alpha_prod.sqrt().to(dtype=self.dtype, device=self.device)
    self.beta_prod_t_sqrt = beta_prod.sqrt().to(dtype=self.dtype, device=self.device)
    noise_lvs = self.beta_prod_t_sqrt.flatten()
    self.noise_lvs = noise_lvs[None, :, None, None, None]
    self.next_noise_lvs = torch.cat([noise_lvs[1:], noise_lvs.new_zeros(1)])[None, :, None, None, None]
def hyper_sd_ddim_predict_x0_step(self, noise_pred, idx, latent):
    alpha = self.alpha_prod_t_sqrt[idx].to(dtype=latent.dtype, device=latent.device).clamp_min(1e-6)
    beta = self.beta_prod_t_sqrt[idx].to(dtype=latent.dtype, device=latent.device)
    return (latent - beta * noise_pred) / alpha

assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator.'
maybe_login_hf(); device, dtype = torch.device('cuda:0'), torch.float16
seed_everything(BASE_SEED)
smd = SemanticDrawPipeline(device=device, dtype=dtype, sd_version='1.5', hf_key=MODEL_ID, has_i2t=False, default_mask_std=MASK_STD, default_mask_strength=MASK_STRENGTH, default_preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA, mask_type=MASK_TYPE)
clear_existing_lora_adapters(smd.pipe)
hyper_sd_path = hf_hub_download(repo_id=HYPER_SD_REPO_ID, filename=HYPER_SD_WEIGHT_NAME)
smd.pipe.load_lora_weights(hyper_sd_path, adapter_name='hyper_sd')
if hasattr(smd.pipe, 'set_adapters'):
    try:
        smd.pipe.set_adapters(['hyper_sd'], adapter_weights=[HYPER_SD_LORA_SCALE])
    except TypeError:
        smd.pipe.set_adapters(['hyper_sd'])
if hasattr(smd.pipe, 'get_list_adapters'):
    assert 'hyper_sd' in str(smd.pipe.get_list_adapters()), smd.pipe.get_list_adapters()
if hasattr(smd.pipe, 'fuse_lora'):
    try:
        smd.pipe.fuse_lora(lora_scale=HYPER_SD_LORA_SCALE, safe_fusing=False)
    except TypeError:
        smd.pipe.fuse_lora(lora_scale=HYPER_SD_LORA_SCALE)
    except Exception as error:
        active = smd.pipe.get_active_adapters() if hasattr(smd.pipe, 'get_active_adapters') else []
        assert 'hyper_sd' in active, (error, active)
        print('[WARN] fuse_lora skipped; verified active Hyper-SD adapter:', error)
try:
    smd.pipe.scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder='scheduler', timestep_spacing='trailing')
except Exception as error:
    print('[WARN] DDIMScheduler.from_pretrained fallback:', error)
    smd.pipe.scheduler = DDIMScheduler.from_config(smd.pipe.scheduler.config, timestep_spacing='trailing')
smd.scheduler = smd.pipe.scheduler
smd.unet = smd.pipe.unet
smd.default_num_inference_steps = HYPER_SD_NUM_INFERENCE_STEPS
smd.default_guidance_scale = SEMANTICDRAW_GUIDANCE_SCALE
smd.prepare_hyper_sd_ddim_schedule = types.MethodType(prepare_hyper_sd_ddim_schedule, smd)
smd.prepare_lcm_schedule = smd.prepare_hyper_sd_ddim_schedule
smd.scheduler_step = types.MethodType(hyper_sd_ddim_predict_x0_step, smd)
smd.prepare_hyper_sd_ddim_schedule(num_inference_steps=HYPER_SD_NUM_INFERENCE_STEPS)
teacher_unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder='unet', torch_dtype=dtype).to(device).eval()
for parameter in teacher_unet.parameters():
    parameter.requires_grad_(False)
actual_timesteps = [int(value) for value in smd.timesteps.detach().cpu().tolist()]
assert type(smd.scheduler).__name__ == 'DDIMScheduler'
assert actual_timesteps == EXPECTED_TIMESTEPS, (actual_timesteps, EXPECTED_TIMESTEPS)
assert abs(float(smd.default_guidance_scale) - SEMANTICDRAW_GUIDANCE_SCALE) < 1e-12
assert FIXED_LAYER['layer_name'] in set(smd.unet.attn_processors)
assert teacher_unet is not smd.unet and teacher_unet.config.in_channels == smd.unet.config.in_channels
print('[OK] Student=SD1.5+Hyper-SD 4-step; teacher=separate base SD1.5; timesteps=', actual_timesteps)

# --- Source notebook cell 5 ---
# 5. Build the full-manifest loader and retain the established W03/L04 generation path.
config = COCORegionConfig(coco_root=COCO_ROOT, split='val2017', instances_json=annotations[0], captions_json=annotations[1], manifest_path=RUN_MANIFEST, profile='multidiffusion_coco_all', model_family='sd15', target_size=TARGET_SIZE, return_image=False, cache_resized_masks=False, cache_dir=None, batch_size=BATCH_SIZE, num_workers=0, pin_memory=False, persistent_workers=False)
loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
assert len(loader.dataset) == len(all_manifest_records) == 1073
def make_payload(batch, index):
    item = batch_item_to_semanticdraw_inputs(batch, index); meta = item['metadata']
    foreground = item['masks'].float().cpu(); background = (1.0 - foreground.sum(dim=0, keepdim=True).clamp(0, 1)).clamp(0, 1)
    prompts = [item['background_prompt'], *item['prompts']]
    return {'sample_id': meta['sample_id'], 'image_id': meta['image_id'], 'prompts': prompts, 'negative_prompts': [NEGATIVE_PROMPT] * len(prompts), 'foreground_prompts': list(item['prompts']), 'foreground_masks': foreground, 'all_masks': torch.cat([background, foreground], dim=0), 'category_names': list(meta['category_names'])}
def select_layer_attention_maps(captured_maps, output_size, layer_name):
    selected = {}
    for key, layer_maps in captured_maps.items():
        matches = [layer_map for layer_map in layer_maps if layer_map.layer_name == layer_name]
        if len(matches) != 1:
            raise RuntimeError(f'Expected one {layer_name} map at {key}, got {len(matches)}')
        values = F.interpolate(matches[0].values.detach().float().cpu()[None, None], size=output_size, mode='bilinear', align_corners=False)[0, 0]
        selected[key] = (values - values.min()) / (values.max() - values.min()).clamp_min(1e-8)
    return selected
def install_l04(runtime):
    def anchors(self, timestep, foreground_masks, *, strategy, topk_percent, layer_index=None, layer_name=None, **kwargs):
        maps = select_layer_attention_maps(self.attention_capture.maps, self.image_size, FIXED_LAYER['layer_name']); result = []
        for region_index, mask in enumerate(foreground_masks):
            measurement = compute_anchor_measurements(maps[(int(timestep), region_index)], mask.cpu(), topk_percent=topk_percent)
            prefix = '' if strategy == 'argmax' else 'topk_'
            result.append((float(measurement[prefix + 'anchor_x']), float(measurement[prefix + 'anchor_y'])))
        return result
    runtime._anchors_from_current_step = types.MethodType(anchors, runtime)
def run_one(payload, seed, lambda_value, use_hook=True):
    token_indices = [find_target_token_indices(smd.tokenizer, prompt, category) for prompt, category in zip(payload['foreground_prompts'], payload['category_names'])]
    refiner = DistillationPPRefiner(teacher_unet=teacher_unet, pipeline=smd, teacher_guidance_scale=lambda_value, guide_steps=GUIDE_STEPS, teacher_cfg_scale=TEACHER_CFG_SCALE, seed_offset=TEACHER_SEED_OFFSET) if use_hook else None
    if refiner is not None:
        refiner.configure_sample(seed)
    seed_everything(seed); sync_cuda(); started = time.perf_counter()
    with SemanticAnchorCapture(smd.unet) as capture:
        capture.configure(token_indices); runtime = SemanticAnchorRuntime(smd, capture, image_size=TARGET_SIZE); install_l04(runtime)
        image, _ = runtime.generate(prompts=payload['prompts'], negative_prompts=payload['negative_prompts'], masks=payload['all_masks'].to(device=device, dtype=torch.float32), foreground_masks=payload['foreground_masks'], mode=ANCHOR_MODE, weight_policy=WEIGHT_POLICY, bootstrap_steps=BOOTSTRAP_STEPS, topk_percent=TOPK_ATTENTION_PERCENT, spatial_sigma_latent=SIGMA_D, semantic_sigma_scale=SEMANTIC_SIGMA_SCALE, mask_stds=MASK_STD, mask_strengths=MASK_STRENGTH, preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA, attention_layer_name=FIXED_LAYER['layer_name'], denoised_refiner=refiner)
    sync_cuda(); elapsed = time.perf_counter() - started
    records = [] if refiner is None else [record.to_dict() for record in refiner.records]
    return image, elapsed, records
print('[OK] Loader and Hyper-SD generation helper ready.')

# --- Source notebook cell 6 ---
# 6. Mandatory parity gate: lambda=0 must be pixel-identical to the original Hyper-SD W03 path.
first_batch = next(iter(loader)); parity_payload = make_payload(first_batch, 0); parity_seed = BASE_SEED
baseline_image, _, _ = run_one(parity_payload, parity_seed, 0.0, use_hook=False)
zero_image, _, zero_records = run_one(parity_payload, parity_seed, 0.0, use_hook=True)
assert np.array_equal(np.asarray(baseline_image), np.asarray(zero_image)), 'lambda=0 changed Hyper-SD W03 pixels'
assert len(zero_records) == len(EXPECTED_TIMESTEPS) and sum(int(record['applied']) for record in zero_records) == 0
del baseline_image, zero_image, zero_records, first_batch, parity_payload; gc.collect(); torch.cuda.empty_cache()
print('[OK] lambda=0 is pixel-identical to the unmodified Hyper-SD W03/L04 path.')

# --- Source notebook cell 7 ---
# 7. Paired lambda sweep. Images stay in RAM only until CLIP scoring.
generated_in_memory = {value: [] for value in LAMBDA_SWEEP}
elapsed_by_lambda = {value: [] for value in LAMBDA_SWEEP}
eval_samples, audit_rows = [], []
for sample_index, batch in enumerate(islice(loader, RUN_SAMPLES)):
    payload = make_payload(batch, 0); sample_id = payload['sample_id']; seed = BASE_SEED + sample_index
    assert sample_id == manifest_records[sample_index]['sample_id']
    eval_samples.append({'sample_id': sample_id, 'background_prompt': payload['prompts'][0], 'foreground_prompts': list(payload['foreground_prompts']), 'foreground_masks': payload['foreground_masks'].bool().cpu().clone()})
    for lambda_value in LAMBDA_SWEEP:
        print(f'[RUN] sample={sample_index + 1}/{RUN_SAMPLES} lambda={lambda_value:.3f}')
        image, elapsed, records = run_one(payload, seed, lambda_value, use_hook=True)
        assert len(records) == len(EXPECTED_TIMESTEPS)
        applied = [record for record in records if record['applied']]
        assert len(applied) == (0 if lambda_value == 0 else GUIDE_STEPS)
        if applied:
            first = applied[0]
            assert first['step_index'] == 0 and first['timestep'] == EXPECTED_TIMESTEPS[0] and first['teacher_timestep'] == EXPECTED_TIMESTEPS[1]
            assert abs(float(first['teacher_guidance_scale']) - lambda_value) < 1e-12
            assert abs(float(first['teacher_cfg_scale']) - TEACHER_CFG_SCALE) < 1e-12
            expected_l2 = lambda_value * float(first['student_teacher_l2'])
            assert abs(float(first['refinement_l2']) - expected_l2) <= max(2e-4, 0.01 * expected_l2)
        generated_in_memory[lambda_value].append(image)
        elapsed_by_lambda[lambda_value].append(float(elapsed))
        audit_rows.append({'sample_index': sample_index, 'sample_id': sample_id, 'lambda': lambda_value, 'applied_steps': len(applied), 'teacher_timestep': (applied[0]['teacher_timestep'] if applied else None), 'teacher_elapsed_sec': (applied[0]['teacher_elapsed_sec'] if applied else 0.0), 'student_teacher_l2': (applied[0]['student_teacher_l2'] if applied else None), 'refinement_l2': (applied[0]['refinement_l2'] if applied else None)})
    del payload, batch; gc.collect(); torch.cuda.empty_cache()
assert len(eval_samples) == RUN_SAMPLES
assert all(len(images) == RUN_SAMPLES for images in generated_in_memory.values())
audit_df = pd.DataFrame(audit_rows)
audit_summary = audit_df.groupby('lambda', as_index=False).agg(samples=('sample_index', 'nunique'), applied_steps=('applied_steps', 'sum'), teacher_timestep=('teacher_timestep', 'first'), teacher_elapsed_mean_sec=('teacher_elapsed_sec', 'mean'), student_teacher_l2_mean=('student_teacher_l2', 'mean'), refinement_l2_mean=('refinement_l2', 'mean'))
audit_summary.to_csv(RUN_ROOT / 'distillationpp_audit_summary.csv', index=False)
print(f'[OK] Generated {RUN_SAMPLES * len(LAMBDA_SWEEP)} paired images in RAM; no generated image was written to disk.')

# --- Source notebook cell 8 ---
# 8. Save method and environment provenance before releasing generation models.
def git_output(*args):
    result = subprocess.run(['git', '-C', str(REPO_ROOT), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None
def installed_version(distribution):
    try:
        return package_metadata.version(distribution)
    except package_metadata.PackageNotFoundError:
        return None
run_config = {
    'experiment_id': EXPERIMENT_ID, 'run_id': RUN_ID, 'purpose': 'paired_hypersd_lambda_sweep_128',
    'paper': 'arXiv:2412.08871v1', 'official_code': 'https://github.com/geonyeong-park/inference_distillation',
    'official_code_audited_commit': '59dd61e6b2393dd16b8cadadef8897dad22e468a',
    'formulation': 'paper Eq.9 clean-estimate interpolation',
    'student': 'SD1.5 + Hyper-SD15-4steps LoRA', 'student_guidance_scale': SEMANTICDRAW_GUIDANCE_SCALE,
    'hyper_sd_repo': HYPER_SD_REPO_ID, 'hyper_sd_weight': HYPER_SD_WEIGHT_NAME, 'hyper_sd_lora_scale': HYPER_SD_LORA_SCALE,
    'teacher': 'separate unfused base SD1.5 UNet', 'teacher_cfg_scale': TEACHER_CFG_SCALE,
    'manifest': str(RUN_MANIFEST), 'manifest_records_total': len(all_manifest_records), 'subset_rule': 'first 128 records in manifest order',
    'samples': RUN_SAMPLES, 'resolution': list(TARGET_SIZE), 'base_seed': BASE_SEED, 'per_sample_seed': 'BASE_SEED + sample_index',
    'timesteps': actual_timesteps, 'bootstrap_steps': BOOTSTRAP_STEPS, 'anchor_mode': ANCHOR_MODE, 'weight_policy': WEIGHT_POLICY,
    'topk_attention_percent': TOPK_ATTENTION_PERCENT, 'fixed_layer': FIXED_LAYER, 'spatial_sigma_latent': SIGMA_D, 'semantic_sigma_scale': SEMANTIC_SIGMA_SCALE,
    'lambda_sweep': LAMBDA_SWEEP, 'guide_steps': GUIDE_STEPS, 'renoise': 'fresh Gaussian noise at next scheduled student timestep',
    'teacher_rng': f'private seed = sample seed + {TEACHER_SEED_OFFSET}', 'metrics': ['clip_fg', 'clip_bg', 'time'],
    'selection_rule': 'maximize arithmetic mean of CLIP(fg)x100 and CLIP(bg)x100; lower time breaks exact ties',
    'generated_images_persisted': False, 'git_commit': git_output('rev-parse', 'HEAD'), 'git_status_short': git_output('status', '--short'),
    'package_versions': {name: installed_version(name) for name in ['torch', 'diffusers', 'transformers', 'accelerate', 'peft', 'huggingface-hub', 'open-clip-torch', 'pandas']},
    'semantic_anchor_sha256': hashlib.sha256((OURS_SRC / 'experiments/semantic_anchor.py').read_bytes()).hexdigest(),
    'distillation_pp_sha256': hashlib.sha256((OURS_SRC / 'experiments/distillation_pp.py').read_bytes()).hexdigest(),
    'baseline_pipeline_sha256': hashlib.sha256(pipeline_path.read_bytes()).hexdigest(),
}
(RUN_ROOT / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
del teacher_unet, smd, loader; gc.collect(); torch.cuda.empty_cache()
print('[OK] Provenance saved and generation models released.')

# --- Source notebook cell 9 ---
# 9. Score in-memory images with the same project CLIP(fg)/CLIP(bg) implementation.
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
display(Markdown('## Hyper-SD 128-sample paired lambda sweep'))
display(metrics_df.style.format({'lambda': '{:.3f}', 'clip_fg_x100': '{:.4f}', 'clip_bg_x100': '{:.4f}', 'selection_score': '{:.4f}', 'delta_clip_fg_x100': '{:+.4f}', 'delta_clip_bg_x100': '{:+.4f}', 'delta_selection_score': '{:+.4f}', 'time_mean_sec': '{:.3f}', 'time_std_sec': '{:.3f}', 'time_total_sec': '{:.1f}'}))

# --- Source notebook cell 10 ---
# 10. Select the raw winner and persist metrics only.
assert set(metrics_df['num_evaluated']) == {RUN_SAMPLES}
assert set(metrics_df['lambda']) == set(LAMBDA_SWEEP)
assert int(audit_summary.loc[audit_summary['lambda'] == 0.0, 'applied_steps'].iloc[0]) == 0
assert all(int(audit_summary.loc[audit_summary['lambda'] == value, 'applied_steps'].iloc[0]) == RUN_SAMPLES for value in LAMBDA_SWEEP if value > 0)
best = metrics_df.iloc[0].to_dict()
selection = {'selected_lambda': float(best['lambda']), 'selection_score': float(best['selection_score']), 'delta_vs_lambda0': float(best['delta_selection_score']), 'selection_rule': run_config['selection_rule'], 'num_samples': RUN_SAMPLES, 'metrics': {key: (int(value) if key == 'num_evaluated' else float(value)) for key, value in best.items()}}
(RUN_ROOT / 'selected_lambda.json').write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding='utf-8')
del clip_accumulator, generated_in_memory, eval_samples; gc.collect(); torch.cuda.empty_cache()
assert not list(RUN_ROOT.rglob('*.png')), 'Generated PNG persistence is disabled for this validation.'
expected_outputs = {'run_config.json', 'distillationpp_audit_summary.csv', 'lambda_metrics_summary.csv', 'lambda_metrics_summary.json', 'selected_lambda.json'}
actual_outputs = {path.name for path in RUN_ROOT.iterdir() if path.is_file()}
assert actual_outputs == expected_outputs, (actual_outputs, expected_outputs)
print('[OK] Method/code gates passed.')
print(f'[RESULT] Raw selected lambda = {selection["selected_lambda"]:.3f} | score = {selection["selection_score"]:.4f} | delta vs lambda0 = {selection["delta_vs_lambda0"]:+.4f}')
print('[OK] Metrics-only output:', RUN_ROOT)
