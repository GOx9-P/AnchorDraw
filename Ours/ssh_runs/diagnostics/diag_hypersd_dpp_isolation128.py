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

# Notebook: Ours/kaggle_semantic_anchor_experiments/hypersd_canonical_ddim_distillpp_l000_vs_l0005_isolation128_kaggle.ipynb
# Notebook SHA256: 40dfeb5a790d085e29f5751d16c4134e86a282c101a13d5dbe6a7fac56ab3ac0

# --- Source notebook cell 2 ---
OURS_SRC = REPO_ROOT / 'Ours/src'
sys.path.insert(0, str(OURS_SRC))
# 2. Locate/clone AnchorDraw only for the frozen manifest and shared metric implementation.
import os, json, time, gc, hashlib, importlib, importlib.util
import importlib.metadata as package_metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
# terminal display provided above

EXPERIMENT_ID = 'HYPSD-CANON-DDIM-DPP-L000-VS-L0005-ISO128'
RUN_ID = 'hypersd_canonical_ddim_distillpp_l000_vs_l0005_isolation128'
RUN_MANIFEST = REPO_ROOT / 'Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl'
RUN_ROOT = WORK_DIR / RUN_ID
REAL_IMAGE_DIR = RUN_ROOT / 'coco_val2017_subset'
RUN_ROOT.mkdir(parents=True, exist_ok=True); REAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ID = 'runwayml/stable-diffusion-v1-5'
HYPER_SD_REPO_ID = 'ByteDance/Hyper-SD'
HYPER_SD_WEIGHT_NAME = 'Hyper-SD15-4steps-lora.safetensors'
NUM_INFERENCE_STEPS = 4
EXPECTED_TIMESTEPS = [999, 749, 499, 249]
GUIDANCE_SCALE = 0.0
ETA = 0.0
LAMBDA_VALUES = [0.0, 0.005]
TEACHER_CFG_SCALE = 7.5
GUIDE_STEPS = 1
TEACHER_SEED_OFFSET = 1_000_003
RUN_SAMPLES = 128
BASE_SEED = 2024
TARGET_SIZE = (512, 512)
METRIC_BATCH_SIZE = 8
IS_SPLITS = 10
records = [json.loads(line) for line in RUN_MANIFEST.read_text(encoding='utf-8').splitlines() if line.strip()]
assert len(records) == 1073 and len({row['sample_id'] for row in records}) == 1073
records = records[:RUN_SAMPLES]
assert all(str(row.get('caption', '')).strip() for row in records)
print('[OK]', EXPERIMENT_ID, '| paired samples:', len(records))

# --- Source notebook cell 3 ---
# Server adaptation of source cell 3: use the mounted COCO images.
import shutil
def valid_image(path):
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False
real_paths = []
for row in records:
    source_image = SSH_COCO_ROOT / 'val2017' / str(row['file_name'])
    destination = REAL_IMAGE_DIR / str(row['file_name'])
    if not valid_image(source_image):
        raise RuntimeError(f'Invalid source COCO image: {source_image}')
    shutil.copy2(source_image, destination)
    real_paths.append(destination)
assert len(real_paths) == RUN_SAMPLES and all(valid_image(path) for path in real_paths)
print('[OK] Real COCO subset ready:', len(real_paths))

# --- Source notebook cell 4 ---
# 4. Load the official canonical Hyper-SD student and an independent base SD1.5 teacher.
importlib.invalidate_caches()
if 'torchao' in sys.modules or importlib.util.find_spec('torchao') is not None:
    raise RuntimeError('torchao remains importable. Restart the Kaggle session, then Run All from cell 1.')
from diffusers import DiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download
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
def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator.'
maybe_login_hf(); device, dtype = torch.device('cuda:0'), torch.float16
try:
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, variant='fp16', safety_checker=None, requires_safety_checker=False).to(device)
    model_variant = 'fp16'
except Exception as error:
    print('[WARN] fp16 variant fallback:', error)
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False).to(device)
    model_variant = None
hyper_sd_path = hf_hub_download(repo_id=HYPER_SD_REPO_ID, filename=HYPER_SD_WEIGHT_NAME)
pipe.load_lora_weights(hyper_sd_path, adapter_name='hyper_sd')
if hasattr(pipe, 'set_adapters'):
    try:
        pipe.set_adapters(['hyper_sd'], adapter_weights=[1.0])
    except TypeError:
        pipe.set_adapters(['hyper_sd'])
if hasattr(pipe, 'get_list_adapters'):
    assert 'hyper_sd' in str(pipe.get_list_adapters()), pipe.get_list_adapters()
pipe.fuse_lora(lora_scale=1.0)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')
pipe.set_progress_bar_config(disable=True)
try:
    teacher_unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder='unet', torch_dtype=dtype, variant='fp16').to(device).eval()
except Exception:
    teacher_unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder='unet', torch_dtype=dtype).to(device).eval()
for parameter in teacher_unet.parameters():
    parameter.requires_grad_(False)
pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
actual_timesteps = [int(value) for value in pipe.scheduler.timesteps.detach().cpu().tolist()]
assert actual_timesteps == EXPECTED_TIMESTEPS, actual_timesteps
assert pipe.scheduler.config.prediction_type == 'epsilon'
assert teacher_unet is not pipe.unet
print('[OK] Canonical Hyper-SD pipeline and separate teacher loaded:', actual_timesteps)

# --- Source notebook cell 5 ---
# 5. Canonical DDIM loop with an optional first-step Distillation++ Eq.9 replacement.
def encode_conditioning(prompt, need_teacher_cfg):
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt=prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=need_teacher_cfg, negative_prompt=('' if need_teacher_cfg else None))
    return prompt_embeds, negative_prompt_embeds
def process_teacher_x0(value):
    if bool(getattr(pipe.scheduler.config, 'thresholding', False)):
        return pipe.scheduler._threshold_sample(value)
    if bool(getattr(pipe.scheduler.config, 'clip_sample', False)):
        limit = float(getattr(pipe.scheduler.config, 'clip_sample_range', 1.0))
        return value.clamp(-limit, limit)
    return value
@torch.no_grad()
def generate_one(prompt, seed, lambda_value):
    need_teacher = lambda_value > 0.0
    sync_cuda(); started = time.perf_counter()
    prompt_embeds, negative_prompt_embeds = encode_conditioning(prompt, need_teacher)
    student_generator = torch.Generator(device=device.type).manual_seed(int(seed))
    teacher_generator = torch.Generator(device=device.type).manual_seed(int(seed) + TEACHER_SEED_OFFSET)
    pipe.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    timesteps = pipe.scheduler.timesteps
    latent_shape = (1, int(pipe.unet.config.in_channels), TARGET_SIZE[0] // pipe.vae_scale_factor, TARGET_SIZE[1] // pipe.vae_scale_factor)
    latents = randn_tensor(latent_shape, generator=student_generator, device=device, dtype=prompt_embeds.dtype) * pipe.scheduler.init_noise_sigma
    audit = []
    for step_index, timestep in enumerate(timesteps):
        model_input = pipe.scheduler.scale_model_input(latents, timestep)
        student_noise = pipe.unet(model_input, timestep, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
        step_output = pipe.scheduler.step(student_noise, timestep, latents, eta=ETA, generator=student_generator, return_dict=True)
        should_apply = need_teacher and step_index < GUIDE_STEPS and step_index + 1 < len(timesteps)
        if should_apply:
            next_timestep = timesteps[step_index + 1]
            next_timestep_index = int(next_timestep.detach().cpu().item()) if isinstance(next_timestep, torch.Tensor) else int(next_timestep)
            alpha_next = pipe.scheduler.alphas_cumprod[next_timestep_index].to(device=device, dtype=latents.dtype)
            alpha_sqrt = alpha_next.sqrt(); sigma_sqrt = (1.0 - alpha_next).clamp_min(0).sqrt()
            canonical_rebuild = alpha_sqrt * step_output.pred_original_sample + sigma_sqrt * student_noise
            assert torch.allclose(canonical_rebuild, step_output.prev_sample, rtol=2e-3, atol=2e-3), 'Eq.9 forwarding does not match canonical DDIM at lambda=0'
            renoise = randn_tensor(step_output.pred_original_sample.shape, generator=teacher_generator, device=device, dtype=latents.dtype)
            teacher_input = alpha_sqrt * step_output.pred_original_sample + sigma_sqrt * renoise
            teacher_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            teacher_model_input = pipe.scheduler.scale_model_input(torch.cat([teacher_input, teacher_input], dim=0), next_timestep)
            teacher_noise = teacher_unet(teacher_model_input, next_timestep, encoder_hidden_states=teacher_embeddings, return_dict=False)[0]
            teacher_uncond, teacher_cond = teacher_noise.chunk(2)
            teacher_noise_cfg = teacher_uncond + TEACHER_CFG_SCALE * (teacher_cond - teacher_uncond)
            teacher_x0 = process_teacher_x0((teacher_input - sigma_sqrt * teacher_noise_cfg) / alpha_sqrt.clamp_min(1e-8))
            refined_x0 = torch.lerp(step_output.pred_original_sample, teacher_x0.to(step_output.pred_original_sample.dtype), float(lambda_value))
            latents = alpha_sqrt * refined_x0 + sigma_sqrt * student_noise
            teacher_delta = float((teacher_x0.float() - step_output.pred_original_sample.float()).pow(2).mean().sqrt().item())
            refinement_delta = float((refined_x0.float() - step_output.pred_original_sample.float()).pow(2).mean().sqrt().item())
            audit.append({'step_index': step_index, 'timestep': int(timestep), 'teacher_timestep': next_timestep_index, 'applied': True, 'student_teacher_l2': teacher_delta, 'refinement_l2': refinement_delta})
        else:
            latents = step_output.prev_sample
            audit.append({'step_index': step_index, 'timestep': int(timestep), 'teacher_timestep': None, 'applied': False, 'student_teacher_l2': None, 'refinement_l2': None})
    image_latent = latents / pipe.vae.config.scaling_factor
    decoded = pipe.vae.decode(image_latent, return_dict=False)[0]
    image = pipe.image_processor.postprocess(decoded, output_type='pil')[0]
    sync_cuda(); elapsed = time.perf_counter() - started
    return image, elapsed, audit
print('[OK] Canonical DDIM + optional Eq.9 loop ready.')

# --- Source notebook cell 6 ---
# 6. Pixel parity gate: our lambda=0 loop must reproduce the official Diffusers pipeline call.
parity_prompt = str(records[0]['caption']); parity_seed = BASE_SEED
official_generator = torch.Generator(device=device.type).manual_seed(parity_seed)
official_image = pipe(prompt=parity_prompt, num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE, eta=ETA, generator=official_generator, output_type='pil').images[0]
manual_image, _, manual_audit = generate_one(parity_prompt, parity_seed, 0.0)
assert np.array_equal(np.asarray(official_image), np.asarray(manual_image)), 'lambda=0 manual loop differs from official canonical Hyper-SD pipeline'
assert len(manual_audit) == NUM_INFERENCE_STEPS and sum(int(row['applied']) for row in manual_audit) == 0
del official_image, manual_image, manual_audit; gc.collect(); torch.cuda.empty_cache()
print('[OK] lambda=0 is pixel-identical to official canonical Hyper-SD DDIM.')

# --- Source notebook cell 7 ---
# 7. Paired generation; all generated images stay in RAM.
generated = {value: [] for value in LAMBDA_VALUES}
elapsed = {value: [] for value in LAMBDA_VALUES}
audit_rows = []
for sample_index, record in enumerate(records):
    prompt = str(record['caption']); seed = BASE_SEED + sample_index
    for lambda_value in LAMBDA_VALUES:
        print(f'[RUN] sample={sample_index + 1}/{RUN_SAMPLES} lambda={lambda_value:.3f}')
        image, seconds, audit = generate_one(prompt, seed, lambda_value)
        assert len(audit) == NUM_INFERENCE_STEPS
        applied = [row for row in audit if row['applied']]
        assert len(applied) == (0 if lambda_value == 0 else GUIDE_STEPS)
        if applied:
            first = applied[0]
            assert first['step_index'] == 0 and first['timestep'] == EXPECTED_TIMESTEPS[0] and first['teacher_timestep'] == EXPECTED_TIMESTEPS[1]
            expected_l2 = lambda_value * first['student_teacher_l2']
            assert abs(first['refinement_l2'] - expected_l2) <= max(2e-4, 0.01 * expected_l2)
        generated[lambda_value].append(image); elapsed[lambda_value].append(float(seconds))
        audit_rows.append({'sample_index': sample_index, 'sample_id': record['sample_id'], 'lambda': lambda_value, 'applied_steps': len(applied), 'teacher_timestep': (applied[0]['teacher_timestep'] if applied else None), 'student_teacher_l2': (applied[0]['student_teacher_l2'] if applied else None), 'refinement_l2': (applied[0]['refinement_l2'] if applied else None)})
assert all(len(images) == RUN_SAMPLES for images in generated.values())
audit_df = pd.DataFrame(audit_rows)
audit_summary = audit_df.groupby('lambda', as_index=False).agg(samples=('sample_index', 'nunique'), applied_steps=('applied_steps', 'sum'), teacher_timestep=('teacher_timestep', 'first'), student_teacher_l2_mean=('student_teacher_l2', 'mean'), refinement_l2_mean=('refinement_l2', 'mean'))
audit_summary.to_csv(RUN_ROOT / 'distillationpp_audit_summary.csv', index=False)
print('[OK] Paired generation complete; no generated PNG was written.')

# --- Source notebook cell 8 ---
# 8. Persist provenance, then release both diffusion UNets before metric loading.
def git_output(*args):
    result = subprocess.run(['git', '-C', str(REPO_ROOT), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None
def installed_version(name):
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None
run_config = {
    'experiment_id': EXPERIMENT_ID, 'run_id': RUN_ID, 'purpose': 'isolate Distillation++ from AnchorDraw and custom random-renoise sampling',
    'sample_source': 'first 128 records of the established full1073 manifest', 'prompt_source': 'record.caption', 'samples': RUN_SAMPLES,
    'student': 'canonical SD1.5 + Hyper-SD15-4steps LoRA', 'model_id': MODEL_ID, 'model_variant': model_variant,
    'hyper_sd_repo': HYPER_SD_REPO_ID, 'hyper_sd_weight': HYPER_SD_WEIGHT_NAME, 'sampler': 'DDIMScheduler.step',
    'timestep_spacing': 'trailing', 'timesteps': actual_timesteps, 'eta': ETA, 'guidance_scale': GUIDANCE_SCALE,
    'lambda_values': LAMBDA_VALUES, 'teacher': 'separate unfused base SD1.5 UNet', 'teacher_cfg_scale': TEACHER_CFG_SCALE,
    'guide_steps': GUIDE_STEPS, 'teacher_rng': f'private seed = sample seed + {TEACHER_SEED_OFFSET}',
    'distillation_update': 'Eq.9 clean-estimate interpolation; deterministic DDIM forwarding with student epsilon',
    'anchor_draw_enabled': False, 'generated_images_persisted': False, 'metrics': ['fid', 'is', 'clip_global', 'time'],
    'git_commit': git_output('rev-parse', 'HEAD'), 'git_status_short': git_output('status', '--short'),
    'package_versions': {name: installed_version(name) for name in ['torch', 'diffusers', 'transformers', 'accelerate', 'peft', 'huggingface-hub', 'torchmetrics', 'torch-fidelity', 'open-clip-torch']},
    'metric_code_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((OURS_SRC / 'metrics').glob('*.py'))},
}
(RUN_ROOT / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
del teacher_unet, pipe; gc.collect(); torch.cuda.empty_cache()
try:
    torch.cuda.ipc_collect()
except Exception:
    pass
print('[OK] Provenance saved; diffusion models released.')

# --- Source notebook cell 9 ---
# 9. Diagnostic FID, IS, global-caption CLIP, and time for each paired branch.
from metrics.inception_metrics import InceptionMetricsAccumulator
from metrics.clip_metrics import CLIPScoreAccumulator
real_images = [Image.open(path).convert('RGB') for path in real_paths]
prompts = [str(record['caption']) for record in records]
metric_rows = []
for lambda_value in LAMBDA_VALUES:
    inception = InceptionMetricsAccumulator(device=device, compute_fid=True, compute_is=True, fid_feature=2048, is_splits=IS_SPLITS)
    clip = CLIPScoreAccumulator(device=device, model_name='ViT-B-32', pretrained='openai', batch_size=16)
    for start in range(0, RUN_SAMPLES, METRIC_BATCH_SIZE):
        stop = min(start + METRIC_BATCH_SIZE, RUN_SAMPLES)
        inception.update(real_images[start:stop], generated[lambda_value][start:stop], TARGET_SIZE)
        clip.update_prompt_global(generated[lambda_value][start:stop], prompts[start:stop])
    values = {**inception.compute(), **clip.compute()}
    times = np.asarray(elapsed[lambda_value], dtype=np.float64)
    metric_rows.append({'lambda': lambda_value, 'fid': values['fid'], 'is_mean': values['is_mean'], 'is_std': values['is_std'], 'clip_global_x100': values['clip_pg_x100'], 'time_mean_sec': float(times.mean()), 'time_std_sec': float(times.std()), 'time_total_sec': float(times.sum()), 'num_evaluated': RUN_SAMPLES})
    del inception, clip; gc.collect(); torch.cuda.empty_cache()
metrics_df = pd.DataFrame(metric_rows).sort_values('lambda').reset_index(drop=True)
baseline = metrics_df.loc[metrics_df['lambda'] == 0.0].iloc[0]
for metric in ['fid', 'is_mean', 'clip_global_x100', 'time_mean_sec']:
    metrics_df['delta_' + metric] = metrics_df[metric] - baseline[metric]
metrics_df.to_csv(RUN_ROOT / 'isolation_metrics.csv', index=False)
(RUN_ROOT / 'isolation_metrics.json').write_text(metrics_df.to_json(orient='records', indent=2), encoding='utf-8')
display(Markdown('## Canonical Hyper-SD DDIM isolation results'))
display(metrics_df.style.format({'lambda': '{:.3f}', 'fid': '{:.4f}', 'is_mean': '{:.4f}', 'is_std': '{:.4f}', 'clip_global_x100': '{:.4f}', 'time_mean_sec': '{:.3f}', 'time_std_sec': '{:.3f}', 'time_total_sec': '{:.1f}', 'delta_fid': '{:+.4f}', 'delta_is_mean': '{:+.4f}', 'delta_clip_global_x100': '{:+.4f}', 'delta_time_mean_sec': '{:+.3f}'}))

# --- Source notebook cell 10 ---
# 10. Remove image data and enforce the final metrics-only contract.
assert set(metrics_df['lambda']) == set(LAMBDA_VALUES) and set(metrics_df['num_evaluated']) == {RUN_SAMPLES}
assert int(audit_summary.loc[audit_summary['lambda'] == 0.0, 'applied_steps'].iloc[0]) == 0
assert int(audit_summary.loc[audit_summary['lambda'] == 0.005, 'applied_steps'].iloc[0]) == RUN_SAMPLES
assert int(audit_summary.loc[audit_summary['lambda'] == 0.005, 'teacher_timestep'].iloc[0]) == EXPECTED_TIMESTEPS[1]
del real_images, generated, prompts; gc.collect(); torch.cuda.empty_cache()
assert REAL_IMAGE_DIR.resolve().is_relative_to(WORK_DIR.resolve())
assert all(path.resolve().is_relative_to(REAL_IMAGE_DIR.resolve()) for path in real_paths)
for path in real_paths:
    if path.exists():
        path.unlink()
if REAL_IMAGE_DIR.exists():
    REAL_IMAGE_DIR.rmdir()
assert not list(RUN_ROOT.rglob('*.png')), 'Image persistence is disabled.'
expected = {'run_config.json', 'distillationpp_audit_summary.csv', 'isolation_metrics.csv', 'isolation_metrics.json'}
actual = {path.name for path in RUN_ROOT.iterdir() if path.is_file()}
assert actual == expected, (actual, expected)
dpp = metrics_df.loc[metrics_df['lambda'] == 0.005].iloc[0]
print('[OK] Canonical DDIM parity, Distillation++ audit, and metric gates passed.')
print(f'[DELTA λ=0.005] FID={dpp["delta_fid"]:+.4f} | IS={dpp["delta_is_mean"]:+.4f} | CLIP-global={dpp["delta_clip_global_x100"]:+.4f} | time={dpp["delta_time_mean_sec"]:+.3f}s')
print('[OK] Metrics-only output:', RUN_ROOT)
