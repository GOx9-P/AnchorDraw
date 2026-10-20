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

# Notebook: Ours/kaggle_semantic_anchor_experiments/pladis_hypersd15_canonical_smoke8_kaggle.ipynb
# Notebook SHA256: 357e1021634589b6b6446e2f4fed5dbb996225e647007c959ee2cf31eedd226b

# --- Source notebook cell 2 ---
# 2. Resolve the exact manifest, select the first eight rows, and freeze the paired protocol.
import os, json, time, hashlib, importlib.util, importlib.metadata
from pathlib import Path
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
# terminal display provided above

MANIFEST_REL = Path('Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl')
MANIFEST = REPO_ROOT / MANIFEST_REL
assert MANIFEST.is_file(), MANIFEST
all_records = [json.loads(line) for line in MANIFEST.read_text(encoding='utf-8').splitlines() if line.strip()]
assert len(all_records) == 1073 and len({r['sample_id'] for r in all_records}) == 1073
records = all_records[:8]
assert len(records) == 8 and all(str(r.get('caption', '')).strip() for r in records)
RUN_DIR = WORK_DIR / 'pladis_hypersd15_canonical_smoke8'
RUN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ID = 'runwayml/stable-diffusion-v1-5'
HYPER_SD_REPO_ID = 'ByteDance/Hyper-SD'
HYPER_SD_WEIGHT_NAME = 'Hyper-SD15-4steps-lora.safetensors'
STEPS = 4
EXPECTED_TIMESTEPS = [999, 749, 499, 249]
GUIDANCE_SCALE = 0.0
PLADIS_SCALE = 1.5
BASE_SEED = 2024
HEIGHT = WIDTH = 512
MANIFEST_SHA256 = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
print('[OK] Manifest:', MANIFEST, '| first eight sample IDs:', [r['sample_id'] for r in records])

# --- Source notebook cell 3 ---
# 3. Load canonical Hyper-SD15 (no teacher and no regional pipeline).
if 'torchao' in sys.modules or importlib.util.find_spec('torchao') is not None:
    raise RuntimeError('torchao is still importable: restart the Kaggle session and Run All.')
assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator.'
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import hf_hub_download
from entmax import entmax15

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
device, dtype = torch.device('cuda:0'), torch.float16
try:
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, variant='fp16', safety_checker=None, requires_safety_checker=False).to(device)
    model_variant = 'fp16'
except Exception as error:
    print('[WARN] fp16 variant fallback:', error)
    pipe = DiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False).to(device)
    model_variant = None
lora_path = hf_hub_download(repo_id=HYPER_SD_REPO_ID, filename=HYPER_SD_WEIGHT_NAME)
pipe.load_lora_weights(lora_path, adapter_name='hyper_sd')
if hasattr(pipe, 'get_list_adapters'):
    assert 'hyper_sd' in str(pipe.get_list_adapters()), pipe.get_list_adapters()
if hasattr(pipe, 'set_adapters'):
    try:
        pipe.set_adapters(['hyper_sd'], adapter_weights=[1.0])
    except TypeError:
        pipe.set_adapters(['hyper_sd'])
pipe.fuse_lora(lora_scale=1.0)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing='trailing')
pipe.scheduler.set_timesteps(STEPS, device=device)
actual_timesteps = [int(t) for t in pipe.scheduler.timesteps.detach().cpu().tolist()]
assert actual_timesteps == EXPECTED_TIMESTEPS, actual_timesteps
assert pipe.scheduler.config.prediction_type == 'epsilon'
pipe.set_progress_bar_config(disable=True)
pipe.unet.eval()
print('[OK] Hyper-SD15; DDIM trailing timesteps:', actual_timesteps, '| guidance:', GUIDANCE_SCALE)

# --- Source notebook cell 4 ---
# 4. Port only PLADIS's SDXL cross-attention equation, not its whole pipeline.
# At scale=0 we call the original processor verbatim for an exact baseline parity gate.
# At scale>0 the Q/K/V preparation and output projection match PLADIS's SparseAttnProcessor.
import torch.nn.functional as F
test_generator = torch.Generator(device=device.type).manual_seed(10)
q = torch.randn((1, 2, 4, 8), generator=test_generator, device=device, dtype=torch.float32)
k = torch.randn((1, 2, 5, 8), generator=test_generator, device=device, dtype=torch.float32)
v = torch.randn((1, 2, 5, 8), generator=test_generator, device=device, dtype=torch.float32)
logits_test = (q @ k.transpose(-2, -1)) * (8 ** -0.5)
mixed_zero = 0.0 * entmax15(logits_test, dim=-1) + torch.softmax(logits_test, dim=-1)
torch.testing.assert_close(mixed_zero @ v, F.scaled_dot_product_attention(q, k, v), atol=2e-5, rtol=2e-5)
del q, k, v, logits_test, mixed_zero
class PLADISCrossAttnProcessor:
    def __init__(self, native, scale):
        self.native = native
        self.scale = float(scale)
        self.calls = 0
        self.modified_calls = 0
        self.attention_delta_sum = 0.0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        self.calls += 1
        if self.scale == 0.0:
            return self.native(attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs)
        if args or kwargs:
            raise RuntimeError(f'Unexpected attention processor arguments: {args}, {list(kwargs)}')
        if encoder_hidden_states is None:
            raise RuntimeError('PLADIS must only replace text cross-attention (attn2).')
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        logits = (query @ key.transpose(-2, -1)) * (head_dim ** -0.5)
        if attention_mask is not None:
            logits = logits + attention_mask
        dense = torch.softmax(logits, dim=-1)
        sparse = entmax15(logits, dim=-1)
        mixed = self.scale * sparse + (1.0 - self.scale) * dense
        if not torch.isfinite(mixed).all():
            raise FloatingPointError('Non-finite PLADIS attention weights.')
        self.attention_delta_sum += float((mixed - dense).abs().mean().item())
        self.modified_calls += 1
        hidden_states = mixed @ value
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor

original_processors = dict(pipe.unet.attn_processors)
cross_names = sorted(name for name in original_processors if '.attn2.processor' in name)
assert cross_names and all(name.startswith(('down_blocks.', 'mid_block.', 'up_blocks.')) for name in cross_names), cross_names
assert all(type(original_processors[name]).__name__ == 'AttnProcessor2_0' for name in cross_names), 'Unexpected native processor: inspect before replacing it.'
assert all('.attn1.processor' not in name for name in cross_names)
@contextmanager
def installed_pladis(scale):
    wrappers = {name: PLADISCrossAttnProcessor(original_processors[name], scale) for name in cross_names}
    pipe.unet.set_attn_processor({name: wrappers.get(name, processor) for name, processor in original_processors.items()})
    try:
        yield wrappers
    finally:
        pipe.unet.set_attn_processor(dict(original_processors))
        assert all(pipe.unet.attn_processors[name] is processor for name, processor in original_processors.items()), 'Original attention processors were not restored.'
print('[OK] PLADIS target attn2 modules:', len(cross_names), '| groups:', sorted({name.split('.')[0] for name in cross_names}))

# --- Source notebook cell 5 ---
# 5. Paired smoke8: native vs installed scale=0 vs scale=1.5.
def run_image(prompt, latent):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        image = pipe(prompt=prompt, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS, guidance_scale=GUIDANCE_SCALE, eta=0.0, latents=latent.clone(), output_type='pil').images[0]
    torch.cuda.synchronize()
    return image, time.perf_counter() - start

rows, image_pairs = [], []
for index, record in enumerate(records):
    seed = BASE_SEED + index
    generator = torch.Generator(device=device.type).manual_seed(seed)
    latent = randn_tensor((1, pipe.unet.config.in_channels, HEIGHT // pipe.vae_scale_factor, WIDTH // pipe.vae_scale_factor), generator=generator, device=device, dtype=dtype)
    prompt = str(record['caption'])
    native_image, native_sec = run_image(prompt, latent)
    with installed_pladis(0.0) as zero_wrappers:
        zero_image, zero_sec = run_image(prompt, latent)
        zero_calls = sum(wrapper.calls for wrapper in zero_wrappers.values())
        assert sum(wrapper.modified_calls for wrapper in zero_wrappers.values()) == 0
    native_pixels = np.asarray(native_image, dtype=np.uint8)
    zero_pixels = np.asarray(zero_image, dtype=np.uint8)
    assert np.array_equal(native_pixels, zero_pixels), f'scale=0 pixel parity failed: {record["sample_id"]}'
    assert zero_calls == STEPS * len(cross_names), (zero_calls, STEPS, len(cross_names))
    with installed_pladis(PLADIS_SCALE) as active_wrappers:
        pladis_image, pladis_sec = run_image(prompt, latent)
        active_calls = sum(wrapper.modified_calls for wrapper in active_wrappers.values())
        attention_delta = sum(wrapper.attention_delta_sum for wrapper in active_wrappers.values()) / active_calls
    assert active_calls == STEPS * len(cross_names), (active_calls, STEPS, len(cross_names))
    assert attention_delta > 0, 'PLADIS attention was a no-op.'
    pladis_pixels = np.asarray(pladis_image, dtype=np.uint8)
    absolute_delta = np.abs(native_pixels.astype(np.int16) - pladis_pixels.astype(np.int16))
    row = {'sample_id': record['sample_id'], 'seed': seed, 'native_vs_scale0_equal': True, 'cross_attn_calls': active_calls, 'mean_attention_abs_delta': attention_delta, 'pixel_mae': float(absolute_delta.mean()), 'changed_pixel_fraction': float(np.mean(np.any(absolute_delta != 0, axis=-1))), 'native_sec': native_sec, 'scale0_sec': zero_sec, 'pladis_sec': pladis_sec}
    rows.append(row)
    image_pairs.append((native_image, pladis_image))
    print(f'[{index+1}/8] parity=exact | attn_calls={active_calls} | attn_delta={attention_delta:.6f} | pixel_MAE={row["pixel_mae"]:.3f}')
assert len(rows) == 8 and any(row['pixel_mae'] > 0 for row in rows), 'No image changed: investigate processor application.'
print('[OK] All eight scale=0 images match native pixel-for-pixel; positive scale changed attention and image output.')

# --- Source notebook cell 6 ---
# 6. Compact visual and audit export. No FID/IS claim from smoke8.
results = pd.DataFrame(rows)
display(results[['sample_id', 'native_vs_scale0_equal', 'cross_attn_calls', 'mean_attention_abs_delta', 'pixel_mae', 'changed_pixel_fraction', 'native_sec', 'pladis_sec']])
results.to_csv(RUN_DIR / 'smoke8_results.csv', index=False)
tile_w, tile_h, label_h = WIDTH, HEIGHT, 28
sheet = Image.new('RGB', (2 * tile_w, len(image_pairs) * (tile_h + label_h)), 'white')
draw = ImageDraw.Draw(sheet)
for index, (native_image, pladis_image) in enumerate(image_pairs):
    y = index * (tile_h + label_h)
    draw.text((8, y + 7), f'{index:02d} native / scale=0', fill='black')
    draw.text((tile_w + 8, y + 7), f'{index:02d} PLADIS scale={PLADIS_SCALE}', fill='black')
    sheet.paste(native_image, (0, y + label_h))
    sheet.paste(pladis_image, (tile_w, y + label_h))
sheet.save(RUN_DIR / 'paired_contact_sheet.png')
git_result = subprocess.run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True, capture_output=True)
run_config = {'experiment': 'PLADIS-HYPSD15-CANONICAL-SMOKE8', 'purpose': 'installation/parity diagnostic, not quality benchmark', 'anchor_draw': False, 'distillation_pp': False, 'model_id': MODEL_ID, 'model_variant': model_variant, 'lora_repo': HYPER_SD_REPO_ID, 'lora_weight': HYPER_SD_WEIGHT_NAME, 'lora_scale': 1.0, 'scheduler': 'DDIMScheduler', 'timestep_spacing': 'trailing', 'timesteps': actual_timesteps, 'num_inference_steps': STEPS, 'guidance_scale': GUIDANCE_SCALE, 'pladis_scale': PLADIS_SCALE, 'sparse_activation': 'entmax15', 'pladis_source': 'https://github.com/cubeyoung/PLADIS/blob/main/pipeline/pipeline_sdxl.py', 'pladis_layers': cross_names, 'manifest': str(MANIFEST_REL), 'manifest_sha256': MANIFEST_SHA256, 'sample_ids': [r['sample_id'] for r in records], 'base_seed': BASE_SEED, 'resolution': [HEIGHT, WIDTH], 'repo_commit': git_result.stdout.strip() if git_result.returncode == 0 else None, 'torch_version': torch.__version__, 'diffusers_version': importlib.metadata.version('diffusers'), 'entmax_version': importlib.metadata.version('entmax')}
(RUN_DIR / 'run_config.json').write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding='utf-8')
display(sheet.resize((512, len(image_pairs) * (tile_h + label_h) // 2)))
print('[OK] Saved:', RUN_DIR, '| scale=0 exact parity: 8/8 | changed images:', sum(row['pixel_mae'] > 0 for row in rows), '/8')
