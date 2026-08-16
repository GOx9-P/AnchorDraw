# Colab MultiDiffusion Experiment Notebooks

```text
colab_multidiffusion_experiments/
|-- README.md                                           # File này mô tả các notebook Colab cho MultiDiffusion.
|-- multidiffusion_sd15_ddim_ref_colab.ipynb            # MD-SD15-DDIM-REF-F1073: MultiDiffusion Ref., SD1.5, DDIM 50-step.
|-- multidiffusion_sdxl_ddim_ref_colab.ipynb            # MD-SDXL-DDIM-REF-F1073: MultiDiffusion Ref., SDXL, DDIM 50-step.
|-- multidiffusion_sdxl_euler_colab.ipynb               # MD-SDXL-EULER-NAIVE-F1073: MultiDiffusion naive, SDXL-Lightning, Euler trailing.
`-- multidiffusion_sd3_flashflowmatch_naive_colab.ipynb # MD-SD3-FFM-NAIVE-F1073: MultiDiffusion naive, SD3 Medium, Flash-SD3.
```

## multidiffusion_sd15_ddim_ref_colab.ipynb

```text
Exp ID      = MD-SD15-DDIM-REF-F1073
Method      = MultiDiffusion Ref.
Model       = runwayml/stable-diffusion-v1-5
Resolution  = 512x512
Sampler     = DDIMScheduler
Steps       = 50
Guidance    = 7.5
Bootstrap   = 20 random-color background latent
Core source = Baseline/MultiDiffusion-master/MultiDiffusion-master/region_based.py
Wrapper     = Ours/src/baselines/multidiffusion_ddim.py
```

## multidiffusion_sdxl_ddim_ref_colab.ipynb

```text
Exp ID      = MD-SDXL-DDIM-REF-F1073
Method      = MultiDiffusion Ref.
Model       = stabilityai/stable-diffusion-xl-base-1.0
Resolution  = 1024x1024
Sampler     = DDIMScheduler
Steps       = 50
Guidance    = 7.5
Bootstrap   = 20 random-color background latent
View config = mặc định full latent window 128, stride 128, tức 1 view ở 1024x1024
Core source = Baseline/MultiDiffusion-master/MultiDiffusion-master/region_based.py
Wrapper     = Ours/src/baselines/multidiffusion_sdxl_ddim.py
```

Notebook này có các mode chính:

```python
RUN_PROFILE = "smoke_bs2"            # smoke_bs2, mini32, mini128, full1073
COLAB_GPU_MODE = "low_vram"          # low_vram, high_vram_24gb, a100_80gb
DDIM_EXPERIMENT_PROFILE = "ref_g75_b20_full_view"
RUN_METRICS = False
```

Khi chạy benchmark chính trên A100 80GB, đổi:

```python
RUN_PROFILE = "full1073"
COLAB_GPU_MODE = "a100_80gb"
RUN_METRICS = True
```

Profile chính dùng full-view để bám mức thời gian benchmark của paper. Profile `diagnostic_g75_b20_v64s8` vẫn có trong notebook để trace sliding-window `64/8 = 81 views`, nhưng không dùng làm benchmark chính vì A100 smoke test đã cho thấy nó có thể mất hàng chục phút mỗi ảnh.

## multidiffusion_sdxl_euler_colab.ipynb

```text
Exp ID       = MD-SDXL-EULER-NAIVE-F1073
Method       = MultiDiffusion
Variant      = MD-naive runtime-safe
Model        = stabilityai/stable-diffusion-xl-base-1.0
Acceleration = ByteDance/SDXL-Lightning/sdxl_lightning_4step_unet.safetensors
Sampler      = EulerDiscreteScheduler(timestep_spacing="trailing")
Resolution   = 1024x1024
View config  = native_full_v128, tức 1 full latent view 128x128
Wrapper      = Ours/src/baselines/multidiffusion_sdxl_euler.py
```

Notebook SDXL Euler hiện dùng profile `native_full_v128` cho ảnh 1024x1024. Profile panorama `64/8` được giữ để debug sliding-window nhưng không dùng làm main metric vì đã tạo mosaic trong smoke test.

## multidiffusion_sd3_flashflowmatch_naive_colab.ipynb

```text
Exp ID       = MD-SD3-FFM-NAIVE-F1073
Method       = MultiDiffusion
Variant      = MD-naive runtime-safe
Model        = stabilityai/stable-diffusion-3-medium-diffusers
Acceleration = jasperai/flash-sd3
Sampler      = FlashFlowMatchEulerDiscreteScheduler
Resolution   = 1024x1024
Steps/index  = schedule_steps=50, t_index_list=[0, 4, 12, 25, 37]
Guidance     = 0.0
Bootstrap    = 2 random-color background latent
View config  = native_full_v128, tức 1 full latent view 128x128 mặc định
Wrapper      = Ours/src/baselines/multidiffusion_sd3_flashflowmatch.py
```

Notebook này cần `HF_TOKEN` có quyền truy cập SD3 Medium trên Hugging Face. Mặc định notebook chạy `smoke_bs2` và không lưu Drive. Khi muốn benchmark full, đổi:

```python
RUN_PROFILE = "full1073"
COLAB_GPU_MODE = "a100_80gb"
RUN_METRICS_AFTER_GENERATION = True
SAVE_EXPORT_TO_GOOGLE_DRIVE = True  # chỉ bật nếu muốn copy export sang Google Drive
```

## Quy Ước Input Cho MultiDiffusion

MultiDiffusion khác SemanticDraw ở chỗ nó cần mask nền nằm chung trong danh sách mask:

```python
masks = [background_mask] + foreground_masks
prompts = [background_prompt] + foreground_prompts
```

Dataloader của `Ours/src/data` chỉ trả foreground masks từ COCO. Vì vậy các notebook MultiDiffusion tự tạo:

```python
background_mask = 1 - union(foreground_masks)
```

rồi ghép background mask vào đầu danh sách trước khi gọi wrapper. Đây là khác biệt quan trọng so với SemanticDraw, nơi pipeline nhận `background_prompt` riêng và chỉ nhận foreground masks.
