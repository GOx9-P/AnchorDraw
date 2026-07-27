# Colab SemanticDraw Smoke

```text
colab_semanticdraw_smoke/
|-- README.md                                      # File này: mô tả notebook Colab và cách đổi profile chạy.
`-- semanticdraw_sdxl_euler_smoke_bs2_colab.ipynb  # Notebook Colab chạy SDXL + Euler Discrete, mặc định smoke test bs2.
```

Notebook mặc định dùng:

```text
RUN_PROFILE    = "smoke_bs2"
COLAB_GPU_MODE = "low_vram"
Manifest       = Ours/test_sets/manifests/smoke/coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_smoke_bs2.jsonl
Model          = stabilityai/stable-diffusion-xl-base-1.0
Acceleration   = ByteDance/SDXL-Lightning/sdxl_lightning_4step_unet.safetensors
Sampler        = EulerDiscreteScheduler(timestep_spacing="trailing")
Resolution     = 1024x1024
```

Để chạy full 1073 sample sau khi smoke chạy ổn, sửa trong cell cấu hình:

```python
RUN_PROFILE = "full1073"
COLAB_GPU_MODE = "high_vram_24gb"
```

Nếu vẫn dùng Colab T4/L4 và muốn ưu tiên tránh OOM, giữ:

```python
COLAB_GPU_MODE = "low_vram"
```

Notebook sẽ sinh ảnh, overlay mask, `generation_summary.json`, export manifest cho metric và file zip tải về ở:

```text
/content/anchordraw_metric_exports/<EXPERIMENT_ID>__metric_export.zip
```
