# Kaggle MultiDiffusion Experiment Notebooks

```text
kaggle_multidiffusion_experiments/
|-- README.md                                      # Mô tả các notebook MultiDiffusion chạy trên Kaggle.
|-- multidiffusion_sd15_lcm_full1073_kaggle.ipynb  # MD + SD1.5 + LCM, adaptive boot/no-boot qua LCM_EXPERIMENT_PROFILE.
`-- multidiffusion_sd15_hypersd_full1073_kaggle.ipynb # MD + SD1.5 + Hyper-SD, đổi smoke/mini/full bằng RUN_PROFILE.
```

## Notebook MD + LCM

File chính:

```text
multidiffusion_sd15_lcm_full1073_kaggle.ipynb
```

Notebook này là bản adaptive cho hai cấu hình:

```python
LCM_EXPERIMENT_PROFILE = "g1_b1"  # CFG-MD-SD15-LCM-G1-B1: guidance=1.0, bootstrapping=1
LCM_EXPERIMENT_PROFILE = "g1_b0"  # CFG-MD-SD15-LCM-G1-B0: guidance=1.0, bootstrapping=0
```

Để chạy đúng config user đang hỏi:

```python
RUN_PROFILE = "full1073"
LCM_EXPERIMENT_PROFILE = "g1_b0"
```

Output, metrics và export được tách theo profile, ví dụ:

```text
multidiffusion_sd15_lcm_g1_b0_full1073_outputs
multidiffusion_sd15_lcm_g1_b0_full1073_metrics
anchordraw_metric_exports/multidiffusion_sd15_lcm_g1_b0_full1073
```

## RUN_PROFILE

```text
smoke    -> 8 sample, validate nhanh.
mini32   -> 32 sample, debug metric nhanh hơn full.
mini128  -> 128 sample, kiểm tra ổn định trung gian.
full1073 -> 1073 sample, benchmark chính thức.
```

## Cấu hình MD + LCM

```text
Method    = MultiDiffusion (MD)
Model     = runwayml/stable-diffusion-v1-5
Sampler   = LCMScheduler
Accel     = latent-consistency/lcm-lora-sdv1-5
Manifest  = Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl
Size      = 512x512
Metrics   = FID, IS, CLIP(fg), CLIP(bg), Time(s)
```

Core fusion bám theo `Baseline/MultiDiffusion-master/MultiDiffusion-master/region_based.py`. Phần thay đổi có chủ đích là thay sampler DDIM gốc bằng LCM/Hyper-SD để so cùng sampler với SemanticDraw trong bảng thí nghiệm.
