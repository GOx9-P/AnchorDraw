# Colab SemanticDraw Smoke

```text
colab_semanticdraw_smoke/
|-- README.md
|   `-- File này. Ghi chú chức năng notebook Colab trong folder.
|-- semanticdraw_sd15_lcm_semanticfull_colab.ipynb
|   `-- SemanticDraw SD1.5 + LCM. Tự build manifest COCO val2017 theo
|       profile semanticdraw_sd15, hỗ trợ smoke/full_val2017 và
|       low_vram/standard.
|-- semanticdraw_sdxl_euler_smoke_bs2_colab.ipynb
|   `-- SemanticDraw SDXL + SDXL-Lightning 4-step UNet + Euler Discrete
|       trailing. Dùng để validate smoke_bs2, có option cho full1073 và
|       chế độ VRAM thấp/cao.
`-- semanticdraw_sd3_flashflowmatch_full1073_colab.ipynb
    `-- SemanticDraw SD3 + Flash Flow Match. Dùng cho các lượt validate/full
        1024x1024 của SD3 trên Colab.
```

## semanticdraw_sd15_lcm_semanticfull_colab.ipynb

```text
Model        : runwayml/stable-diffusion-v1-5
Sampler      : LCMScheduler bên trong baseline SemanticDrawPipeline
Acceleration : latent-consistency/lcm-lora-sdv1-5
Resolution   : 512x512
Baseline     : Baseline/semantic-draw-main/src/model/pipeline_semantic_draw.py
Data profile : semanticdraw_sd15
Default mode : RUN_PROFILE = "full_val2017", LOW_VRAM = False
Input API    : prompts = [caption] + foreground prompts
               masks   = [background mask] + foreground masks
```

Notebook này dùng để chạy SemanticDraw trên protocol COCO val2017 rộng hơn,
không phải manifest 1073 mẫu theo filter `multidiffusion_coco_all`. Nó giữ mọi
ảnh có ít nhất một object segmentation hợp lệ và không phải crowd. Với annotation
COCO val2017 hiện tại, profile này có khoảng 4952 ảnh hợp lệ.

Mặc định notebook hiện đã được đặt để chạy full trên A100 80GB:

```python
RUN_PROFILE = "full_val2017"
LOW_VRAM = False
REBUILD_MANIFEST = False
RUN_SANITY_CHECK = True
RUN_METRICS = False
RUN_EXPORT_ZIP = True
SKIP_EXISTING = True
```

Khi `LOW_VRAM=False`, notebook tự suy ra:

```python
MAX_OBJECTS_PER_IMAGE = 80
BATCH_SIZE = 8
METRIC_BATCH_SIZE = 8
CLIP_BATCH_SIZE = 16
MAX_DISPLAY_RESULTS = 4
```

`BATCH_SIZE` ở đây là batch của dataloader. Pipeline SemanticDraw baseline vẫn
sinh từng sample một trong vòng lặp generation.

Nếu chỉ muốn validate nhanh bằng smoke test trên GPU yếu hơn, đổi lại:

```python
RUN_PROFILE = "smoke"
LOW_VRAM = True
```

Output được lưu trong runtime Colab:

```text
/content/anchordraw_runs/<RUN_ID>/
|-- generated_images/
|-- mask_overlays/
|-- manifests/
|-- mask_cache/
|-- metrics/
|-- generation_summary.json
|-- metric_generated_manifest.jsonl
|-- metric_generated_manifest.csv
`-- export_summary.json
```

File zip để tải về nằm cạnh folder run:

```text
/content/anchordraw_runs/<RUN_ID>__metric_export.zip
```

Hãy tải file zip này trước khi kết thúc session Colab, vì output local trong
`/content` sẽ mất sau khi runtime bị reset.
