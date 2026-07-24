# Kaggle SemanticDraw Mini32 Test

Thư mục này chứa notebook chạy thử end-to-end baseline SemanticDraw trên Kaggle, sau đó đo metric sanity check ngay trong notebook. Mặc định notebook chạy cấu hình:

```text
model   = Stable Diffusion 1.5
sampler = LCM
input   = mini32 manifest, 32 sample, 512x512
metric  = FID, IS, CLIP(fg), CLIP(pg), Time(s)
```

Manifest mặc định:

```text
Ours/test_sets/manifests/mini32/coco_val2017_multidiffusion_coco_all_512x512_mini32.jsonl
```

Notebook này chưa dùng để benchmark chính thức. Mục tiêu là kiểm tra toàn bộ đường chạy:

```text
GitHub repo
-> COCO val2017 images + annotations
-> Ours COCORegionDataset/DataLoader
-> prompt/mask input đúng format SemanticDraw
-> baseline SemanticDrawPipeline SD1.5 + LCM
-> generated images + overlay preview
-> metrics JSON/CSV
```

## Cây thư mục

```text
kaggle_semanticdraw_smoke/
|-- README.md                                   # File này: giải thích notebook và các file/folder liên quan.
`-- semanticdraw_sd15_smoke_kaggle.ipynb         # Notebook Kaggle chạy SemanticDraw SD1.5 + LCM, mặc định mini32.
```

## File/folder notebook đọc từ repo

Notebook không tự sửa các file dưới đây, nhưng có đọc/import chúng khi chạy:

```text
Ours/test_sets/manifests/mini32/
`-- coco_val2017_multidiffusion_coco_all_512x512_mini32.jsonl
```

Manifest mini32 cung cấp 32 record input COCO cho test.

```text
Ours/src/data/
|-- __init__.py
|-- adapters.py
|-- coco_region_config.py
|-- coco_region_dataset.py
|-- coco_region_collate.py
|-- coco_region_manifest.py
|-- coco_mask_utils.py
`-- visualize.py
```

Package data loader tạo batch, mask tensor, metadata và overlay preview cho notebook.

```text
Ours/src/metrics/
```

Package metric đo FID, IS, CLIP(fg), CLIP(pg) và Time(s) từ ảnh đã sinh.

```text
Baseline/semantic-draw-main/src/model/pipeline_semantic_draw.py
Baseline/semantic-draw-main/src/model/semantic_draw.py
```

Source baseline chạy SemanticDraw SD1.5. Với `sd_version="1.5"`, baseline tự dùng `LCMScheduler` và gắn LCM LoRA.

## File/folder notebook ghi khi chạy trên Kaggle

Các output này nằm trong Kaggle runtime, không nằm trong repo:

```text
/kaggle/working/COCO/
```

COCO val2017 images và annotations nếu runtime chưa có sẵn.

```text
/kaggle/working/semanticdraw_mask_cache/
```

Cache mask đã resize để giảm chi phí decode/resize lại.

```text
/kaggle/working/semanticdraw_mini32_outputs/
|-- *_generated.png
|-- *_overlay.png
`-- generation_summary.json
```

Ảnh sinh ra, overlay mask và summary JSON của lần chạy mini32.

```text
/kaggle/working/semanticdraw_mini32_metrics/
|-- semanticdraw_sd15_lcm_mini32_metrics.json
`-- semanticdraw_sd15_lcm_mini32_metrics.csv
```

Report metric sau khi notebook chạy xong.

## Ghi chú đổi về smoke8

Nếu muốn chạy test ngắn hơn, đổi trong cell cấu hình:

```python
SMOKE_MANIFEST = REPO_ROOT / "Ours" / "test_sets" / "manifests" / "smoke" / "coco_val2017_multidiffusion_coco_all_512x512_smoke_bs8.jsonl"
OUTPUT_DIR = Path("/kaggle/working/semanticdraw_smoke_outputs")
METRICS_OUTPUT_DIR = Path("/kaggle/working/semanticdraw_smoke_metrics")
METRICS_REPORT_PREFIX = "semanticdraw_sd15_lcm_smoke_metrics"
```

`BATCH_SIZE = 8` chỉ là số sample mỗi batch của dataloader. Tổng số ảnh được sinh bằng số record trong manifest.
