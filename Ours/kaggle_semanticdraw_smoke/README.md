# Kaggle SemanticDraw SD1.5 + LCM

Thư mục này chứa notebook chạy baseline SemanticDraw trên Kaggle bằng cấu hình:

```text
model   = Stable Diffusion 1.5
sampler = LCM
metric  = FID, IS, CLIP(fg), CLIP(pg), Time(s)
```

Có 2 notebook chính:

```text
kaggle_semanticdraw_smoke/
|-- README.md                                      # File này: giải thích notebook và các file/folder liên quan.
|-- semanticdraw_sd15_smoke_kaggle.ipynb            # Notebook debug nhanh, mặc định mini128 gồm 128 sample.
`-- semanticdraw_sd15_lcm_full1073_kaggle.ipynb     # Notebook chạy full manifest 1073 sample để đo metric.
```

## Notebook Mini128

Notebook:

```text
semanticdraw_sd15_smoke_kaggle.ipynb
```

Manifest mặc định:

```text
Ours/test_sets/manifests/mini128/coco_val2017_multidiffusion_coco_all_512x512_mini128.jsonl
```

Output trên Kaggle:

```text
/kaggle/working/semanticdraw_mini128_outputs/
/kaggle/working/semanticdraw_mini128_metrics/
```

Mục tiêu của Mini128 là kiểm tra nhanh toàn bộ đường chạy trước khi benchmark chính thức.

## Notebook Full1073

Notebook:

```text
semanticdraw_sd15_lcm_full1073_kaggle.ipynb
```

Manifest mặc định:

```text
Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl
```

Output trên Kaggle:

```text
/kaggle/working/semanticdraw_full1073_outputs/
/kaggle/working/semanticdraw_full1073_metrics/
```

Notebook này dùng để chạy đủ 1073 sample hợp lệ của COCO val2017 theo manifest chính SD1.5 512x512.

## Luồng Chạy

Hai notebook đều đi qua cùng một pipeline:

```text
GitHub repo
-> COCO val2017 images + annotations
-> Ours COCORegionDataset/DataLoader
-> prompt/mask input đúng format SemanticDraw
-> baseline SemanticDrawPipeline SD1.5 + LCM
-> generated images + overlay preview
-> metrics JSON/CSV
```

`MAX_DISPLAY_RESULTS = 8` chỉ giới hạn số ảnh preview hiển thị trong notebook. Tổng số ảnh được sinh bằng số record trong manifest.

## File/Folder Notebook Đọc Từ Repo

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

## File/Folder Notebook Ghi Khi Chạy Trên Kaggle

```text
/kaggle/working/COCO/
```

COCO val2017 images và annotations nếu runtime chưa có sẵn.

```text
/kaggle/working/semanticdraw_mask_cache/
```

Cache mask đã resize để giảm chi phí decode/resize lại.

```text
/kaggle/working/semanticdraw_*_outputs/
|-- *_generated.png
|-- *_overlay.png
`-- generation_summary.json
```

Ảnh sinh ra, overlay mask và summary JSON của mỗi lần chạy.

```text
/kaggle/working/semanticdraw_*_metrics/
|-- *.json
`-- *.csv
```

Report metric sau khi notebook chạy xong.

## Ghi Chú

`BATCH_SIZE = 8` chỉ là số sample mỗi batch của dataloader. Baseline SemanticDraw hiện vẫn generate tuần tự từng ảnh, nên full 1073 sẽ lâu hơn Mini128 khoảng 8.4 lần nếu giữ cùng cấu hình.

Nếu chỉ muốn debug, chạy Mini128 trước. Nếu muốn đo metric gần benchmark chính hơn, chạy Full1073.
