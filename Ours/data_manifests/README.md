# Manifest Index

Folder này chỉ giữ các **manifest chính** cho benchmark trên COCO `val2017`.

Manifest là danh sách sample cố định cho thí nghiệm. Nó không chứa ảnh gốc và không chứa mask pixel trực tiếp; nó lưu `annotation_ids` để dataloader trace về `instances_val2017.json`, decode `segmentation`, resize mask, rồi tạo tensor.

## Cây thư mục

```text
data_manifests/
|-- README.md                                                  # Index ngắn cho các manifest chính.
|-- EDA.md                                                     # Giải thích chi tiết cấu trúc manifest và cách dataloader dùng nó.
|-- coco_val2017_multidiffusion_coco_all_512x512_all.jsonl     # Manifest chính cho SD1.5 512x512.
|-- coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_all.jsonl # Manifest chính cho SDXL 1024x1024.
`-- coco_val2017_multidiffusion_coco_all_sd3_1024x1024_all.jsonl  # Manifest chính cho SD3 1024x1024.
```

## Ba manifest chính

| Manifest | Model family | Target size | Batch size mặc định | Số record | Dùng cho |
|---|---|---:|---:|---:|---|
| `coco_val2017_multidiffusion_coco_all_512x512_all.jsonl` | `sd15` | `512x512` | `8` | `1073` | Benchmark SD1.5. |
| `coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_all.jsonl` | `sdxl` | `1024x1024` | `2` | `1073` | Benchmark SDXL. |
| `coco_val2017_multidiffusion_coco_all_sd3_1024x1024_all.jsonl` | `sd3` | `1024x1024` | `2` | `1073` | Benchmark SD3. |

## Vì sao chỉ cần 3 manifest?

Các bảng experiment hiện tại khác nhau ở:

```text
method: MultiDiffusion / SemanticDraw / AnchorDraw
sampler: DDIM / LCM / Hyper-SD / Euler Discret / Flash Flow Match
ablation: Semantic Anchor / Adaptive Bilateral Masking / Distillation++
```

Những thứ đó là cấu hình của **runner/model**, không phải cấu hình của input data.

Manifest chỉ cần khác khi input contract khác:

```text
model_family
target_size
data protocol/filter
```

Với benchmark chính, data protocol/filter dùng chung là:

```text
multidiffusion_coco_all
```

Do đó chỉ cần 3 manifest tương ứng với 3 model family/resolution:

```text
SD1.5 -> 512x512
SDXL  -> 1024x1024
SD3   -> 1024x1024
```

## Protocol của manifest chính

Tất cả manifest chính đều dùng cùng protocol:

```text
multidiffusion_coco_all
```

Rule chính:

```text
COCO val2017
background prompt = caption COCO đầu tiên
foreground prompt = "a {category_name}"
exclude category "person"
drop iscrowd annotation
min_mask_area_ratio = 0.05
min_objects = 2
max_objects = 4
truncate_objects = false
subset_size = None
seed = null
```

Kết quả:

```text
1073 sample hợp lệ
2-object samples: 717
3-object samples: 258
4-object samples: 98
```

## Ghi chú

- Smoke test nên được trích từ 3 manifest chính này.
- Không tạo manifest riêng cho từng sampler.
- Không tạo manifest riêng cho từng method nếu method dùng cùng input.
- Không tạo manifest riêng cho ablation nếu ablation chỉ bật/tắt module trong model.
- Các manifest theo method/module riêng đã được loại khỏi folder này để tránh nhầm với benchmark chính.
