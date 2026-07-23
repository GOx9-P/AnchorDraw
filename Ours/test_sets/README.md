# Test Sets

Folder này chứa các subset nhỏ được trích từ full manifest chính trong `Ours/data_manifests/`.

Các subset này dùng để validate hệ thống trước khi benchmark full `1073` sample. Chúng không thay đổi protocol data; chúng chỉ lấy ít dòng hơn từ full manifest.

## Cây thư mục

```text
test_sets/
|-- README.md                                                     # Giải thích smoke/mini32 và cách dùng.
|-- manifests/                                                    # Subset manifest dùng trực tiếp bởi dataloader.
|   |-- smoke/                                                    # Test cực nhanh, đúng 1 batch theo config mặc định.
|   |   |-- coco_val2017_multidiffusion_coco_all_512x512_smoke_bs8.jsonl      # SD1.5, 8 sample.
|   |   |-- coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_smoke_bs2.jsonl # SDXL, 2 sample.
|   |   `-- coco_val2017_multidiffusion_coco_all_sd3_1024x1024_smoke_bs2.jsonl  # SD3, 2 sample.
|   `-- mini32/                                                   # Test dài hơn smoke, dùng để check loop/save/memory.
|       |-- coco_val2017_multidiffusion_coco_all_512x512_mini32.jsonl          # SD1.5, 32 sample.
|       |-- coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_mini32.jsonl   # SDXL, 32 sample.
|       `-- coco_val2017_multidiffusion_coco_all_sd3_1024x1024_mini32.jsonl    # SD3, 32 sample.
|-- reports/                                                      # Summary JSON cho từng subset manifest.
|   |-- smoke/                                                    # Report cho smoke manifests.
|   `-- mini32/                                                   # Report cho mini32 manifests.
`-- previews/                                                     # Overlay ảnh COCO + mask, sinh local nếu môi trường có đủ torch/pycocotools.
    |-- smoke/                                                    # Preview smoke theo model family.
    `-- mini32/                                                   # Preview mini32 theo model family.
```

## Loại test set

| Loại | Số sample | Mục đích |
|---|---:|---|
| `smoke` SD1.5 | `8` | Đúng 1 batch với `batch_size=8`, dùng để kiểm tra dataloader/runner/model không crash. |
| `smoke` SDXL | `2` | Đúng 1 batch với `batch_size=2`, dùng để kiểm tra pipeline high-res SDXL. |
| `smoke` SD3 | `2` | Đúng 1 batch với `batch_size=2`, dùng để kiểm tra pipeline high-res SD3. |
| `mini32` | `32` | Test dài hơn smoke để kiểm tra loop, save output, logging, và lỗi memory theo thời gian. |

## Cách chọn sample

Các subset được chọn cố định từ full manifest chính:

```text
Ours/data_manifests/coco_val2017_multidiffusion_coco_all_*.jsonl
```

Smoke SD1.5:

```text
8 sample = 3 ảnh có 2 object + 3 ảnh có 3 object + 2 ảnh có 4 object
```

Smoke SDXL/SD3:

```text
2 sample = 1 ảnh có 2 object + 1 ảnh có 4 object
```

Mini32:

```text
32 sample = 16 ảnh có 2 object + 10 ảnh có 3 object + 6 ảnh có 4 object
```

Lý do chọn như vậy:

```text
kiểm tra padding trong collate
kiểm tra valid_regions
kiểm tra sample có số mask khác nhau
kiểm tra case tối đa 4 object của protocol
```

## Cách build lại

Từ root repo `AnchorDraw/`:

```powershell
$env:PYTHONPATH="Ours\src"
python -m data.build_test_sets --repo-root . --manifest-dir Ours\data_manifests --output-dir Ours\test_sets --preview-limit 8
```

Nếu đang dùng Python chưa cài `torch`, builder vẫn tạo được:

```text
manifests/
reports/
```

nhưng preview overlay sẽ không được export. Khi dùng đúng môi trường đã cài requirements, chạy lại lệnh trên để sinh thêm:

```text
previews/
```

## Ghi chú

- Không tạo smoke/mini riêng cho sampler như DDIM, LCM, Hyper-SD, Euler Discret, Flash Flow Match.
- Không tạo smoke/mini riêng cho method như MultiDiffusion, SemanticDraw, AnchorDraw.
- Không tạo smoke/mini riêng cho ablation module nếu ablation dùng cùng input.
- Runner/model sẽ dùng cùng subset manifest để so sánh công bằng giữa các method/sampler.
