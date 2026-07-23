# README cho `Ours/src/data`

Folder `Ours/src/data/` chứa toàn bộ phần data loading cho thí nghiệm nâng cấp SemanticDraw trên COCO. Mục tiêu chính của folder này là biến COCO annotations gốc thành input dùng được cho SemanticDraw:

```text
COCO annotations + manifest
  -> Dataset/DataLoader
  -> batch prompt + mask + metadata
  -> adapter sang input SemanticDraw
```

Nếu chỉ cần nhớ một câu: **file chính để chạy dataloader là `coco_region_dataset.py`; các file còn lại hoặc tạo config, hoặc build manifest, hoặc hỗ trợ decode mask/collate/adapter.**

## 1. Cây thư mục

```text
Ours/src/data/
|-- README.md
|-- __init__.py
|-- adapters.py
|-- build_manifest.py
|-- build_test_sets.py
|-- coco_mask_utils.py
|-- coco_profiles.py
|-- coco_region_collate.py
|-- coco_region_config.py
|-- coco_region_dataset.py
|-- coco_region_manifest.py
|-- coco_region_sampler.py
|-- download_coco.py
`-- visualize.py
```

Folder `__pycache__/` có thể xuất hiện sau khi chạy Python. Đây là cache tự sinh của Python, không phải source code chính.

## 2. File chính là file nào?

File chính để chạy dataloader là:

```text
coco_region_dataset.py
```

File này chứa:

```python
class COCORegionDataset(Dataset)
def build_coco_region_dataloader(...)
```

Trong experiment script, thường dùng như sau:

```python
from data.coco_profiles import multidiffusion_coco_all
from data.coco_region_dataset import build_coco_region_dataloader

config = multidiffusion_coco_all("annotations_trainval2017")
dataloader = build_coco_region_dataloader(config, shuffle=False)

for batch in dataloader:
    ...
```

Nói cách khác:

```text
coco_region_dataset.py = nơi biến manifest + COCO gốc thành PyTorch sample/batch thật.
```

## 3. Tầm ảnh hưởng của từng file

| File | Mức ảnh hưởng | Chức năng chính |
|---|---|---|
| `coco_region_dataset.py` | Rất chính | PyTorch `Dataset` và `DataLoader`. Đọc manifest, trace `annotation_ids`, decode mask, tạo sample và batch. |
| `coco_profiles.py` | Rất chính | Định nghĩa các preset/profile thí nghiệm như `multidiffusion_coco_all`, `semanticdraw_sd15`, `ours_weighted_mask`. |
| `coco_region_config.py` | Rất chính | Định nghĩa `COCORegionConfig`, nơi chứa toàn bộ tham số data loading và sampling. |
| `coco_region_sampler.py` | Chính khi build manifest | Filter COCO, chọn ảnh/object/caption hợp lệ, tạo record manifest. |
| `build_manifest.py` | Chính khi build manifest | CLI để chạy build manifest từ terminal. |
| `build_test_sets.py` | Chính khi build smoke/mini | CLI trích `smoke` và `mini32` từ full manifest chính, kèm summary report và preview optional. |
| `coco_region_manifest.py` | Hỗ trợ bắt buộc | Load COCO JSON, tạo index tra cứu `image_id`, `annotation_id`, `category_id`, `caption`. |
| `coco_mask_utils.py` | Hỗ trợ bắt buộc | Decode COCO segmentation thành binary mask, resize mask, chuyển mask sang tensor, rescale bbox. |
| `coco_region_collate.py` | Hỗ trợ bắt buộc khi batch | Gom nhiều sample có số object khác nhau thành batch bằng padding và `valid_regions`. |
| `adapters.py` | Cầu nối sang SemanticDraw | Chuyển output batch đầy đủ thành input gọn cho SemanticDraw pipeline. |
| `visualize.py` | Debug/EDA | Xuất ảnh overlay mask để kiểm tra sample, mask, prompt. |
| `download_coco.py` | Tiện ích phụ trợ | Hỗ trợ tải và extract COCO. Không cần dùng nếu data đã có sẵn. |
| `__init__.py` | API package | Export các class/function quan trọng để import gọn từ package `data`. |

## 4. Luồng chạy dataloader

Khi chạy dataloader, luồng chính là:

```text
experiment script
  -> coco_profiles.py
  -> coco_region_config.py
  -> coco_region_dataset.py
  -> coco_region_manifest.py
  -> coco_mask_utils.py
  -> coco_region_collate.py
  -> adapters.py
  -> SemanticDraw pipeline
```

Diễn giải:

1. `experiment script` chọn profile, ví dụ `multidiffusion_coco_all`.
2. `coco_profiles.py` tạo một `COCORegionConfig`.
3. `coco_region_config.py` giữ toàn bộ tham số như `target_size`, `batch_size`, `min_objects`, `max_objects`, path annotation, path manifest.
4. `coco_region_dataset.py` dùng config để đọc manifest và COCO annotations.
5. `coco_region_manifest.py` tạo index để tra nhanh `annotation_id -> annotation`.
6. `coco_mask_utils.py` decode `annotation["segmentation"]` thành mask tensor.
7. `coco_region_collate.py` gom nhiều sample thành batch.
8. `adapters.py` rút batch về format SemanticDraw cần.

## 5. Luồng build manifest

Manifest là file `.jsonl` nằm ở:

```text
Ours/data_manifests/
```

Nếu cần build hoặc rebuild manifest, luồng là:

```text
build_manifest.py
  -> coco_profiles.py
  -> coco_region_config.py
  -> coco_region_sampler.py
  -> coco_region_manifest.py
  -> save .jsonl manifest
```

Lệnh mẫu:

```powershell
$env:PYTHONPATH="Ours\src"
python -m data.build_manifest --coco-root annotations_trainval2017 --profile multidiffusion_coco_all --model-family sd15 --overwrite
```

Manifest hiện tại tương ứng với:

```text
coco_val2017_multidiffusion_coco_all_512x512_all.jsonl
```

Nó chứa toàn bộ `1073` ảnh hợp lệ sau filter, không random subset.

## 6. Vai trò chi tiết từng file

### 6.1. `coco_region_config.py`

File này định nghĩa dataclass:

```python
class COCORegionConfig
```

Đây là trung tâm chứa tham số.

Các nhóm tham số chính:

```text
Đường dẫn:
  coco_root
  split
  instances_json
  captions_json
  manifest_dir
  cache_dir

Profile/model:
  profile
  model_family
  target_size

Sampling/filter:
  min_objects
  max_objects
  truncate_objects
  exclude_categories
  min_mask_area_ratio
  drop_iscrowd

Prompt policy:
  prompt_template
  caption_policy
  object_policy

DataLoader performance:
  batch_size
  num_workers
  pin_memory
  persistent_workers
  prefetch_factor

Mask/cache:
  mask_resize_mode
  mask_dtype
  cache_resized_masks
```

Nếu muốn đổi tham số hệ thống, thường sửa qua profile hoặc override config, không sửa trực tiếp trong dataset.

Ví dụ:

```python
config = multidiffusion_coco_all(
    "annotations_trainval2017",
    batch_size=4,
    num_workers=4,
)
```

### 6.2. `coco_profiles.py`

File này định nghĩa các preset cấu hình.

Các profile hiện có:

```text
semanticdraw_sd15
semanticdraw_sdxl
semanticdraw_sd3
multidiffusion_coco_all
multidiffusion_coco_1k
ours_weighted_mask
ours_overlap_stress
```

Profile quan trọng nhất hiện tại:

```python
multidiffusion_coco_all(...)
```

Profile này đang set:

```text
profile = "multidiffusion_coco_all"
model_family = "sd15"
target_size = (512, 512)
subset_size = None
min_objects = 2
max_objects = 4
truncate_objects = False
exclude_categories = ("person",)
min_mask_area_ratio = 0.05
drop_iscrowd = True
prompt_template = "a {label}"
caption_policy = "first"
object_policy = "largest"
batch_size = 8
```

Nói dễ hiểu:

```text
coco_profiles.py = nơi đặt tên và đóng gói các protocol/config thí nghiệm.
```

### 6.3. `coco_region_sampler.py`

File này dùng khi tạo manifest.

Nó làm các việc:

1. Duyệt toàn bộ ảnh trong COCO validation.
2. Lấy annotation của từng ảnh.
3. Loại annotation không hợp lệ.
4. Chọn caption.
5. Tạo foreground prompt từ category.
6. Ghi record ra manifest.

Logic filter chính:

```text
drop iscrowd
exclude category "person"
area_ratio >= min_mask_area_ratio
phải có segmentation
số object hợp lệ nằm trong [min_objects, max_objects]
```

Với manifest hiện tại:

```text
min_objects = 2
max_objects = 4
min_mask_area_ratio = 0.05
exclude_categories = ("person",)
truncate_objects = False
```

Nếu ảnh có hơn 4 object hợp lệ và `truncate_objects=False`, ảnh đó bị loại.

### 6.4. `build_manifest.py`

Đây là CLI wrapper cho `coco_region_sampler.py`.

Nó cho phép build manifest từ terminal thay vì viết Python script riêng.

Ví dụ:

```powershell
$env:PYTHONPATH="Ours\src"
python -m data.build_manifest --coco-root annotations_trainval2017 --profile multidiffusion_coco_all --model-family sd15 --overwrite
```

Nên dùng file này khi:

```text
muốn tạo manifest mới
muốn đổi profile
muốn đổi min/max object
muốn đổi threshold area
muốn regenerate manifest sau khi sửa code sampling
```

### 6.5. `build_test_sets.py`

Đây là CLI để build các subset test nhỏ từ full manifest chính.

Nó tạo:

```text
Ours/test_sets/manifests/smoke/
Ours/test_sets/manifests/mini32/
Ours/test_sets/reports/
Ours/test_sets/previews/
```

Mục đích:

```text
smoke  = đúng 1 batch để test hệ thống không crash
mini32 = 32 sample để test loop/save/logging trước benchmark full
```

Lệnh mẫu:

```powershell
$env:PYTHONPATH="Ours\src"
python -m data.build_test_sets --repo-root . --manifest-dir Ours\data_manifests --output-dir Ours\test_sets --preview-limit 8
```

### 6.6. `coco_region_manifest.py`

File này xử lý việc đọc COCO JSON và manifest.

Nó tạo object:

```python
class COCOIndex
```

Bên trong có các index quan trọng:

```text
images_by_id
annotations_by_id
annotations_by_image
categories_by_id
captions_by_image
```

Index quan trọng nhất để trace mask:

```text
annotations_by_id[annotation_id] -> full COCO annotation
```

Nhờ đó dataset có thể đi từ:

```text
manifest["annotation_ids"]
  -> annotation gốc trong instances_val2017.json
  -> annotation["segmentation"]
  -> mask
```

### 6.7. `coco_mask_utils.py`

File này xử lý mask-level logic.

Các hàm chính:

```text
decode_coco_mask
resize_mask_nearest
mask_to_tensor
rescale_bbox_xyxy
cache_path_for_mask
stack_masks
```

Vai trò quan trọng nhất:

```text
COCO segmentation -> binary mask -> resized torch tensor
```

Mask được resize bằng nearest neighbor để giữ biên mask sắc nét, tránh tạo giá trị trung gian như khi dùng bilinear.

### 6.8. `coco_region_dataset.py`

Đây là file runtime chính.

Nó chứa:

```python
class COCORegionDataset(Dataset)
def build_coco_region_dataloader(...)
```

`COCORegionDataset.__getitem__()` trả về một sample giàu thông tin:

```python
{
    "sample_id": str,
    "image_id": int,
    "file_name": str,
    "background_prompt": str,
    "foreground_prompts": list[str],
    "category_names": list[str],
    "category_ids": list[int],
    "masks": Tensor[P, 1, H, W],
    "boxes_xyxy": Tensor[P, 4],
    "area_ratios": Tensor[P],
    "original_size": tuple[int, int],
    "target_size": tuple[int, int],
    "annotation_ids": list[int],
    "metadata": dict,
}
```

Trong đó:

```text
P = số object/region trong ảnh
H, W = target size, ví dụ 512x512
```

File này cần tồn tại dù đã có manifest, vì manifest chỉ chứa `annotation_ids`, còn dataset mới decode được mask thật từ `instances_val2017.json`.

### 6.9. `coco_region_collate.py`

File này xử lý batch.

Vấn đề:

```text
Ảnh A có 2 object
Ảnh B có 4 object
Ảnh C có 3 object
```

Không thể stack trực tiếp thành tensor nếu số object khác nhau.

Giải pháp:

```text
pad lên Pmax
tạo valid_regions để biết slot nào là thật
```

Output batch có dạng:

```python
{
    "masks": Tensor[B, Pmax, 1, H, W],
    "valid_regions": Tensor[B, Pmax],
    "category_ids": Tensor[B, Pmax],
    "boxes_xyxy": Tensor[B, Pmax, 4],
    ...
}
```

Với manifest hiện tại, `Pmax <= 4`.

### 6.10. `adapters.py`

File này chuyển batch đầy đủ thành input gọn cho SemanticDraw.

Dataset/DataLoader output nhiều field để phục vụ research:

```text
category_ids
category_names
bbox
area_ratios
metadata
annotation_ids
valid_regions
```

SemanticDraw chủ yếu cần:

```text
background_prompt
foreground prompts
masks
height
width
metadata
```

Adapter làm việc:

```text
lấy một item trong batch
bỏ padding dựa trên valid_regions
đổi masks sang float
trả format gọn cho SemanticDraw
```

Output của adapter:

```python
{
    "background_prompt": str,
    "prompts": list[str],
    "masks": Tensor[P, 1, H, W],
    "height": int,
    "width": int,
    "metadata": dict,
}
```

### 6.11. `visualize.py`

File này phục vụ debug/EDA.

Nó có thể tạo overlay mask để nhìn xem:

```text
mask có đúng object không
mask resize có ổn không
prompt có khớp category không
sample có bị lệch không
```

Không bắt buộc khi chạy experiment chính, nhưng rất hữu ích trước khi chạy hàng trăm/hàng nghìn sample.

### 6.12. `download_coco.py`

File này hỗ trợ tải COCO.

Nó không nằm trong runtime chính nếu bạn đã có data local.

Dùng khi:

```text
chưa có annotations
chưa có val2017 images
muốn setup COCO từ đầu
```

### 6.13. `__init__.py`

File này giúp import gọn hơn.

Thay vì import sâu từ từng file, ta có thể dùng package `data`.

Ví dụ:

```python
from data import multidiffusion_coco_all, build_coco_manifest
```

Nó cũng dùng lazy import cho một số object nặng để tránh import `torch`/dataset quá sớm khi không cần.

## 7. Khi nào cần chỉnh file nào?

| Muốn làm gì? | Chỉnh file nào? |
|---|---|
| Đổi batch size, workers, target size mặc định | `coco_profiles.py` hoặc override `COCORegionConfig` |
| Thêm profile thí nghiệm mới | `coco_profiles.py` |
| Đổi logic filter ảnh/object | `coco_region_sampler.py` |
| Đổi schema manifest | `coco_region_sampler.py`, `coco_region_manifest.py`, có thể cả `coco_region_dataset.py` |
| Build smoke/mini32 test set | `build_test_sets.py` |
| Đổi cách decode/resize mask | `coco_mask_utils.py` |
| Đổi output sample của dataset | `coco_region_dataset.py` |
| Đổi cách batch padding | `coco_region_collate.py` |
| Đổi format input đưa sang SemanticDraw | `adapters.py` |
| Thêm debug visualization | `visualize.py` |
| Thêm CLI build data | `build_manifest.py` |
| Thêm CLI build smoke/mini32 | `build_test_sets.py` |

## 8. Tóm tắt cực ngắn

```text
coco_profiles.py
  Chọn preset/config thí nghiệm.

coco_region_config.py
  Lưu toàn bộ tham số.

coco_region_sampler.py
  Build manifest từ COCO gốc.

build_test_sets.py
  Trích smoke/mini32 từ full manifest.

coco_region_dataset.py
  File chính chạy dataloader.

coco_region_manifest.py
  Load COCO JSON và tạo index tra cứu ID.

coco_mask_utils.py
  Decode segmentation thành mask tensor.

coco_region_collate.py
  Gom sample thành batch.

adapters.py
  Chuyển batch sang input SemanticDraw.

visualize.py
  Debug mask/prompt/sample.

download_coco.py
  Tải COCO nếu cần.
```

## 9. Công thức nhớ

```text
Manifest trả lời:
  dùng ảnh nào, object nào, caption nào, prompt nào.

Dataset trả lời:
  load sample đó thành tensor thật như thế nào.

Collate trả lời:
  gom nhiều sample khác số object thành batch như thế nào.

Adapter trả lời:
  đưa batch đó vào SemanticDraw theo format nào.
```
