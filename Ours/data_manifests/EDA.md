# EDA cho `Ours/data_manifests`

Tài liệu này giải thích folder `Ours/data_manifests/`: nó chứa gì, cấu trúc từng dòng dữ liệu ra sao, khác gì so với COCO gốc, và dataloader trong `Ours/src/data/` dùng nó như thế nào để đưa input vào hệ thống SemanticDraw.

Scope của file này là cho người mới đọc. Nếu chỉ cần nhớ một câu: **manifest không phải COCO gốc, mà là một bản chỉ mục thí nghiệm đã được filter và đóng gói lại từ COCO validation để hệ thống có thể load nhiều sample một cách cố định, tái lập, và đúng protocol.**

## 1. Cấu trúc folder hiện tại

```text
Ours/data_manifests/
├── EDA.md
└── coco_val2017_multidiffusion_coco_all_512x512_all.jsonl
```

Trong đó:

- `EDA.md`: file giải thích này.
- `coco_val2017_multidiffusion_coco_all_512x512_all.jsonl`: manifest đang dùng cho thí nghiệm COCO validation theo protocol `multidiffusion_coco_all`.

Hiện tại folder này chỉ có một manifest. Về sau nếu tạo thêm thí nghiệm khác, folder này có thể có nhiều file `.jsonl`, ví dụ cho `sdxl`, `sd3`, profile overlap stress, weighted mask, hoặc subset đặc biệt.

## 2. Manifest là gì?

Manifest là một file **JSONL**.

JSONL nghĩa là:

- Mỗi dòng là một JSON object hoàn chỉnh.
- Không có dấu phẩy giữa các dòng.
- Không bọc toàn bộ file trong mảng `[...]`.
- Có thể đọc từng dòng một, rất tiện cho dataset lớn.

Ví dụ ý tưởng:

```json
{"sample_id": "...", "image_id": 776, "...": "..."}
{"sample_id": "...", "image_id": 785, "...": "..."}
{"sample_id": "...", "image_id": 802, "...": "..."}
```

Với file hiện tại, **mỗi dòng tương ứng với một ảnh COCO hợp lệ được đưa vào thí nghiệm**. Mỗi ảnh đó có một background prompt và từ 2 đến 4 foreground object prompts đi kèm mask.

## 3. Ý nghĩa tên file manifest

File hiện tại:

```text
coco_val2017_multidiffusion_coco_all_512x512_all.jsonl
```

Tên này có thể đọc như sau:

| Thành phần | Ý nghĩa |
|---|---|
| `coco` | Dataset gốc là COCO. |
| `val2017` | Split sử dụng là COCO validation 2017. |
| `multidiffusion_coco_all` | Protocol/filter đang bám theo setup MultiDiffusion-style cho COCO, nhưng lấy toàn bộ sample hợp lệ. |
| `512x512` | Mask sẽ được resize về kích thước target `[512, 512]`, phù hợp với SD 1.5. |
| `all` | Không lấy subset random; giữ toàn bộ ảnh hợp lệ sau filter. |
| `.jsonl` | Mỗi dòng là một JSON record độc lập. |

Điểm quan trọng: file này không có `_seed42` trong tên vì hiện tại không dùng random sampling. Ta lấy toàn bộ `1073` ảnh hợp lệ, theo thứ tự tăng dần của `image_id`.

## 4. Thống kê nhanh của manifest hiện tại

Manifest hiện tại có:

| Thống kê | Giá trị |
|---|---:|
| Số sample/ảnh hợp lệ | `1073` |
| Tổng số object annotation được dùng | `2600` |
| Target size | `[512, 512]` |
| `model_family` | `sd15` |
| `profile` | `multidiffusion_coco_all` |
| `protocol` | `multidiffusion_coco_all` |
| `seed` | `null` |

Phân bố số object hợp lệ trên mỗi ảnh:

| Số object trong một ảnh | Số ảnh |
|---:|---:|
| `2` | `717` |
| `3` | `258` |
| `4` | `98` |

Top 15 category xuất hiện nhiều nhất trong manifest:

| Rank | Category | Số lần xuất hiện |
|---:|---|---:|
| 1 | `dining table` | `241` |
| 2 | `cat` | `119` |
| 3 | `bowl` | `117` |
| 4 | `couch` | `106` |
| 5 | `chair` | `102` |
| 6 | `car` | `92` |
| 7 | `laptop` | `80` |
| 8 | `tv` | `80` |
| 9 | `pizza` | `76` |
| 10 | `dog` | `74` |
| 11 | `zebra` | `72` |
| 12 | `bed` | `70` |
| 13 | `sandwich` | `70` |
| 14 | `elephant` | `68` |
| 15 | `teddy bear` | `62` |

## 5. Manifest khác gì so với COCO gốc?

COCO gốc nằm trong các file annotation như:

```text
annotations_trainval2017/annotations/
├── instances_val2017.json
└── captions_val2017.json
```

COCO gốc tổ chức dữ liệu theo nhiều bảng lớn:

| File | Thành phần chính | Ý nghĩa |
|---|---|---|
| `instances_val2017.json` | `images` | Metadata của từng ảnh: `id`, `file_name`, `height`, `width`, ... |
| `instances_val2017.json` | `annotations` | Mỗi object annotation: `id`, `image_id`, `category_id`, `segmentation`, `bbox`, `area`, ... |
| `instances_val2017.json` | `categories` | Bảng tra category: `id` -> `name`, `supercategory`. |
| `captions_val2017.json` | `annotations` | Caption của ảnh: `id`, `image_id`, `caption`. |

Manifest trong `Ours/data_manifests` không thay thế COCO gốc. Nó chỉ là bản đã được xử lý lại cho thí nghiệm:

| COCO gốc | Manifest của ta |
|---|---|
| Dữ liệu nằm rải ở nhiều list: `images`, `annotations`, `categories`, `captions`. | Mỗi dòng gom thông tin cần thiết cho một sample thí nghiệm. |
| Một ảnh thường có nhiều caption. | Manifest chọn đúng một caption theo `caption_policy`. |
| Một ảnh có thể có nhiều object annotation, cả object nhỏ, crowd, person, v.v. | Manifest chỉ giữ object thỏa điều kiện filter. |
| Category chỉ có dạng `category_id`, muốn biết tên phải tra bảng `categories`. | Manifest lưu sẵn `category_ids`, `category_names`, và `foreground_prompts`. |
| Mask nằm trong trường `segmentation` của annotation gốc. | Manifest chỉ lưu `annotation_ids`; mask thật sẽ được decode lại từ `instances_val2017.json`. |
| Không có thông tin protocol thí nghiệm. | Manifest lưu `profile`, `protocol`, `sampling`, `target_size`, `version`. |

Nói ngắn gọn: **manifest là bản index/recipe cho thí nghiệm**, không phải bản copy đầy đủ của COCO.

## 6. Manifest có thay đổi dữ liệu COCO gốc không?

Không.

Manifest không sửa `instances_val2017.json` và cũng không sửa `captions_val2017.json`.

Các ID vẫn là ID gốc của COCO:

- `image_id` vẫn là ID ảnh gốc trong COCO.
- `annotation_ids` vẫn là ID object annotation gốc trong COCO.
- `category_ids` vẫn là ID category gốc trong COCO.
- `caption_id` vẫn là ID caption annotation gốc trong COCO.

Manifest chỉ thêm các field tiện cho thí nghiệm, ví dụ:

- `sample_id`
- `category_names`
- `foreground_prompts`
- `area_ratios`
- `target_size`
- `profile`
- `protocol`
- `sampling`

Vì vậy, nếu cần kiểm chứng lại một sample, ta luôn có thể lấy `image_id`, `annotation_ids`, `caption_id`, `category_ids` để truy ngược về COCO gốc.

## 7. Điều kiện để một ảnh được đưa vào manifest

Với manifest hiện tại, một ảnh COCO validation được giữ lại nếu thỏa toàn bộ điều kiện sau:

1. Ảnh thuộc split `val2017`.
2. Ảnh có ít nhất một caption trong `captions_val2017.json`.
3. Sau khi filter object, ảnh còn từ `2` đến `4` object hợp lệ.
4. Object có `segmentation`.
5. Object không phải `crowd`, tức `iscrowd == 0`.
6. Object không thuộc category bị loại, hiện tại loại `person`.
7. Object có tỉ lệ diện tích mask đủ lớn: `area / (image_height * image_width) >= 0.05`.

Trong code, các điều kiện này được lưu ngay trong field `sampling` của từng record:

```json
"sampling": {
  "caption_policy": "first",
  "drop_iscrowd": true,
  "exclude_categories": ["person"],
  "max_objects": 4,
  "min_mask_area_ratio": 0.05,
  "min_objects": 2,
  "object_policy": "largest",
  "prompt_template": "a {label}",
  "truncate_objects": false
}
```

Điểm cần chú ý:

- `truncate_objects: false` nghĩa là nếu một ảnh có hơn `4` object hợp lệ thì ảnh đó bị loại, không cắt xuống còn 4.
- `object_policy: largest` nghĩa là các object được sắp theo diện tích giảm dần, tie-break bằng `annotation_id`.
- `caption_policy: first` nghĩa là chọn caption có `caption_id` nhỏ nhất trong các caption của ảnh.
- `seed: null` vì không có bước chọn random trong manifest hiện tại.

## 8. Schema từng dòng trong manifest

Một dòng trong manifest hiện tại có các key sau:

```text
annotation_ids
area_ratios
caption
caption_id
category_ids
category_names
file_name
foreground_prompts
image_id
model_family
original_size
profile
protocol
sample_id
sampling
seed
target_size
version
```

Do file được lưu bằng `sort_keys=True`, thứ tự key trong JSONL có thể là thứ tự alphabet. Code không phụ thuộc vào thứ tự key, chỉ phụ thuộc vào tên key.

### 8.1. `sample_id`

Ví dụ:

```json
"sample_id": "coco_val2017_000000000776_multidiffusion_coco_all"
```

Đây là ID nội bộ cho thí nghiệm của ta.

Cấu trúc:

```text
coco_{split}_{image_id 12 chữ số}_{protocol}
```

Với sample đầu tiên:

- Dataset: `coco`
- Split: `val2017`
- Image ID: `000000000776`
- Protocol: `multidiffusion_coco_all`

Vì không dùng random seed nên `sample_id` không có đoạn `_seed42`.

### 8.2. `image_id`

Ví dụ:

```json
"image_id": 776
```

Đây là ID ảnh gốc trong COCO.

Trong COCO:

- `instances_val2017.json/images[*].id` dùng ID này.
- `instances_val2017.json/annotations[*].image_id` dùng ID này để nói object thuộc ảnh nào.
- `captions_val2017.json/annotations[*].image_id` dùng ID này để nói caption thuộc ảnh nào.

Nói cách khác, `image_id` là khóa nối giữa ảnh, object annotation, và caption.

### 8.3. `file_name`

Ví dụ:

```json
"file_name": "000000000776.jpg"
```

Đây là tên file ảnh gốc trong COCO.

Manifest không chứa pixel của ảnh. Nếu cần load ảnh gốc, dataloader sẽ tìm:

```text
{coco_root}/val2017/000000000776.jpg
```

Trong config hiện tại, `return_image` mặc định là `False`, nên dataloader không bắt buộc phải load ảnh gốc để tạo mask và prompt. Ảnh gốc chỉ cần thiết nếu ta muốn visualize, debug, hoặc dùng pixel ảnh làm input phụ.

### 8.4. `caption_id`

Ví dụ:

```json
"caption_id": 637709
```

Đây là ID caption annotation trong `captions_val2017.json`.

COCO thường có khoảng 5 caption cho mỗi ảnh. Manifest hiện tại chọn một caption bằng policy:

```text
caption_policy = "first"
```

Nghĩa là lấy caption có ID nhỏ nhất sau khi sort theo `id`.

### 8.5. `caption`

Ví dụ:

```json
"caption": "Three teddy bears, each a different color, snuggling together."
```

Đây là background prompt cho SemanticDraw.

Trong adapter:

```python
"background_prompt": batch["background_prompts"][index]
```

Nói dễ hiểu:

- `caption` mô tả toàn cảnh ảnh.
- SemanticDraw dùng nó như prompt nền/toàn ảnh.
- Nó không phải mask.
- Nó cũng không phải ảnh nền pixel.

### 8.6. `annotation_ids`

Ví dụ:

```json
"annotation_ids": [1161486, 1161607, 1159354, 1611634]
```

Đây là danh sách ID object annotation gốc trong `instances_val2017.json`.

Mỗi `annotation_id` trỏ tới một object instance cụ thể.

Ví dụ một ảnh có 3 con gấu bông và 1 cái giường thì sẽ có 4 annotation khác nhau:

- Annotation cho gấu bông thứ nhất.
- Annotation cho gấu bông thứ hai.
- Annotation cho gấu bông thứ ba.
- Annotation cho cái giường.

Điểm rất quan trọng: **manifest không lưu trực tiếp `segmentation`**. Khi dataloader cần mask, nó dùng từng `annotation_id` để tra lại annotation gốc, rồi decode trường `segmentation`.

### 8.7. `category_ids`

Ví dụ:

```json
"category_ids": [88, 88, 88, 65]
```

Đây là ID category COCO của từng object annotation.

Trong ví dụ:

- `88` là `teddy bear`.
- `65` là `bed`.

Một ảnh có thể có nhiều object cùng category. Vì vậy `category_ids` có thể lặp lại.

### 8.8. `category_names`

Ví dụ:

```json
"category_names": ["teddy bear", "teddy bear", "teddy bear", "bed"]
```

Đây là tên category đã được tra từ bảng `categories` trong `instances_val2017.json`.

Field này giúp ta không phải tra lại `category_id -> name` mỗi lần phân tích hoặc tạo prompt.

### 8.9. `foreground_prompts`

Ví dụ:

```json
"foreground_prompts": ["a teddy bear", "a teddy bear", "a teddy bear", "a bed"]
```

Đây là prompt riêng cho từng region/object.

Nó được tạo từ:

```text
prompt_template = "a {label}"
```

Với `label = "teddy bear"`, prompt trở thành:

```text
a teddy bear
```

Quan hệ giữa các field:

```text
annotation_ids[i]  <->  category_ids[i]  <->  category_names[i]  <->  foreground_prompts[i]
```

Nghĩa là phần tử thứ `i` trong bốn list này cùng mô tả một object.

### 8.10. `area_ratios`

Ví dụ:

```json
"area_ratios": [
  0.3098613200934579,
  0.2911494542202103,
  0.24064060382593458,
  0.0988291492041472
]
```

Đây là tỉ lệ diện tích object so với ảnh gốc:

```text
area_ratio = annotation["area"] / (image_height * image_width)
```

Trong COCO, `annotation["area"]` là diện tích object/mask theo annotation, không phải toàn bộ ảnh.

Field này được dùng để:

- Filter object nhỏ: phải `>= 0.05`.
- Phân tích kích thước object.
- Có thể dùng về sau cho weighting hoặc stress test.

Vì `object_policy = "largest"`, danh sách object trong manifest hiện tại thường được sắp theo `area_ratios` giảm dần.

### 8.11. `original_size`

Ví dụ:

```json
"original_size": [640, 428]
```

Đây là kích thước ảnh gốc theo thứ tự:

```text
[height, width]
```

Không phải `[width, height]`.

Field này rất quan trọng khi decode mask, vì segmentation gốc của COCO nằm trong hệ tọa độ ảnh gốc.

### 8.12. `target_size`

Ví dụ:

```json
"target_size": [512, 512]
```

Đây là kích thước mà mask sẽ được resize tới trước khi đưa vào SemanticDraw.

Thứ tự cũng là:

```text
[height, width]
```

Với SD 1.5, target size mặc định là `512x512`. Với SDXL hoặc SD3, thường dùng `1024x1024`, nhưng manifest hiện tại là cho `sd15`.

### 8.13. `profile`

Ví dụ:

```json
"profile": "multidiffusion_coco_all"
```

`profile` là tên cấu hình được dùng trong code.

Nó đến từ `COCORegionConfig.profile`.

Profile quyết định các tham số như:

- Số object tối thiểu/tối đa.
- Có loại `person` không.
- Ngưỡng diện tích mask.
- Target size.
- Batch size mặc định.
- Caption/object selection policy.

### 8.14. `protocol`

Ví dụ:

```json
"protocol": "multidiffusion_coco_all"
```

`protocol` là nhãn protocol thí nghiệm được ghi vào manifest.

Thông thường `protocol` gần giống `profile`, nhưng nó có vai trò là nhãn chuẩn để tracking experiment. Ví dụ trong code, profile cũ `multidiffusion_coco_1k` được map về protocol `multidiffusion_coco_all` để giữ backward compatibility nhưng vẫn phản ánh setup hiện tại là lấy toàn bộ ảnh hợp lệ.

### 8.15. `model_family`

Ví dụ:

```json
"model_family": "sd15"
```

Cho biết manifest đang được build cho họ model nào.

Với manifest hiện tại:

- `model_family = "sd15"`
- `target_size = [512, 512]`

Nếu sau này chạy SDXL hoặc SD3, cần tạo manifest/config tương ứng với target size `1024x1024`.

### 8.16. `seed`

Ví dụ:

```json
"seed": null
```

`seed` là `null` vì manifest hiện tại không dùng random sampling.

Dù config có default `seed = 42`, seed chỉ thật sự được ghi vào manifest nếu có bước chọn ngẫu nhiên, ví dụ:

- `subset_size` khác `None`.
- `caption_policy = "seeded_random"`.
- `object_policy = "seeded_random"`.

Ở manifest hiện tại:

- Không lấy subset random.
- Không random caption.
- Không random object.

Do đó seed không tham gia quyết định sample.

### 8.17. `version`

Ví dụ:

```json
"version": 1
```

Đây là version của schema manifest.

Nếu sau này thêm field mới hoặc thay đổi format, ta nên tăng `version` để tránh nhầm lẫn giữa các manifest cũ và mới.

### 8.18. `sampling`

Ví dụ:

```json
"sampling": {
  "caption_policy": "first",
  "drop_iscrowd": true,
  "exclude_categories": ["person"],
  "max_objects": 4,
  "min_mask_area_ratio": 0.05,
  "min_objects": 2,
  "object_policy": "largest",
  "prompt_template": "a {label}",
  "truncate_objects": false
}
```

Đây là metadata cực kỳ quan trọng để biết record này được tạo ra theo luật nào.

Ý nghĩa từng field:

| Field | Ý nghĩa |
|---|---|
| `caption_policy` | Cách chọn caption cho ảnh. Hiện tại là `first`. |
| `drop_iscrowd` | Có loại annotation crowd hay không. Hiện tại loại crowd. |
| `exclude_categories` | Category bị loại khỏi object hợp lệ. Hiện tại loại `person`. |
| `max_objects` | Số object tối đa trong một sample. |
| `min_mask_area_ratio` | Ngưỡng diện tích object tối thiểu so với ảnh. |
| `min_objects` | Số object tối thiểu trong một sample. |
| `object_policy` | Cách sắp/chọn object. Hiện tại ưu tiên object lớn. |
| `prompt_template` | Template tạo foreground prompt từ category name. |
| `truncate_objects` | Nếu ảnh có quá nhiều object hợp lệ thì cắt bớt hay loại ảnh. Hiện tại là loại ảnh. |

## 9. Ví dụ record đầu tiên

Record đầu tiên trong manifest hiện tại có nội dung:

```json
{
  "annotation_ids": [1161486, 1161607, 1159354, 1611634],
  "area_ratios": [
    0.3098613200934579,
    0.2911494542202103,
    0.24064060382593458,
    0.0988291492041472
  ],
  "caption": "Three teddy bears, each a different color, snuggling together.",
  "caption_id": 637709,
  "category_ids": [88, 88, 88, 65],
  "category_names": ["teddy bear", "teddy bear", "teddy bear", "bed"],
  "file_name": "000000000776.jpg",
  "foreground_prompts": ["a teddy bear", "a teddy bear", "a teddy bear", "a bed"],
  "image_id": 776,
  "model_family": "sd15",
  "original_size": [640, 428],
  "profile": "multidiffusion_coco_all",
  "protocol": "multidiffusion_coco_all",
  "sample_id": "coco_val2017_000000000776_multidiffusion_coco_all",
  "sampling": {
    "caption_policy": "first",
    "drop_iscrowd": true,
    "exclude_categories": ["person"],
    "max_objects": 4,
    "min_mask_area_ratio": 0.05,
    "min_objects": 2,
    "object_policy": "largest",
    "prompt_template": "a {label}",
    "truncate_objects": false
  },
  "seed": null,
  "target_size": [512, 512],
  "version": 1
}
```

Cách đọc sample này:

- Ảnh gốc là `000000000776.jpg`.
- Background prompt là caption: `Three teddy bears, each a different color, snuggling together.`
- Có 4 foreground regions.
- 3 region đầu là `teddy bear`, region cuối là `bed`.
- Mỗi region có một `annotation_id` để quay lại COCO gốc decode mask.
- Mask sẽ được resize từ kích thước gốc `[640, 428]` sang `[512, 512]`.

## 10. Cách dataloader dùng manifest

Luồng chính trong code:

```text
COCORegionConfig
  -> resolved_manifest_path()
  -> COCORegionDataset
  -> load_manifest()
  -> load_coco_index()
  -> __getitem__()
  -> collate_coco_region_batch()
  -> batch_item_to_semanticdraw_inputs()
```

Các file liên quan:

```text
Ours/src/data/coco_region_config.py
Ours/src/data/coco_region_manifest.py
Ours/src/data/coco_region_sampler.py
Ours/src/data/coco_region_dataset.py
Ours/src/data/coco_region_collate.py
Ours/src/data/adapters.py
Ours/src/data/coco_mask_utils.py
```

### 10.1. Khi tạo dataset

`COCORegionDataset(config)` làm các việc chính:

1. Validate config.
2. Tìm manifest path bằng `config.resolved_manifest_path()`.
3. Nếu manifest chưa tồn tại và `build_manifest_if_missing=True`, tự build manifest.
4. Load toàn bộ record trong manifest bằng `load_manifest()`.
5. Load lại COCO index từ `instances_val2017.json` và `captions_val2017.json`.

Điểm quan trọng: dù manifest đã có `annotation_ids`, dataloader vẫn cần `instances_val2017.json` để lấy `segmentation`, `bbox`, `area`, v.v.

### 10.2. Khi lấy một sample

Trong `COCORegionDataset.__getitem__`, với một record:

1. Lấy `annotation_ids`.
2. Với từng `annotation_id`, tra annotation gốc trong `index.annotations_by_id`.
3. Decode mask từ `annotation["segmentation"]`.
4. Resize mask bằng nearest neighbor về `target_size`.
5. Chuyển mask thành tensor.
6. Rescale bbox từ tọa độ ảnh gốc sang tọa độ target.
7. Trả về sample dict.

Sample trả ra có dạng logic:

```python
{
    "sample_id": "...",
    "image_id": 776,
    "file_name": "000000000776.jpg",
    "background_prompt": "Three teddy bears...",
    "foreground_prompts": ["a teddy bear", "a teddy bear", "a teddy bear", "a bed"],
    "category_names": ["teddy bear", "teddy bear", "teddy bear", "bed"],
    "category_ids": [88, 88, 88, 65],
    "masks": Tensor[P, 1, H, W],
    "boxes_xyxy": Tensor[P, 4],
    "area_ratios": Tensor[P],
    "original_size": (640, 428),
    "target_size": (512, 512),
    "annotation_ids": [1161486, 1161607, 1159354, 1611634],
    "metadata": record
}
```

Trong đó:

- `P` là số object/region của ảnh, từ 2 đến 4.
- `H = 512`, `W = 512` với manifest hiện tại.
- `masks` là binary mask tensor.

### 10.3. Khi gom batch

Vì mỗi ảnh có thể có số object khác nhau, collate function phải pad theo số object lớn nhất trong batch.

`collate_coco_region_batch(samples)` tạo batch dạng:

```python
{
    "sample_ids": [...],
    "image_ids": [...],
    "background_prompts": [...],
    "foreground_prompts": List[List[str]],
    "masks": Tensor[B, Pmax, 1, H, W],
    "valid_regions": Tensor[B, Pmax],
    "category_ids": Tensor[B, Pmax],
    "boxes_xyxy": Tensor[B, Pmax, 4],
    "area_ratios": Tensor[B, Pmax],
    "metadata": [...]
}
```

Trong đó:

- `B` là batch size.
- `Pmax` là số object lớn nhất trong batch.
- Với manifest này, `Pmax <= 4`.
- `valid_regions` cho biết region nào là thật, region nào chỉ là padding.

Ví dụ nếu batch có 2 ảnh:

- Ảnh A có 2 object.
- Ảnh B có 4 object.

Thì `Pmax = 4`, ảnh A sẽ được pad thêm 2 region rỗng, và `valid_regions` sẽ đánh dấu chỉ 2 region đầu là hợp lệ.

### 10.4. Khi chuyển sang input cho SemanticDraw

Adapter `batch_item_to_semanticdraw_inputs(batch, index)` lấy một item trong batch và biến thành input gần với pipeline SemanticDraw:

```python
{
    "background_prompt": batch["background_prompts"][index],
    "prompts": batch["foreground_prompts"][index][:p],
    "masks": masks,
    "height": target_h,
    "width": target_w,
    "metadata": batch["metadata"][index],
}
```

Ở đây:

- `background_prompt` đến từ COCO caption.
- `prompts` đến từ COCO object category names qua template `a {label}`.
- `masks` đến từ COCO object segmentation.
- `height`, `width` đến từ `target_size`.

Đây chính là format cần để đưa nhiều input COCO vào SemanticDraw một cách tự động.

## 11. Manifest có chứa mask không?

Không.

Manifest chỉ chứa:

```json
"annotation_ids": [...]
```

Mask thật nằm trong COCO gốc:

```text
instances_val2017.json -> annotations[*].segmentation
```

Khi chạy dataloader:

```text
annotation_id
  -> tìm annotation gốc
  -> lấy segmentation
  -> decode thành binary mask
  -> resize nearest neighbor
  -> chuyển thành torch tensor
```

Lý do không lưu mask trực tiếp trong manifest:

- File manifest nhẹ hơn.
- Không duplicate dữ liệu COCO.
- Có thể đổi `target_size` và regenerate mask theo model.
- Giữ traceability về COCO gốc.

## 12. Vì sao dùng nearest neighbor khi resize mask?

Mask là nhãn rời rạc: pixel thuộc object hoặc không thuộc object.

Nếu resize bằng bilinear/bicubic, biên mask có thể sinh giá trị trung gian như `0.2`, `0.5`, `0.7`, làm mask bị mềm và sai nhãn.

Nearest neighbor giữ mask sắc nét hơn:

```text
0 vẫn là 0
1 vẫn là 1
```

Điều này bám sát cách paper baseline xử lý object mask khi đưa về độ phân giải của diffusion model.

## 13. Tại sao lấy toàn bộ 1073 ảnh thay vì random 1000?

Ban đầu có thể sample subset cố định bằng seed. Nhưng để robust hơn và tránh tranh luận về random subset, code hiện tại lấy **toàn bộ ảnh hợp lệ**:

```text
subset_size = None
```

Hệ quả:

- Không cần seed random cho việc chọn ảnh.
- Tất cả sample hợp lệ đều được dùng.
- Kết quả dễ tái lập hơn.
- Manifest luôn có `1073` record nếu dùng cùng COCO val2017 và cùng filter.
- Thứ tự record được sort theo `image_id`.

Đây là lý do `seed` trong manifest là `null`.

## 14. Cách kiểm tra nhanh manifest

Không nên mở file `.jsonl` lớn bằng editor nặng nếu máy bị lag. Dùng lệnh đọc nhanh.

### 14.1. Đếm số dòng

```powershell
(Get-Content -LiteralPath "Ours\data_manifests\coco_val2017_multidiffusion_coco_all_512x512_all.jsonl").Count
```

Kỳ vọng:

```text
1073
```

### 14.2. In dòng đầu tiên

```powershell
Get-Content -LiteralPath "Ours\data_manifests\coco_val2017_multidiffusion_coco_all_512x512_all.jsonl" -TotalCount 1
```

### 14.3. In đẹp dòng đầu tiên bằng Python

```powershell
python -c "import json, pathlib; p=pathlib.Path(r'Ours\data_manifests\coco_val2017_multidiffusion_coco_all_512x512_all.jsonl'); r=json.loads(next(p.open(encoding='utf-8'))); print(json.dumps(r, ensure_ascii=False, indent=2))"
```

### 14.4. Kiểm tra phân bố số object

```powershell
python -c "import json, pathlib; from collections import Counter; p=pathlib.Path(r'Ours\data_manifests\coco_val2017_multidiffusion_coco_all_512x512_all.jsonl'); rows=[json.loads(l) for l in p.open(encoding='utf-8')]; print(Counter(len(r['annotation_ids']) for r in rows))"
```

Kỳ vọng:

```text
Counter({2: 717, 3: 258, 4: 98})
```

## 15. Cách regenerate manifest hiện tại

Từ root project:

```powershell
$env:PYTHONPATH="Ours\src"
python -m data.build_manifest --coco-root annotations_trainval2017 --profile multidiffusion_coco_all --model-family sd15 --overwrite
```

Lệnh này giả định COCO annotation nằm ở:

```text
annotations_trainval2017/annotations/instances_val2017.json
annotations_trainval2017/annotations/captions_val2017.json
```

Nếu không dùng `--overwrite`, code sẽ thấy manifest đã tồn tại và không build lại.

## 16. Cách dùng manifest trong dataloader

Ví dụ tối thiểu:

```python
from data.coco_profiles import multidiffusion_coco_all
from data.coco_region_dataset import build_coco_region_dataloader

config = multidiffusion_coco_all(
    coco_root="annotations_trainval2017",
    build_manifest_if_missing=False,
)

dataloader = build_coco_region_dataloader(
    config,
    shuffle=False,
)

batch = next(iter(dataloader))

print(batch["sample_ids"])
print(batch["background_prompts"])
print(batch["foreground_prompts"])
print(batch["masks"].shape)
print(batch["valid_regions"])
```

Với config mặc định hiện tại:

- `batch_size = 8`
- `num_workers = 8`
- `pin_memory = True`
- `persistent_workers = True`
- `prefetch_factor = 4`
- `cache_resized_masks = True`

Những tham số này phù hợp với máy có RTX 4090 24GB VRAM và CPU i9 13900K/32GB RAM ở mức khởi đầu hợp lý. Nếu chạy SDXL/SD3 hoặc pipeline nặng hơn, batch size có thể cần giảm.

## 17. Những lỗi hiểu nhầm thường gặp

### 17.1. "Manifest có phải COCO gốc không?"

Không. Manifest là file đã filter và đóng gói lại từ COCO gốc.

### 17.2. "Manifest có chứa ảnh không?"

Không. Nó chỉ chứa `file_name` để trỏ tới ảnh nếu cần.

### 17.3. "Manifest có chứa mask không?"

Không. Nó chứa `annotation_ids`. Mask được decode lại từ `instances_val2017.json`.

### 17.4. "Một ảnh có nhiều annotation là sao?"

Một ảnh có thể có nhiều object. Mỗi object instance là một annotation riêng.

Ví dụ một ảnh có 3 `teddy bear` và 1 `bed` thì có 4 annotation:

```text
annotation 1 -> teddy bear thứ nhất
annotation 2 -> teddy bear thứ hai
annotation 3 -> teddy bear thứ ba
annotation 4 -> bed
```

### 17.5. "Category và annotation khác nhau thế nào?"

`category` là loại object, ví dụ `cat`, `dog`, `bed`.

`annotation` là một object instance cụ thể trong một ảnh cụ thể.

Ví dụ:

```text
category_id = 88 -> teddy bear
annotation_id = 1161486 -> một teddy bear cụ thể trong ảnh 000000000776.jpg
```

Nhiều annotation có thể cùng category.

### 17.6. "Caption có phải foreground prompt không?"

Không.

- `caption` là background/global prompt.
- `foreground_prompts` là prompt riêng cho từng object region.

Với sample đầu tiên:

```text
background prompt:
Three teddy bears, each a different color, snuggling together.

foreground prompts:
a teddy bear
a teddy bear
a teddy bear
a bed
```

## 18. Checklist khi dùng manifest cho thí nghiệm

Trước khi chạy experiment, nên kiểm tra:

- Manifest có đúng số dòng `1073`.
- `profile` và `protocol` đúng với thí nghiệm muốn chạy.
- `target_size` đúng với model: `512x512` cho SD 1.5, `1024x1024` cho SDXL/SD3.
- COCO annotation gốc vẫn còn ở đúng path.
- Nếu cần visualize ảnh gốc, thư mục `val2017/` chứa file `.jpg` phải có sẵn.
- Nếu chỉ cần prompt + mask, annotation JSON là phần quan trọng nhất.
- Nếu đổi filter, phải regenerate manifest và ghi lại protocol mới.
- Không trộn kết quả từ hai manifest khác protocol mà không ghi chú rõ.

## 19. Kết luận

`Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl` là manifest cố định cho thí nghiệm COCO validation hiện tại.

Nó lấy toàn bộ `1073` ảnh hợp lệ sau filter, không random subset, không dùng seed, và mỗi dòng đã gom đủ:

- Caption làm background prompt.
- Object categories làm foreground prompts.
- Annotation IDs để decode object masks.
- Metadata về target size, protocol, profile, và sampling rule.

Nhờ manifest này, các thí nghiệm sau có thể dùng chung dataloader, đảm bảo input nhất quán, dễ debug, dễ tái lập, và bám sát tinh thần baseline SemanticDraw/MultiDiffusion trên COCO.
