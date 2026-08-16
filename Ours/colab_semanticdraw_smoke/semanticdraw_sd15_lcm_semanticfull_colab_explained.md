# Giải thích chi tiết notebook `semanticdraw_sd15_lcm_semanticfull_colab.ipynb`

File này giải thích từng cell và từng cụm dòng code quan trọng trong notebook:

`AnchorDraw/Ours/colab_semanticdraw_smoke/semanticdraw_sd15_lcm_semanticfull_colab.ipynb`

Mục tiêu của notebook là chạy **SemanticDraw SD1.5 + LCM** trên COCO Val2017 bằng Colab. Notebook dùng **pipeline gốc của SemanticDraw** từ baseline, còn phần dataloader, manifest, export và metric là phần hỗ trợ trong `Ours`.

Đối tượng đọc là người mới bắt đầu, nên phần giải thích sẽ đi chậm: vừa giải thích cú pháp Python, vừa giải thích ý nghĩa research của từng bước.

## 1. Bức tranh tổng thể

Notebook này làm theo luồng sau:

1. Cài các thư viện cần thiết trên Colab.
2. Clone hoặc tìm repo `AnchorDraw`.
3. Khai báo cấu hình thí nghiệm.
4. Tải COCO Val2017 nếu runtime chưa có dữ liệu.
5. Import dataloader của `Ours` và import pipeline baseline SemanticDraw gốc.
6. Tạo manifest và dataloader cho tập COCO cần chạy.
7. Định nghĩa các hàm hỗ trợ để chuẩn bị input, hiển thị ảnh, lưu log và resume.
8. Load model SemanticDraw SD1.5 + LCM.
9. Kiểm tra notebook thật sự đang dùng LCM.
10. Chạy sanity check để xác nhận model sinh ảnh được.
11. Sinh ảnh cho từng sample trong manifest.
12. Export ảnh sinh, overlay, manifest dùng cho metric và file zip.
13. Giải thích cell metric tùy chọn.
14. Đo metric nếu `RUN_METRICS = True`.

Điểm quan trọng nhất về input:

- COCO caption được dùng làm **background prompt**.
- Object category của mỗi mask được đổi thành foreground prompt dạng `a {label}`.
- Dataloader ban đầu trả về **foreground masks**.
- Notebook tự tạo thêm **background mask** bằng công thức:

```python
background_mask = 1 - union(foreground_masks)
```

Sau đó notebook ghép:

```python
all_masks = [background_mask] + foreground_masks
prompts = [background_prompt] + foreground_prompts
```

Đây là cách notebook đưa input vào `SemanticDrawPipeline`.

## 2. Cell 0: Markdown giới thiệu notebook

Cell đầu tiên là cell Markdown, không phải code Python. Nó dùng để ghi chú:

- Notebook chạy `SemanticDraw SD1.5 + LCM`.
- Notebook dành cho Colab.
- Mặc định hiện tại hướng tới chạy full trên GPU mạnh như A100 80GB.
- Nếu muốn validate nhanh thì có thể đổi sang `RUN_PROFILE = "smoke"` và `LOW_VRAM = True`.
- Notebook không implement lại lõi SemanticDraw, mà gọi pipeline baseline.

Cú pháp Markdown cơ bản:

- `#` tạo tiêu đề cấp 1.
- `##` tạo tiêu đề cấp 2.
- Dấu backtick như `` `RUN_PROFILE` `` dùng để hiển thị tên biến hoặc code inline.
- Dấu `**...**` dùng để in đậm.

Cell này không ảnh hưởng trực tiếp đến việc chạy code, nhưng rất quan trọng để người đọc biết notebook đang đo thí nghiệm nào.

## 3. Cell 1: Cài thư viện cần thiết

Mục tiêu của cell này là chuẩn bị môi trường Python trên Colab.

```python
import sys
import subprocess
```

`import sys` nạp module hệ thống của Python. Ở đây dùng `sys.executable` để biết notebook đang chạy bằng Python executable nào.

`import subprocess` nạp module cho phép Python gọi lệnh shell, ví dụ gọi `pip install`.

```python
packages = [
    "diffusers>=0.30.0",
    "transformers>=4.44.0",
    ...
]
```

Đây là một list Python. List được đặt trong dấu `[]`. Mỗi phần tử là một chuỗi package cần cài.

Ý nghĩa các package chính:

- `diffusers`: thư viện Hugging Face để chạy Stable Diffusion, scheduler, pipeline diffusion.
- `transformers`: thư viện text encoder, tokenizer và các model Hugging Face.
- `accelerate`: hỗ trợ chạy model lớn trên GPU.
- `peft`: hỗ trợ LoRA, cần cho LCM LoRA.
- `huggingface_hub`: tải model từ Hugging Face.
- `safetensors`: đọc checkpoint dạng `.safetensors`.
- `pycocotools`: đọc annotation và segmentation mask của COCO.
- `matplotlib`: hiển thị ảnh trong notebook.
- `pandas`: hiển thị bảng kết quả.
- `open-clip-torch`, `torch-fidelity`, `torchmetrics`: phục vụ đo metric.

```python
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])
```

Dòng này gọi lệnh tương đương:

```bash
python -m pip install -q package1 package2 ...
```

Giải thích cú pháp:

- `subprocess.check_call(...)` chạy lệnh ngoài Python.
- Nếu lệnh lỗi, `check_call` sẽ báo lỗi và dừng cell.
- `sys.executable` là đường dẫn tới Python đang chạy notebook.
- `"-m", "pip"` nghĩa là chạy module `pip` bằng chính Python hiện tại.
- `"-q"` là quiet mode, làm log cài đặt ngắn hơn.
- `*packages` là cú pháp unpack list. Thay vì truyền cả list như một phần tử, Python bung từng package thành từng argument riêng.

```python
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)
```

Dòng này gỡ `torchao`.

Lý do: trong một số runtime Colab/Kaggle, `torchao` có thể gây conflict với `peft` khi load LoRA. SemanticDraw SD1.5 + LCM cần load LCM LoRA, nên notebook chủ động gỡ `torchao` để tránh lỗi.

`check=False` nghĩa là nếu gỡ không thành công hoặc `torchao` không tồn tại thì cũng không dừng notebook.

```python
print("[OK] ...")
```

`print` in thông báo ra màn hình. Đây là log cho người chạy biết cell đã xong.

## 4. Cell 2: Clone hoặc tìm repo AnchorDraw

Cell này đảm bảo runtime Colab có source code.

```python
from pathlib import Path
import os
import subprocess
```

`Path` là class giúp thao tác đường dẫn file/folder rõ ràng hơn so với string thường.

Ví dụ:

```python
Path("/content") / "AnchorDraw"
```

sẽ tạo đường dẫn:

```text
/content/AnchorDraw
```

```python
REPO_URL = "https://github.com/GOx9-P/AnchorDraw.git"
WORK_DIR = Path("/content")
```

`REPO_URL` là link GitHub cần clone.

`WORK_DIR` là thư mục làm việc mặc định trên Colab.

```python
def is_repo_root(path: Path) -> bool:
```

Đây là khai báo function Python.

- `def` nghĩa là định nghĩa hàm.
- `path: Path` là type hint, báo rằng tham số `path` nên là object `Path`.
- `-> bool` là type hint cho output, báo rằng hàm trả về `True` hoặc `False`.

```python
return (
    (path / "Baseline" / "semantic-draw-main" / "src" / "model" / "pipeline_semantic_draw.py").exists()
    and (path / "Ours" / "src" / "data").exists()
)
```

Hàm kiểm tra một folder có đúng là repo root hay không.

Điều kiện thứ nhất: phải có file baseline:

```text
Baseline/semantic-draw-main/src/model/pipeline_semantic_draw.py
```

Điều kiện thứ hai: phải có folder dataloader:

```text
Ours/src/data
```

`and` nghĩa là cả hai điều kiện đều phải đúng.

`.exists()` kiểm tra file hoặc folder có tồn tại không.

```python
def find_repo_root() -> Path | None:
```

Hàm này tìm repo root. `Path | None` nghĩa là hàm có thể trả về một `Path`, hoặc `None` nếu không tìm thấy.

```python
starts = [
    Path.cwd(),
    Path.cwd() / "AnchorDraw",
    WORK_DIR / "AnchorDraw",
    WORK_DIR / "AnchorDraw" / "AnchorDraw",
]
```

Đây là các vị trí có khả năng chứa repo.

`Path.cwd()` là current working directory của notebook.

```python
checked = set()
```

`set()` tạo một tập hợp. Nó dùng để tránh kiểm tra trùng cùng một path nhiều lần.

```python
for start in starts:
```

Vòng lặp qua từng folder bắt đầu.

```python
if not start.exists():
    continue
```

Nếu folder không tồn tại thì bỏ qua.

`continue` nghĩa là chuyển sang vòng lặp tiếp theo.

```python
for path in [start, *start.parents]:
```

Dòng này kiểm tra cả folder `start` và các folder cha của nó.

`*start.parents` là unpack danh sách folder cha.

```python
path = path.resolve()
```

`.resolve()` chuẩn hóa đường dẫn thành dạng tuyệt đối.

```python
if path in checked:
    continue
checked.add(path)
```

Nếu path đã kiểm tra rồi thì bỏ qua. Nếu chưa, thêm vào `checked`.

```python
if is_repo_root(path):
    return path
```

Nếu path là repo root thì trả về ngay.

```python
return None
```

Nếu kiểm tra hết mà không thấy thì trả về `None`.

```python
REPO_ROOT = find_repo_root()
```

Gọi hàm tìm repo và lưu kết quả vào biến `REPO_ROOT`.

```python
if REPO_ROOT is None:
```

Nếu chưa tìm thấy repo thì clone từ GitHub.

```python
clone_target = WORK_DIR / "AnchorDraw"
```

Vị trí clone là `/content/AnchorDraw`.

```python
if not clone_target.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(clone_target)], check=True)
```

Nếu folder clone chưa có thì chạy `git clone`.

`--depth 1` nghĩa là clone shallow, chỉ lấy commit mới nhất để nhanh hơn.

`str(clone_target)` đổi `Path` thành string vì `subprocess.run` cần argument dạng string.

`check=True` nghĩa là nếu `git clone` lỗi thì dừng.

```python
REPO_ROOT = find_repo_root()
```

Sau khi clone, tìm lại repo root.

```python
assert REPO_ROOT is not None and is_repo_root(REPO_ROOT), "Không tìm thấy repo root sau khi clone."
```

`assert` là câu lệnh kiểm tra điều kiện. Nếu điều kiện sai, Python ném lỗi với message phía sau.

Dòng này bảo vệ notebook: nếu repo không đúng cấu trúc thì dừng sớm.

```python
print(f"[OK] Repo root: {REPO_ROOT}")
```

Đây là f-string. Cú pháp `f"...{REPO_ROOT}..."` cho phép nhúng giá trị biến vào chuỗi.

## 5. Cell 3: Cấu hình thí nghiệm

Đây là cell quan trọng nhất để đổi mode chạy.

```python
from pathlib import Path
import json
```

`json` dùng để lưu config và summary ra file `.json`.

### 5.1. Các cờ điều khiển chính

```python
RUN_PROFILE = "full_val2017"
```

Chọn tập chạy.

- `"smoke"`: chạy ít sample để validate nhanh.
- `"full_val2017"`: chạy toàn bộ tập semantic-full được build từ COCO Val2017.

```python
LOW_VRAM = False
```

Chọn chế độ VRAM.

- `False`: dành cho GPU mạnh như A100 80GB, giữ nhiều object mask hơn và batch lớn hơn.
- `True`: dành cho T4/L4 hoặc validate nhẹ, giảm batch và số object tối đa.

```python
REBUILD_MANIFEST = False
```

Nếu `False`, notebook dùng lại manifest cũ nếu đã tồn tại.

Nếu `True`, notebook build lại manifest. Chỉ nên bật khi bạn đổi rule sampling, ví dụ đổi `MAX_OBJECTS_PER_IMAGE`.

```python
RUN_SANITY_CHECK = True
```

Bật sanity check trước khi chạy manifest. Sanity check giúp xác nhận pipeline, model, scheduler và VAE không hỏng.

```python
RUN_METRICS = True
```

Hiện tại notebook mặc định bật đo metric sau generation.

Lưu ý research: metric nội bộ trong `Ours/src/metrics` hữu ích để so sánh nội bộ, nhưng protocol CLIP có thể chưa khớp hoàn toàn với paper. Nếu chỉ muốn sinh ảnh và export để đo bằng pipeline khác, đổi thành `False`.

```python
RUN_EXPORT_ZIP = True
```

Bật export zip. Sau khi chạy xong, notebook nén output để tải về.

```python
SKIP_EXISTING = True
```

Nếu ảnh đã sinh tồn tại, notebook không sinh lại mà dùng ảnh cũ. Điều này giúp resume sau khi runtime bị ngắt.

### 5.2. Đường dẫn và model

```python
COCO_ROOT = Path(os.environ.get("COCO_ROOT", "/content/COCO"))
```

Dòng này đọc biến môi trường `COCO_ROOT`.

Nếu biến môi trường không tồn tại, dùng mặc định `/content/COCO`.

`os.environ.get(key, default)` nghĩa là lấy biến môi trường, nếu không có thì trả về `default`.

```python
BASE_OUTPUT_DIR = Path("/content/anchordraw_runs")
```

Folder cha để lưu toàn bộ output.

```python
MODEL_ID = "runwayml/stable-diffusion-v1-5"
```

Đây là base model SD1.5. Pipeline baseline sẽ kết hợp model này với LCM scheduler và LCM LoRA.

```python
TARGET_SIZE = (512, 512)
```

Tuple Python. Tuple dùng dấu `()`.

Ở đây nghĩa là sinh ảnh 512 x 512.

```python
BASE_SEED = 2024
MANIFEST_SEED = 2026
```

`BASE_SEED` dùng cho generation.

`MANIFEST_SEED` dùng khi build manifest có sampling.

Trong full mode, nếu lấy toàn bộ record hợp lệ thì seed manifest ít ảnh hưởng hơn. Tuy nhiên vẫn lưu lại để reproducibility.

### 5.3. Số object và batch

```python
SMOKE_SUBSET_SIZE = 8
SMOKE_MAX_OBJECTS_PER_IMAGE = 8
FULL_MAX_OBJECTS_PER_IMAGE = 80
```

`SMOKE_SUBSET_SIZE = 8` nghĩa là smoke test lấy 8 record.

`SMOKE_MAX_OBJECTS_PER_IMAGE = 8` giới hạn tối đa 8 object mask mỗi ảnh khi smoke hoặc low VRAM.

`FULL_MAX_OBJECTS_PER_IMAGE = 80` cho phép lấy gần như toàn bộ object mask hợp lệ trong COCO Val2017, vì tập hiện tại tối đa khoảng 62 object hợp lệ trên một ảnh.

```python
MAX_OBJECTS_PER_IMAGE = SMOKE_MAX_OBJECTS_PER_IMAGE if (RUN_PROFILE == "smoke" or LOW_VRAM) else FULL_MAX_OBJECTS_PER_IMAGE
```

Đây là toán tử điều kiện một dòng của Python:

```python
value_if_true if condition else value_if_false
```

Nếu đang smoke hoặc low VRAM, dùng `SMOKE_MAX_OBJECTS_PER_IMAGE`.

Ngược lại, dùng `FULL_MAX_OBJECTS_PER_IMAGE`.

```python
DROP_ISCROWD = True
```

COCO có annotation `iscrowd`. Nếu `True`, notebook bỏ các annotation crowd để mask rõ ràng hơn và dễ so sánh hơn.

```python
BATCH_SIZE = 1 if LOW_VRAM else 8
METRIC_BATCH_SIZE = 1 if LOW_VRAM else 8
CLIP_BATCH_SIZE = 4 if LOW_VRAM else 16
NUM_WORKERS = 0
```

Nếu `LOW_VRAM=True`, batch nhỏ để tiết kiệm VRAM.

Nếu `LOW_VRAM=False`, batch lớn hơn cho GPU mạnh.

Lưu ý rất quan trọng: `BATCH_SIZE` ở đây là batch của **dataloader**, không phải chắc chắn là pipeline sinh 8 ảnh cùng lúc. Trong generation loop, notebook vẫn gọi `smd(...)` cho từng sample. Vì mỗi sample có số mask khác nhau, batch hóa generation qua nhiều ảnh là khó hơn.

`NUM_WORKERS = 0` nghĩa là dataloader không tạo subprocess phụ. Trên notebook cloud, đặt 0 thường ổn định hơn.

### 5.4. Tham số SemanticDraw

```python
BOOTSTRAP_STEPS = 1
```

Số bước bootstrap của SemanticDraw. Bootstrap là giai đoạn khởi tạo nền hoặc latent ban đầu theo mask/prompt trước khi denoise chính.

```python
MASK_STD = 0.0
```

Độ mờ hoặc độ mềm của mask. `0.0` nghĩa là mask rời rạc, không làm mềm biên.

```python
MASK_STRENGTH = 1.0
```

Độ mạnh của mask. `1.0` nghĩa là mask ảnh hưởng đầy đủ.

```python
PREPROCESS_MASK_COVER_ALPHA = 0.0
```

Tham số tiền xử lý vùng mask. Ở đây để 0 để không thêm cover alpha.

```python
MASK_TYPE = "discrete"
```

Dùng mask dạng rời rạc. Đây là lựa chọn phù hợp với mask COCO resize bằng nearest neighbor.

```python
NEGATIVE_PROMPT = ""
```

Negative prompt rỗng. Tức là notebook không thêm mô tả cần tránh.

```python
NUM_INFERENCE_STEPS = None
GUIDANCE_SCALE = None
```

`None` trong Python nghĩa là không có giá trị cụ thể.

Ở đây `None` có nghĩa là để pipeline baseline tự dùng default của nó.

### 5.5. Chạy một đoạn con của dataset

```python
START_INDEX = 0
MAX_SAMPLES = None
```

`START_INDEX` là vị trí bắt đầu trong manifest.

`MAX_SAMPLES = None` nghĩa là chạy hết từ `START_INDEX` đến cuối dataset.

Nếu muốn chạy 100 ảnh từ index 200, có thể đặt:

```python
START_INDEX = 200
MAX_SAMPLES = 100
```

```python
MAX_DISPLAY_RESULTS = 8 if RUN_PROFILE == "smoke" else 4
```

Smoke thì hiển thị nhiều ảnh hơn để kiểm tra trực quan.

Full thì chỉ hiển thị 4 ảnh đầu để tránh notebook quá nặng.

### 5.6. Preset A100

```python
A100_80GB_FULL_PRESET = {
    ...
}
```

Đây là dictionary Python. Dictionary dùng dấu `{}` và lưu dữ liệu dạng key-value.

Ví dụ:

```python
"BATCH_SIZE": 8
```

Key là `"BATCH_SIZE"`, value là `8`.

Preset này ghi lại cấu hình tham chiếu cho A100 80GB. Nó giúp người đọc biết cấu hình full nên là gì.

Lưu ý: trong preset, `"RUN_METRICS": False`, nhưng biến thật đang chạy trong notebook hiện tại là:

```python
RUN_METRICS = True
```

Nghĩa là preset chỉ là ghi chú tham chiếu, còn giá trị thật nằm ở các biến phía trên.

### 5.7. Xác định subset size

```python
if RUN_PROFILE == "smoke":
    subset_size = SMOKE_SUBSET_SIZE
elif RUN_PROFILE == "full_val2017":
    subset_size = None
else:
    raise ValueError(...)
```

`if`, `elif`, `else` là cấu trúc rẽ nhánh.

Nếu `RUN_PROFILE` là `"smoke"`, chỉ lấy 8 sample.

Nếu là `"full_val2017"`, lấy toàn bộ record hợp lệ.

Nếu nhập sai tên profile, `raise ValueError` sẽ ném lỗi để dừng sớm.

### 5.8. Tạo tên run và folder output

```python
mode_name = "low_vram" if LOW_VRAM else "standard"
run_suffix = "smoke" if RUN_PROFILE == "smoke" else "full_val2017"
RUN_ID = f"semanticdraw_sd15_lcm_semanticfull_512x512_{run_suffix}_{mode_name}"
```

Các dòng này tạo tên thí nghiệm dễ đọc.

Ví dụ nếu full và không low VRAM:

```text
semanticdraw_sd15_lcm_semanticfull_512x512_full_val2017_standard
```

```python
RUN_ROOT = BASE_OUTPUT_DIR / RUN_ID
GENERATED_IMAGES_DIR = RUN_ROOT / "generated_images"
OVERLAY_DIR = RUN_ROOT / "mask_overlays"
...
```

Các dòng này tạo path tới từng folder con:

- `generated_images`: ảnh sinh ra.
- `mask_overlays`: ảnh COCO gốc phủ màu mask để kiểm tra.
- `manifests`: manifest được build.
- `mask_cache`: cache mask đã resize.
- `metrics`: kết quả metric.

```python
MANIFEST_NAME = (...)
RUN_MANIFEST = MANIFEST_DIR / MANIFEST_NAME
```

Tên manifest phụ thuộc vào `subset_size`.

Nếu smoke, tên manifest có số sample.

Nếu full, tên manifest kết thúc bằng `full_val2017.jsonl`.

```python
for path in [RUN_ROOT, GENERATED_IMAGES_DIR, ...]:
    path.mkdir(parents=True, exist_ok=True)
```

Vòng lặp này tạo tất cả folder cần dùng.

`parents=True` nghĩa là nếu folder cha chưa tồn tại thì tạo luôn.

`exist_ok=True` nghĩa là nếu folder đã tồn tại thì không báo lỗi.

### 5.9. Lưu `run_config.json`

```python
run_config = {
    ...
}
```

Dictionary này lưu toàn bộ cấu hình quan trọng của run.

```python
with RUN_CONFIG_PATH.open("w", encoding="utf-8") as f:
    json.dump(run_config, f, ensure_ascii=False, indent=2)
```

`with ... as f` là context manager. Nó mở file và tự đóng file sau khi xong.

`"w"` nghĩa là write mode.

`encoding="utf-8"` giúp lưu tiếng Việt có dấu đúng.

`json.dump` ghi dictionary ra file JSON.

`ensure_ascii=False` giữ nguyên tiếng Việt có dấu thay vì escape thành `\u...`.

`indent=2` làm file JSON dễ đọc.

```python
print(json.dumps(run_config, ensure_ascii=False, indent=2))
```

`json.dumps` đổi dictionary thành string JSON để in ra màn hình.

### 5.10. Cảnh báo khi chạy full

```python
if RUN_PROFILE == "full_val2017":
    print("[WARN] ...")
```

Nếu chạy full, notebook nhắc rằng output nằm trong runtime Colab. Nếu tắt session mà chưa tải zip thì có thể mất dữ liệu.

## 6. Cell 4: Tải COCO Val2017

Cell này đảm bảo có dữ liệu COCO đúng chuẩn.

```python
import ssl
import urllib.request
import zipfile
```

- `ssl`: tạo context bỏ kiểm tra SSL khi tải nếu server SSL gây lỗi.
- `urllib.request`: tải file bằng Python.
- `zipfile`: giải nén file `.zip`.

```python
COCO_ROOT.mkdir(parents=True, exist_ok=True)
```

Tạo folder `/content/COCO` nếu chưa có.

```python
VAL_ZIP_URLS = [...]
ANN_ZIP_URLS = [...]
```

Hai list URL:

- `val2017.zip`: ảnh validation.
- `annotations_trainval2017.zip`: annotation gồm `instances_val2017.json` và `captions_val2017.json`.

```python
val_zip = COCO_ROOT / "val2017.zip"
ann_zip = COCO_ROOT / "annotations_trainval2017.zip"
```

Đường dẫn lưu file zip sau khi tải.

```python
def run_download_command(cmd: list[str]) -> bool:
```

Hàm chạy một command tải file, trả về `True` nếu thành công, `False` nếu lỗi.

```python
try:
    subprocess.run(cmd, check=True)
    return True
except Exception as exc:
    print(...)
    return False
```

`try/except` dùng để bắt lỗi.

Nếu command lỗi, notebook không dừng ngay mà thử cách tải khác.

```python
def download_file(urls: list[str], dst: Path) -> None:
```

Hàm tải một file từ nhiều URL dự phòng.

```python
if dst.exists() and dst.stat().st_size > 0:
    print(...)
    return
```

Nếu file đã tồn tại và có dung lượng lớn hơn 0 thì bỏ qua, tránh tải lại.

```python
for url in urls:
```

Thử từng URL.

```python
run_download_command(["wget", "-c", "--no-check-certificate", "-O", str(dst), url])
```

Thử tải bằng `wget`.

`-c` cho phép resume download.

`--no-check-certificate` bỏ kiểm tra SSL.

`-O` chỉ định file output.

```python
run_download_command(["curl", "-L", "-k", "--retry", "3", "-o", str(dst), url])
```

Nếu `wget` không được thì thử `curl`.

`-L` cho phép follow redirect.

`-k` bỏ kiểm tra SSL.

`--retry 3` thử lại 3 lần.

```python
context = ssl._create_unverified_context()
with urllib.request.urlopen(url, context=context, timeout=120) as response:
    with dst.open("wb") as f:
        f.write(response.read())
```

Nếu `wget` và `curl` đều lỗi, dùng Python để tải.

`"wb"` nghĩa là write binary, phù hợp để ghi file zip.

```python
raise RuntimeError(...)
```

Nếu thử hết cách vẫn không tải được, ném lỗi.

```python
def unzip_if_missing(zip_path: Path, marker_path: Path) -> None:
```

Hàm giải nén nếu chưa có marker file.

Marker file là một file đại diện để biết folder đã giải nén chưa.

```python
if marker_path.exists():
    return
```

Nếu marker tồn tại thì không giải nén lại.

```python
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(COCO_ROOT)
```

Mở zip ở read mode và giải nén vào `COCO_ROOT`.

```python
download_file(VAL_ZIP_URLS, val_zip)
download_file(ANN_ZIP_URLS, ann_zip)
```

Tải ảnh và annotation.

```python
unzip_if_missing(val_zip, COCO_ROOT / "val2017" / "000000000139.jpg")
unzip_if_missing(ann_zip, COCO_ROOT / "annotations" / "instances_val2017.json")
```

Giải nén nếu chưa thấy file marker.

```python
assert ...
```

Ba dòng `assert` cuối xác nhận đủ:

- folder `val2017`
- `instances_val2017.json`
- `captions_val2017.json`

## 7. Cell 5: Import dataloader và pipeline baseline gốc

Cell này nối code của `Ours` với code baseline.

```python
import sys
import importlib.util
import time
import math
import csv
import shutil
import gc
```

Các module chuẩn:

- `sys`: chỉnh `sys.path`.
- `importlib.util`: import một file Python bằng đường dẫn cụ thể.
- `time`: đo thời gian.
- `math`: xử lý số, ví dụ `math.isnan`.
- `csv`: ghi file CSV.
- `shutil`: nén zip.
- `gc`: garbage collection, dọn bộ nhớ Python.

```python
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from IPython.display import display, Markdown
```

Các thư viện bên ngoài:

- `torch`: tensor và GPU.
- `matplotlib.pyplot`: vẽ ảnh.
- `pandas`: bảng dữ liệu.
- `PIL.Image`: đọc, ghi, resize ảnh.
- `display`, `Markdown`: hiển thị đẹp trong notebook.

```python
OURS_SRC = REPO_ROOT / "Ours" / "src"
BASELINE_SRC = REPO_ROOT / "Baseline" / "semantic-draw-main" / "src"
```

Đường dẫn tới source code của `Ours` và baseline.

```python
sys.path.insert(0, str(OURS_SRC))
```

Thêm `Ours/src` vào đầu đường dẫn import của Python.

`insert(0, ...)` đặt ở vị trí ưu tiên cao nhất.

```python
from data import semanticdraw_sd15, build_coco_manifest, build_coco_region_dataloader, batch_item_to_semanticdraw_inputs
from data.visualize import make_mask_overlay
```

Import các hàm dataloader của `Ours`.

Ý nghĩa:

- `semanticdraw_sd15`: tạo config mặc định cho SemanticDraw SD1.5.
- `build_coco_manifest`: tạo file manifest từ COCO.
- `build_coco_region_dataloader`: tạo PyTorch dataloader.
- `batch_item_to_semanticdraw_inputs`: chuyển một item trong batch sang input thuận tiện cho pipeline.
- `make_mask_overlay`: tạo ảnh overlay mask để kiểm tra trực quan.

```python
sys.path.insert(0, str(BASELINE_SRC))
```

Thêm source baseline vào đường dẫn import.

```python
pipeline_path = BASELINE_SRC / "model" / "pipeline_semantic_draw.py"
```

Đường dẫn tới file lõi baseline cần dùng.

```python
spec = importlib.util.spec_from_file_location("pipeline_semantic_draw_original", pipeline_path)
```

Tạo import spec từ file cụ thể.

Tên `"pipeline_semantic_draw_original"` là tên module tạm thời trong Python runtime.

```python
pipeline_module = importlib.util.module_from_spec(spec)
```

Tạo một module rỗng theo spec.

```python
assert spec.loader is not None
```

Đảm bảo Python biết cách load module.

```python
spec.loader.exec_module(pipeline_module)
```

Thực thi file `pipeline_semantic_draw.py`, nạp class và function bên trong vào `pipeline_module`.

```python
SemanticDrawPipeline = pipeline_module.SemanticDrawPipeline
```

Lấy class `SemanticDrawPipeline` từ baseline.

Điểm research quan trọng: notebook không viết lại lõi SemanticDraw ở đây. Nó import class gốc từ baseline.

## 8. Cell 6: Tạo manifest và dataloader

Cell này quyết định sample nào được đưa vào thí nghiệm.

```python
base_config = semanticdraw_sd15(
    COCO_ROOT,
    seed=MANIFEST_SEED,
    subset_size=subset_size,
    manifest_path=RUN_MANIFEST,
)
```

Gọi hàm tạo config mặc định cho SemanticDraw SD1.5.

Các argument:

- `COCO_ROOT`: nơi chứa COCO.
- `seed=MANIFEST_SEED`: seed nếu cần sampling.
- `subset_size=subset_size`: số sample cần lấy, hoặc `None` nếu full.
- `manifest_path=RUN_MANIFEST`: nơi lưu manifest.

```python
config = base_config.copy_with(...)
```

`copy_with` tạo bản config mới dựa trên `base_config`, nhưng ghi đè một số trường.

Ý tưởng: giữ profile gốc, chỉ thay các tham số cần cho thí nghiệm này.

```python
instances_json=COCO_ROOT / "annotations" / "instances_val2017.json"
captions_json=COCO_ROOT / "annotations" / "captions_val2017.json"
```

Chỉ rõ hai file annotation COCO cần dùng.

`instances_val2017.json` chứa object annotation và segmentation.

`captions_val2017.json` chứa caption.

```python
min_objects=1
```

Mỗi ảnh phải có ít nhất 1 object hợp lệ.

```python
max_objects=MAX_OBJECTS_PER_IMAGE
truncate_objects=True
```

Nếu ảnh có quá nhiều object, chỉ lấy tối đa `MAX_OBJECTS_PER_IMAGE`.

`truncate_objects=True` nghĩa là cắt bớt thay vì bỏ ảnh đó.

```python
exclude_categories=()
```

Tuple rỗng. Nghĩa là không loại category nào.

```python
min_mask_area_ratio=0.0
```

Không đặt ngưỡng diện tích mask tối thiểu.

```python
drop_iscrowd=DROP_ISCROWD
```

Bỏ annotation crowd nếu `DROP_ISCROWD=True`.

```python
prompt_template="a {label}"
```

Template foreground prompt.

Ví dụ label là `teddy bear`, prompt thành:

```text
a teddy bear
```

```python
caption_policy="first"
```

COCO thường có nhiều caption cho một ảnh. Policy này chọn caption đầu tiên.

```python
object_policy="largest"
```

Nếu cần sắp xếp hoặc chọn object, ưu tiên object có diện tích lớn.

```python
return_image=True
```

Dataloader trả về cả ảnh COCO gốc để hiển thị và export.

```python
cache_resized_masks=True
cache_dir=MASK_CACHE_DIR
```

Mask sau khi resize được cache lại để chạy lại nhanh hơn.

```python
batch_size=BATCH_SIZE
num_workers=NUM_WORKERS
pin_memory=False
persistent_workers=False
```

Thiết lập cho PyTorch dataloader.

`pin_memory=False` và `persistent_workers=False` giúp notebook ổn định hơn trong môi trường Colab.

```python
records = build_coco_manifest(config, overwrite=REBUILD_MANIFEST)
```

Build manifest từ COCO theo config.

`overwrite=REBUILD_MANIFEST` quyết định có ghi đè manifest cũ không.

```python
loader = build_coco_region_dataloader(config, shuffle=False, drop_last=False)
```

Tạo dataloader.

`shuffle=False` giữ thứ tự cố định, quan trọng cho reproducibility.

`drop_last=False` giữ cả batch cuối dù không đủ `BATCH_SIZE`.

```python
dataset_size = len(loader.dataset)
num_batches = len(loader)
preview_batch = next(iter(loader))
```

- `len(loader.dataset)`: số record.
- `len(loader)`: số batch.
- `next(iter(loader))`: lấy batch đầu để kiểm tra shape.

```python
print(...)
```

Các dòng print cuối giúp xác nhận manifest, số record, số batch và shape mask.

## 9. Cell 7: Hàm hỗ trợ

Cell này định nghĩa các hàm dùng lại ở generation và export.

### 9.1. `md_escape`

```python
def md_escape(text: object) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")
```

Hàm này làm sạch text trước khi đưa vào bảng Markdown.

`str(text)` ép input thành string.

`.replace("\n", " ")` thay xuống dòng bằng khoảng trắng.

`.replace("|", "\\|")` escape dấu `|`, vì trong Markdown `|` là ký tự phân cột bảng.

### 9.2. `seed_everything`

```python
def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Hàm đặt seed để kết quả dễ tái lập.

`torch.manual_seed(seed)` đặt seed cho PyTorch CPU.

`torch.cuda.manual_seed_all(seed)` đặt seed cho toàn bộ GPU CUDA nếu có.

### 9.3. `image_stats`

```python
def image_stats(image: Image.Image) -> dict:
```

Hàm nhận một ảnh PIL và trả về thống kê pixel.

```python
import numpy as np
```

Import NumPy bên trong function. Cách này hợp lệ, module chỉ cần dùng khi function được gọi.

```python
arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
```

Chuyển ảnh sang RGB rồi thành mảng NumPy kiểu `uint8`.

Pixel ảnh RGB chuẩn nằm trong khoảng 0 đến 255.

```python
return {
    "min": int(arr.min()),
    "max": int(arr.max()),
    "mean": float(arr.mean()),
    "std": float(arr.std()),
}
```

Trả về dictionary gồm min, max, mean và độ lệch chuẩn.

Nếu ảnh đen toàn bộ, thường `min = max = mean = std = 0`.

### 9.4. `make_semanticdraw_payload`

Đây là hàm quan trọng nhất trong cell helper.

```python
def make_semanticdraw_payload(batch: dict, index: int) -> dict:
```

Hàm nhận một batch từ dataloader và vị trí sample trong batch.

```python
item = batch_item_to_semanticdraw_inputs(batch, index)
metadata = item["metadata"]
```

Chuyển batch item sang format thuận tiện.

`metadata` chứa các thông tin như `sample_id`, `image_id`, category, annotation id.

```python
fg_masks = item["masks"].float().cpu()
```

Lấy foreground masks.

Shape được comment là:

```text
(P, 1, H, W)
```

Trong đó:

- `P`: số object mask.
- `1`: số channel mask.
- `H`, `W`: chiều cao và chiều rộng.

`.float()` đổi tensor về float32.

`.cpu()` đưa tensor về CPU để chuẩn bị logic chung. Khi gọi pipeline, notebook sẽ đưa sang GPU sau.

```python
fg_union = fg_masks.sum(dim=0, keepdim=True).clamp(0, 1)
```

Gộp tất cả foreground mask thành một mask union.

`sum(dim=0)` cộng theo chiều object.

`keepdim=True` giữ lại chiều object để shape vẫn thuận lợi khi concat.

`.clamp(0, 1)` ép giá trị nằm trong `[0, 1]`. Nếu nhiều mask overlap làm tổng lớn hơn 1, nó bị cắt về 1.

```python
background_mask = (1.0 - fg_union).clamp(0, 1)
```

Tạo background mask bằng phần còn lại ngoài foreground.

Nếu pixel thuộc object thì `fg_union = 1`, background ở đó bằng 0.

Nếu pixel không thuộc object thì `fg_union = 0`, background ở đó bằng 1.

```python
all_masks = torch.cat([background_mask, fg_masks], dim=0)
```

Ghép background mask và foreground masks thành một tensor.

`torch.cat(..., dim=0)` nối theo chiều region.

Kết quả:

```text
(1 + P, 1, H, W)
```

```python
prompts = [item["background_prompt"], *item["prompts"]]
```

Tạo list prompt.

Phần tử đầu là background prompt.

`*item["prompts"]` unpack list foreground prompt.

Ví dụ:

```python
item["background_prompt"] = "Three teddy bears, each a different color..."
item["prompts"] = ["a teddy bear", "a teddy bear", "a bed"]
```

thì:

```python
prompts = [
    "Three teddy bears, each a different color...",
    "a teddy bear",
    "a teddy bear",
    "a bed",
]
```

```python
negative_prompts = [NEGATIVE_PROMPT for _ in prompts]
```

List comprehension. Nó tạo một list negative prompt có cùng số phần tử với `prompts`.

`_` là biến tạm, thể hiện rằng ta không cần dùng giá trị từng phần tử.

```python
return {
    ...
}
```

Trả về một dictionary đầy đủ cho một sample, gồm:

- id ảnh
- prompt
- mask
- metadata
- foreground masks để overlay
- all masks để đưa vào SemanticDraw

### 9.5. Hàm tạo đường dẫn ảnh

```python
def generated_path_for(index: int, sample_id: str) -> Path:
    return GENERATED_IMAGES_DIR / f"{index:06d}_{sample_id}_generated.png"
```

`{index:06d}` format số nguyên thành 6 chữ số, thêm số 0 phía trước.

Ví dụ `index = 7` thành `000007`.

Điều này giúp file sort theo đúng thứ tự.

```python
def overlay_path_for(index: int, sample_id: str) -> Path:
    return OVERLAY_DIR / f"{index:06d}_{sample_id}_overlay.png"
```

Tương tự, tạo path cho ảnh overlay mask.

### 9.6. `display_result`

Hàm này hiển thị kết quả từng sample.

```python
rows = ["| Vùng | Prompt | Annotation | Tỷ lệ diện tích |", "|---|---|---:|---:|"]
```

Tạo hai dòng đầu của bảng Markdown.

`---:` nghĩa là căn phải cột đó.

```python
rows.append(...)
```

Thêm dòng background vào bảng.

```python
for label, prompt, ann_id, area in zip(...):
```

`zip` ghép nhiều list lại để lặp song song.

Mỗi vòng lặp lấy:

- tên category
- prompt object
- annotation id
- tỷ lệ diện tích mask

```python
elapsed_text = "dùng lại ảnh cũ" if elapsed is None else f"{elapsed:.2f}s"
```

Toán tử điều kiện một dòng.

Nếu `elapsed is None`, nghĩa là không sinh lại mà dùng ảnh có sẵn.

Ngược lại, format thời gian với 2 chữ số sau dấu phẩy.

```python
display(Markdown(...))
```

Hiển thị thông tin sample bằng Markdown.

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
```

Tạo figure gồm 1 hàng, 3 cột.

```python
axes[0].imshow(original)
axes[1].imshow(overlay)
axes[2].imshow(generated)
```

Hiển thị ảnh gốc, overlay mask và ảnh sinh.

```python
for ax in axes:
    ax.axis("off")
```

Tắt trục tọa độ cho đẹp.

```python
plt.tight_layout()
plt.show()
```

Căn layout và hiển thị figure.

### 9.7. `load_existing_summary`

```python
def load_existing_summary() -> dict[int, dict]:
```

Hàm đọc lại summary cũ để resume.

```python
if not RUN_SUMMARY_PATH.exists():
    return {}
```

Nếu chưa có summary thì trả về dictionary rỗng.

```python
with RUN_SUMMARY_PATH.open("r", encoding="utf-8") as f:
    rows = json.load(f)
```

Đọc file JSON.

```python
out = {}
for row in rows:
    if "index" in row:
        out[int(row["index"])] = row
return out
```

Chuyển list row thành dictionary theo `index`.

Việc này giúp truy cập nhanh theo index khi resume.

## 10. Cell 8: Load Hugging Face token và pipeline baseline

```python
def maybe_login_to_huggingface() -> None:
```

Định nghĩa hàm login Hugging Face nếu có token.

```python
token = os.environ.get("HF_TOKEN")
```

Thử lấy token từ biến môi trường.

```python
if token is None:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception:
        token = None
```

Nếu chưa có token, thử lấy từ Colab Secrets.

`try/except` giúp notebook không lỗi nếu không chạy trong Colab hoặc chưa setup secrets.

```python
if token:
    from huggingface_hub import login
    login(token=token)
```

Nếu có token thì login.

```python
else:
    print("[INFO] ... public.")
```

Nếu không có token thì dùng public access.

```python
assert torch.cuda.is_available(), "..."
```

Bắt buộc phải có GPU. Nếu chưa bật GPU trong Colab, notebook dừng tại đây.

```python
device = torch.device("cuda:0")
dtype = torch.float16
```

`device` là GPU đầu tiên.

`dtype=torch.float16` dùng half precision để tiết kiệm VRAM và tăng tốc.

```python
maybe_login_to_huggingface()
print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
```

Login nếu cần và in tên GPU.

```python
import importlib
import importlib.util
importlib.invalidate_caches()
```

Xóa cache import để kiểm tra package mới chính xác hơn.

```python
if "torchao" in sys.modules or importlib.util.find_spec("torchao") is not None:
    raise RuntimeError(...)
```

Nếu `torchao` vẫn còn import được thì dừng.

Lý do: tránh lỗi PEFT khi nạp LoRA.

```python
seed_everything(BASE_SEED)
```

Đặt seed trước khi load và sinh ảnh.

```python
smd = SemanticDrawPipeline(
    device=device,
    dtype=dtype,
    sd_version="1.5",
    hf_key=MODEL_ID,
    has_i2t=False,
    default_mask_std=MASK_STD,
    default_mask_strength=MASK_STRENGTH,
    default_preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA,
    mask_type=MASK_TYPE,
)
```

Tạo pipeline SemanticDraw baseline.

Ý nghĩa từng tham số:

- `device=device`: chạy trên GPU.
- `dtype=dtype`: dùng float16.
- `sd_version="1.5"`: chọn nhánh SD1.5 trong baseline.
- `hf_key=MODEL_ID`: dùng `runwayml/stable-diffusion-v1-5`.
- `has_i2t=False`: không dùng image-to-text, vì COCO caption đã có sẵn.
- `default_mask_std=MASK_STD`: mặc định độ mềm mask.
- `default_mask_strength=MASK_STRENGTH`: mặc định độ mạnh mask.
- `default_preprocess_mask_cover_alpha=...`: mặc định tiền xử lý mask.
- `mask_type=MASK_TYPE`: dùng mask discrete.

Với `sd_version="1.5"`, baseline tự setup LCM scheduler và LCM LoRA theo logic trong `pipeline_semantic_draw.py`.

```python
if hasattr(smd.pipe, "enable_attention_slicing"):
    smd.pipe.enable_attention_slicing()
```

`hasattr` kiểm tra object có method hay không.

Nếu pipeline hỗ trợ attention slicing thì bật để giảm VRAM.

```python
if LOW_VRAM and hasattr(smd.pipe, "enable_vae_slicing"):
    smd.pipe.enable_vae_slicing()
```

Chỉ bật VAE slicing khi low VRAM. VAE slicing tiết kiệm VRAM nhưng có thể chậm hơn.

## 11. Cell 9: Kiểm tra đúng SD1.5 + LCM

Cell này in ra metadata của pipeline.

```python
print("Họ model: SD1.5")
print("Model id:", MODEL_ID)
```

Xác nhận họ model và checkpoint.

```python
print("Scheduler:", type(smd.scheduler).__name__)
print("Scheduler trong pipeline:", type(smd.pipe.scheduler).__name__)
```

In scheduler đang được dùng.

`type(obj).__name__` lấy tên class của object.

Với notebook này, scheduler phải là `LCMScheduler`.

```python
print("Số bước inference mặc định:", smd.default_num_inference_steps)
print("Guidance scale mặc định:", smd.default_guidance_scale)
```

In default inference steps và guidance scale từ baseline.

```python
print("Kích thước sinh:", TARGET_SIZE)
print("Manifest:", RUN_MANIFEST)
print("Số record:", dataset_size)
```

In kích thước ảnh, đường dẫn manifest và số record.

```python
assert type(smd.scheduler).__name__ == "LCMScheduler", "..."
```

Bảo vệ notebook khỏi chạy nhầm sampler.

Nếu scheduler không phải LCM thì dừng.

## 12. Cell 10: Sanity check

Sanity check kiểm tra pipeline trước khi chạy hàng nghìn ảnh.

```python
if RUN_SANITY_CHECK:
```

Nếu cờ bật thì chạy sanity.

```python
seed_everything(BASE_SEED)
```

Đặt seed để sanity reproducible.

```python
sanity_mask = torch.ones(1, 1, TARGET_SIZE[0], TARGET_SIZE[1], dtype=torch.float32, device=device)
```

Tạo một mask toàn 1.

Shape:

```text
(1, 1, 512, 512)
```

Nghĩa là có 1 region, 1 channel, phủ toàn bộ ảnh.

`torch.ones(...)` tạo tensor toàn số 1.

`dtype=torch.float32` dùng float32 cho mask.

`device=device` tạo trực tiếp trên GPU.

```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
```

Đồng bộ GPU trước khi đo thời gian.

GPU chạy bất đồng bộ, nên nếu không synchronize thì đo thời gian có thể sai.

```python
tic = time.perf_counter()
```

Lưu thời điểm bắt đầu. `perf_counter` phù hợp để đo thời gian.

```python
sanity_image = smd(
    prompts=["a studio photo of a teddy bear on a clean table"],
    negative_prompts=[NEGATIVE_PROMPT],
    masks=sanity_mask,
    ...
)
```

Gọi pipeline SemanticDraw để sinh một ảnh đơn giản.

Ở đây chỉ có một prompt và một mask phủ toàn ảnh, nên đây gần giống text-to-image qua wrapper SemanticDraw.

```python
height=TARGET_SIZE[0]
width=TARGET_SIZE[1]
```

Chiều cao và chiều rộng lấy từ tuple `TARGET_SIZE`.

```python
num_inference_steps=NUM_INFERENCE_STEPS
guidance_scale=GUIDANCE_SCALE
```

Đang là `None`, nên baseline dùng default.

```python
bootstrap_steps=BOOTSTRAP_STEPS
mask_stds=MASK_STD
mask_strengths=MASK_STRENGTH
preprocess_mask_cover_alpha=PREPROCESS_MASK_COVER_ALPHA
do_blend=False
```

Các tham số SemanticDraw được truyền vào pipeline.

`do_blend=False` nghĩa là không blend ảnh nền có sẵn. Notebook đang sinh từ prompt và mask, không hòa trộn với background image thật.

```python
print("[SANITY] Thời gian:", round(time.perf_counter() - tic, 2), "giây")
print("[SANITY] Thống kê ảnh:", image_stats(sanity_image))
display(sanity_image.resize((384, 384)))
```

In thời gian, thống kê pixel, và hiển thị ảnh sanity.

Nếu sanity sinh ảnh đen hoặc lỗi, không nên chạy full.

## 13. Cell 11: Sinh ảnh cho manifest

Đây là cell chính của notebook.

```python
summary_by_index = load_existing_summary()
```

Đọc summary cũ nếu có. Dùng để resume.

```python
end_index = dataset_size if MAX_SAMPLES is None else min(dataset_size, START_INDEX + int(MAX_SAMPLES))
```

Nếu `MAX_SAMPLES=None`, chạy đến hết dataset.

Nếu có `MAX_SAMPLES`, chạy đến `START_INDEX + MAX_SAMPLES`, nhưng không vượt quá `dataset_size`.

```python
stop = False
global_index = 0
```

`stop` dùng để dừng vòng lặp ngoài khi đã đủ sample.

`global_index` là index tuyệt đối của sample trong manifest.

```python
for batch_index, batch in enumerate(loader):
```

Lặp qua dataloader.

`enumerate(loader)` trả về cả số thứ tự batch và batch data.

```python
print(f"[BATCH] {batch_index + 1}/{len(loader)} - {len(batch['sample_ids'])} sample")
```

In tiến độ batch.

```python
for local_index, sample_id in enumerate(batch["sample_ids"]):
```

Lặp qua từng sample trong batch.

`local_index` là index trong batch.

`sample_id` là id của sample.

```python
if global_index < START_INDEX:
    global_index += 1
    continue
```

Nếu chưa tới vị trí bắt đầu, bỏ qua sample.

```python
if global_index >= end_index:
    stop = True
    break
```

Nếu đã đủ sample thì dừng vòng lặp trong.

```python
payload = make_semanticdraw_payload(batch, local_index)
```

Chuyển sample thành input SemanticDraw:

- prompts có background + foreground.
- masks có background mask + foreground masks.
- metadata dùng cho log.

```python
generated_path = generated_path_for(global_index, sample_id)
overlay_path = overlay_path_for(global_index, sample_id)
```

Tạo đường dẫn output cho ảnh sinh và overlay.

```python
original = batch["images"][local_index].resize((payload["width"], payload["height"]), Image.Resampling.BILINEAR)
```

Lấy ảnh COCO gốc và resize về kích thước target.

`Image.Resampling.BILINEAR` là phương pháp nội suy ảnh RGB.

```python
overlay = make_mask_overlay(original, payload["foreground_masks"], payload["category_names"], alpha=0.45)
```

Tạo ảnh overlay mask để kiểm tra vùng object.

`alpha=0.45` nghĩa là màu mask bán trong suốt.

```python
overlay.save(overlay_path)
```

Lưu ảnh overlay.

```python
seed = BASE_SEED + global_index
```

Mỗi sample có seed riêng, nhưng seed được xác định cố định theo index.

Điều này giúp kết quả reproducible và tránh mọi ảnh dùng cùng latent noise.

```python
elapsed = None
skipped_existing = False
```

Khởi tạo biến thời gian và cờ reuse ảnh cũ.

```python
if SKIP_EXISTING and generated_path.exists():
    generated = Image.open(generated_path).convert("RGB")
    skipped_existing = True
```

Nếu ảnh đã có và bật `SKIP_EXISTING`, notebook đọc ảnh cũ thay vì sinh lại.

`.convert("RGB")` đảm bảo ảnh có 3 channel RGB.

```python
else:
    seed_everything(seed)
```

Nếu cần sinh mới, đặt seed.

```python
masks = payload["all_masks"].to(device=device, dtype=torch.float32)
```

Đưa mask sang GPU và float32.

Mask để float32 giúp tránh lỗi dtype không cần thiết.

```python
torch.cuda.synchronize()
tic = time.perf_counter()
```

Đồng bộ GPU và bắt đầu đo thời gian.

```python
generated = smd(
    prompts=payload["prompts"],
    negative_prompts=payload["negative_prompts"],
    masks=masks,
    ...
)
```

Gọi baseline SemanticDraw để sinh ảnh.

Đây là điểm notebook thật sự chạy hệ thống.

Input quan trọng:

- `prompts`: số lượng phải khớp với số lượng masks.
- `negative_prompts`: cùng số lượng với prompts.
- `masks`: background mask + foreground masks.

```python
elapsed = time.perf_counter() - tic
generated.save(generated_path)
```

Tính thời gian và lưu ảnh sinh.

```python
row = {
    "index": global_index,
    ...
}
```

Tạo một dictionary log cho sample.

Các trường quan trọng:

- `index`: vị trí trong manifest.
- `sample_id`: id ổn định của sample.
- `image_id`: COCO image id.
- `seed`: seed generation.
- `model_family`: `sd15`.
- `sampler`: `LCMScheduler`.
- `num_regions_including_background`: số region gồm background.
- `num_foreground_regions`: số object mask.
- `background_prompt`: caption COCO.
- `foreground_prompts`: prompt object.
- `annotation_ids`: id annotation COCO.
- `elapsed_sec`: thời gian sinh ảnh.
- `generated_path`: đường dẫn ảnh sinh.
- `overlay_path`: đường dẫn overlay.

```python
summary_by_index[global_index] = row
```

Lưu row vào dictionary theo index.

```python
if MAX_DISPLAY_RESULTS is None or global_index < START_INDEX + MAX_DISPLAY_RESULTS:
    display_result(...)
```

Chỉ hiển thị một số ảnh đầu để notebook không quá nặng.

```python
with RUN_SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump([...], f, ensure_ascii=False, indent=2)
```

Ghi summary sau mỗi sample.

Đây là logic quan trọng: nếu runtime bị ngắt, những ảnh đã sinh vẫn có log.

```python
global_index += 1
del generated, overlay, original
torch.cuda.empty_cache()
```

Tăng index, xóa biến ảnh lớn, và giải phóng cache GPU.

```python
if stop:
    break
```

Nếu đã đủ sample, thoát vòng lặp batch.

```python
summary = [summary_by_index[k] for k in sorted(summary_by_index)]
```

Chuyển dictionary summary thành list được sort theo index.

```python
pd.DataFrame(summary).tail()
```

Tạo bảng Pandas và hiển thị vài dòng cuối.

## 14. Cell 12: Export và nén zip

Cell này tạo package output để đo metric hoặc tải về.

```python
if RUN_EXPORT_ZIP:
```

Chỉ chạy nếu bật export.

```python
EXPORT_MANIFEST_JSONL = RUN_ROOT / "metric_generated_manifest.jsonl"
EXPORT_MANIFEST_CSV = RUN_ROOT / "metric_generated_manifest.csv"
EXPORT_SUMMARY_JSON = RUN_ROOT / "export_summary.json"
ZIP_PATH = BASE_OUTPUT_DIR / f"{RUN_ID}__metric_export.zip"
```

Các file export:

- `metric_generated_manifest.jsonl`: manifest liên kết ảnh sinh với ảnh gốc.
- `metric_generated_manifest.csv`: bản CSV dễ xem.
- `export_summary.json`: summary của export.
- `*.zip`: toàn bộ folder run được nén lại.

```python
manifest_by_sample_id = {}
with RUN_MANIFEST.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            manifest_by_sample_id[record["sample_id"]] = record
```

Đọc manifest gốc theo từng dòng JSONL.

JSONL nghĩa là mỗi dòng là một JSON object.

`line.strip()` kiểm tra dòng không rỗng.

`json.loads(line)` chuyển string JSON thành dictionary.

`manifest_by_sample_id` giúp tra cứu record gốc bằng `sample_id`.

```python
export_records = []
for row in summary:
```

Tạo list record export từ summary generation.

```python
manifest_record = manifest_by_sample_id.get(sample_id, {})
```

Lấy record manifest tương ứng. Nếu không thấy thì dùng dictionary rỗng.

```python
generated_path = Path(row["generated_path"])
if not generated_path.exists():
    continue
```

Nếu ảnh sinh không tồn tại thì bỏ qua record đó.

```python
coco_original_path = COCO_ROOT / "val2017" / str(file_name)
```

Tạo path tới ảnh COCO gốc.

```python
export_records.append({
    ...
})
```

Thêm record export.

Các field quan trọng:

- `metric_index`: index dùng cho metric.
- `experiment_id`: tên run.
- `sample_id`: id sample.
- `generated_image_path`: path ảnh sinh.
- `generated_image_relative_path`: path tương đối trong folder run.
- `coco_original_path`: path ảnh COCO gốc.
- `background_prompt`, `foreground_prompts`: prompt.
- `annotation_ids`: trace lại mask/object.
- `target_size`, `original_size`: kích thước ảnh.
- `generation_metadata`: toàn bộ row generation.

```python
with EXPORT_MANIFEST_JSONL.open("w", encoding="utf-8") as f:
    for record in export_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

Ghi export manifest dạng JSONL.

Mỗi record một dòng.

```python
csv_fields = [...]
```

Chọn các cột cần ghi vào CSV.

```python
with EXPORT_MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
```

Mở file CSV và ghi header.

`newline=""` giúp tránh lỗi dòng trống thừa trong CSV.

```python
writer.writerow({
    key: json.dumps(record.get(key), ensure_ascii=False) if isinstance(record.get(key), (list, dict)) else record.get(key)
    for key in csv_fields
})
```

Đây là dictionary comprehension.

Nếu value là list hoặc dictionary, convert thành JSON string để ghi vào một ô CSV.

Nếu value là số hoặc string bình thường thì ghi trực tiếp.

```python
export_summary = {
    ...
}
```

Tạo summary cho export.

```python
if ZIP_PATH.exists():
    ZIP_PATH.unlink()
```

Nếu zip cũ tồn tại thì xóa trước để tránh dùng nhầm bản cũ.

`unlink()` xóa file.

```python
shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=RUN_ROOT)
```

Nén toàn bộ `RUN_ROOT` thành file zip.

`with_suffix("")` bỏ suffix `.zip` vì `make_archive` tự thêm `.zip`.

```python
export_summary["zip_size_mb"] = round(ZIP_PATH.stat().st_size / (1024 * 1024), 2)
```

Tính dung lượng zip theo MB.

```python
display(Markdown(...))
display(pd.DataFrame(export_records).head())
```

Hiển thị thông tin export và vài dòng đầu của manifest export.

## 15. Cell 13: Markdown giải thích metric tùy chọn

Cell này nhắc rằng metric hiện tại nằm trong:

```text
Ours/src/metrics
```

Ý chính:

- Metric hữu ích để so sánh nội bộ giữa các experiment.
- CLIP hiện tại là region-masked protocol.
- Có thể chưa khớp hoàn toàn với protocol trong paper.
- Nếu chỉ muốn sinh ảnh và đo bằng pipeline khác, nên đặt `RUN_METRICS = False`.

## 16. Cell 14: Đo metric tùy chọn

Cell này chỉ chạy nếu:

```python
RUN_METRICS = True
```

### 16.1. Giải phóng model diffusion khỏi GPU

```python
for var_name in ("smd", "preview_batch", "batch"):
    if var_name in globals():
        del globals()[var_name]
```

Xóa các biến lớn khỏi global namespace.

`globals()` trả về dictionary các biến global hiện tại.

`del globals()[var_name]` xóa biến theo tên.

```python
gc.collect()
```

Yêu cầu Python thu gom bộ nhớ không còn dùng.

```python
torch.cuda.empty_cache()
```

Giải phóng cache GPU của PyTorch.

```python
torch.cuda.ipc_collect()
```

Dọn bộ nhớ IPC CUDA nếu có. Được đặt trong `try/except` vì không phải runtime nào cũng hỗ trợ.

### 16.2. Tạo config metric

```python
from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report
```

Import metric code từ `Ours/src/metrics`.

```python
metric_device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

Nếu có GPU thì đo trên GPU, nếu không thì dùng CPU.

```python
metric_config = MetricEvaluationConfig(
    manifest_path=RUN_MANIFEST,
    coco_root=COCO_ROOT,
    generated_dir=GENERATED_IMAGES_DIR,
    generation_summary=RUN_SUMMARY_PATH,
    output_dir=METRICS_OUTPUT_DIR,
    model_family="sd15",
    target_size=TARGET_SIZE,
    metrics=("fid", "is", "clip_fg", "clip_bg", "time"),
    batch_size=METRIC_BATCH_SIZE,
    num_workers=0,
    pin_memory=False,
    device=metric_device,
    clip_batch_size=CLIP_BATCH_SIZE,
    is_splits=10,
)
```

Tạo object config cho metric.

Các tham số:

- `manifest_path`: manifest gốc của dataset.
- `coco_root`: nơi chứa ảnh COCO gốc.
- `generated_dir`: folder ảnh sinh.
- `generation_summary`: log generation.
- `output_dir`: nơi lưu metric report.
- `model_family`: họ model.
- `target_size`: kích thước ảnh.
- `metrics`: danh sách metric cần đo.
- `batch_size`: batch size cho FID/IS.
- `clip_batch_size`: batch size riêng cho CLIP.
- `is_splits=10`: số split khi tính Inception Score.

### 16.3. Chạy và ghi metric

```python
metric_report = run_evaluation(metric_config)
```

Chạy toàn bộ metric.

```python
metrics_json, metrics_csv = write_metrics_report(metric_report, METRICS_OUTPUT_DIR, prefix=f"{RUN_ID}_metrics")
```

Ghi kết quả ra JSON và CSV.

```python
values = metric_report["metrics"]
```

Lấy dictionary chứa giá trị metric.

### 16.4. Hàm format số

```python
def fmt(value: object, digits: int = 4) -> str:
```

Hàm format số thành string đẹp.

```python
if value is None:
    return "-"
```

Nếu metric không có thì hiển thị `-`.

```python
value = float(value)
if math.isnan(value):
    return "-"
return f"{value:.{digits}f}"
```

Ép về float, kiểm tra NaN, rồi format 4 chữ số sau dấu phẩy.

```python
except Exception:
    return str(value)
```

Nếu không ép số được thì trả về string gốc.

### 16.5. Tạo bảng metric

```python
metrics_table = pd.DataFrame([
    {"Metric": "FID", "Value": fmt(values.get("fid"))},
    ...
])
```

Tạo bảng Pandas gồm các metric:

- FID
- IS
- IS std
- CLIP(fg) x100
- CLIP(bg) x100
- Time mean sec
- Total time sec

`values.get("fid")` lấy metric theo key. Nếu key không tồn tại thì trả về `None`.

```python
display(Markdown(...))
display(metrics_table)
```

Hiển thị báo cáo metric.

Nếu `RUN_METRICS=False`, cell chỉ in:

```python
print("[INFO] RUN_METRICS=False nên bỏ qua bước đo metric.")
```

## 17. Những điểm dễ nhầm

### 17.1. `BATCH_SIZE` không đồng nghĩa với sinh nhiều ảnh cùng lúc

Trong notebook này, `BATCH_SIZE` là batch size của dataloader. Generation loop vẫn xử lý từng sample trong batch:

```python
for local_index, sample_id in enumerate(batch["sample_ids"]):
    generated = smd(...)
```

Vì mỗi ảnh có số mask khác nhau, việc batch hóa nhiều ảnh vào một lần gọi pipeline khó hơn nhiều so với batch dataloader.

### 17.2. Background không phải ảnh nền thật

Notebook không dùng ảnh COCO gốc làm background image để blend.

Background ở đây là:

- một prompt nền, lấy từ COCO caption;
- một mask nền, là vùng không thuộc foreground object.

Nó không truyền ảnh nền thật vào pipeline.

### 17.3. Vì sao cần background mask

Nếu chỉ đưa foreground masks, toàn bộ vùng còn lại không có prompt kiểm soát rõ ràng.

Notebook tạo background mask để caption COCO có vùng ảnh riêng.

Nhờ đó số `prompts` khớp số `masks`.

### 17.4. `RUN_METRICS=True` có thể làm notebook chạy lâu hơn

Sau generation, notebook còn load metric model và đo FID, IS, CLIP.

Nếu chỉ muốn kiểm tra ảnh sinh hoặc export để đo ở notebook khác, có thể đặt:

```python
RUN_METRICS = False
```

### 17.5. `SKIP_EXISTING=True` giúp resume

Nếu runtime bị ngắt giữa chừng, chạy lại notebook có thể dùng lại ảnh đã sinh.

Điều kiện là folder output và `generation_summary.json` vẫn còn trong runtime hoặc được mount lại đúng chỗ.

## 18. Tóm tắt các file output quan trọng

Sau khi chạy, output chính nằm trong:

```text
/content/anchordraw_runs/{RUN_ID}/
```

Các file và folder quan trọng:

```text
generated_images/
```

Chứa ảnh được sinh ra bởi SemanticDraw.

```text
mask_overlays/
```

Chứa ảnh COCO gốc có overlay foreground masks để kiểm tra input.

```text
manifests/
```

Chứa manifest được build từ COCO.

```text
generation_summary.json
```

Log từng ảnh sinh: seed, prompt, category, annotation id, path, thời gian.

```text
metric_generated_manifest.jsonl
```

Manifest export dùng để liên kết ảnh sinh với ảnh gốc khi đo metric.

```text
metric_generated_manifest.csv
```

Bản CSV dễ mở bằng spreadsheet.

```text
export_summary.json
```

Tóm tắt export.

```text
{RUN_ID}__metric_export.zip
```

File zip để tải về.

## 19. Kết luận

Notebook này là một wrapper Colab cho thí nghiệm **SemanticDraw SD1.5 + LCM**. Lõi generation vẫn đến từ baseline `SemanticDrawPipeline`. Phần `Ours` đảm nhiệm các việc phục vụ research:

- chuẩn hóa COCO thành manifest;
- tạo dataloader linh hoạt;
- chuyển foreground masks thành input có background mask;
- log output đầy đủ;
- export ảnh và metadata;
- đo metric tùy chọn.

Nếu cần thay đổi thí nghiệm, nơi nên chỉnh đầu tiên là **Cell 3: Cấu hình thí nghiệm**. Các cell sau chủ yếu sử dụng những biến đã khai báo ở cell này.
