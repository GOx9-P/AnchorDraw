# Metrics Evaluation

Package này dùng để đo metric cho ảnh đã sinh bởi SemanticDraw/AnchorDraw. Nó không chạy generation; nó chỉ đọc output đã có, đối chiếu với manifest COCO, rồi xuất report JSON/CSV.

## Cây thư mục

```text
metrics/
|-- README.md              # Giải thích mục đích, input/output và cách chạy.
|-- __init__.py            # Export API chính của package.
|-- config.py              # Dataclass cấu hình evaluation.
|-- io.py                  # Đọc manifest/summary và tìm path ảnh đã generate.
|-- image_ops.py           # Resize ảnh, chuyển tensor, crop foreground và tạo background image bằng mask.
|-- time_metrics.py        # Tính Time(s) từ generation summary.
|-- clip_metrics.py        # Tính CLIP(bg), CLIP(fg) và CLIP(pg) phụ bằng open_clip.
|-- inception_metrics.py   # Tính FID và Inception Score bằng torchmetrics.
|-- reporting.py           # Ghi report JSON/CSV.
`-- evaluate_metrics.py    # CLI chạy toàn bộ evaluation.
```

## Metric được hỗ trợ

```text
FID
```

So sánh phân phối ảnh generate với ảnh COCO gốc đã resize về cùng target size. Metric này càng thấp càng tốt.

```text
IS
```

Inception Score trên ảnh generate. Metric này càng cao càng tốt.

```text
CLIP(bg)
```

Độ tương đồng CLIP giữa vùng background của ảnh generate và background/global caption của COCO. Vùng background được tạo bằng cách lấy `1 - union(foreground_masks)`, tức là phần không thuộc các object mask; mặc định các vùng foreground bị tô trắng trước khi đưa vào CLIP. Trong report lưu cả `clip_bg` dạng cosine và `clip_bg_x100`.

```text
CLIP(pg)
```

Metric phụ để debug global alignment: đo CLIP giữa toàn bộ ảnh generate và background/global caption của COCO. Metric này vẫn được hỗ trợ nếu truyền `clip_pg`, nhưng không phải default chính vì không tương đương `CLIP_bg` trong paper.

```text
CLIP(fg)
```

Độ tương đồng CLIP giữa từng foreground region generate và foreground prompt tương ứng. Region được crop theo COCO mask đã resize; mặc định vùng ngoài mask được tô trắng để giảm nhiễu background. Trong report lưu cả `clip_fg` dạng cosine và `clip_fg_x100`.

```text
Time(s)
```

Thống kê thời gian generate từ `generation_summary.json`: mean, std, total, min, max.

## Input cần có

```text
manifest JSONL
```

Ví dụ mini128 SD1.5:

```text
Ours/test_sets/manifests/mini128/coco_val2017_multidiffusion_coco_all_512x512_mini128.jsonl
```

```text
COCO root
```

Folder phải chứa:

```text
COCO/
|-- val2017/
`-- annotations/
    |-- instances_val2017.json
    `-- captions_val2017.json
```

```text
generated output dir
```

Folder chứa ảnh sinh ra từ notebook/runner:

```text
semanticdraw_mini128_outputs/
|-- *_generated.png
|-- *_overlay.png
`-- generation_summary.json
```

## Cách chạy trên Kaggle

Sau khi notebook generation đã sinh xong ảnh mini128:

```bash
export PYTHONPATH=/kaggle/working/AnchorDraw/Ours/src

python -m metrics.evaluate_metrics \
  --manifest-path /kaggle/working/AnchorDraw/Ours/test_sets/manifests/mini128/coco_val2017_multidiffusion_coco_all_512x512_mini128.jsonl \
  --coco-root /kaggle/working/COCO \
  --generated-dir /kaggle/working/semanticdraw_mini128_outputs \
  --output-dir /kaggle/working/semanticdraw_mini128_metrics \
  --model-family sd15 \
  --metrics fid,is,clip_fg,clip_bg,time \
  --is-splits 10
```

## Cách chạy local trên Windows

Từ root repo `AnchorDraw/`:

```powershell
$env:PYTHONPATH="Ours\src"

python -m metrics.evaluate_metrics `
  --manifest-path Ours\test_sets\manifests\mini128\coco_val2017_multidiffusion_coco_all_512x512_mini128.jsonl `
  --coco-root D:\datasets\COCO `
  --generated-dir D:\outputs\semanticdraw_mini128_outputs `
  --output-dir Ours\eval_outputs\semanticdraw_sd15_lcm_mini128 `
  --model-family sd15 `
  --metrics fid,is,clip_fg,clip_bg,time `
  --is-splits 10
```

## Output

```text
Ours/eval_outputs/.../
|-- metrics.json
`-- metrics.csv
```

`metrics.json` giữ đầy đủ metadata, danh sách sample đã evaluate và các sample bị thiếu ảnh generate nếu có. `metrics.csv` là một dòng phẳng để copy nhanh vào bảng thí nghiệm.

## Ghi chú quan trọng

- FID và IS cần `torchmetrics` + `torch-fidelity`.
- CLIP metrics cần `open-clip-torch`.
- Lần đầu chạy FID/IS/CLIP có thể cần Internet để tải weight.
- Smoke/mini32/mini128 chỉ dùng để kiểm tra pipeline metric có chạy đúng; kết quả báo cáo chính thức nên đo trên full manifest 1073 sample.
- `--is-splits` sẽ tự được giới hạn không vượt quá số ảnh evaluate, nên smoke8 không bị lỗi vì split lớn hơn số sample.
- `BATCH_SIZE` trong dataloader không có nghĩa là generation batch size; ở đây nó chỉ quyết định số sample được load mỗi lượt khi đo metric.
- Để so sánh công bằng, mọi method/sampler phải dùng cùng manifest và cùng tập ảnh generate tương ứng.
