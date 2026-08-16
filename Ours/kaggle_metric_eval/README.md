# Kaggle Metric Eval

Folder này chứa notebook dùng để đo FID trên Kaggle.

Notebook hiện có:

```text
cleanfid_fid_eval_kaggle.ipynb
  Đo FID bằng clean-fid. Dùng để đo nhiều export nếu dataset có dạng:
  <dataset_root>/<experiment_name>/generated_images
  <dataset_root>/reference_images/<experiment_name>

pytorch_fid_semanticdraw_sd15_lcm_full1073_kaggle.ipynb
  Đo FID bằng mseitzer/pytorch-fid cho export:
  semanticdraw_sd15_lcm_full1073__metric_export

ttur_tensorflow_fid_semanticdraw_sd15_lcm_full1073_kaggle.ipynb
  Đo FID bằng TensorFlow fid.py gốc từ bioinf-jku/TTUR cho export:
  semanticdraw_sd15_lcm_full1073__metric_export
```

Input kỳ vọng là một Kaggle Dataset đã upload từ `Ours/experiment_exports/`, có dạng:

```text
<dataset_root>/
  <experiment_name>/
    generated_images/
  reference_images/
    <experiment_name>/
```

Notebook `cleanfid_fid_eval_kaggle.ipynb` sẽ tự dò các cặp `generated_images` và `reference_images/<experiment_name>`, sau đó lưu kết quả vào:

```text
/kaggle/working/cleanfid_eval/
```

Notebook `pytorch_fid_semanticdraw_sd15_lcm_full1073_kaggle.ipynb` lưu kết quả vào:

```text
/kaggle/working/pytorch_fid_eval/semanticdraw_sd15_lcm_full1073__metric_export/
```

Notebook `ttur_tensorflow_fid_semanticdraw_sd15_lcm_full1073_kaggle.ipynb` lưu kết quả vào:

```text
/kaggle/working/ttur_fid_eval/semanticdraw_sd15_lcm_full1073__metric_export/
```

Metric trong các notebook này chỉ là FID; không tính IS, CLIP(fg), hoặc CLIP(bg).

## Ghi chú cho notebook pytorch-fid

Notebook `pytorch_fid_semanticdraw_sd15_lcm_full1073_kaggle.ipynb` cần:

```text
semanticdraw_sd15_lcm_full1073__metric_export/
|-- generated_images/
|-- metric_generated_manifest.jsonl
`-- export_summary.json
```

Nếu chưa có:

```text
reference_images/
`-- semanticdraw_sd15_lcm_full1073__metric_export/
```

thì notebook có thể tự build reference từ COCO `val2017`, miễn là Kaggle session có access tới COCO images.

## Ghi chú cho notebook TTUR TensorFlow FID

Notebook `ttur_tensorflow_fid_semanticdraw_sd15_lcm_full1073_kaggle.ipynb` tải `fid.py` trực tiếp từ repo `bioinf-jku/TTUR`, sau đó thêm alias tương thích TensorFlow 2.x trên Kaggle cho `tf.Session` và `tf.global_variables_initializer`.

Notebook đặt `TTUR_BATCH_SIZE = 1` vì trên Kaggle/TensorFlow 2.x, graph Inception của TTUR thường expose input tensor dạng `(1, None, None, 3)`. Nếu feed batch lớn hơn `1`, ví dụ `(29, 512, 512, 3)`, TensorFlow sẽ báo lỗi shape. Cấu hình `batch=1` không đổi công thức FID hay feature Inception, chỉ làm quá trình tính activation chậm hơn.
