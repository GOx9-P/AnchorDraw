# AnchorDraw

```text
AnchorDraw/
|-- .gitignore                                           # Quy tắc bỏ qua cache, dataset lớn, checkpoint, output thí nghiệm và file tạm.
|-- README.md                                            # Cây thư mục của repository và ý nghĩa ngắn của từng file/folder.
|-- Baseline/                                            # Chứa source code baseline gốc dùng để kế thừa và đối chiếu.
|   `-- semantic-draw-main/                              # Source chính của project SemanticDraw gốc.
|       |-- assets/                                      # Tài nguyên minh họa/demo đi kèm baseline.
|       |-- demo/                                        # Script hoặc notebook demo của baseline.
|       |-- notebooks/                                   # Notebook thử nghiệm/phân tích của baseline.
|       |-- src/                                         # Source code lõi của baseline SemanticDraw.
|       |   |-- model/                                   # Pipeline/model SemanticDraw và các module generation chính.
|       |   |-- data.py                                  # Helper data/input của baseline.
|       |   |-- ipython_util.py                          # Tiện ích chạy trong môi trường notebook/IPython.
|       |   |-- prompt_util.py                           # Tiện ích xử lý prompt trong baseline.
|       |   |-- util.py                                  # Hàm tiện ích chung của baseline.
|       |   `-- __init__.py                              # Khai báo package Python cho `src`.
|       |-- .gitignore                                   # Quy tắc ignore riêng của baseline gốc.
|       |-- LICENSE                                      # License của baseline SemanticDraw.
|       |-- README.md                                    # Hướng dẫn chính của baseline SemanticDraw.
|       |-- README_old.md                                # README cũ/phiên bản lưu trữ của baseline.
|       `-- requirement.txt                              # Danh sách thư viện của baseline gốc.
`-- Ours/                                                # Phần đề xuất, tài liệu và code mới của AnchorDraw.
    |-- data_manifests/                                  # Ba manifest chính cho input COCO benchmark theo model family.
    |   |-- README.md                                    # Index ba manifest chính và protocol dùng chung.
    |   |-- EDA.md                                       # Giải thích cấu trúc manifest và cách dùng.
    |   `-- coco_val2017_multidiffusion_coco_all_*.jsonl # Full manifest chính cho SD1.5, SDXL và SD3.
    |-- documents/                                       # Tài liệu research/proposal/flowchart của hướng cải tiến.
    |   |-- Flow_chart.png                               # Flowchart nháp của hệ thống AnchorDraw.
    |   |-- Plan_SemanticAnchor.pdf                      # Bản plan/proposal dạng PDF cho hướng SemanticAnchor.
    |   |-- Proposal.pdf                                 # Proposal nghiên cứu chính.
    |   `-- proposal.txt                                 # Proposal dạng text để đọc/search nhanh.
    |-- kaggle_semanticdraw_smoke/                       # Notebook Kaggle chạy SemanticDraw SD1.5 + LCM.
    |   |-- README.md                                    # Mô tả notebook, output và các file/folder liên quan.
    |   |-- semanticdraw_sd15_smoke_kaggle.ipynb          # Notebook debug nhanh, mặc định chạy mini128 rồi đo FID/IS/CLIP/Time.
    |   `-- semanticdraw_sd15_lcm_full1073_kaggle.ipynb   # Notebook chạy full manifest 1073 sample để đo metric chính.
    |-- src/                                             # Source code xây dựng cho phần thí nghiệm.
    |   |-- data/                                        # Package dataloader COCO cho SemanticDraw/AnchorDraw.
    |   |   |-- README.md                                 # Giải thích riêng từng file trong package dataloader.
    |   |   |-- __init__.py                               # Export API package `data`.
    |   |   |-- adapters.py                               # Chuyển batch dataloader sang input gọn cho SemanticDraw.
    |   |   |-- build_manifest.py                         # CLI build manifest COCO từ annotations gốc.
    |   |   |-- build_test_sets.py                        # CLI build smoke/mini32/mini128 subset từ full manifest chính.
    |   |   |-- coco_mask_utils.py                        # Decode segmentation, resize mask, chuyển mask sang tensor.
    |   |   |-- coco_profiles.py                          # Preset/profile thí nghiệm như `multidiffusion_coco_all`.
    |   |   |-- coco_region_collate.py                    # Collate batch có số region khác nhau bằng padding.
    |   |   |-- coco_region_config.py                     # Dataclass chứa toàn bộ config dataloader/sampling.
    |   |   |-- coco_region_dataset.py                    # File chính chạy PyTorch Dataset/DataLoader.
    |   |   |-- coco_region_manifest.py                   # Load manifest và tạo COCO index để trace ID.
    |   |   |-- coco_region_sampler.py                    # Filter COCO và tạo record manifest.
    |   |   |-- download_coco.py                          # Tiện ích tải COCO nếu cần setup lại data.
    |   |   `-- visualize.py                            # Xuất preview/overlay mask để debug data.
    |   `-- metrics/                                     # Package đo FID, IS, CLIP(fg), CLIP(pg) và Time(s).
    |       |-- README.md                                 # Giải thích input/output và cách chạy metric.
    |       |-- __init__.py                               # Export API chính của package `metrics`.
    |       |-- clip_metrics.py                           # CLIP(pg) toàn ảnh và CLIP(fg) theo mask foreground.
    |       |-- config.py                                 # Dataclass cấu hình metric evaluation.
    |       |-- evaluate_metrics.py                       # CLI chạy evaluation từ manifest + generated outputs.
    |       |-- evaluator.py                              # Orchestrator nối dataloader, ảnh generate và metric accumulators.
    |       |-- image_ops.py                              # Resize ảnh, tensor uint8, crop foreground theo mask.
    |       |-- inception_metrics.py                      # FID và Inception Score bằng torchmetrics.
    |       |-- io.py                                     # Đọc summary, tìm ảnh generate theo sample_id.
    |       |-- reporting.py                              # Xuất report JSON/CSV.
    |       `-- time_metrics.py                          # Tổng hợp Time(s) từ generation summary.
    |-- test_sets/                                      # Smoke, mini32 và mini128 manifest để validate trước benchmark full.
    |   |-- README.md                                    # Giải thích cách dùng smoke/mini32/mini128.
    |   |-- manifests/                                  # Subset manifest dùng trực tiếp bởi dataloader.
    |   |-- reports/                                    # Summary JSON của từng subset manifest.
    |   `-- previews/                                   # Overlay ảnh COCO + mask, sinh local nếu đủ dependency.
    |-- first_2_instances_val2017_annotations.json       # Mẫu JSON trích từ `instances_val2017.json` để EDA/debug.
    `-- requirements.txt                                 # Thư viện cần thiết cho phần code trong `Ours`.
```
