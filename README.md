# AnchorDraw

```text
AnchorDraw/
|-- .gitignore                                           # Quy tắc bỏ qua cache, dataset lớn, checkpoint, output thí nghiệm và file tạm.
|-- README.md                                            # Cây thư mục của repository và ý nghĩa ngắn của từng file/folder.
|-- Baseline/                                            # Chứa source code baseline gốc dùng để kế thừa và đối chiếu.
|   |-- MultiDiffusion-master/                           # Source baseline MultiDiffusion gốc để chạy/đối chiếu method MultiDiffusion.
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
    |-- experiment_exports/                              # Nơi lưu generated images, logs, summary và reference images để đo metric.
    |   |-- README.md                                    # Giải thích loại artifact nên lưu trong folder export.
    |   |-- reference_images/                            # Ảnh COCO gốc đã resize tương ứng từng export để đo clean-fid.
    |   |-- sdraw_sdxl_lightning4_euler_1024_full1073_b2_bt2_colab_24gb/ # Export SDXL Lightning Euler full1073.
    |   `-- semanticdraw_sd15_hypersd_full1073__metric_export/ # Export SD1.5 Hyper-SD full1073.
    |-- kaggle_semanticdraw_smoke/                       # Notebook Kaggle chạy SemanticDraw với LCM/Hyper-SD/Euler.
    |   |-- README.md                                    # Mô tả notebook, output và các file/folder liên quan.
    |   |-- semanticdraw_sd15_smoke_kaggle.ipynb          # Notebook debug nhanh, mặc định chạy mini128 rồi đo FID/IS/CLIP/Time.
    |   |-- semanticdraw_sd15_lcm_full1073_kaggle.ipynb   # Notebook chạy full manifest 1073 sample với sampler LCM.
    |   |-- semanticdraw_sd15_hypersd_full1073_kaggle.ipynb # Notebook chạy full manifest 1073 sample với sampler Hyper-SD.
    |   `-- semanticdraw_sdxl_euler_full1073_kaggle.ipynb # Notebook chạy full manifest 1073 sample với SDXL + Euler Discrete.
    |-- kaggle_multidiffusion_experiments/               # Notebook Kaggle chạy baseline MultiDiffusion cho SD1.5 + LCM/Hyper-SD.
    |   |-- README.md                                    # Mô tả notebook, config và mapping với baseline MultiDiffusion.
    |   |-- multidiffusion_sd15_lcm_full1073_kaggle.ipynb # Notebook chạy MultiDiffusion (MD) + SD1.5 + LCM, đổi smoke/mini/full bằng `RUN_PROFILE`.
    |   `-- multidiffusion_sd15_hypersd_full1073_kaggle.ipynb # Notebook chạy MultiDiffusion (MD) + SD1.5 + Hyper-SD, đổi smoke/mini/full bằng `RUN_PROFILE`.
    |-- colab_multidiffusion_experiments/                # Notebook Colab chạy baseline MultiDiffusion cho SDXL + Euler Discrete.
    |   |-- README.md                                    # Mô tả notebook Colab MultiDiffusion, profile và mode GPU.
    |   `-- multidiffusion_sdxl_euler_colab.ipynb        # Notebook chạy MultiDiffusion (MD) + SDXL-Lightning + Euler Discrete.
    |-- kaggle_metric_eval/                              # Notebook Kaggle đo FID bằng clean-fid từ generated/reference images.
    |   |-- README.md                                    # Hướng dẫn cấu trúc dataset upload lên Kaggle để đo clean-fid.
    |   `-- cleanfid_fid_eval_kaggle.ipynb               # Notebook đo FID cho các export trong experiment_exports.
    |-- colab_semanticdraw_smoke/                        # Notebook Colab validate/chạy SemanticDraw cho SDXL và SD3.
    |   |-- README.md                                    # Mô tả các notebook Colab, profile smoke/full và output zip.
    |   |-- semanticdraw_sdxl_euler_smoke_bs2_colab.ipynb # Notebook Colab chạy SDXL + Euler Discrete.
    |   `-- semanticdraw_sd3_flashflowmatch_full1073_colab.ipynb # Notebook chạy SD3 Medium + jasperai/flash-sd3.
    |-- src/                                             # Source code xây dựng cho phần thí nghiệm.
    |   |-- baselines/                                   # Wrapper/adaptation để chạy baseline method trong framework thí nghiệm chung.
    |   |   |-- __init__.py                              # Export API baseline wrappers.
    |   |   |-- multidiffusion_lcm.py                    # MultiDiffusion fusion gốc + sampler LCM cho SD1.5.
    |   |   |-- multidiffusion_hypersd.py                # MultiDiffusion fusion gốc + sampler Hyper-SD cho SD1.5.
    |   |   `-- multidiffusion_sdxl_euler.py            # MultiDiffusion fusion gốc + SDXL-Lightning + Euler Discrete.
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
    |   `-- metrics/                                     # Package đo FID, IS, CLIP(fg), CLIP(bg) và Time(s).
    |       |-- README.md                                 # Giải thích input/output và cách chạy metric.
    |       |-- __init__.py                               # Export API chính của package `metrics`.
    |       |-- clip_metrics.py                           # CLIP(bg) theo vùng nền và CLIP(fg) theo mask foreground.
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
