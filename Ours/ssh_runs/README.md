# Chạy thí nghiệm AnchorDraw trên máy chủ SSH

**Audit 2026-10-20:** các profile Hyper-SD dừng nếu không gỡ được LCM LoRA mặc định. Mã mask AnchorDraw chuẩn hóa ở vùng chồng lấn bằng log-softmax để tránh mất toàn bộ trọng số tiền cảnh khi các trọng số bilateral quá nhỏ. Benchmark kiểm tra dependency trước khi tải mô hình, không tự cài `open-clip-torch` giữa run và so snapshot mã/Python trước và sau khi sinh ảnh. Cặp đối chứng cùng tag chỉ được chấp nhận nếu run trước đã hoàn tất hậu kiểm. Các sửa đổi này chưa được kiểm chứng bằng một run GPU mới; các metric lịch sử không phải kết quả của mã hiện tại.

**Chỉ chạy một file Bash: `Ours/ssh_runs/run_experiment.sh`.** Nó chọn thí nghiệm bằng `--experiment`, bao gồm cả các phép FID. Thư mục này có chương trình Python cho toàn bộ 38 notebook ngoài SD3 trong `Ours/`: 14 benchmark 1.073 ảnh, 21 phép smoke/ablation và 3 công cụ FID phụ. Các file Python trong `profiles/`, `diagnostics/` và `metric_profiles/` là mã nội bộ do Bash gọi; khi chạy trên máy chủ, chúng không đọc notebook. `export_notebooks.py` chỉ dùng khi cần đồng bộ lại mã nguồn notebook. Đầu mỗi chương trình ghi tên notebook và SHA-256 của nguồn.

## Chuẩn bị

- Python 3.11+, PyTorch CUDA tương thích GPU và driver. Cài `Ours/requirements.txt`; các cấu hình PLADIS cần thêm `entmax`. Công cụ FID phụ cần `clean-fid`, `pytorch-fid` hoặc TensorFlow + `imageio` tương ứng. Không chạy các ô cài gói của notebook trên máy chủ.
- Nhiều profile giữ guard tương thích của notebook và sẽ dừng nếu Python còn import được `torchao`; môi trường SSH dùng các profile đó cần không cài gói này. Thông báo lỗi có thể vẫn nhắc Kaggle/Colab vì là mã nguồn notebook được tách ra.
- COCO val2017 trong một thư mục có `val2017/`, `annotations/instances_val2017.json` và `annotations/captions_val2017.json`. Lệnh benchmark kiểm tra đủ 1.073 bản ghi và ảnh trước khi tải mô hình.
- Đặt `HF_TOKEN` trong môi trường SSH nếu checkpoint yêu cầu xác thực. Chạy trong `tmux` nếu cần giữ tiến trình sau khi ngắt SSH.

Từ thư mục gốc repository:

```bash
bash Ours/ssh_runs/run_experiment.sh --list
bash Ours/ssh_runs/run_experiment.sh \
  --experiment sem_sd15_lcm \
  --coco-root /data/coco \
  --output-root /data/anchordraw_runs \
  --tag rerun_maskstd1
```

Sau đó chạy cặp AnchorDraw SD1.5 tương ứng với **cùng dữ liệu và tag**:

```bash
bash Ours/ssh_runs/run_experiment.sh \
  --experiment ours_sd15_lcm \
  --coco-root /data/coco \
  --output-root /data/anchordraw_runs \
  --tag rerun_maskstd1
```

Mỗi ID có thư mục riêng dưới `OUTPUT_ROOT/TAG/ID`. Lệnh từ chối ghi vào thư mục run đã có. `--python /path/to/venv/bin/python` chọn Python; `--gpu-mode low_vram|high_vram_24gb|a100_80gb` chỉ ảnh hưởng notebook nguồn có biến `COLAB_GPU_MODE`. Với benchmark `md_sd15_lcm`, Bash khóa `LCM_EXPERIMENT_PROFILE=g1_b1`, đúng hàng benchmark chính trong checklist; notebook nguồn mặc định `g1_b0` là ablation khác.

## Nhóm cấu hình

| ID | Nội dung |
|---|---|
| `sem_sd15_lcm`, `sem_sd15_hypersd` | SemanticDraw SD1.5; baseline chạy lại với `MASK_STD=1.0` |
| `sem_sdxl_euler` | SemanticDraw SDXL/Lightning/Euler; bootstrap 1 để ghép cặp với AnchorDraw SDXL hiện tại |
| `md_sd15_lcm`, `md_sd15_hypersd`, `md_sd15_ddim` | MultiDiffusion SD1.5: hai adapter nhanh và bản DDIM reference |
| `md_sdxl_euler`, `md_sdxl_ddim` | MultiDiffusion SDXL: Euler và DDIM reference |
| `ours_sd15_lcm`, `ours_sd15_hypersd` | AnchorDraw SD1.5 dùng mã căn tọa độ attention hiện tại |
| `ours_sd15_lcm_dpp`, `ours_sd15_hypersd_dpp` | AnchorDraw + Distillation++ |
| `ours_sd15_hypersd_pladis` | AnchorDraw + Hyper-SD + PLADIS scale 0.5 |
| `ours_sdxl_euler` | AnchorDraw SDXL/Lightning/Euler |
| `diag_*` | 21 phép smoke, parity, sigma/layer/lambda sweep và PLADIS ablation; xem danh sách bằng `--list` |

Benchmark luôn đặt `RUN_PROFILE=full1073`. Diagnostic giữ profile gốc của notebook, trừ hai notebook tên smoke nhưng mặc định chạy full: `diag_ours_sd15_lcm_anchor_smoke` được đặt `smoke8`, `diag_sem_sdxl_euler_smoke` được đặt `smoke_bs2`. Chúng có assertion và artifact riêng; không được dùng đầu ra diagnostic như báo cáo benchmark 1.073 ảnh. Nếu phép kiểm tra cuối thất bại, Bash trả mã lỗi và giữ `console.log` để truy vết.

Riêng `diag_sem_sd15_lcm_semanticfull` giữ `RUN_PROFILE=full_val2017` của notebook và kiểm tra toàn bộ ảnh COCO val2017 trước khi chạy. Có thể dùng `--run-profile smoke` cho phép smoke của notebook. Phép này có tập mẫu và giao thức khác benchmark 1.073 ảnh. `--run-profile` chỉ cho phép thay profile của diagnostic; benchmark luôn cố định `full1073`.

Sau benchmark, `verify_run.py` kiểm số lượng, thứ tự mẫu, seed, tên/đường dẫn và định dạng ảnh, kích thước, báo cáo metric và số mẫu đánh giá. `prepare_metric_export.py` tạo `metric_generated_manifest.jsonl`, CSV và summary cho các công cụ FID phụ. Nếu ảnh nguồn nằm trực tiếp trong thư mục output, bước này ưu tiên hardlink vào `metric_export/generated_images/`, rồi thử symlink hoặc copy nếu filesystem không hỗ trợ. Thư mục run còn có `console.log`, `verification.log`, mã commit, `pip freeze` và thông tin GPU nếu có.

## Công cụ FID phụ

Hai notebook FID cũ dành cho SemanticDraw SD1.5 LCM đọc `metric_generated_manifest.jsonl` và `generated_images/`. Với run mới, dùng đường dẫn in trong `metric_export.log` của `sem_sd15_lcm` làm `--export-dir`:

```bash
bash Ours/ssh_runs/run_experiment.sh \
  --experiment pytorch_fid \
  --export-dir /data/anchordraw_runs/rerun_maskstd1/sem_sd15_lcm/metric_export \
  --coco-root /data/coco \
  --output-root /data/fid_runs
```

`ttur_fid` dùng cùng đối số, có thể thêm `--ttur-fid-file /path/to/fid.py` để cố định mã TTUR. Nếu không có file, chương trình giữ hành vi notebook: tải `fid.py` từ URL gốc và ghi SHA-256. `cleanfid` dùng bố cục dataset cũ gồm `reference_images/EXPERIMENT/` và `EXPERIMENT/generated_images/`:

```bash
bash Ours/ssh_runs/run_experiment.sh \
  --experiment cleanfid \
  --dataset-root /data/fid_pairs \
  --output-root /data/fid_runs
```

Ba công cụ FID này có giao thức riêng; không trộn số của chúng với FID do `Ours/src/metrics` tính nếu chưa kiểm tra cùng reference, tiền xử lý và phiên bản thư viện.

## Kiểm tra mã

```bash
python Ours/ssh_runs/validate_ports.py
bash -n Ours/ssh_runs/run_experiment.sh
```

`run_experiment.sh` tự chạy `validate_ports.py` trước khi tạo run; vì vậy bản sao trên máy chủ phải có notebook nguồn để đối chiếu. Profile Python sinh ảnh không thực thi notebook. `validate_ports.py` so từng chương trình với các ô nguồn đã chọn, kiểm tra cú pháp, biến toàn cục, đường dẫn cloud và thư mục ảnh mà evaluator đọc. Đây là kiểm tra tĩnh. Chưa có xác nhận chạy GPU hoặc tái lập các metric trên máy chủ SSH; chỉ coi kết quả mới là đối chứng sau khi `verification.log` báo PASS và kiểm tra ảnh/metric thực tế.

Runner khóa cấu hình DDIM/precision của hai benchmark MultiDiffusion DDIM và `g1_b1` cho MultiDiffusion LCM, không nhận các giá trị cũ vô tình còn trong môi trường SSH. Mặc định `--gpu-mode high_vram_24gb`; riêng SemanticDraw SDXL dùng cờ low VRAM chỉ khi chọn `--gpu-mode low_vram`. Hai baseline SD1.5 chạy lại với `MASK_STD=1` và tắt attention slicing để khớp đường đi AnchorDraw. Source SemanticDraw SD1.5 cũng được vá lỗi đếm batch sau khi thêm vùng nền qua `background_prompt`; `validate_ports.py` kiểm tra bản vá này trước khi chạy. Ba notebook SDXL có metric đã sửa `generated_dir` thành đúng thư mục chứa PNG. Metric evaluator mặc định giải mã mặt nạ trực tiếp từ annotation COCO của run; chỉ `md_sdxl_ddim` bật cache riêng trong `WORK_DIR`. Preflight ghi SHA-256 của manifest, hai annotation JSON và 1.073 ảnh COCO vào `input_verification.json`; hậu kiểm đọc lại các file đó để bắt thay đổi trong lúc chạy. Nếu thấy cặp đối chứng trong cùng tag, hậu kiểm còn so hash dữ liệu, snapshot mã và `pip freeze` giữa hai run. Mỗi run cũng ghi `repo_status.txt`, `profile_sha256.txt` và thông tin GPU nếu có.

Các model Hugging Face trong notebook gốc chưa được khóa bằng commit revision cụ thể. Trước khi dùng số liệu cho paper, cần kiểm tra các cặp chạy trên cùng snapshot checkpoint, thư viện và phần cứng, cùng no-op parity của nhánh AnchorDraw; checksum mã nguồn và `verification.log` không tự chứng minh điều đó. Đặc biệt, AnchorDraw SD1.5 dùng processor riêng để thu attention; diagnostic hiện có chỉ kiểm sai khác đầu ra một processor với ngưỡng `0.02`, chưa xác nhận no-op parity toàn pipeline so với SemanticDraw SD1.5. Cần chạy và lưu phép kiểm ảnh/latent cuối trên GPU trước khi coi delta SD1.5 là hiệu quả nhân quả của phương pháp.
