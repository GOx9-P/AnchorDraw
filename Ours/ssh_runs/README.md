# Chạy thí nghiệm AnchorDraw trên máy chủ SSH

**Audit 2026-10-20:** các profile Hyper-SD dừng nếu không gỡ được LCM LoRA mặc định. Mã mask AnchorDraw chuẩn hóa ở vùng chồng lấn bằng log-softmax để tránh mất toàn bộ trọng số tiền cảnh khi các trọng số bilateral quá nhỏ. Benchmark kiểm tra dependency trước khi tải mô hình, không tự cài `open-clip-torch` giữa run và so snapshot mã/Python trước và sau khi sinh ảnh. Cặp đối chứng cùng tag chỉ được chấp nhận nếu run trước đã hoàn tất hậu kiểm. Các sửa đổi này chưa được kiểm chứng bằng một run GPU mới; các metric lịch sử không phải kết quả của mã hiện tại.

**Chỉ chạy một file Bash: `Ours/ssh_runs/run_experiment.sh`.** Gọi file này không có đối số sẽ kiểm tra 21 phép smoke/ablation, 17 benchmark 1.073 ảnh (gồm ba profile SD3) và 3 công cụ FID phụ cho từng seed trong ba seed `2024`, `10002024`, `20002024` (tổng 81 profile). File tự tải/kiểm tra COCO, dùng một tag chung, ghép đúng cặp baseline/AnchorDraw trong từng seed và chuẩn bị ảnh tham chiếu cho CleanFID. Nếu có suite cũ phù hợp trong cùng output root, Bash tiếp tục tag đó và bỏ qua từng profile đã hoàn tất hậu kiểm. Nếu cần xử lý lỗi một mục riêng lẻ, vẫn có thể chọn bằng `--experiment`. Các file Python trong `profiles/`, `diagnostics/` và `metric_profiles/` là mã nội bộ do Bash gọi; khi chạy trên máy chủ, chúng không đọc notebook. `export_notebooks.py` chỉ dùng khi cần đồng bộ lại mã nguồn notebook. Đầu mỗi chương trình ghi tên notebook và SHA-256 của nguồn.

## Giao thức ba seed

Với seed nền `s`, ảnh ở vị trí `i` trong manifest dùng seed `s+i` (`i` bắt đầu từ 0). Ba dải seed cách nhau 10 triệu để không giao với seed giáo viên Distillation++ (`s+i+1.000.003`) hoặc seed bootstrap SDXL (`s+500.000`); baseline và AnchorDraw cùng cấu hình nhận đúng cùng seed tại mỗi vị trí. Mỗi seed có đầu ra ở `OUTPUT_ROOT/TAG/seed_S/ID/`, gồm `base_seed.txt`, ảnh, báo cáo metric và hậu kiểm. Diagnostic chạy một lần bằng giao thức gốc; nó không được cộng vào thống kê ba seed. SD3 đã nằm trong Bash, nhưng chưa được xác nhận bằng GPU; không coi số liệu là kết quả paper trước khi chạy thật và kiểm tra ảnh.

Sau khi đủ ba nhóm, `three_seed_metrics.json` và `three_seed_metrics.csv` ghi từng giá trị FID, IS trung bình, CLIP foreground/background và thời gian theo seed, cùng trung bình và **độ lệch chuẩn mẫu giữa ba kết quả seed** (`ddof=1`). Báo cáo còn có hiệu số ghép cặp theo seed và thống kê riêng cho CleanFID, PyTorch-FID, TTUR-FID. Không diễn giải `is_std` của các split trong một run là độ lệch chuẩn giữa seed. Bộ tổng hợp từ chối nhóm thiếu 1.073 mẫu, sai seed, thiếu metric, khác dữ liệu COCO trong cùng họ mô hình, hoặc khác snapshot mã/môi trường giữa bất kỳ profile nào của suite. Chạy riêng một benchmark bằng `--experiment ID --base-seed 10002024`; cách này chỉ tạo một seed, không tạo báo cáo ba seed.

Ba run SD3 cho mỗi seed được tổng hợp riêng trong `sd3_three_seed_metrics.json/csv` khi cả chín ZIP và báo cáo metric đều hợp lệ. Bộ kiểm tra dùng ánh xạ tên PNG trong manifest export của từng phương pháp; MultiDiffusion SD3 dùng tên ảnh export khác ảnh nguồn. Mỗi ảnh benchmark được thử tối đa ba lần với **cùng index và cùng seed**. Lỗi được ghi vào `failed_images.jsonl`, sau đó vòng lặp chuyển sang ảnh tiếp theo. Nếu sau ba lần vẫn thiếu ảnh, run bị đánh dấu chưa hoàn tất: không xuất metric hợp lệ, không tổng hợp seed, không thay ảnh lỗi bằng mẫu khác. Thời gian của lần thử thất bại được cộng vào thời gian ảnh thành công khi phải thử lại; log giữ chi tiết từng lỗi. Việc thử lại không che được lỗi lặp lại có tính hệ thống.

Nếu một SD3 run còn thiếu ảnh, suite tự thử thêm một lượt `--resume` trên cùng tag. Lượt này xác minh seed, COCO, hash profile, hash mã và `pip freeze` trước khi giữ các ảnh cũ; sau đó chỉ sinh lại index còn thiếu. Có thể tự gọi `--experiment sem_sd3_ffm --base-seed 2024 --tag TAG --resume` cho một run SD3 chưa hoàn tất. Các benchmark ngoài SD3 không hỗ trợ tiếp tục một run cũ bằng `--resume`.

## Chuẩn bị

- Python 3.11+, PyTorch CUDA tương thích GPU và driver. Cài `Ours/requirements.txt`; bản chạy toàn bộ cần thêm `entmax`, `clean-fid`, `pytorch-fid`, TensorFlow và `imageio`. Mỗi profile tự kiểm tra các dependency cần thiết; thiếu package sẽ được ghi là lỗi của profile đó. Không chạy các ô cài gói của notebook trên máy chủ.
- Nhiều profile giữ guard tương thích của notebook và sẽ dừng nếu Python còn import được `torchao`; môi trường SSH dùng các profile đó cần không cài gói này. Thông báo lỗi có thể vẫn nhắc Kaggle/Colab vì là mã nguồn notebook được tách ra.
- File Bash tự tải và giải nén COCO 2017 validation images cùng hai annotation JSON nếu thiếu. Mặc định dữ liệu ở `<REPO_ROOT>/anchordraw_runs/_coco2017`, đầu ra ở `<REPO_ROOT>/anchordraw_runs` (với đường dẫn repo SSH hiện tại là `/media/ldkhoan/sourcecode/AnchorDraw/anchordraw_runs`); có thể đổi bằng `ANCHORDRAW_OUTPUT_ROOT`, `--coco-root` và `--output-root`. Archive được giữ trong `_coco2017/.downloads/` để dùng lại hoặc sửa lần giải nén dở. URL HTTP đến từ trang COCO; trước khi giải nén, Bash kiểm kích thước và SHA-256 cố định của cả hai archive theo checksum của TensorFlow Datasets. Máy chủ cần truy cập Internet và đủ dung lượng cho archive, ảnh đã giải nén và đầu ra. Runner kiểm tra 1.073 bản ghi và ảnh trước khi tải mô hình. COCO không cần đưa lên Git.
- SD3 Medium là model giới hạn quyền truy cập. Trước khi chạy, đăng nhập vào [trang model](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) bằng tài khoản Hugging Face sẽ dùng trên SSH và chấp nhận điều kiện truy cập. Trên máy chủ, chạy `hf auth login` một lần bằng token của đúng tài khoản đó, hoặc cung cấp `HF_TOKEN` qua môi trường chạy. Không ghi token vào Bash, cấu hình trong Git hay log. Runner kiểm tra khả năng tải `model_index.json` ở commit đã khóa trước khi chạy từng profile SD3; nếu quyền truy cập thiếu, chỉ các profile SD3 báo lỗi, suite vẫn tiếp tục các thí nghiệm khác. Chạy trong `tmux` nếu cần giữ tiến trình sau khi ngắt SSH.

Từ thư mục gốc repository:

```bash
./Ours/ssh_runs/run_experiment.sh
```

Lệnh trên xét 81 profile theo thứ tự. Một profile lỗi sẽ ghi `runner_logs/seed/ID.log` và một dòng trong `suite_failures.tsv`; các profile độc lập vẫn tiếp tục. Metric phụ thiếu đầu vào được bỏ qua. Suite trả mã lỗi nếu có run chưa hoàn tất, và chỉ tổng hợp ba seed khi các run cần thiết đều đạt hậu kiểm. Xem danh sách trước khi chạy bằng `./Ours/ssh_runs/run_experiment.sh --dry-run`. Đầu ra nằm dưới `OUTPUT_ROOT/TAG/`; mặc định `OUTPUT_ROOT` là `anchordraw_runs` ngay trong repo. Chạy đủ gồm diagnostic SemanticDraw trên 5.000 ảnh COCO và 51 benchmark 1.073 ảnh, nên cần dự trù thời gian và dung lượng tương ứng.

Khi chạy lại cùng output root và preset GPU, Bash chọn tag phù hợp mới nhất, kể cả tag được tạo trước khi có checkpoint nếu tìm thấy một run thành công để xác nhận cấu hình. Trước khi bỏ qua một profile, runner đối chiếu hash mã của chính profile, mã sinh ảnh/metric dùng chung, `pip freeze`, COCO/manifest, seed, preset GPU và hậu kiểm lại artifact. Với run cũ chưa có `gpu_mode.txt`, preset được đọc từ dòng `[RUN]` trong `runner_logs/seed/ID.log`. Thay đổi ở một diagnostic độc lập không buộc benchmark khác chạy lại. Run cũ không còn hợp lệ được chuyển nguyên trạng vào `_superseded/`, rồi profile đó chạy mới; nếu benchmark nguồn chạy lại thì FID phụ thuộc cũng được chạy lại. SD3 thiếu ảnh chỉ resume khi toàn bộ mã và môi trường vẫn khớp với run dở; nếu mã đã đổi, run dở được lưu trong `_superseded/` trước khi tạo run mới. Chọn `--tag TAG_MOI` để bắt đầu suite mới có chủ ý. Không chạy hai suite cùng lúc trên một output root; runner dùng khóa `flock`.

Nếu cần chạy riêng một mục để chẩn đoán:

```bash
./Ours/ssh_runs/run_experiment.sh --experiment sem_sd15_lcm --tag debug_sem_sd15_lcm
```

Có thể tải và kiểm tra cấu trúc COCO trước bằng chính file đó: `bash Ours/ssh_runs/run_experiment.sh --prepare-coco`. Nếu dữ liệu đã nằm ở máy chủ, dùng `--coco-root /đường/dẫn/coco`; runner dùng lại khi 1.073 mẫu vượt kiểm tra. Có thể đặt `--output-root` đến ổ đĩa lớn; khi không chỉ định `--coco-root`, dữ liệu sẽ nằm trong `_coco2017/` của output root ấy. Tải bị ngắt có thể chạy lại cùng lệnh; file `.part` được tải tiếp khi máy chủ nguồn hỗ trợ HTTP Range. Nếu dữ liệu có lỗi, runner thử khôi phục từ archive chính thức một lần rồi kiểm tra lại.

Mỗi benchmark hoặc FID có thư mục dưới `OUTPUT_ROOT/TAG/seed_S/ID`; diagnostic ở `OUTPUT_ROOT/TAG/ID`. Chạy riêng một profile bằng `--experiment` vẫn từ chối ghi đè thư mục run đã có. `--python /path/to/venv/bin/python` chọn Python; `--gpu-mode low_vram|high_vram_24gb|a100_80gb` chỉ ảnh hưởng notebook nguồn có biến `COLAB_GPU_MODE`. Với benchmark `md_sd15_lcm`, Bash khóa `LCM_EXPERIMENT_PROFILE=g1_b1`, đúng hàng benchmark chính trong checklist; notebook nguồn mặc định `g1_b0` là ablation khác. Bản chạy toàn bộ thực thi PyTorch-FID và TTUR-FID trên `sem_sd15_lcm` ở từng seed, đúng nguồn của hai notebook đó. CleanFID đánh giá cả 14 benchmark bằng ảnh tham chiếu COCO được đổi kích thước bilinear theo độ phân giải 512 hoặc 1024; layout tự tạo nằm trong `TAG/seed_S/_cleanfid_dataset`.

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
| `sem_sd3_ffm`, `ours_sd3_ffm`, `md_sd3_ffm` | SemanticDraw, AnchorDraw và MultiDiffusion adapter trên SD3 Medium với Flash Flow Match; chưa có GPU run được xác nhận |
| `diag_*` | 21 phép smoke, parity, sigma/layer/lambda sweep và PLADIS ablation; xem danh sách bằng `--list` |

Benchmark luôn đặt `RUN_PROFILE=full1073`. Diagnostic giữ profile gốc của notebook, trừ hai notebook tên smoke nhưng mặc định chạy full: `diag_ours_sd15_lcm_anchor_smoke` được đặt `smoke8`, `diag_sem_sdxl_euler_smoke` được đặt `smoke_bs2`. Chúng có assertion và artifact riêng; không được dùng đầu ra diagnostic như báo cáo benchmark 1.073 ảnh. Nếu phép kiểm tra cuối thất bại, Bash trả mã lỗi và giữ `console.log` để truy vết.

Riêng `diag_sem_sd15_lcm_semanticfull` giữ `RUN_PROFILE=full_val2017` của notebook và kiểm tra toàn bộ ảnh COCO val2017 trước khi chạy. Có thể dùng `--run-profile smoke` cho phép smoke của notebook. Phép này có tập mẫu và giao thức khác benchmark 1.073 ảnh. `--run-profile` chỉ cho phép thay profile của diagnostic; benchmark luôn cố định `full1073`.

Sau benchmark, `verify_run.py` kiểm số lượng, thứ tự mẫu, seed, tên/đường dẫn và định dạng ảnh, kích thước, báo cáo metric và số mẫu đánh giá. `prepare_metric_export.py` tạo `metric_generated_manifest.jsonl`, CSV và summary cho các công cụ FID phụ. Nếu ảnh nguồn nằm trực tiếp trong thư mục output, bước này ưu tiên hardlink vào `metric_export/generated_images/`, rồi thử symlink hoặc copy nếu filesystem không hỗ trợ. Thư mục run còn có `console.log`, `verification.log`, mã commit, `pip freeze` và thông tin GPU nếu có.

## Công cụ FID phụ

Hai notebook FID cũ dành cho SemanticDraw SD1.5 LCM đọc `metric_generated_manifest.jsonl` và `generated_images/`. Với run mới, dùng đường dẫn in trong `metric_export.log` của `sem_sd15_lcm` làm `--export-dir`:

```bash
bash Ours/ssh_runs/run_experiment.sh \
  --experiment pytorch_fid \
  --export-dir /data/anchordraw_runs/rerun_maskstd1/seed_2024/sem_sd15_lcm/metric_export \
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

`run_experiment.sh` tự chạy `validate_ports.py` trước khi tạo run; vì vậy bản sao trên máy chủ phải có notebook nguồn để đối chiếu. Profile Python sinh ảnh không thực thi notebook. `validate_ports.py` so từng chương trình với các ô nguồn đã chọn, kiểm tra cú pháp, biến toàn cục, đường dẫn cloud và thư mục ảnh mà evaluator đọc. Đây là kiểm tra tĩnh. Chưa có xác nhận chạy GPU hoặc tái lập các metric trên máy chủ SSH; chỉ coi kết quả mới là đối chứng sau khi Bash in `[PASS]`, `verification.log` có báo cáo hợp lệ và ảnh/metric thực tế được kiểm tra.

Runner khóa cấu hình DDIM/precision của hai benchmark MultiDiffusion DDIM và `g1_b1` cho MultiDiffusion LCM, không nhận các giá trị cũ vô tình còn trong môi trường SSH. Mặc định `--gpu-mode a100_80gb`; riêng SemanticDraw SDXL dùng cờ low VRAM chỉ khi chọn `--gpu-mode low_vram`. Hai baseline SD1.5 chạy lại với `MASK_STD=1` và tắt attention slicing để khớp đường đi AnchorDraw. Source SemanticDraw SD1.5 cũng được vá lỗi đếm batch sau khi thêm vùng nền qua `background_prompt`; `validate_ports.py` kiểm tra bản vá này trước khi chạy. Ba notebook SDXL có metric đã sửa `generated_dir` thành đúng thư mục chứa PNG. Metric evaluator mặc định giải mã mặt nạ trực tiếp từ annotation COCO của run; chỉ `md_sdxl_ddim` bật cache riêng trong `WORK_DIR`. Preflight ghi SHA-256 của manifest, hai annotation JSON và 1.073 ảnh COCO vào `input_verification.json`; hậu kiểm đọc lại các file đó để bắt thay đổi trong lúc chạy. Nếu thấy cặp đối chứng trong cùng tag, hậu kiểm còn so hash dữ liệu, snapshot mã và `pip freeze` giữa hai run. Mỗi run cũng ghi `repo_status.txt`, `profile_sha256.txt` và thông tin GPU nếu có.

Chế độ mặc định A100 80 GB được truyền nguyên dạng tới mọi profile. Các profile MultiDiffusion có preset `a100_80gb` dùng batch tương ứng. SemanticDraw/AnchorDraw SD3 nhận trực tiếp `a100_80gb` và ghi chế độ này vào `run_config.json` cùng chữ ký run; `full1073` của hai phương pháp vẫn dùng dataloader batch 1 và sinh tuần tự từng ảnh theo giao thức gốc, với hai worker ở nhánh VRAM cao. Diagnostic SemanticDraw SDXL 2 ảnh nhận nhãn `a100_80gb` nhưng vẫn dùng batch 2 của nhánh VRAM cao; batch 2 là toàn bộ tập smoke. Các profile không đọc `COLAB_GPU_MODE` giữ cấu hình gốc của notebook. Vì vậy nhãn A100 không có nghĩa tất cả profile dùng batch 80 GB hoặc tăng tốc như nhau.

Các model Hugging Face trong notebook gốc chưa được khóa bằng commit revision cụ thể. Trước khi dùng số liệu cho paper, cần kiểm tra các cặp chạy trên cùng snapshot checkpoint, thư viện và phần cứng, cùng no-op parity của nhánh AnchorDraw; checksum mã nguồn và `verification.log` không tự chứng minh điều đó. Đặc biệt, AnchorDraw SD1.5 dùng processor riêng để thu attention; diagnostic hiện có chỉ kiểm sai khác đầu ra một processor với ngưỡng `0.02`, chưa xác nhận no-op parity toàn pipeline so với SemanticDraw SD1.5. Cần chạy và lưu phép kiểm ảnh/latent cuối trên GPU trước khi coi delta SD1.5 là hiệu quả nhân quả của phương pháp.
