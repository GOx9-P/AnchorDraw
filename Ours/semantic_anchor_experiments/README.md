# Thí nghiệm Semantic Anchor

| File | Chức năng |
|---|---|
| `semantic_anchor_sd15_lcm_smoke8_colab.ipynb` | Kiểm chứng Semantic Anchor bằng SemanticDraw SD1.5 + LCM của baseline; hỗ trợ cả `smoke8` và `full1073`. |
| `semantic_anchor_sd15_hypersd_smoke8_colab.ipynb` | Chạy cùng phép kiểm chứng bằng SD1.5 + Hyper-SD 4-step để so sánh với LCM; hỗ trợ cả `smoke8` và `full1073`. |
| `semantic_anchor_sd15_lcm_anchor_runtime_smoke8_colab.ipynb` | Ablation nhân quả: so sánh baseline, bbox-control, raw attention argmax và top-k attention center để centering ở các step sau bootstrap; mặc định `smoke8`, hỗ trợ `full1073`. |
| `semantic_anchor_sd15_lcm_weighted_mask_all_ids_colab.ipynb` | Ablation Weighted Mask `WM-00` đến `WM-05` cho SemanticDraw SD1.5 + LCM. Cùng input/seed, chỉ thay reference và cách phân bổ trọng số ở pixel foreground overlap; mặc định `smoke8`, hỗ trợ `full1073`. |

Notebook LCM mặc định dùng `RUN_PROFILE = "full1073"`; notebook Hyper-SD mặc
định dùng `RUN_PROFILE = "smoke8"` để kiểm tra nhanh 8 mẫu trước khi mở rộng.
Hai notebook dùng cùng manifest, thứ tự sample, seed, bootstrap, protocol
prompt/mask, bộ bắt attention và công thức metric. Khác biệt có chủ đích là:

| Thành phần | LCM | Hyper-SD |
|---|---|---|
| LoRA | `latent-consistency/lcm-lora-sdv1-5` | `ByteDance/Hyper-SD/Hyper-SD15-4steps-lora.safetensors` |
| Scheduler | `LCMScheduler` | `DDIMScheduler(timestep_spacing="trailing")` |
| Số denoising step | 5 theo baseline | 4 theo Hyper-SD model card |
| Guidance trong SemanticDraw | Mặc định baseline | `1.0` để lấy conditional branch trong công thức CFG thủ công |

Constructor SemanticDraw baseline luôn nạp LCM trước. Notebook Hyper-SD gỡ LoRA
LCM, nạp Hyper-SD và thay scheduler/method dự đoán `x0` trên instance runtime;
không sửa source baseline trên đĩa.

Metric được tính trên mọi sample và mọi foreground region của profile. Để ZIP
không phình quá lớn, heatmap, attention array và visualization chi tiết chỉ được
lưu cho `MAX_ARTIFACT_SAMPLES` sample đầu; ảnh sinh vẫn được lưu cho toàn bộ
sample.

Input generation dùng cùng protocol đã chạy ổn trong notebook SemanticDraw 1073:
COCO caption và `background_mask = 1 - union(foreground_masks)` được đặt ở đầu
danh sách, nhờ đó số prompt luôn bằng số mask. Cách này tránh lỗi đếm prompt
của nhánh truyền `background_prompt` riêng trong pipeline baseline.

Notebook xuất toàn bộ artifact vào một thư mục chạy gồm:

```text
semantic_anchor_sem_sd15_<accelerator>_<profile>/
|-- generated_images/          # Ảnh do SemanticDraw sinh ra.
|-- mask_overlays/             # Overlay của các sample được lưu artifact.
|-- attention_maps/            # Heatmap chi tiết của các sample artifact.
|-- visualizations/            # Mask, heatmap và anchor của sample artifact.
|-- intermediate_step_images/ # Ảnh decode từ latent sau từng denoising step của sample artifact.
|-- attention_arrays/          # Attention map NPZ của sample artifact.
|-- mask_cache/                # Mask COCO đã resize dùng trong lần chạy.
|-- anchor_metrics_<profile>.csv   # Mọi phép đo anchor của profile.
|-- anchor_metrics_<profile>.jsonl
|-- anchor_debug_<profile>.jsonl   # Debug chi tiết cho sample artifact.
|-- anchor_metrics_by_step.csv # Metric tổng hợp riêng cho từng timestep.
|-- hybrid_anchor_schedule_metrics.csv # Điểm bbox-center/anchor của lịch hybrid.
|-- semantic_anchor_post_bootstrap_summary.csv # Metric chỉ từ bước thực sự dùng Semantic Anchor; tách riêng bước chuyển từ baseline.
|-- metric_summary.csv         # Bảng metric tổng hợp.
|-- metric_assessment.md       # Nhận định tự động bằng tiếng Việt.
|-- generation_summary.json
|-- summary.json
`-- run_config.json
```

Cell cuối nén nguyên thư mục trên thành
`semantic_anchor_sem_sd15_<accelerator>_<profile>__export.zip` và tự mở hộp thoại tải file
trên Google Colab khi `AUTO_DOWNLOAD_ZIP = True`.

Phần nhận định phân biệt rõ ba ý: tính toàn vẹn của lần chạy, mức global
cross-attention peak tự nhiên nằm trong mask, và độ ổn định của anchor qua các
timestep. Anchor luôn nằm trong mask theo định nghĩa, vì vậy notebook không dùng
điều đó làm bằng chứng rằng cơ chế anchor chính xác hay tốt hơn baseline.

Notebook còn mô phỏng offline lịch chọn điểm `bbox center -> attention anchor`:
bbox center được dùng khi `step_index < BOOTSTRAP_STEPS`, còn masked attention
argmax được dùng ở các bước sau. Lịch này chỉ được tính từ attention đã thu để
kiểm tra giả thuyết; nó chưa được đưa ngược vào core generation. Cách chọn bbox
center ở bước bootstrap khớp với hàm centering của baseline SemanticDraw.

Mỗi visualization có thêm cột `Ảnh trung gian sau step`. Đây là ảnh được VAE
decode từ global canvas latent ngay sau reverse-diffusion step tương ứng và trước
khi pipeline thêm noise để chuyển sang step kế tiếp. Latent trung gian chỉ được
thu cho `MAX_ARTIFACT_SAMPLES` sample đầu. Export cũ không lưu các latent này nên
không thể dựng lại cột ảnh trung gian nếu không chạy generation lại.

## Ablation runtime: LCM

`semantic_anchor_sd15_lcm_anchor_runtime_smoke8_colab.ipynb` là phép thử tiếp
theo sau phép quan sát attention. Nó không chỉ đo attention mà thực sự dùng điểm
chọn để dịch latent foreground trong quá trình denoise.

| Nhánh | Step 0 | Step 1 đến step 4 | Mục tiêu |
|---|---|---|---|
| `baseline` | Bootstrap và bbox-centering đúng baseline | Không center thêm | Mốc so sánh công bằng. |
| `bbox_control` | Giống baseline | Bbox center | Đối chứng cho giả thuyết rằng chỉ cần tiếp tục kéo về tâm hình học. |
| `semantic_anchor` | Giống baseline | Masked attention argmax thu ở step ngay trước đó | Đối chứng cho anchor một điểm, nhạy với peak nhiễu. |
| `semantic_topk_anchor` | Giống baseline | Centroid của top `k%` attention trong mask ở step ngay trước đó, chiếu về pixel top-k gần nhất | Kiểm tra anchor vùng có ổn định và hữu ích hơn argmax hay không. |

Với LCM, attention chỉ được biết sau UNet của một step. Vì vậy anchor ở step
`i` được dùng để dịch latent cho step `i + 1`; không dùng attention tương lai và
không bỏ qua step cuối. Ba nhánh dùng cùng sample, cùng prompt/mask, cùng seed và
cùng cấu hình scheduler. `TOPK_ATTENTION_PERCENT = 10.0` là biến cấu hình của
nhánh `semantic_topk_anchor`; top-k luôn được chọn sau khi giới hạn attention
vào foreground mask, lấy centroid của các pixel top-k, rồi chiếu về pixel top-k gần nhất để anchor luôn thuộc mask. Notebook lưu riêng `generated_images/`, ảnh trung gian,
attention map, CSV/JSON metric của từng nhánh rồi nén toàn bộ `RUN_ROOT` thành ZIP
để tải từ Colab.
