# Experiment Versions Checklist

File này dùng để theo dõi các phiên bản thí nghiệm đã chạy, chưa chạy, notebook tương ứng, và metric để so sánh.

Ghi chú trạng thái:

- `Done`: đã chạy và có metric.
- `Pending`: chưa chạy hoặc chưa có metric.
- `Rerun`: đã có notebook/kết quả cũ nhưng cần chạy lại vì config đã đổi.
- `Excluded`: chỉ giữ để tham khảo/debug, không dùng làm baseline chính.

Quy ước `Exp ID`:

- `SEM-*`: SemanticDraw baseline.
- `MD-*`: MultiDiffusion baseline/reimplementation.
- `OURS-*`: AnchorDraw.
- `CFG-*`: config ablation để khóa protocol.
- `DIAG-*`: kết quả diagnostic hoặc excluded.
- `PAPER-*`: số tham chiếu chép từ paper.

Các số đã điền là kết quả hiện có từ các lần chạy/screenshot. Khi chạy lại, cập nhật trực tiếp vào bảng này.

## Protocol Notes

- MultiDiffusion gốc trong `Baseline/MultiDiffusion-master/MultiDiffusion-master/region_based.py` dùng `get_random_background(bootstrapping)`: bootstrap bằng các latent nền màu ngẫu nhiên, sau đó add noise theo scheduler. Nó không dùng white image latent.
- SemanticDraw mới dùng white/background bootstrap: white image latent có thể được trộn với background latent, kèm mask-centering, leak suppression, mask blur/std và xử lý background riêng.
- Vì vậy các dòng `MD-*` trong bảng này phải được đọc là MultiDiffusion hoặc adapter của MultiDiffusion. Không được hiểu là đã dùng các cải tiến ổn định của SemanticDraw, trừ khi note ghi rõ.
- Các dòng `MD-*` dùng LCM, Hyper-SD, SDXL-Lightning/Euler là adapter để chạy cùng sampler/acceleration với SemanticDraw trong cùng bảng thí nghiệm. Chúng không phải official code phát hành bởi paper MultiDiffusion 2023.
- Nếu một bản MD dùng white bootstrap, mask-centering, leak suppression, hoặc sửa denoise/mixing theo SemanticDraw thì phải chuyển sang `DIAG-*`, `CFG-*`, hoặc một variant mới; không ghi là `MD-naive`.

## Main Benchmark Checklist

| Exp ID | Check | Status | Method | Variant | Model | Resolution | Sampler / Accel | Notebook | Manifest / Profile | FID↓ | IS↑ | IS std | CLIP(fg)↑ | CLIP(bg)↑ | Time(s)↓ | Total time(s) | Note |
|---|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SEM-SD15-LCM-F1073` | [x] | Done | SemanticDraw | Baseline | SD1.5 | 512 | LCM LoRA | [semanticdraw_sd15_lcm_full1073_kaggle.ipynb](kaggle_semanticdraw_smoke/semanticdraw_sd15_lcm_full1073_kaggle.ipynb) | `full1073` | 77.5989 | 16.3658 | 1.3661 | 25.9368 | 27.0449 | 5.4353 | 5832.0934 | CLIP(bg) đã đo lại đúng nghĩa background prompt. |
| `SEM-SD15-HYPSD-F1073` | [x] | Done | SemanticDraw | Baseline | SD1.5 | 512 | Hyper-SD 4-step LoRA | [semanticdraw_sd15_hypersd_full1073_kaggle.ipynb](kaggle_semanticdraw_smoke/semanticdraw_sd15_hypersd_full1073_kaggle.ipynb) | `full1073` | 73.9058 | 16.3010 |  | 27.0101 | 28.1058 |  |  | Cần xác nhận lại Time(s) và đảm bảo CLIP(bg), không phải CLIP(pg). |
| `SEM-SDXL-EULER-F1073` | [x] | Done | SemanticDraw | Baseline | SDXL | 1024 | SDXL-Lightning 4-step UNet + Euler trailing | [semanticdraw_sdxl_euler_full1073_kaggle.ipynb](kaggle_semanticdraw_smoke/semanticdraw_sdxl_euler_full1073_kaggle.ipynb) | `full1073` | 69.4239 | 18.3598 |  | 26.2673 | 27.9198 |  |  | Đã fix safe bootstrap cho SDXL VAE fp16. |
| `SEM-SD3-FFM-F1073` | [ ] | Pending | SemanticDraw | Baseline | SD3 | 1024 | Flash Flow Match | [semanticdraw_sd3_flashflowmatch_full1073_colab.ipynb](colab_semanticdraw_smoke/semanticdraw_sd3_flashflowmatch_full1073_colab.ipynb) | `full1073` |  |  |  |  |  |  |  | Notebook đã tạo, cần verify/run full. |
| `MD-SD15-HYPSD-CUR-F1073` | [x] | Done | MultiDiffusion | Naive adapter | SD1.5 | 512 | Hyper-SD 4-step LoRA | [multidiffusion_sd15_hypersd_full1073_kaggle.ipynb](kaggle_multidiffusion_experiments/multidiffusion_sd15_hypersd_full1073_kaggle.ipynb) | `full1073` | 74.5424 | 16.4142 | 2.1497 | 26.6949 | 27.8592 | 3.8352 | 4115.1574 | Adapter theo fusion/mask của MD, dùng random-background bootstrap; không dùng white bootstrap/mask-centering của SemanticDraw. Kết quả tốt bất thường so với paper, cần ablation guidance/bootstrap và kiểm tra metric. |
| `MD-SD15-LCM-NAIVE-F1073` | [x] | Done | MultiDiffusion | Naive adapter | SD1.5 | 512 | LCM LoRA | [multidiffusion_sd15_lcm_full1073_kaggle.ipynb](kaggle_multidiffusion_experiments/multidiffusion_sd15_lcm_full1073_kaggle.ipynb) | `full1073`, `LCM_EXPERIMENT_PROFILE="g1_b1"` | 77.2768 | 16.8130 | 2.0850 | 25.9738 | 27.0796 | 5.5294 | 5933.0998 | Measured run sau khi hết kẹt batch 111. Adapter LCM, không phải official MD release; giữ random-background bootstrap theo MD gốc (`bootstrapping=1`), không dùng white bootstrap của SemanticDraw. Kết quả lệch mạnh so với paper reference `PAPER-LCM-MD-LCM-SD15`, cần kiểm tra protocol/metric trước khi đưa vào báo cáo chính. |
| `MD-SDXL-EULER-NAIVE-F1073` | [x] | Done | MultiDiffusion | Naive adapter | SDXL | 1024 | SDXL-Lightning 4-step UNet + Euler trailing | [multidiffusion_sdxl_euler_colab.ipynb](colab_multidiffusion_experiments/multidiffusion_sdxl_euler_colab.ipynb) | `full1073`, `CFG-MD-SDXL-EULER-G1-B2-V128FULL` | 70.8000 | 17.4631 | 1.3640 | 26.3942 | 27.9071 | 4.0686 | 4365.6318 | Full1073 measured. Naive MD adapter với 1 full latent view, `global_time_ids_v2`, random-background bootstrap; không dùng SemanticDraw white bootstrap/mask-centering. Kết quả rất tốt so với paper `PAPER-T3-MD-EULER-SDXL`, cần ghi rõ đây là adapter/runtime hiện tại khi báo cáo. |
| `MD-SD15-DDIM-REF-F1073` | [x] | Done | MultiDiffusion | Ref/original | SD1.5 | 512 | DDIM 50-step | [multidiffusion_sd15_ddim_ref_colab.ipynb](colab_multidiffusion_experiments/multidiffusion_sd15_ddim_ref_colab.ipynb) | `full1073`, `DDIM_EXPERIMENT_PROFILE="ref_g75_b20"` | 64.5922 | 18.7259 | 1.6212 | 27.2731 | 27.8353 | 8.8414 | 9486.8338 | Full1073 measured. Gần nhất với MultiDiffusion gốc/paper Ref.; dùng random-background bootstrap `bootstrapping=20`, `guidance=7.5`, `steps=50`, không phải white bootstrap. So với paper `PAPER-T2-MD-REF-DDIM-SD15`: FID tốt hơn 6.3378, IS cao hơn 2.4859, CLIP(fg) cao hơn 3.1831, CLIP(bg) cao hơn 0.2853, Time nhanh hơn 5.2586s/ảnh. Cần ghi chú khi báo cáo vì metric/runtime implementation và phần cứng có thể khác paper. |
| `MD-SDXL-DDIM-REF-F1073` | [x] | Done | MultiDiffusion | Ref/original | SDXL | 1024 | DDIM 50-step | [multidiffusion_sdxl_ddim_ref_colab.ipynb](colab_multidiffusion_experiments/multidiffusion_sdxl_ddim_ref_colab.ipynb) | `full1073`, `DDIM_EXPERIMENT_PROFILE="ref_g75_b20_full_view"` | 85.1970 | 13.1230 | 0.9334 | 27.4549 | 27.7253 | 37.4222 | 40154.0122 | Full1073 measured. SDXL base + DDIM 50-step + guidance=7.5 + random-background bootstrapping=20 + full latent view 128/128, tức 1 view ở 1024. So với paper Ref. DDIM SDXL `PAPER-T3-MD-REF-DDIM-SDXL`: FID tệ hơn 11.4270, IS thấp hơn 3.1870, CLIP(fg) cao hơn 3.2949, CLIP(bg) thấp hơn 0.3847, Time nhanh hơn 13.1778s/ảnh. |
| `MD-SD3-DDIM-REF-F1073` | [ ] | Pending | MultiDiffusion | Ref/original | SD3 | 1024 | DDIM / paper default |  | `full1073` hoặc mini trước |  |  |  |  |  |  |  | Chưa có notebook. |
| `MD-SD3-FFM-NAIVE-F1073` | [ ] | Notebook ready | MultiDiffusion | Naive runtime-safe | SD3 | 1024 | Flash Flow Match | [multidiffusion_sd3_flashflowmatch_naive_colab.ipynb](colab_multidiffusion_experiments/multidiffusion_sd3_flashflowmatch_naive_colab.ipynb) | `smoke_bs2` -> `full1073`, `CFG-MD-SD3-FFM-G0-B2-V128FULL` |  |  |  |  |  |  |  | Notebook đã tạo. SD3 Medium + Flash-SD3, guidance=0.0, `t_index_list=[0, 4, 12, 25, 37]`, random-background bootstrap=2, background mask ở đầu danh sách mask/prompt, default full latent view 128/128. Cần `HF_TOKEN` có quyền truy cập SD3 Medium. |
| `OURS-SD15-LCM-F1073` | [ ] | Pending | AnchorDraw | Ours full | SD1.5 | 512 | LCM LoRA |  | `full1073` |  |  |  |  |  |  |  | Chưa implement/run. |
| `OURS-SD15-HYPSD-F1073` | [ ] | Pending | AnchorDraw | Ours full | SD1.5 | 512 | Hyper-SD 4-step LoRA |  | `full1073` |  |  |  |  |  |  |  | Chưa implement/run. |
| `OURS-SDXL-EULER-F1073` | [ ] | Pending | AnchorDraw | Ours full | SDXL | 1024 | SDXL-Lightning 4-step UNet + Euler trailing |  | `full1073` |  |  |  |  |  |  |  | Chưa implement/run. |
| `OURS-SD3-FFM-F1073` | [ ] | Pending | AnchorDraw | Ours full | SD3 | 1024 | Flash Flow Match |  | `full1073` |  |  |  |  |  |  |  | Chưa implement/run. |

## Diagnostic / Excluded Results

Các dòng dưới đây không nên dùng làm dòng baseline chính trong bảng paper, nhưng nên giữ lại để debug và phân tích protocol.

| Exp ID | Check | Status | Method | Variant | Model | Resolution | Sampler / Accel | FID↓ | IS↑ | IS std | CLIP(fg)↑ | CLIP(bg)↑ | Time(s)↓ | Total time(s) | Why excluded |
|---|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `DIAG-MD-SDXL-EULER-REPAIRED-F1073` | [x] | Excluded | MultiDiffusion | Repaired/optimized wrapper | SDXL | 1024 | SDXL-Lightning 4-step UNet + Euler trailing | 70.8000 | 17.4631 | 1.3640 | 26.3942 | 27.9071 | 3.7816 | 4057.6739 | Bản diagnostic cũ; tên `128/16` gây nhầm dù với latent 128x128 chỉ còn 1 view. Không dùng làm MD-naive paper-style. |

## Configs Cần Chạy Để Khóa Protocol

| Exp ID | Check | Status | Target | Model | Sampler | Steps / t_index | Guidance | Bootstrap | View config | Expected purpose | FID↓ | IS↑ | CLIP(fg)↑ | CLIP(bg)↑ | Time(s)↓ |
|---|---|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|
| `CFG-MD-SD15-HYPSD-G0-B1` | [ ] | Pending | MD Hyper-SD H1 | SD1.5 | Hyper-SD | 4 steps | 0.0 | 1 | 1 view at 512 | Kiểm tra adapter Hyper-SD với guidance thấp hơn. Bootstrap ở đây là random-background của MD, không phải white. |  |  |  |  |  |
| `CFG-MD-SD15-HYPSD-G1-B1` | [x] | Done | MD Hyper-SD H2 | SD1.5 | Hyper-SD | 4 steps | 1.0 | 1 | 1 view at 512 | Current run; random-background bootstrap. Chất lượng/metric tốt bất thường nên cần đối chiếu bằng ablation. | 74.5424 | 16.4142 | 26.6949 | 27.8592 | 3.8352 |
| `CFG-MD-SD15-HYPSD-G0-B0` | [ ] | Pending | MD Hyper-SD H0 | SD1.5 | Hyper-SD | 4 steps | 0.0 | 0 | 1 view at 512 | Tách ảnh hưởng bootstrap bằng cách tắt random-background bootstrap. |  |  |  |  |  |
| `CFG-MD-SD15-HYPSD-G0-B2` | [ ] | Pending | MD Hyper-SD H2B2 | SD1.5 | Hyper-SD | 4 steps | 0.0 | 2 | 1 view at 512 | Tách ảnh hưởng số bước random-background bootstrap. |  |  |  |  |  |
| `CFG-MD-SD15-DDIM-G75-B20` | [x] | Done | MD DDIM Ref. | SD1.5 | DDIM | 50 steps | 7.5 | 20 | 1 view at 512 | Protocol gần nhất với `Baseline/MultiDiffusion-master/MultiDiffusion-master/region_based.py` và paper `MultiDiffusion (Ref.) DDIM`. Dùng [multidiffusion_sd15_ddim_ref_colab.ipynb](colab_multidiffusion_experiments/multidiffusion_sd15_ddim_ref_colab.ipynb), đặt `DDIM_EXPERIMENT_PROFILE="ref_g75_b20"`. Full1073 measured; so với paper Ref. DDIM SD1.5 thì tốt hơn ở tất cả metric hiện đo được, nhưng cần note khác biệt phần cứng/metric runtime. | 64.5922 | 18.7259 | 27.2731 | 27.8353 | 8.8414 |
| `CFG-MD-SD15-LCM-G1-B1` | [x] | Done | MD LCM L1 | SD1.5 | LCM | `[0, 4, 12, 25, 37]` | 1.0 | 1 | 1 view at 512 | Dùng [multidiffusion_sd15_lcm_full1073_kaggle.ipynb](kaggle_multidiffusion_experiments/multidiffusion_sd15_lcm_full1073_kaggle.ipynb) với `LCM_EXPERIMENT_PROFILE="g1_b1"`. Measured full1073: adapter LCM với random-background bootstrap; so cùng sampler LCM với SemanticDraw nhưng không phải official MD release. | 77.2768 | 16.8130 | 25.9738 | 27.0796 | 5.5294 |
| `CFG-MD-SD15-LCM-G1-B0` | [ ] | Pending | MD LCM L0 | SD1.5 | LCM | `[0, 4, 12, 25, 37]` | 1.0 | 0 | 1 view at 512 | Dùng cùng notebook MD+LCM adaptive; đặt `RUN_PROFILE="full1073"` và `LCM_EXPERIMENT_PROFILE="g1_b0"`. Mục tiêu là tắt random-background bootstrap để tách ảnh hưởng bootstrap so với `CFG-MD-SD15-LCM-G1-B1`. |  |  |  |  |  |
| `CFG-MD-SDXL-DDIM-G75-B20-V128FULL` | [x] | Done | MD SDXL DDIM Ref. | SDXL | DDIM | 50 steps | 7.5 | 20 | `128/full`, 1 view | Full1073 measured cho `MD-SDXL-DDIM-REF-F1073`. Giữ masks/prompts có background ở đầu và random-background bootstrap theo MultiDiffusion gốc; dùng full latent view cho benchmark SDXL vuông 1024. | 85.1970 | 13.1230 | 27.4549 | 27.7253 | 37.4222 |
| `CFG-MD-SDXL-DDIM-G75-B20-V64S8` | [x] | Excluded | MD SDXL DDIM sliding-window diagnostic | SDXL | DDIM | 50 steps | 7.5 | 20 | `64/8`, 81 views | Diagnostic/stress profile, đặt `DDIM_EXPERIMENT_PROFILE="diagnostic_g75_b20_v64s8"`. A100 smoke test mất khoảng 37 phút/sample nên không dùng làm benchmark chính cho bảng paper. |  |  |  |  |  |
| `CFG-MD-SDXL-EULER-G1-B2-V64S8` | [x] | Excluded | MD SDXL Euler panorama stress | SDXL | Euler trailing + Lightning | `[0, 4, 12, 25, 37]` | 1.0 | 2 | `64/8`, 81 views | Smoke `global_time_ids_v2` vẫn mosaic; dùng để debug/stress sliding-window, không dùng làm paper main metric cho SDXL native 1024. |  |  |  |  |  |
| `CFG-MD-SDXL-EULER-G1-B2-V128FULL` | [x] | Done | MD SDXL Euler native full-view | SDXL | Euler trailing + Lightning | `[0, 4, 12, 25, 37]` | 1.0 | 2 | `128/full`, 1 view | Full1073 measured cho `MD-SDXL-EULER-NAIVE-F1073`: không đen/NaN/mosaic. Dùng random-background bootstrap theo MD, không dùng SemanticDraw white bootstrap. | 70.8000 | 17.4631 | 26.3942 | 27.9071 | 4.0686 |
| `CFG-MD-SDXL-EULER-G1-B2-V128S16` | [x] | Excluded | MD SDXL Euler old naming | SDXL | Euler trailing + Lightning | `[0, 4, 12, 25, 37]` | 1.0 | 2 | `128/16`, 1 view | Excluded để tránh nhầm tên; với latent 128x128 thì window 128 vẫn chỉ có 1 view, tương đương `V128FULL` về view count. | 70.8000 | 17.4631 | 26.3942 | 27.9071 | 3.7816 |
| `CFG-MD-SD3-FFM-G0-B2-V128FULL` | [ ] | Notebook ready | MD SD3 Flash-SD3 naive | SD3 | Flash Flow Match | `[0, 4, 12, 25, 37]` | 0.0 | 2 | `128/full`, 1 view | Profile chính cho [multidiffusion_sd3_flashflowmatch_naive_colab.ipynb](colab_multidiffusion_experiments/multidiffusion_sd3_flashflowmatch_naive_colab.ipynb). Mặc định chạy `smoke_bs2`; đổi `RUN_PROFILE="full1073"`, `COLAB_GPU_MODE="a100_80gb"`, `RUN_METRICS_AFTER_GENERATION=True` để benchmark. |  |  |  |  |  |
| `CFG-MD-SD3-FFM-G0-B2-V64S8` | [ ] | Diagnostic only | MD SD3 Flash-SD3 sliding-window | SD3 | Flash Flow Match | `[0, 4, 12, 25, 37]` | 0.0 | 2 | `64/8`, 81 views | Chỉ dùng để trace sliding-window/stress test. Không đặt làm profile mặc định vì SDXL 64/8 từng rất chậm và dễ làm lệch runtime benchmark. |  |  |  |  |  |

## Paper Reference Numbers

Các số dưới đây được chép từ bảng paper/screenshot để đối chiếu nhanh. Trước khi đưa vào báo cáo chính thức, nên verify lại trực tiếp từ PDF.

| Paper ID | Paper table | Method | Model | Resolution | Sampler | FID↓ | IS↑ | CLIP(fg)↑ | CLIP(bg)↑ | Time(s)↓ |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|
| `PAPER-T2-MD-REF-DDIM-SD15` | Table 2 | MultiDiffusion Ref. | SD1.5 | 512 | DDIM | 70.93 | 16.24 | 24.09 | 27.55 | 14.1 |
| `PAPER-T2-MD-HYPSD-SD15` | Table 2 | MultiDiffusion MD | SD1.5 | 512 | Hyper-SD | 168.34 | 10.12 | 20.08 | 15.90 | 1.7 |
| `PAPER-T2-SEM-HYPSD-SD15` | Table 2 | SemanticDraw | SD1.5 | 512 | Hyper-SD | 98.60 | 14.90 | 24.48 | 23.31 | 1.3 |
| `PAPER-LCM-MD-REF-DDIM-SD15` | LCM table | MultiDiffusion Ref. | SD1.5 | 512 | DDIM | 70.93 | 16.24 | 24.09 | 27.55 | 14.1 |
| `PAPER-LCM-MD-LCM-SD15` | LCM table | MultiDiffusion MD | SD1.5 | 512 | LCM | 270.55 | 2.653 | 22.53 | 19.63 | 1.7 |
| `PAPER-LCM-SEM-LCM-SD15` | LCM table | SemanticDraw | SD1.5 | 512 | LCM | 93.93 | 14.12 | 24.14 | 24.00 | 1.3 |
| `PAPER-T3-MD-REF-DDIM-SDXL` | Table 3 | MultiDiffusion Ref. | SDXL | 1024 | DDIM | 73.77 | 16.31 | 24.16 | 28.11 | 50.6 |
| `PAPER-T3-MD-EULER-SDXL` | Table 3 | MultiDiffusion MD | SDXL | 1024 | Euler Discrete | 572.95 | 1.328 | 21.02 | 17.36 | 4.3 |
| `PAPER-T3-SEM-EULER-SDXL` | Table 3 | SemanticDraw | SDXL | 1024 | Euler Discrete | 84.27 | 15.04 | 24.19 | 24.22 | 3.6 |

## Ablation Checklist Cho AnchorDraw

| Exp ID | Check | Status | Variant | Semantic Anchor | Adaptive Bilateral Masking | Distillation++ | FID↓ | IS↑ | CLIP(fg)↑ | CLIP(bg)↑ | Note |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|
| `ABL-NOSTAB` | [ ] | Pending | No stabilization |  |  |  |  |  |  |  | Baseline nội bộ cho ablation. |
| `ABL-LATENT-PREAVG` | [ ] | Pending | + Latent pre-averaging |  |  |  |  |  |  |  | Tương ứng proposal stabilization cơ bản. |
| `ABL-MASK-CENTER` | [ ] | Pending | + Mask-centering bootstrapping |  |  |  |  |  |  |  | Kiểm tra đóng góp của mask-centering. |
| `ABL-QMASK-S4` | [ ] | Pending | + Quantized masks, sigma = 4 |  |  |  |  |  |  |  | Kiểm tra mask quantization. |
| `ABL-ANCHOR` | [ ] | Pending | Semantic Anchor only | Yes |  |  |  |  |  |  | Ablation module riêng. |
| `ABL-ABM` | [ ] | Pending | Adaptive Bilateral Masking only |  | Yes |  |  |  |  |  | Ablation module riêng. |
| `ABL-DISTILL` | [ ] | Pending | Distillation++ only |  |  | Yes |  |  |  |  | Ablation module riêng. |
| `ABL-ANCHOR-ABM` | [ ] | Pending | Semantic Anchor + Adaptive Bilateral Masking | Yes | Yes |  |  |  |  |  | Combination ablation. |
| `ABL-ABM-DISTILL` | [ ] | Pending | Adaptive Bilateral Masking + Distillation++ |  | Yes | Yes |  |  |  |  | Combination ablation. |
| `ABL-ANCHOR-DISTILL` | [ ] | Pending | Semantic Anchor + Distillation++ | Yes |  | Yes |  |  |  |  | Combination ablation. |
| `ABL-FULL` | [ ] | Pending | Full AnchorDraw | Yes | Yes | Yes |  |  |  |  | Bản đề xuất đầy đủ. |
