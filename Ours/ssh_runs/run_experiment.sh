#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
cd -- "$REPO_ROOT"

usage() {
  printf '%s\n' \
    'Usage:' \
    '  bash Ours/ssh_runs/run_experiment.sh --list' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment ID --coco-root DIR --output-root DIR [--tag NAME] [--python BIN] [--gpu-mode MODE] [--run-profile PROFILE]' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment cleanfid --dataset-root DIR --output-root DIR [--experiments ID1,ID2] [--tag NAME]' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment pytorch_fid --export-dir DIR --coco-root DIR --output-root DIR [--reference-dir DIR] [--tag NAME]' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment ttur_fid --export-dir DIR --coco-root DIR --output-root DIR [--reference-dir DIR] [--ttur-fid-file FILE] [--tag NAME]' \
    '' \
    'This is the only Bash entrypoint. Use --list for all experiment IDs.'
}

list_profiles() {
  for directory in profiles diagnostics metric_profiles; do
    for source in "$SCRIPT_DIR/$directory"/*.py; do
      source="${source##*/}"
      printf '%s\n' "${source%.py}"
    done
  done
}

experiment=""; coco_root=""; output_root=""; run_tag=""
dataset_root=""; export_dir=""; reference_dir=""; metric_experiments=""; ttur_fid_file=""
python_bin="${PYTHON_BIN:-python3}"
gpu_mode="high_vram_24gb"
run_profile_override=""

while (($#)); do
  case "$1" in
    --list) list_profiles; exit 0 ;;
    --experiment) experiment="${2:?Missing experiment ID}"; shift 2 ;;
    --coco-root) coco_root="${2:?Missing COCO directory}"; shift 2 ;;
    --output-root) output_root="${2:?Missing output directory}"; shift 2 ;;
    --dataset-root) dataset_root="${2:?Missing dataset root}"; shift 2 ;;
    --export-dir) export_dir="${2:?Missing export directory}"; shift 2 ;;
    --reference-dir) reference_dir="${2:?Missing reference directory}"; shift 2 ;;
    --experiments) metric_experiments="${2:?Missing experiment names}"; shift 2 ;;
    --ttur-fid-file) ttur_fid_file="${2:?Missing TTUR file}"; shift 2 ;;
    --tag) run_tag="${2:?Missing run tag}"; shift 2 ;;
    --python) python_bin="${2:?Missing Python command}"; shift 2 ;;
    --gpu-mode) gpu_mode="${2:?Missing GPU mode}"; shift 2 ;;
    --run-profile) run_profile_override="${2:?Missing run profile}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$experiment" || -z "$output_root" || ! "$experiment" =~ ^[a-z0-9_]+$ ]]; then
  usage >&2; exit 2
fi
if [[ -f "$SCRIPT_DIR/profiles/$experiment.py" ]]; then
  profile_kind="full1073"
  profile_script="$SCRIPT_DIR/profiles/$experiment.py"
elif [[ -f "$SCRIPT_DIR/diagnostics/$experiment.py" ]]; then
  profile_kind="diagnostic"
  profile_script="$SCRIPT_DIR/diagnostics/$experiment.py"
elif [[ -f "$SCRIPT_DIR/metric_profiles/$experiment.py" ]]; then
  profile_kind="metric"
  profile_script="$SCRIPT_DIR/metric_profiles/$experiment.py"
else
  echo "Unknown experiment: $experiment" >&2; exit 2
fi
if [[ -n "$run_tag" && ! "$run_tag" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
  echo "Invalid run tag: $run_tag" >&2; exit 2
fi
if [[ -n "$run_profile_override" && ! "$run_profile_override" =~ ^[a-z0-9_]+$ ]]; then
  echo "Invalid run profile: $run_profile_override" >&2; exit 2
fi
if [[ "$gpu_mode" != "low_vram" && "$gpu_mode" != "high_vram_24gb" && "$gpu_mode" != "a100_80gb" ]]; then
  echo "Invalid GPU mode: $gpu_mode" >&2; exit 2
fi
if [[ "$profile_kind" == "full1073" && -n "$run_profile_override" && "$run_profile_override" != "full1073" ]]; then
  echo "Benchmark profile is fixed to full1073" >&2; exit 2
fi
if [[ "$profile_kind" == "metric" && -n "$run_profile_override" ]]; then
  echo "--run-profile applies only to generation diagnostics" >&2; exit 2
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python command not found: $python_bin" >&2; exit 2
fi
for command_name in sha256sum find sort xargs cmp; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required provenance command not found: $command_name" >&2; exit 2
  fi
done
"$python_bin" "$SCRIPT_DIR/validate_ports.py" >/dev/null
echo "[OK] All notebook-to-terminal ports match their source cells"
if [[ "$profile_kind" == "full1073" ]]; then
  "$python_bin" -c 'import torch, diffusers, open_clip, pycocotools; from torchmetrics.image.fid import FrechetInceptionDistance; from torchmetrics.image.inception import InceptionScore'
  if [[ "$experiment" == "ours_sd15_hypersd_pladis" ]]; then "$python_bin" -c 'import entmax'; fi
fi

if [[ "$profile_kind" == "metric" && "$experiment" == "cleanfid" ]]; then
  if [[ ! -d "$dataset_root/reference_images" ]]; then
    echo "CleanFID needs --dataset-root with reference_images/: $dataset_root" >&2; exit 2
  fi
  dataset_root="$(cd -- "$dataset_root" && pwd -P)"
  coco_root="$dataset_root"
elif [[ "$profile_kind" == "metric" ]]; then
  if [[ ! -d "$export_dir/generated_images" || ! -f "$export_dir/metric_generated_manifest.jsonl" ]]; then
    echo "FID export needs generated_images/ and metric_generated_manifest.jsonl: $export_dir" >&2; exit 2
  fi
  if [[ ! -d "$coco_root/val2017" ]]; then
    echo "COCO root needs val2017/: $coco_root" >&2; exit 2
  fi
  export_dir="$(cd -- "$export_dir" && pwd -P)"
  coco_root="$(cd -- "$coco_root" && pwd -P)"
  if [[ -n "$reference_dir" ]]; then
    if [[ ! -d "$reference_dir" ]]; then
      echo "Reference directory missing: $reference_dir" >&2; exit 2
    fi
    reference_dir="$(cd -- "$reference_dir" && pwd -P)"
  fi
else
  if [[ ! -d "$coco_root/val2017" || ! -f "$coco_root/annotations/instances_val2017.json" || ! -f "$coco_root/annotations/captions_val2017.json" ]]; then
    echo "COCO root must contain val2017/ and both val2017 annotation JSON files: $coco_root" >&2; exit 2
  fi
  coco_root="$(cd -- "$coco_root" && pwd -P)"
fi
if [[ -n "$ttur_fid_file" && ! -f "$ttur_fid_file" ]]; then
  echo "TTUR fid.py missing: $ttur_fid_file" >&2; exit 2
fi

mkdir -p -- "$output_root"
output_root="$(cd -- "$output_root" && pwd -P)"
if [[ -z "$run_tag" ]]; then run_tag="$(date -u +%Y%m%dT%H%M%SZ)"; fi
unset PYTHONOPTIMIZE DDIM_EXPERIMENT_PROFILE PRECISION_PROFILE LCM_EXPERIMENT_PROFILE
if [[ "$profile_kind" != "metric" ]]; then
  input_report="$("$python_bin" -u "$SCRIPT_DIR/verify_inputs.py" "$experiment" "$coco_root" "$run_profile_override")"
  printf '%s\n' "$input_report"
fi
run_dir="$output_root/$run_tag/$experiment"
mkdir -p -- "${run_dir%/*}"
if ! mkdir -- "$run_dir"; then
  echo "Run directory already exists; choose another --tag: $run_dir" >&2; exit 2
fi

export ANCHORDRAW_WORK_DIR="$run_dir" COCO_ROOT="$coco_root"
export MPLBACKEND="Agg" PYTHONUNBUFFERED="1"
if [[ "$profile_kind" == "full1073" ]]; then
  export RUN_PROFILE="full1073"
elif [[ "$profile_kind" == "diagnostic" ]]; then
  unset RUN_PROFILE
  case "$experiment" in
    diag_ours_sd15_lcm_anchor_smoke) export RUN_PROFILE="smoke8" ;;
    diag_sem_sdxl_euler_smoke) export RUN_PROFILE="smoke_bs2" ;;
  esac
  if [[ -n "$run_profile_override" ]]; then export RUN_PROFILE="$run_profile_override"; fi
else
  unset RUN_PROFILE
  export METRIC_DATASET_ROOT="$dataset_root" METRIC_EXPERIMENTS="$metric_experiments"
  export METRIC_EXPORT_DIR="$export_dir" METRIC_REFERENCE_DIR="$reference_dir"
  if [[ -n "$ttur_fid_file" ]]; then export TTUR_FID_FILE="$ttur_fid_file"; fi
fi
case "$experiment" in
  md_sd15_lcm) export LCM_EXPERIMENT_PROFILE="g1_b1" ;;
  md_sd15_ddim) export DDIM_EXPERIMENT_PROFILE="ref_g75_b20" PRECISION_PROFILE="fp32_original_autocast" ;;
  md_sdxl_ddim) export DDIM_EXPERIMENT_PROFILE="ref_g75_b20_full_view" ;;
esac
export COLAB_GPU_MODE="$gpu_mode"

echo "[RUN] $experiment | kind=$profile_kind | tag=$run_tag | output=$run_dir"
if [[ "$profile_kind" != "metric" ]]; then printf '%s\n' "$input_report" > "$run_dir/input_verification.json"; fi
git -C "$REPO_ROOT" rev-parse HEAD > "$run_dir/repo_commit.txt" 2>/dev/null || true
git -C "$REPO_ROOT" status --short > "$run_dir/repo_status.txt" 2>/dev/null || true
find "$REPO_ROOT/Baseline" "$REPO_ROOT/Ours/src" "$SCRIPT_DIR" -type f \( -name '*.py' -o -name '*.sh' \) -print0 \
  | sort -z | xargs -0 -r sha256sum > "$run_dir/code_sha256.txt"
sha256sum "$profile_script" > "$run_dir/profile_sha256.txt"
"$python_bin" -m pip freeze > "$run_dir/pip_freeze.txt"
if [[ ! -s "$run_dir/code_sha256.txt" || ! -s "$run_dir/pip_freeze.txt" ]]; then
  echo "Missing code or Python environment snapshot: $run_dir" >&2; exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$run_dir/nvidia_smi.txt" 2>&1 || true
fi

"$python_bin" -u "$profile_script" 2>&1 | tee "$run_dir/console.log"
if [[ "$profile_kind" == "full1073" ]]; then
  find "$REPO_ROOT/Baseline" "$REPO_ROOT/Ours/src" "$SCRIPT_DIR" -type f \( -name '*.py' -o -name '*.sh' \) -print0 \
    | sort -z | xargs -0 -r sha256sum > "$run_dir/code_sha256_after.txt"
  "$python_bin" -m pip freeze > "$run_dir/pip_freeze_after.txt"
  if ! cmp -s "$run_dir/code_sha256.txt" "$run_dir/code_sha256_after.txt" ||
     ! cmp -s "$run_dir/pip_freeze.txt" "$run_dir/pip_freeze_after.txt"; then
    echo "Code or Python environment changed during the benchmark: $run_dir" >&2; exit 2
  fi
fi
case "$profile_kind" in
  full1073)
    "$python_bin" -u "$SCRIPT_DIR/verify_run.py" "$experiment" "$run_dir" 2>&1 | tee "$run_dir/verification.log"
    "$python_bin" -u "$SCRIPT_DIR/prepare_metric_export.py" "$experiment" "$run_dir" 2>&1 | tee "$run_dir/metric_export.log"
    ;;
  diagnostic)
    "$python_bin" -u "$SCRIPT_DIR/verify_diagnostic.py" "$experiment" "$run_dir" 2>&1 | tee "$run_dir/verification.log"
    ;;
  metric)
    "$python_bin" -u "$SCRIPT_DIR/verify_metric.py" "$experiment" "$run_dir" 2>&1 | tee "$run_dir/verification.log"
    ;;
esac
if [[ "$profile_kind" == "full1073" ]]; then printf '%s\n' "$experiment" > "$run_dir/benchmark_complete.txt"; fi
echo "[PASS] $experiment: $run_dir"
