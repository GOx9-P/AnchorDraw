#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
cd -- "$REPO_ROOT"

usage() {
  printf '%s\n' \
    'Usage:' \
    '  ./Ours/ssh_runs/run_experiment.sh [--output-root DIR] [--coco-root DIR] [--tag NAME] [--python BIN] [--gpu-mode MODE]' \
    '  ./Ours/ssh_runs/run_experiment.sh --dry-run' \
    '  bash Ours/ssh_runs/run_experiment.sh --list' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment ID [--base-seed N] [--coco-root DIR] [--output-root DIR] [--tag NAME] [--python BIN] [--gpu-mode MODE] [--run-profile PROFILE]' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment sem_sd3_ffm|ours_sd3_ffm|md_sd3_ffm --tag NAME --base-seed N --resume' \
    '  bash Ours/ssh_runs/run_experiment.sh --prepare-coco [--coco-root DIR] [--output-root DIR] [--python BIN]' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment cleanfid --dataset-root DIR [--output-root DIR] [--experiments ID1,ID2] [--tag NAME]' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment pytorch_fid --export-dir DIR [--coco-root DIR] [--output-root DIR] [--reference-dir DIR] [--tag NAME]' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment ttur_fid --export-dir DIR [--coco-root DIR] [--output-root DIR] [--reference-dir DIR] [--ttur-fid-file FILE] [--tag NAME]' \
    '' \
    'No arguments: run 21 diagnostics once, then 17 benchmarks and 3 FID tools for each of 3 base seeds (2024, 10002024, 20002024).' \
    'This is the only Bash entrypoint. Use --list for all experiment IDs.' \
    'Default GPU preset: a100_80gb (A100 80 GB).' \
    'SD3 requires accepted access to stabilityai/stable-diffusion-3-medium-diffusers and HF_TOKEN or a saved hf auth login on the SSH host.'
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
gpu_mode="a100_80gb"
run_profile_override=""
base_seed="2024"
base_seed_explicit=false
prepare_coco_only=false
dry_run=false
resume_run=false

while (($#)); do
  case "$1" in
    --list) list_profiles; exit 0 ;;
    --prepare-coco) prepare_coco_only=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    --resume) resume_run=true; shift ;;
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
    --base-seed) base_seed="${2:?Missing base seed}"; base_seed_explicit=true; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$output_root" ]]; then output_root="${ANCHORDRAW_OUTPUT_ROOT:-$REPO_ROOT/anchordraw_runs}"; fi
if [[ -z "$coco_root" ]]; then coco_root="${ANCHORDRAW_COCO_ROOT:-$output_root/_coco2017}"; fi

if [[ "$prepare_coco_only" == true && -n "$experiment" ]]; then
  echo "--prepare-coco cannot be combined with --experiment" >&2; exit 2
fi
if [[ -n "$experiment" && ! "$experiment" =~ ^[a-z0-9_]+$ ]]; then
  echo "Invalid experiment ID: $experiment" >&2; exit 2
fi
if [[ "$prepare_coco_only" == true ]]; then
  profile_kind="prepare_coco"
elif [[ -z "$experiment" ]]; then
  profile_kind="suite"
elif [[ -f "$SCRIPT_DIR/profiles/$experiment.py" ]]; then
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
if [[ "$profile_kind" == "suite" && ( -n "$dataset_root" || -n "$export_dir" || -n "$reference_dir" || -n "$metric_experiments" || -n "$ttur_fid_file" || -n "$run_profile_override" ) ]]; then
  echo "Dataset/export/reference and run-profile options require --experiment" >&2; exit 2
fi
if [[ "$profile_kind" == "suite" && "$base_seed_explicit" == true ]]; then
  echo "The complete suite uses its fixed three base seeds; --base-seed is for one experiment" >&2; exit 2
fi
if [[ ( "$profile_kind" == "diagnostic" || "$profile_kind" == "prepare_coco" ) && "$base_seed_explicit" == true ]]; then
  echo "--base-seed applies only to benchmark and FID profiles" >&2; exit 2
fi
if [[ ! "$base_seed" =~ ^[0-9]+$ ]] || (( ${#base_seed} > 10 )); then
  echo "Invalid base seed: $base_seed" >&2; exit 2
fi
base_seed="$((10#$base_seed))"
if (( base_seed < 0 || base_seed + 1072 >= 4294967296 )); then
  echo "Base seed must leave room for 1073 distinct per-image seeds" >&2; exit 2
fi
if [[ "$dry_run" == true && "$profile_kind" != "suite" ]]; then
  echo "--dry-run applies only to the complete suite" >&2; exit 2
fi
if [[ "$resume_run" == true && "$experiment" != *_sd3_ffm ]]; then
  echo "--resume is supported only for an incomplete SD3 benchmark" >&2; exit 2
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

check_sd3_hf_access() {
  "$python_bin" - "$1" <<'PY'
import ast
import os
import sys
from pathlib import Path

profile = Path(sys.argv[1])
assignments = {}
for node in ast.parse(profile.read_text(encoding="utf-8")).body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"MODEL_ID", "MODEL_REVISION"}:
                assignments[target.id] = ast.literal_eval(node.value)

model_id = assignments.get("MODEL_ID")
revision = assignments.get("MODEL_REVISION")
if not isinstance(model_id, str) or not isinstance(revision, str):
    sys.exit("[HF] Cannot read the pinned SD3 model ID and revision from the profile")

try:
    from huggingface_hub import get_token, hf_hub_download
except ImportError:
    sys.exit("[HF] huggingface_hub is missing from the selected Python environment")

token = get_token()
if not token:
    sys.exit(
        "[HF] SD3 requires Hugging Face access. Accept the model conditions at "
        "https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers "
        "and use HF_TOKEN or run `hf auth login` on the SSH host."
    )
if os.environ.get("HF_HUB_DISABLE_IMPLICIT_TOKEN", "").lower() in {"1", "true", "yes", "on"}:
    sys.exit("[HF] Unset HF_HUB_DISABLE_IMPLICIT_TOKEN so SD3 model loading can use your token")

try:
    hf_hub_download(
        repo_id=model_id,
        revision=revision,
        filename="model_index.json",
        token=token,
    )
except Exception as exc:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    detail = f"HTTP {status}" if status is not None else type(exc).__name__
    sys.exit(
        f"[HF] Cannot access SD3 model at pinned revision ({detail}). "
        "Check model access, token permissions, and network/cache availability."
    )
print("[HF] SD3 model access verified at the pinned revision")
PY
}

prepare_coco() {
  "$python_bin" - "$coco_root" "${1:-normal}" <<'PY'
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError
from urllib.request import Request, urlopen

root = Path(sys.argv[1]).expanduser().resolve()
repair = sys.argv[2] == "repair"
root.mkdir(parents=True, exist_ok=True)
val_dir = root / "val2017"
ann_dir = root / "annotations"
cache_dir = root / ".downloads"
val_url = os.environ.get("ANCHORDRAW_COCO_VAL_URL", "http://images.cocodataset.org/zips/val2017.zip")
ann_url = os.environ.get("ANCHORDRAW_COCO_ANN_URL", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
# Archive lengths and SHA-256 values from TensorFlow Datasets:
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/coco_captions.txt
val_size = 815585330
ann_size = 252907541
val_sha256 = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
ann_sha256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
required_annotations = ("instances_val2017.json", "captions_val2017.json")


def valid_zip(path: Path, expected_size: int, expected_sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            return False
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def download(url: str, target: Path, expected_size: int, expected_sha256: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if valid_zip(target, expected_size, expected_sha256):
        print(f"[COCO] Reusing verified archive: {target}", flush=True)
        return target
    partial = target.with_name(target.name + ".part")
    if valid_zip(partial, expected_size, expected_sha256):
        os.replace(partial, target)
        return target
    for attempt in range(1, 4):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        print(f"[COCO] Download {url} (attempt {attempt}/3, resume={offset} bytes)", flush=True)
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                status = getattr(response, "status", None)
                if offset and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"Unexpected resume range: {content_range}")
                    mode = "ab"
                else:
                    mode = "wb"
                with partial.open(mode) as output:
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        except HTTPError as exc:
            if exc.code == 416:
                if valid_zip(partial, expected_size, expected_sha256):
                    os.replace(partial, target)
                    return target
                partial.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"COCO download failed: {url}") from exc
            continue
        except OSError as exc:
            if attempt == 3:
                raise RuntimeError(f"COCO download failed: {url}") from exc
            continue
        if valid_zip(partial, expected_size, expected_sha256):
            os.replace(partial, target)
            return target
        print(f"[COCO] Archive incomplete or invalid; retrying: {partial}", flush=True)
    raise RuntimeError(f"COCO archive is incomplete or invalid: {partial}")


def extract_member(archive: zipfile.ZipFile, member: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".anchordraw_part")
    try:
        with archive.open(member) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def annotations_usable() -> bool:
    for name in required_annotations:
        target = ann_dir / name
        try:
            with target.open(encoding="utf-8") as source:
                payload = json.load(source)
            if not isinstance(payload, dict) or not isinstance(payload.get("images"), list) or not isinstance(payload.get("annotations"), list):
                return False
        except (OSError, UnicodeError, ValueError):
            return False
    return True


with (root / ".anchordraw_coco.lock").open("a+b") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    image_count = sum(1 for path in val_dir.glob("*.jpg") if path.is_file())
    if repair or image_count < 5000:
        archive_path = download(val_url, cache_dir / "val2017.zip", val_size, val_sha256)
        with zipfile.ZipFile(archive_path) as archive:
            images = []
            for member in archive.infolist():
                parts = PurePosixPath(member.filename).parts
                if len(parts) == 2 and parts[0] == "val2017" and re.fullmatch(r"[0-9]{12}\.jpg", parts[1]):
                    images.append((member, parts[1]))
            if len(images) != 5000 or len({name for _, name in images}) != 5000:
                raise RuntimeError("Unexpected val2017 archive: expected 5000 unique JPEG names")
            for member, name in images:
                target = val_dir / name
                if repair or not target.is_file() or target.stat().st_size != member.file_size:
                    extract_member(archive, member, target)
        print("[COCO] val2017 images extracted", flush=True)

    if repair or not annotations_usable():
        archive_path = download(ann_url, cache_dir / "annotations_trainval2017.zip", ann_size, ann_sha256)
        with zipfile.ZipFile(archive_path) as archive:
            for name in required_annotations:
                member_name = f"annotations/{name}"
                try:
                    member = archive.getinfo(member_name)
                except KeyError as exc:
                    raise RuntimeError(f"COCO annotation archive lacks {member_name}") from exc
                target = ann_dir / name
                extract_member(archive, member, target)
        print("[COCO] val2017 annotations extracted", flush=True)

    if sum(1 for path in val_dir.glob("*.jpg") if path.is_file()) < 5000 or not annotations_usable():
        raise RuntimeError(f"Incomplete COCO val2017 data: {root}")
    print(f"[COCO] Ready: {root}", flush=True)
PY
}

if [[ "$prepare_coco_only" == true ]]; then
  prepare_coco
  exit 0
fi

if [[ "$profile_kind" == "suite" ]]; then
  # Baselines precede their matched AnchorDraw runs so verify_run can compare them.
  benchmark_ids=(
    sem_sd15_lcm ours_sd15_lcm ours_sd15_lcm_dpp
    sem_sd15_hypersd ours_sd15_hypersd ours_sd15_hypersd_dpp ours_sd15_hypersd_pladis
    sem_sdxl_euler ours_sdxl_euler
    md_sd15_lcm md_sd15_hypersd md_sd15_ddim md_sdxl_euler md_sdxl_ddim
    sem_sd3_ffm ours_sd3_ffm md_sd3_ffm
  )
  fid_benchmark_ids=("${benchmark_ids[@]:0:14}")
  diagnostic_ids=()
  for source in "$SCRIPT_DIR"/diagnostics/*.py; do
    diagnostic_ids+=("${source##*/}")
  done
  diagnostic_ids=("${diagnostic_ids[@]%.py}")
  if [[ "${#benchmark_ids[@]}" -ne 17 || "${#diagnostic_ids[@]}" -ne 21 ]]; then
    echo "Unexpected benchmark or diagnostic profile count; inspect the suite before running" >&2; exit 2
  fi
  for id in "${benchmark_ids[@]}"; do
    if [[ ! -f "$SCRIPT_DIR/profiles/$id.py" ]]; then
      echo "Missing benchmark profile: $id" >&2; exit 2
    fi
  done
  for id in cleanfid pytorch_fid ttur_fid; do
    if [[ ! -f "$SCRIPT_DIR/metric_profiles/$id.py" ]]; then
      echo "Missing metric profile: $id" >&2; exit 2
    fi
  done
  suite_seeds=(2024 10002024 20002024)
  # A no-argument invocation picks up the latest compatible suite, including
  # suites made before checkpointing. A new --tag remains an explicit reset.
  output_root="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$output_root")"
  coco_root="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$coco_root")"
  if [[ -z "$run_tag" ]]; then
    run_tag="$("$python_bin" "$SCRIPT_DIR/suite_checkpoint.py" select "$output_root" "$gpu_mode" "$coco_root")"
    if [[ -z "$run_tag" ]]; then run_tag="$(date -u +%Y%m%dT%H%M%SZ)"; fi
  fi
  printf '[SUITE] tag=%s | gpu_mode=%s | diagnostics=%s once | benchmarks=%s x 3 seeds | FID tools=3 x 3 seeds\n' \
    "$run_tag" "$gpu_mode" "${#diagnostic_ids[@]}" "${#benchmark_ids[@]}"
  if [[ "$dry_run" == true ]]; then
    printf '[PLAN] diagnostic %s\n' "${diagnostic_ids[@]}"
    for planned_seed in "${suite_seeds[@]}"; do
      for planned_id in "${benchmark_ids[@]}" pytorch_fid cleanfid ttur_fid; do
        printf '[PLAN] seed_%s %s\n' "$planned_seed" "$planned_id"
      done
    done
    printf '[PLAN] aggregate 3 seed reports\n'
    exit 0
  fi

  # A stale generated profile invalidates every experiment. Detect that once
  # before downloading COCO or starting any GPU work, rather than reporting
  # the same failure for each seed/profile.
  if ! "$python_bin" "$SCRIPT_DIR/validate_ports.py" >/dev/null; then
    echo "[SUITE ERROR] Notebook-to-terminal profile validation failed; no experiments started" >&2
    exit 2
  fi
  echo "[SUITE] Notebook-to-terminal profiles verified"

  # A missing optional package is recorded against its profile, so later
  # independent experiments can still run.
  mkdir -p -- "$output_root"
  output_root="$(cd -- "$output_root" && pwd -P)"
  if ! command -v flock >/dev/null 2>&1; then
    echo "Suite checkpointing requires flock on the SSH host" >&2; exit 2
  fi
  exec 9>"$output_root/.anchordraw_suite.lock"
  if ! flock -n 9; then
    echo "Another AnchorDraw suite is using this output root: $output_root" >&2; exit 2
  fi
  suite_root="$output_root/$run_tag"
  if [[ -e "$suite_root" && ! -d "$suite_root" ]]; then
    echo "Suite output is not a directory: $suite_root" >&2; exit 2
  fi
  if [[ -d "$suite_root" ]]; then
    echo "[SUITE] Continuing existing tag: $run_tag"
  else
    mkdir -- "$suite_root"
  fi
  "$python_bin" - "$suite_root/suite_metadata.json" "$gpu_mode" "$coco_root" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metadata = {"gpu_mode": sys.argv[2], "coco_root": sys.argv[3]}
if path.is_file():
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved != metadata:
        raise SystemExit(f"Suite tag has different GPU/COCO configuration: {path}")
else:
    path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
PY
  echo "[SUITE] output=$suite_root | COCO=$coco_root"
  if ! check_sd3_hf_access "$SCRIPT_DIR/profiles/sem_sd3_ffm.py"; then
    echo "[SUITE WARN] SD3 access is unavailable now; non-SD3 profiles will still run" >&2
  fi
  prepare_coco
  "$python_bin" "$SCRIPT_DIR/suite_checkpoint.py" context \
    "$suite_root/.checkpoint_context.json" "$coco_root" "$gpu_mode"
  if [[ -f "$suite_root/suite_complete.txt" ]]; then
    mkdir -p -- "$suite_root/_superseded/suite_markers"
    mv -- "$suite_root/suite_complete.txt" \
      "$suite_root/_superseded/suite_markers/suite_complete_$(date -u +%Y%m%dT%H%M%S)_$$.txt"
  fi

  profile_ran=false
  force_profile=false
  archive_attempt() {
    local id="$1" seed_label="$2" run_dir="$3" log_file="$4"
    local archive="$suite_root/_superseded/$seed_label/$id/$(date -u +%Y%m%dT%H%M%S)_${BASHPID}_$RANDOM"
    mkdir -p -- "$archive"
    if [[ -e "$run_dir" ]]; then mv -- "$run_dir" "$archive/run"; fi
    if [[ -e "$log_file" ]]; then mv -- "$log_file" "$archive/runner.log"; fi
    if [[ -e "$log_file.ok" ]]; then mv -- "$log_file.ok" "$archive/runner.log.ok"; fi
    echo "[ARCHIVE] $seed_label/$id -> $archive"
  }

  run_suite_profile() {
    local id="$1"
    shift
    local seed_label="${active_base_seed:-diagnostic}"
    local log_dir="$suite_root/runner_logs/$seed_label"
    local log_file="$log_dir/$id.log"
    local profile_kind="full1073"
    local run_dir="$suite_root/seed_$seed_label/$id"
    if [[ "$seed_label" == "diagnostic" ]]; then
      profile_kind="diagnostic"
      run_dir="$suite_root/$id"
    elif [[ "$id" == cleanfid || "$id" == pytorch_fid || "$id" == ttur_fid ]]; then
      profile_kind="metric"
    fi
    profile_ran=false
    mkdir -p -- "$log_dir"
    local -a seed_args=()
    if [[ -n "${active_base_seed:-}" ]]; then seed_args=(--base-seed "$active_base_seed"); fi
    local -a command=(bash "$SCRIPT_DIR/run_experiment.sh" --experiment "$id" \
      --coco-root "$coco_root" --output-root "$output_root" \
      --tag "$run_tag" --python "$python_bin" --gpu-mode "$gpu_mode" "${seed_args[@]}" "$@")
    if [[ ( -f "$log_file.ok" || -s "$run_dir/verification.log" ) && "$force_profile" != true ]]; then
      if "$python_bin" "$SCRIPT_DIR/suite_checkpoint.py" check \
          "$id" "$profile_kind" "$run_dir" "$seed_label" "$suite_root/.checkpoint_context.json"; then
        touch -- "$log_file.ok"
        echo "[SKIP VERIFIED] $seed_label/$id | output=$run_dir"
        return 0
      fi
      echo "[STALE] $seed_label/$id: saved result no longer passes the checkpoint"
    fi
    if [[ -d "$run_dir" && ! -f "$log_file.ok" && "$id" == *_sd3_ffm && "$force_profile" != true ]] &&
       "$python_bin" "$SCRIPT_DIR/suite_checkpoint.py" resume \
         "$run_dir" "$seed_label" "$suite_root/.checkpoint_context.json" >/dev/null 2>&1; then
      echo "[RESUME] $seed_label/$id: continuing incomplete SD3 images"
      profile_ran=true
      if "${command[@]}" --resume >>"$log_file" 2>&1; then
        touch -- "$log_file.ok"
        echo "[PASS] $seed_label/$id after resume | log=$log_file"
        return 0
      else
        local resume_status=$?
        printf '%s\t%s\t%s\t%s\n' "$seed_label" "$id" "$resume_status" "$log_file" >> "$suite_root/suite_failures.tsv"
        echo "[FAIL] $seed_label/$id (resume exit=$resume_status) | log=$log_file" >&2
        tail -n 3 -- "$log_file" >&2 || true
        return 1
      fi
    fi
    if [[ -e "$run_dir" || -e "$log_file" || -e "$log_file.ok" ]]; then
      archive_attempt "$id" "$seed_label" "$run_dir" "$log_file"
    fi
    profile_ran=true
    echo "[SUITE] Starting $seed_label/$id"
    if "${command[@]}" >"$log_file" 2>&1; then
      touch -- "$log_file.ok"
      echo "[PASS] $seed_label/$id | log=$log_file"
      return 0
    else
      local status=$?
      if [[ "$id" == *_sd3_ffm ]]; then
        local image_failure_log=""
        image_failure_log="$(find "$suite_root/seed_$active_base_seed/$id/anchordraw_runs" -name failed_images.jsonl -type f -print -quit 2>/dev/null || true)"
        if [[ -n "$image_failure_log" ]]; then
          echo "[RETRY] $seed_label/$id: resuming only missing SD3 indices"
          if "${command[@]}" --resume >>"$log_file" 2>&1; then
            touch -- "$log_file.ok"
            echo "[PASS] $seed_label/$id after resume | log=$log_file"
            return 0
          else
            status=$?
          fi
        fi
      fi
      printf '%s\t%s\t%s\t%s\n' "$seed_label" "$id" "$status" "$log_file" >> "$suite_root/suite_failures.tsv"
      echo "[FAIL] $seed_label/$id (exit=$status) | log=$log_file" >&2
      tail -n 3 -- "$log_file" >&2 || true
      return 1
    fi
  }

  printf 'seed\tprofile\texit_code\tlog\n' > "$suite_root/suite_failures.tsv"
  suite_failed=false
  for id in "${diagnostic_ids[@]}"; do
    if ! run_suite_profile "$id"; then suite_failed=true; fi
  done
  for active_base_seed in "${suite_seeds[@]}"; do
    seed_root="$suite_root/seed_$active_base_seed"
    seed_benchmarks_ok=true
    seed_benchmarks_reran=false
    sem_lcm_reran=false
    pytorch_reran=false
    if [[ -f "$seed_root/seed_complete.txt" ]]; then
      mkdir -p -- "$suite_root/_superseded/$active_base_seed/seed_markers"
      mv -- "$seed_root/seed_complete.txt" \
        "$suite_root/_superseded/$active_base_seed/seed_markers/seed_complete_$(date -u +%Y%m%dT%H%M%S)_$$.txt"
    fi
    for id in "${benchmark_ids[@]}"; do
      if ! run_suite_profile "$id"; then
        suite_failed=true
        if [[ "$id" != *_sd3_ffm ]]; then seed_benchmarks_ok=false; fi
      fi
      if [[ "$profile_ran" == true && "$id" != *_sd3_ffm ]]; then seed_benchmarks_reran=true; fi
      if [[ "$profile_ran" == true && "$id" == sem_sd15_lcm ]]; then sem_lcm_reran=true; fi
    done

  # These two notebooks specifically evaluate the verified SemanticDraw SD1.5 LCM export.
  export_dir="$seed_root/sem_sd15_lcm/metric_export"
  if [[ -f "$export_dir/metric_generated_manifest.jsonl" ]]; then
    force_profile="$sem_lcm_reran"
    if ! run_suite_profile pytorch_fid --export-dir "$export_dir"; then suite_failed=true; fi
    pytorch_reran="$profile_ran"
    force_profile=false
  else
    echo "[SKIP] seed_$active_base_seed/pytorch_fid: SemanticDraw SD1.5 LCM export missing" >&2
    suite_failed=true
  fi
  reference_dir="$seed_root/pytorch_fid/pytorch_fid_eval/sem_sd15_lcm/reference_images/sem_sd15_lcm"
  if [[ -f "$suite_root/runner_logs/$active_base_seed/pytorch_fid.log.ok" && -d "$reference_dir" ]]; then
    force_profile="${pytorch_reran:-false}"
    if ! run_suite_profile ttur_fid --export-dir "$export_dir" --reference-dir "$reference_dir"; then suite_failed=true; fi
    force_profile=false
  else
    echo "[SKIP] seed_$active_base_seed/ttur_fid: PyTorch FID reference missing" >&2
    suite_failed=true
  fi

  # CleanFID's source notebook expects reference_images/ID and ID/generated_images.
  # Build both resolutions from the verified COCO manifest and link all 14 exports.
  dataset_root="$seed_root/_cleanfid_dataset"
  prepare_cleanfid_dataset() {
  "$python_bin" - "$seed_root" "$coco_root" "$dataset_root" "${fid_benchmark_ids[@]}" <<'PY'
import json
import sys
from pathlib import Path
from PIL import Image

suite, coco, dataset = (Path(arg).resolve() for arg in sys.argv[1:4])
ids = sys.argv[4:]
dataset.mkdir(parents=True, exist_ok=False)
(dataset / "reference_images").mkdir()
canonical = {}

def link(target, path):
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"CleanFID link already exists: {path}")
    path.symlink_to(target.resolve(), target_is_directory=True)

for profile in ids:
    run_dir = suite / profile
    summaries = list(run_dir.rglob("metric_export_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one metric export for {profile}, found {len(summaries)}")
    export = summaries[0].parent
    manifest = export / "metric_generated_manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    family = "sdxl" if "sdxl" in profile else "sd15"
    expected_size = (1024, 1024) if family == "sdxl" else (512, 512)
    if len(rows) != 1073 or len(list((export / "generated_images").glob("*.png"))) != 1073:
        raise RuntimeError(f"Incomplete CleanFID export: {profile}")
    identity = [(int(row["image_id"]), row["file_name"], tuple(row["target_size"])) for row in rows]
    if any(size != expected_size for _, _, size in identity):
        raise RuntimeError(f"Wrong reference size for {profile}")
    if family not in canonical:
        canonical[family] = identity
        ref_dir = dataset / ("_reference_" + family)
        ref_dir.mkdir()
        for index, (_, file_name, _) in enumerate(identity):
            source = coco / "val2017" / file_name
            target = ref_dir / f"{index:04d}_reference.png"
            with Image.open(source) as image:
                image.convert("RGB").resize(expected_size, Image.Resampling.BILINEAR).save(target)
        if len(list(ref_dir.glob("*.png"))) != 1073:
            raise RuntimeError(f"Incomplete {family} CleanFID references")
    elif identity != canonical[family]:
        raise RuntimeError(f"Different COCO samples or order for {profile}")
    link(export, dataset / profile)
    link(dataset / ("_reference_" + family), dataset / "reference_images" / profile)
print(f"[CleanFID] Prepared {len(ids)} verified experiment pairs: {dataset}", flush=True)
PY
  }
  verify_cleanfid_reports() {
  "$python_bin" - "$seed_root/cleanfid" "${fid_benchmark_ids[@]}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected = set(sys.argv[2:])
reports = list(run_dir.rglob("cleanfid_results.json"))
if len(reports) != 1:
    raise SystemExit(f"Expected one CleanFID report, found {len(reports)}")
rows = json.loads(reports[0].read_text(encoding="utf-8"))
found = [row.get("experiment") for row in rows]
if len(rows) != len(expected) or set(found) != expected:
    raise SystemExit(f"CleanFID experiment list mismatch: {found}")
if any(row.get("num_generated") != 1073 or row.get("num_reference") != 1073 for row in rows):
    raise SystemExit("CleanFID did not evaluate 1073 generated and reference images for every benchmark")
print("[CleanFID] Verified all 14 benchmark reports", flush=True)
PY
  }
  if [[ "$seed_benchmarks_ok" == true ]]; then
    cleanfid_reusable=false
    cleanfid_run_dir="$seed_root/cleanfid"
    cleanfid_ok="$suite_root/runner_logs/$active_base_seed/cleanfid.log.ok"
    if [[ "$seed_benchmarks_reran" == false && -f "$cleanfid_ok" ]] &&
       "$python_bin" "$SCRIPT_DIR/suite_checkpoint.py" check cleanfid metric \
         "$cleanfid_run_dir" "$active_base_seed" "$suite_root/.checkpoint_context.json" &&
       verify_cleanfid_reports >/dev/null 2>&1; then
      cleanfid_reusable=true
    fi
    if [[ "$cleanfid_reusable" == false && -e "$dataset_root" ]]; then
      dataset_archive="$suite_root/_superseded/$active_base_seed/_cleanfid_dataset/$(date -u +%Y%m%dT%H%M%S)_${BASHPID}_$RANDOM"
      mkdir -p -- "$dataset_archive"
      mv -- "$dataset_root" "$dataset_archive/dataset"
      echo "[ARCHIVE] seed_$active_base_seed/_cleanfid_dataset -> $dataset_archive"
    fi
    if [[ "$cleanfid_reusable" == true ]] ||
       prepare_cleanfid_dataset >"$suite_root/runner_logs/$active_base_seed/cleanfid_prepare.log" 2>&1; then
      if [[ "$cleanfid_reusable" == false ]]; then force_profile=true; fi
      if run_suite_profile cleanfid --dataset-root "$dataset_root"; then
        if ! verify_cleanfid_reports >"$suite_root/runner_logs/$active_base_seed/cleanfid_verify.log" 2>&1; then
          echo "[FAIL] seed_$active_base_seed/cleanfid verification; see runner_logs" >&2
          suite_failed=true
        fi
      else
        suite_failed=true
      fi
      force_profile=false
    else
      echo "[FAIL] seed_$active_base_seed/cleanfid preparation; see runner_logs" >&2
      suite_failed=true
    fi
  else
    echo "[SKIP] seed_$active_base_seed/cleanfid: one or more 14 source benchmarks missing" >&2
    suite_failed=true
  fi
  seed_complete=true
  for id in "${fid_benchmark_ids[@]}" pytorch_fid cleanfid ttur_fid; do
    if [[ ! -f "$suite_root/runner_logs/$active_base_seed/$id.log.ok" ]]; then seed_complete=false; fi
  done
  if [[ ! -s "$suite_root/runner_logs/$active_base_seed/cleanfid_verify.log" ]] ||
     ! grep -q 'Verified all 14 benchmark reports' "$suite_root/runner_logs/$active_base_seed/cleanfid_verify.log"; then
    seed_complete=false
  fi
  if [[ "$seed_complete" == true ]]; then
    printf '%s\n' "${fid_benchmark_ids[@]}" pytorch_fid cleanfid ttur_fid > "$seed_root/seed_complete.txt"
  fi
  done
  non_sd3_ready=true
  sd3_ready=true
  for seed in "${suite_seeds[@]}"; do
    if [[ ! -f "$suite_root/seed_$seed/seed_complete.txt" ]]; then non_sd3_ready=false; fi
    for id in sem_sd3_ffm ours_sd3_ffm md_sd3_ffm; do
      if [[ ! -f "$suite_root/runner_logs/$seed/$id.log.ok" ]]; then sd3_ready=false; fi
    done
  done
  if [[ "$non_sd3_ready" == true ]]; then
    if ! "$python_bin" -u "$SCRIPT_DIR/aggregate_seeds.py" "$suite_root" "${suite_seeds[@]}" >"$suite_root/aggregate_non_sd3.log" 2>&1; then
      echo "[FAIL] Non-SD3 three-seed aggregation; see aggregate_non_sd3.log" >&2
      suite_failed=true
    fi
  else
    echo "[SKIP] Non-SD3 three-seed aggregation: incomplete seed set" >&2
    suite_failed=true
  fi
  if [[ "$sd3_ready" == true ]]; then
    if ! "$python_bin" -u "$SCRIPT_DIR/aggregate_sd3_seeds.py" \
      --semanticdraw "$suite_root" --anchordraw "$suite_root" --multidiffusion "$suite_root" \
      --output "$suite_root" >"$suite_root/aggregate_sd3.log" 2>&1; then
      echo "[FAIL] SD3 three-seed aggregation; see aggregate_sd3.log" >&2
      suite_failed=true
    fi
  else
    echo "[SKIP] SD3 three-seed aggregation: incomplete seed set" >&2
    suite_failed=true
  fi
  if [[ "$suite_failed" == true ]]; then
    echo "[SUITE INCOMPLETE] Completed independent profiles; inspect $suite_root/suite_failures.tsv and runner_logs" >&2
    exit 1
  fi
  printf '%s\n' "${diagnostic_ids[@]}" "${suite_seeds[@]}" > "$suite_root/suite_complete.txt"
  echo "[SUITE PASS] All three-seed profiles completed: $suite_root"
  exit 0
fi

for command_name in sha256sum find sort xargs cmp; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required provenance command not found: $command_name" >&2; exit 2
  fi
done
"$python_bin" "$SCRIPT_DIR/validate_ports.py" >/dev/null
echo "[OK] All notebook-to-terminal ports match their source cells"
if [[ "$profile_kind" != "metric" || "$experiment" != "cleanfid" ]]; then
  "$python_bin" -c 'from PIL import Image'
fi
if [[ "$profile_kind" == "full1073" ]]; then
  "$python_bin" -c 'import torch, diffusers, open_clip, pycocotools; from torchmetrics.image.fid import FrechetInceptionDistance; from torchmetrics.image.inception import InceptionScore'
  if [[ "$experiment" == "ours_sd15_hypersd_pladis" ]]; then "$python_bin" -c 'import entmax'; fi
  if [[ "$experiment" == *_sd3_ffm ]]; then check_sd3_hf_access "$profile_script"; fi
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
  prepare_coco
  if ! "$python_bin" -u "$SCRIPT_DIR/verify_inputs.py" sem_sd15_lcm "$coco_root" >/dev/null 2>&1; then
    echo "[COCO] Metric reference validation failed; restoring official val2017 files once" >&2
    prepare_coco repair
    "$python_bin" -u "$SCRIPT_DIR/verify_inputs.py" sem_sd15_lcm "$coco_root" >/dev/null
  fi
  export_dir="$(cd -- "$export_dir" && pwd -P)"
  export_seed_file="${export_dir%/*}/base_seed.txt"
  if [[ -f "$export_seed_file" && "$(<"$export_seed_file")" != "$base_seed" ]]; then
    echo "FID export belongs to a different base seed: $export_dir" >&2; exit 2
  fi
  coco_root="$(cd -- "$coco_root" && pwd -P)"
  if [[ -n "$reference_dir" ]]; then
    if [[ ! -d "$reference_dir" ]]; then
      echo "Reference directory missing: $reference_dir" >&2; exit 2
    fi
    reference_dir="$(cd -- "$reference_dir" && pwd -P)"
  fi
else
  if [[ ! -d "$coco_root/val2017" || ! -f "$coco_root/annotations/instances_val2017.json" || ! -f "$coco_root/annotations/captions_val2017.json" ]]; then
    prepare_coco
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
  if ! input_report="$("$python_bin" -u "$SCRIPT_DIR/verify_inputs.py" "$experiment" "$coco_root" "$run_profile_override" 2>/dev/null)"; then
    echo "[COCO] Input validation failed; restoring official val2017 files once" >&2
    prepare_coco repair
    input_report="$("$python_bin" -u "$SCRIPT_DIR/verify_inputs.py" "$experiment" "$coco_root" "$run_profile_override")"
  fi
  printf '%s\n' "$input_report"
fi
if [[ "$profile_kind" == "full1073" || "$profile_kind" == "metric" ]]; then
  run_dir="$output_root/$run_tag/seed_$base_seed/$experiment"
else
  run_dir="$output_root/$run_tag/$experiment"
fi
mkdir -p -- "${run_dir%/*}"
if [[ "$resume_run" == true ]]; then
  if [[ ! -d "$run_dir" || -f "$run_dir/benchmark_complete.txt" ]]; then
    echo "--resume requires an existing incomplete SD3 run: $run_dir" >&2; exit 2
  fi
else
  if ! mkdir -- "$run_dir"; then
    echo "Run directory already exists; choose another --tag: $run_dir" >&2; exit 2
  fi
fi

export ANCHORDRAW_WORK_DIR="$run_dir" COCO_ROOT="$coco_root"
export ANCHORDRAW_BASE_SEED="$base_seed"
export MPLBACKEND="Agg" PYTHONUNBUFFERED="1"
if [[ "$experiment" == *_sd3_ffm ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
fi
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

echo "[RUN] $experiment | kind=$profile_kind | tag=$run_tag | requested_gpu_mode=$gpu_mode | env_gpu_mode=$COLAB_GPU_MODE | output=$run_dir"
if [[ "$resume_run" == true ]]; then
  if [[ "$(<"$run_dir/base_seed.txt")" != "$base_seed" ]] ||
     ! cmp -s <(printf '%s\n' "$input_report") "$run_dir/input_verification.json" ||
     ! cmp -s <(sha256sum "$profile_script") "$run_dir/profile_sha256.txt" ||
     ! cmp -s <("$python_bin" -m pip freeze) "$run_dir/pip_freeze.txt" ||
     ! cmp -s <(find "$REPO_ROOT/Baseline" "$REPO_ROOT/Ours/src" "$SCRIPT_DIR" -type f \( -name '*.py' -o -name '*.sh' \) -print0 \
       | sort -z | xargs -0 -r sha256sum) "$run_dir/code_sha256.txt"; then
    echo "SD3 resume rejected: seed, COCO input, profile, code or Python environment changed: $run_dir" >&2; exit 2
  fi
else
  printf '%s\n' "$base_seed" > "$run_dir/base_seed.txt"
  printf '%s\n' "$gpu_mode" > "$run_dir/gpu_mode.txt"
  if [[ "$profile_kind" != "metric" ]]; then printf '%s\n' "$input_report" > "$run_dir/input_verification.json"; fi
  git -C "$REPO_ROOT" rev-parse HEAD > "$run_dir/repo_commit.txt" 2>/dev/null || true
  git -C "$REPO_ROOT" status --short > "$run_dir/repo_status.txt" 2>/dev/null || true
  find "$REPO_ROOT/Baseline" "$REPO_ROOT/Ours/src" "$SCRIPT_DIR" -type f \( -name '*.py' -o -name '*.sh' \) -print0 \
    | sort -z | xargs -0 -r sha256sum > "$run_dir/code_sha256.txt"
  sha256sum "$profile_script" > "$run_dir/profile_sha256.txt"
  "$python_bin" -m pip freeze > "$run_dir/pip_freeze.txt"
fi
if [[ ! -s "$run_dir/code_sha256.txt" || ! -s "$run_dir/pip_freeze.txt" ]]; then
  echo "Missing code or Python environment snapshot: $run_dir" >&2; exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$run_dir/nvidia_smi.txt" 2>&1 || true
fi

if [[ "$resume_run" == true ]]; then
  "$python_bin" -u "$profile_script" 2>&1 | tee -a "$run_dir/console.log"
else
  "$python_bin" -u "$profile_script" 2>&1 | tee "$run_dir/console.log"
fi
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
    if [[ "$experiment" != *_sd3_ffm ]]; then
      "$python_bin" -u "$SCRIPT_DIR/prepare_metric_export.py" "$experiment" "$run_dir" 2>&1 | tee "$run_dir/metric_export.log"
    fi
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
