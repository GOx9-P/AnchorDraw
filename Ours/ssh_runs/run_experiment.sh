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
    '  bash Ours/ssh_runs/run_experiment.sh --experiment ID [--coco-root DIR] [--output-root DIR] [--tag NAME] [--python BIN] [--gpu-mode MODE] [--run-profile PROFILE]' \
    '  bash Ours/ssh_runs/run_experiment.sh --prepare-coco [--coco-root DIR] [--output-root DIR] [--python BIN]' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment cleanfid --dataset-root DIR [--output-root DIR] [--experiments ID1,ID2] [--tag NAME]' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment pytorch_fid --export-dir DIR [--coco-root DIR] [--output-root DIR] [--reference-dir DIR] [--tag NAME]' \
    '  bash Ours/ssh_runs/run_experiment.sh --experiment ttur_fid --export-dir DIR [--coco-root DIR] [--output-root DIR] [--reference-dir DIR] [--ttur-fid-file FILE] [--tag NAME]' \
    '' \
    'No arguments: run all 38 non-SD3 experiments sequentially (14 benchmarks, 21 diagnostics, 3 FID tools).' \
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
prepare_coco_only=false
dry_run=false

while (($#)); do
  case "$1" in
    --list) list_profiles; exit 0 ;;
    --prepare-coco) prepare_coco_only=true; shift ;;
    --dry-run) dry_run=true; shift ;;
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

if [[ -z "$output_root" ]]; then output_root="${ANCHORDRAW_OUTPUT_ROOT:-${HOME:?HOME is required}/anchordraw_runs}"; fi
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
if [[ "$dry_run" == true && "$profile_kind" != "suite" ]]; then
  echo "--dry-run applies only to the complete suite" >&2; exit 2
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
  )
  diagnostic_ids=()
  for source in "$SCRIPT_DIR"/diagnostics/*.py; do
    diagnostic_ids+=("${source##*/}")
  done
  diagnostic_ids=("${diagnostic_ids[@]%.py}")
  if [[ "${#benchmark_ids[@]}" -ne 14 || "${#diagnostic_ids[@]}" -ne 21 ]]; then
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
  if [[ -z "$run_tag" ]]; then run_tag="$(date -u +%Y%m%dT%H%M%SZ)"; fi
  printf '[SUITE] tag=%s | diagnostics=%s | benchmarks=%s | FID tools=3\n' \
    "$run_tag" "${#diagnostic_ids[@]}" "${#benchmark_ids[@]}"
  if [[ "$dry_run" == true ]]; then
    printf '[PLAN] %s\n' "${diagnostic_ids[@]}" "${benchmark_ids[@]}" pytorch_fid cleanfid ttur_fid
    exit 0
  fi

  # Fail before any GPU work when packages needed by later FID/PLADIS stages are absent.
  "$python_bin" - <<'PY'
import importlib.util
modules = ("torch", "diffusers", "open_clip", "pycocotools", "torchmetrics",
           "entmax", "cleanfid", "pytorch_fid", "tensorflow", "imageio", "PIL")
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing suite dependencies: " + ", ".join(missing))
PY
  mkdir -p -- "$output_root"
  output_root="$(cd -- "$output_root" && pwd -P)"
  coco_root="$("$python_bin" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$coco_root")"
  suite_root="$output_root/$run_tag"
  if [[ -e "$suite_root" ]]; then
    echo "Suite output already exists; choose a new --tag: $suite_root" >&2; exit 2
  fi
  mkdir -- "$suite_root"
  echo "[SUITE] output=$suite_root | COCO=$coco_root"
  prepare_coco

  run_suite_profile() {
    local id="$1"
    shift
    echo "[SUITE] Starting $id"
    bash "$SCRIPT_DIR/run_experiment.sh" --experiment "$id" \
      --coco-root "$coco_root" --output-root "$output_root" \
      --tag "$run_tag" --python "$python_bin" --gpu-mode "$gpu_mode" "$@"
  }

  for id in "${diagnostic_ids[@]}"; do run_suite_profile "$id"; done
  for id in "${benchmark_ids[@]}"; do run_suite_profile "$id"; done

  # These two notebooks specifically evaluate the verified SemanticDraw SD1.5 LCM export.
  export_dir="$suite_root/sem_sd15_lcm/metric_export"
  if [[ ! -f "$export_dir/metric_generated_manifest.jsonl" ]]; then
    echo "SemanticDraw SD1.5 LCM metric export is missing: $export_dir" >&2; exit 2
  fi
  run_suite_profile pytorch_fid --export-dir "$export_dir"
  reference_dir="$suite_root/pytorch_fid/pytorch_fid_eval/sem_sd15_lcm/reference_images/sem_sd15_lcm"
  if [[ ! -d "$reference_dir" ]]; then
    echo "PyTorch FID reference images are missing: $reference_dir" >&2; exit 2
  fi

  # CleanFID's source notebook expects reference_images/ID and ID/generated_images.
  # Build both resolutions from the verified COCO manifest and link all 14 exports.
  dataset_root="$suite_root/_cleanfid_dataset"
  "$python_bin" - "$suite_root" "$coco_root" "$dataset_root" "${benchmark_ids[@]}" <<'PY'
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
  run_suite_profile cleanfid --dataset-root "$dataset_root"
  "$python_bin" - "$suite_root/cleanfid" "${benchmark_ids[@]}" <<'PY'
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
  run_suite_profile ttur_fid --export-dir "$export_dir" --reference-dir "$reference_dir"
  printf '%s\n' "${diagnostic_ids[@]}" "${benchmark_ids[@]}" pytorch_fid cleanfid ttur_fid \
    > "$suite_root/suite_complete.txt"
  echo "[SUITE PASS] All 38 non-SD3 experiments completed: $suite_root"
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
