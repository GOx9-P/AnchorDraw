#!/usr/bin/env python3
"""Standalone SSH benchmark; generated from the source notebook named below."""
from __future__ import annotations
import importlib.util
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
WORK_DIR = Path(os.environ["ANCHORDRAW_WORK_DIR"]).resolve()
SSH_COCO_ROOT = Path(os.environ["COCO_ROOT"]).resolve()
COCO_ROOT = SSH_COCO_ROOT
WORK_DIR.mkdir(parents=True, exist_ok=True)

def Markdown(value):
    return value

def display(*values):
    for value in values:
        if isinstance(value, str):
            print(value)

# Notebook: Ours/kaggle_metric_eval/pytorch_fid_semanticdraw_sd15_lcm_full1073_kaggle.ipynb
# Notebook SHA256: eb603379284631ce30984461fdde580973868bff5be817502a7e8c6fe0d6f014

# --- Source notebook cell 4 ---
from pathlib import Path
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any

import pandas as pd
from PIL import Image

try:
    import torch
except Exception:
    torch = None

EXPERIMENT_NAME = (Path(os.environ["METRIC_EXPORT_DIR"]).parent.name if Path(os.environ["METRIC_EXPORT_DIR"]).name == "metric_export" else Path(os.environ["METRIC_EXPORT_DIR"]).name)
EXPECTED_NUM_IMAGES = 1073
TARGET_SIZE = (512, 512)  # (height, width) cho SD1.5 trong experiment này.

# Để rỗng để notebook tự tìm trong /kaggle/input và /kaggle/working.
EXPORT_DIR_OVERRIDE = ""
REFERENCE_DIR_OVERRIDE = ""
COCO_VAL2017_DIR_OVERRIDE = ""

# Nếu reference chưa có, notebook sẽ build vào /kaggle/working.
AUTO_BUILD_REFERENCE_IF_MISSING = True
REBUILD_REFERENCE = False

# Config cho pytorch-fid.
FID_DIMS = 2048
FID_BATCH_SIZE = 50
FID_DEVICE = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"

OUTPUT_ROOT = Path(str(WORK_DIR / 'pytorch_fid_eval')) / EXPERIMENT_NAME
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print("EXPERIMENT_NAME  :", EXPERIMENT_NAME)
print("EXPECTED_IMAGES  :", EXPECTED_NUM_IMAGES)
print("TARGET_SIZE      :", TARGET_SIZE)
print("FID_DIMS         :", FID_DIMS)
print("FID_BATCH_SIZE   :", FID_BATCH_SIZE)
print("FID_DEVICE       :", FID_DEVICE)
print("OUTPUT_ROOT      :", OUTPUT_ROOT)
EXPORT_DIR_OVERRIDE = os.environ["METRIC_EXPORT_DIR"]
REFERENCE_DIR_OVERRIDE = os.environ.get("METRIC_REFERENCE_DIR", "")
COCO_VAL2017_DIR_OVERRIDE = str(SSH_COCO_ROOT / "val2017")

# --- Source notebook cell 6 ---
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def find_export_dir() -> Path:
    if EXPORT_DIR_OVERRIDE:
        path = Path(EXPORT_DIR_OVERRIDE)
        if not (path / "generated_images").exists():
            raise FileNotFoundError(f"EXPORT_DIR_OVERRIDE không có generated_images: {path}")
        return path

    search_roots = [Path("/kaggle/input"), Path(str(WORK_DIR))]
    candidates = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob(EXPERIMENT_NAME):
            if path.is_dir() and (path / "generated_images").exists():
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy export folder. Hãy upload dataset chứa folder "
            f"{EXPERIMENT_NAME}/generated_images hoặc set EXPORT_DIR_OVERRIDE."
        )

    candidates = sorted(candidates, key=lambda p: (str(p).startswith("/kaggle/input"), len(str(p))), reverse=True)
    return candidates[0]


EXPORT_DIR = find_export_dir()
GENERATED_DIR = EXPORT_DIR / "generated_images"
MANIFEST_PATH = EXPORT_DIR / "metric_generated_manifest.jsonl"

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Thiếu metric_generated_manifest.jsonl trong export: {MANIFEST_PATH}")

records = read_jsonl(MANIFEST_PATH)
generated_files = sorted(GENERATED_DIR.glob("*.png"))

print("EXPORT_DIR       :", EXPORT_DIR)
print("GENERATED_DIR    :", GENERATED_DIR)
print("MANIFEST_PATH    :", MANIFEST_PATH)
print("manifest records :", len(records))
print("generated images :", len(generated_files))

if len(records) != EXPECTED_NUM_IMAGES:
    print(f"[WARN] Manifest có {len(records)} record, khác EXPECTED_NUM_IMAGES={EXPECTED_NUM_IMAGES}.")
if len(generated_files) != len(records):
    print(f"[WARN] Số ảnh generated={len(generated_files)} khác số record={len(records)}.")

# --- Source notebook cell 8 ---
def generated_to_reference_name(generated_name: str) -> str:
    stem = Path(generated_name).stem
    if stem.endswith("__generated"):
        stem = stem[: -len("__generated")] + "__reference"
    elif stem.endswith("_generated"):
        stem = stem[: -len("_generated")] + "_reference"
    else:
        stem = stem + "__reference"
    return stem + ".png"


def resolve_generated_path(record: dict[str, Any]) -> Path:
    rel = record.get("generated_image_relative_path")
    if rel:
        path = EXPORT_DIR / str(rel)
        if path.exists():
            return path

    image_id = int(record["image_id"])
    sample_id = str(record.get("sample_id", ""))
    patterns = [
        f"*{sample_id}*generated*.png",
        f"*{image_id:012d}*generated*.png",
        f"*{image_id:012d}*.png",
    ]
    for pattern in patterns:
        matches = sorted(GENERATED_DIR.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Không tìm thấy generated image cho image_id={image_id}")


def valid_image(path: Path, expected_size: tuple[int, int] | None = None) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
        if expected_size is not None:
            eh, ew = expected_size
            return (w, h) == (ew, eh)
        return True
    except Exception:
        return False


def find_reference_dir() -> Path | None:
    if REFERENCE_DIR_OVERRIDE:
        path = Path(REFERENCE_DIR_OVERRIDE)
        if not path.exists():
            raise FileNotFoundError(f"REFERENCE_DIR_OVERRIDE không tồn tại: {path}")
        return path

    candidates = [
        EXPORT_DIR.parent / "reference_images" / EXPERIMENT_NAME,
        EXPORT_DIR / "reference_images",
    ]
    for root in [Path("/kaggle/input"), Path(str(WORK_DIR))]:
        if root.exists():
            candidates.extend(root.rglob(f"reference_images/{EXPERIMENT_NAME}"))

    for path in candidates:
        if path.exists() and any(path.glob("*.png")):
            return path
    return None


def find_coco_val2017_dir() -> Path:
    first_file = str(records[0].get("file_name", "000000000776.jpg"))
    if COCO_VAL2017_DIR_OVERRIDE:
        path = Path(COCO_VAL2017_DIR_OVERRIDE)
        if not (path / first_file).exists():
            raise FileNotFoundError(f"COCO_VAL2017_DIR_OVERRIDE không chứa {first_file}: {path}")
        return path

    candidates = [
        Path(str(WORK_DIR / 'COCO/val2017')),
        Path("/kaggle/input/coco-2017-dataset/coco2017/val2017"),
        Path("/kaggle/input/coco-2017-dataset/val2017"),
        Path("/kaggle/input/coco2017/val2017"),
        Path("/kaggle/input/coco-val2017/val2017"),
    ]
    for root in [Path("/kaggle/input"), Path(str(WORK_DIR))]:
        if root.exists():
            candidates.extend(root.rglob("val2017"))

    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and (path / first_file).exists():
            return path

    raise FileNotFoundError(
        "Không tìm thấy COCO val2017 để build reference. "
        "Hãy attach/upload COCO val2017 hoặc upload sẵn reference_images/<experiment_name>."
    )


reference_dir = find_reference_dir()

if reference_dir is None:
    if not AUTO_BUILD_REFERENCE_IF_MISSING:
        raise FileNotFoundError(
            f"Thiếu reference_images/{EXPERIMENT_NAME}. "
            "Bật AUTO_BUILD_REFERENCE_IF_MISSING hoặc upload reference folder."
        )
    reference_dir = OUTPUT_ROOT / "reference_images" / EXPERIMENT_NAME

print("REFERENCE_DIR    :", reference_dir)

if REBUILD_REFERENCE and reference_dir.exists() and str(reference_dir).startswith(str(WORK_DIR)):
    shutil.rmtree(reference_dir)

existing_refs = sorted(reference_dir.glob("*.png")) if reference_dir.exists() else []
need_build = len(existing_refs) < len(records)
if need_build and not str(reference_dir).startswith(str(WORK_DIR)):
    print("[WARN] Reference folder trong /kaggle/input bị thiếu ảnh và không thể ghi thêm.")
    reference_dir = OUTPUT_ROOT / "reference_images" / EXPERIMENT_NAME
    existing_refs = sorted(reference_dir.glob("*.png")) if reference_dir.exists() else []
    need_build = len(existing_refs) < len(records)
    print("[INFO] Chuyển sang build reference tại:", reference_dir)

if need_build:
    coco_val_dir = find_coco_val2017_dir()
    reference_dir.mkdir(parents=True, exist_ok=True)
    print("COCO_VAL2017_DIR :", coco_val_dir)
    print("[INFO] Building reference images...")

    manifest_rows = []
    missing = []
    for idx, record in enumerate(records):
        gen_path = resolve_generated_path(record)
        file_name = str(record.get("file_name") or f"{int(record['image_id']):012d}.jpg")
        src_path = coco_val_dir / file_name
        if not src_path.exists():
            missing.append(file_name)
            continue

        target_size = record.get("target_size") or list(TARGET_SIZE)
        target_h, target_w = int(target_size[0]), int(target_size[1])
        ref_path = reference_dir / generated_to_reference_name(gen_path.name)

        if not valid_image(ref_path, (target_h, target_w)):
            tmp_path = ref_path.with_suffix(ref_path.suffix + ".tmp")
            with Image.open(src_path) as img:
                img = img.convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)
                img.save(tmp_path, format="PNG")
            tmp_path.replace(ref_path)

        manifest_rows.append({
            "index": idx,
            "sample_id": record.get("sample_id"),
            "image_id": int(record["image_id"]),
            "file_name": file_name,
            "generated_image": gen_path.name,
            "reference_image": ref_path.name,
            "target_height": target_h,
            "target_width": target_w,
        })

    if missing:
        raise FileNotFoundError(f"Thiếu {len(missing)} ảnh COCO, ví dụ: {missing[:10]}")

    with (reference_dir / "reference_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
else:
    print("[OK] Reference folder đã có sẵn, không cần build lại.")

reference_files = sorted(reference_dir.glob("*.png"))
print("reference images:", len(reference_files))
if len(reference_files) != len(generated_files):
    print(f"[WARN] reference images={len(reference_files)} khác generated images={len(generated_files)}.")

# --- Source notebook cell 10 ---
def inspect_folder(folder: Path, expected_size: tuple[int, int] | None = None) -> dict[str, Any]:
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.jpeg"))
    invalid = []
    sizes = {}
    for path in files:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                size = img.size
            sizes[size] = sizes.get(size, 0) + 1
            if expected_size is not None:
                eh, ew = expected_size
                if size != (ew, eh):
                    invalid.append((str(path), f"size={size}"))
        except Exception as exc:
            invalid.append((str(path), repr(exc)))
    return {"folder": str(folder), "num_images": len(files), "sizes": sizes, "invalid": invalid}


gen_report = inspect_folder(GENERATED_DIR, TARGET_SIZE)
ref_report = inspect_folder(reference_dir, TARGET_SIZE)

print("Generated report:", gen_report["num_images"], gen_report["sizes"])
print("Reference report:", ref_report["num_images"], ref_report["sizes"])

if gen_report["invalid"]:
    raise RuntimeError(f"Generated folder có ảnh lỗi/sai size: {gen_report['invalid'][:5]}")
if ref_report["invalid"]:
    raise RuntimeError(f"Reference folder có ảnh lỗi/sai size: {ref_report['invalid'][:5]}")
if gen_report["num_images"] != ref_report["num_images"]:
    raise RuntimeError(
        f"Số ảnh không khớp: generated={gen_report['num_images']}, reference={ref_report['num_images']}"
    )

print("[OK] Folder ảnh hợp lệ để đo pytorch-fid.")

# --- Source notebook cell 12 ---
cmd = [
    sys.executable,
    "-m",
    "pytorch_fid",
    str(GENERATED_DIR),
    str(reference_dir),
    "--dims",
    str(FID_DIMS),
    "--batch-size",
    str(FID_BATCH_SIZE),
    "--device",
    FID_DEVICE,
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, text=True, capture_output=True)
print("STDOUT:\n", result.stdout)
print("STDERR:\n", result.stderr)

if result.returncode != 0:
    raise RuntimeError(f"pytorch-fid failed with return code {result.returncode}")

combined = result.stdout + "\n" + result.stderr
matches = re.findall(r"FID:\s*([0-9]+(?:\.[0-9]+)?)", combined)
if not matches:
    matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)", result.stdout.strip())
if not matches:
    raise RuntimeError("Không parse được FID từ output của pytorch-fid.")

fid_score = float(matches[-1])
print("FID =", fid_score)

# --- Source notebook cell 14 ---
report = {
    "experiment": EXPERIMENT_NAME,
    "metric": "FID",
    "tool": "mseitzer/pytorch-fid",
    "fid": fid_score,
    "dims": FID_DIMS,
    "batch_size": FID_BATCH_SIZE,
    "device": FID_DEVICE,
    "target_size_hw": list(TARGET_SIZE),
    "export_dir": str(EXPORT_DIR),
    "generated_dir": str(GENERATED_DIR),
    "reference_dir": str(reference_dir),
    "manifest_path": str(MANIFEST_PATH),
    "num_generated": int(gen_report["num_images"]),
    "num_reference": int(ref_report["num_images"]),
    "generated_sizes": {str(k): v for k, v in gen_report["sizes"].items()},
    "reference_sizes": {str(k): v for k, v in ref_report["sizes"].items()},
    "command": cmd,
}

json_path = OUTPUT_ROOT / "pytorch_fid_results.json"
csv_path = OUTPUT_ROOT / "pytorch_fid_results.csv"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(report.keys()))
    writer.writeheader()
    writer.writerow(report)

zip_base = Path(str(WORK_DIR)) / f"{EXPERIMENT_NAME}__pytorch_fid_eval"
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=OUTPUT_ROOT)

display(pd.DataFrame([{
    "Experiment": EXPERIMENT_NAME,
    "Metric": "FID↓",
    "Tool": "pytorch-fid",
    "Value": fid_score,
    "Generated": gen_report["num_images"],
    "Reference": ref_report["num_images"],
    "Zip": zip_path,
}]))

print("Saved JSON:", json_path)
print("Saved CSV :", csv_path)
print("Saved ZIP :", zip_path)
