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

# Notebook: Ours/kaggle_metric_eval/ttur_tensorflow_fid_semanticdraw_sd15_lcm_full1073_kaggle.ipynb
# Notebook SHA256: bc4102c968348ce1a86a6d6f4a9ff9756848b9fc8081d97ff08bb0ae776ec5b1
import tensorflow as tf

# --- Source notebook cell 4 ---
from pathlib import Path
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
import urllib.request
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from imageio.v2 import imread

EXPERIMENT_NAME = (Path(os.environ["METRIC_EXPORT_DIR"]).parent.name if Path(os.environ["METRIC_EXPORT_DIR"]).name == "metric_export" else Path(os.environ["METRIC_EXPORT_DIR"]).name)
EXPECTED_NUM_IMAGES = 1073
TARGET_SIZE = (512, 512)  # (height, width), đúng với SD1.5 trong experiment này.

# Để rỗng để notebook tự dò trong /kaggle/input và /kaggle/working.
EXPORT_DIR_OVERRIDE = ""
REFERENCE_DIR_OVERRIDE = ""
COCO_VAL2017_DIR_OVERRIDE = ""

# Nếu reference chưa có trong Kaggle Dataset, notebook sẽ tự build từ COCO val2017 nếu COCO được attach/upload.
AUTO_BUILD_REFERENCE_IF_MISSING = True
REBUILD_REFERENCE = False

# TTUR gốc lấy Inception pool_3, 2048 chiều.
TTUR_DIMS = 2048

# Trên Kaggle/TF2, Inception graph TTUR thường giữ input shape batch=1.
# Dùng batch=1 để tránh lỗi: Cannot feed value of shape (B,H,W,3) for tensor shape (1,None,None,3).
# Việc này không đổi công thức FID, chỉ chậm hơn batching.
TTUR_BATCH_SIZE = 1

# Với 1073 ảnh 512x512, low_profile=False ổn trên Kaggle RAM và tránh nhánh low_profile cũ của TTUR.
TTUR_LOW_PROFILE = False

# GPU cho TensorFlow. Để "0" dùng GPU đầu tiên, để "" chạy CPU.
TTUR_GPU = "0"

TTUR_RAW_FID_URL = "https://raw.githubusercontent.com/bioinf-jku/TTUR/master/fid.py"
TTUR_DIR = Path(str(WORK_DIR / 'ttur_tensorflow_fid'))
TTUR_FID_FILE = TTUR_DIR / "fid.py"
INCEPTION_CACHE_DIR = Path(str(WORK_DIR / 'ttur_inception'))
OUTPUT_ROOT = Path(str(WORK_DIR / 'ttur_fid_eval')) / EXPERIMENT_NAME

TTUR_DIR.mkdir(parents=True, exist_ok=True)
INCEPTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

try:
    physical_gpus = tf.config.list_physical_devices("GPU")
    if TTUR_GPU == "":
        tf.config.set_visible_devices([], "GPU")
    elif physical_gpus:
        gpu_index = int(TTUR_GPU)
        tf.config.set_visible_devices(physical_gpus[gpu_index], "GPU")
        tf.config.experimental.set_memory_growth(physical_gpus[gpu_index], True)
except Exception as exc:
    print("[WARN] Không thể set TensorFlow GPU visibility sau khi runtime đã init:", repr(exc))


print("EXPERIMENT_NAME  :", EXPERIMENT_NAME)
print("EXPECTED_IMAGES  :", EXPECTED_NUM_IMAGES)
print("TARGET_SIZE      :", TARGET_SIZE)
print("TTUR_DIMS        :", TTUR_DIMS)
print("TTUR_BATCH_SIZE  :", TTUR_BATCH_SIZE)
print("TTUR_LOW_PROFILE :", TTUR_LOW_PROFILE)
print("TTUR_GPU         :", TTUR_GPU if TTUR_GPU else "CPU only")
print("OUTPUT_ROOT      :", OUTPUT_ROOT)
EXPORT_DIR_OVERRIDE = os.environ["METRIC_EXPORT_DIR"]
REFERENCE_DIR_OVERRIDE = os.environ.get("METRIC_REFERENCE_DIR", "")
COCO_VAL2017_DIR_OVERRIDE = str(SSH_COCO_ROOT / "val2017")
TTUR_FID_FILE = Path(os.environ.get("TTUR_FID_FILE", str(TTUR_FID_FILE))).resolve()

# --- Source notebook cell 6 ---
if not TTUR_FID_FILE.exists():
    print("[INFO] Downloading TTUR fid.py...")
    urllib.request.urlretrieve(TTUR_RAW_FID_URL, TTUR_FID_FILE)

spec = importlib.util.spec_from_file_location("ttur_fid", TTUR_FID_FILE)
ttur_fid = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ttur_fid)

# TF2 compatibility cho fid.py gốc.
tf.compat.v1.disable_eager_execution()
ttur_fid.tf.Session = tf.compat.v1.Session
ttur_fid.tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
if not hasattr(ttur_fid.tf, "import_graph_def"):
    ttur_fid.tf.import_graph_def = tf.compat.v1.import_graph_def

with TTUR_FID_FILE.open("rb") as f:
    import hashlib
    ttur_sha256 = hashlib.sha256(f.read()).hexdigest()

print("TTUR fid.py file:", TTUR_FID_FILE)
print("TTUR fid.py sha256:", ttur_sha256)
print("[OK] TTUR fid.py imported with TF compat aliases.")

# --- Source notebook cell 8 ---
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

    candidates = []
    for root in [Path("/kaggle/input"), Path(str(WORK_DIR))]:
        if root.exists():
            for path in root.rglob(EXPERIMENT_NAME):
                if path.is_dir() and (path / "generated_images").exists():
                    candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy export folder. Hãy upload dataset chứa folder "
            f"{EXPERIMENT_NAME}/generated_images hoặc set EXPORT_DIR_OVERRIDE."
        )

    return sorted(candidates, key=lambda p: (str(p).startswith("/kaggle/input"), len(str(p))), reverse=True)[0]


EXPORT_DIR = find_export_dir()
GENERATED_DIR = EXPORT_DIR / "generated_images"
MANIFEST_PATH = EXPORT_DIR / "metric_generated_manifest.jsonl"

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Thiếu metric_generated_manifest.jsonl trong export: {MANIFEST_PATH}")

records = read_jsonl(MANIFEST_PATH)
generated_files = sorted(GENERATED_DIR.glob("*.png")) + sorted(GENERATED_DIR.glob("*.jpg")) + sorted(GENERATED_DIR.glob("*.jpeg"))

print("EXPORT_DIR       :", EXPORT_DIR)
print("GENERATED_DIR    :", GENERATED_DIR)
print("MANIFEST_PATH    :", MANIFEST_PATH)
print("manifest records :", len(records))
print("generated images :", len(generated_files))

if len(records) != EXPECTED_NUM_IMAGES:
    print(f"[WARN] Manifest có {len(records)} record, khác EXPECTED_NUM_IMAGES={EXPECTED_NUM_IMAGES}.")
if len(generated_files) != len(records):
    print(f"[WARN] Số ảnh generated={len(generated_files)} khác số record={len(records)}.")

# --- Source notebook cell 10 ---
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
    for pattern in [f"*{sample_id}*generated*.png", f"*{image_id:012d}*generated*.png", f"*{image_id:012d}*.png"]:
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
            mode = img.mode
        if mode != "RGB":
            return False
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
        Path(str(WORK_DIR / 'COCO/val2017/val2017')),
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
        raise FileNotFoundError(f"Thiếu reference_images/{EXPERIMENT_NAME}.")
    reference_dir = OUTPUT_ROOT / "reference_images" / EXPERIMENT_NAME

print("REFERENCE_DIR    :", reference_dir)

if REBUILD_REFERENCE and reference_dir.exists() and str(reference_dir).startswith(str(WORK_DIR)):
    shutil.rmtree(reference_dir)

existing_refs = sorted(reference_dir.glob("*.png")) if reference_dir.exists() else []
need_build = len(existing_refs) < len(records)
if need_build and not str(reference_dir).startswith(str(WORK_DIR)):
    print("[WARN] Reference trong /kaggle/input bị thiếu ảnh và không thể ghi thêm.")
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

reference_files = sorted(reference_dir.glob("*.png")) + sorted(reference_dir.glob("*.jpg")) + sorted(reference_dir.glob("*.jpeg"))
print("reference images:", len(reference_files))

# --- Source notebook cell 12 ---
IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")


def list_images(folder: Path) -> list[Path]:
    files = []
    for ext in IMAGE_EXTS:
        files.extend(folder.glob(ext))
    return sorted(files)


def inspect_folder(folder: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    files = list_images(folder)
    invalid = []
    sizes: dict[str, int] = {}
    modes: dict[str, int] = {}
    for path in files:
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                w, h = img.size
                mode = img.mode
            sizes[f"{w}x{h}"] = sizes.get(f"{w}x{h}", 0) + 1
            modes[mode] = modes.get(mode, 0) + 1
            eh, ew = expected_size
            if (w, h) != (ew, eh):
                invalid.append((str(path), f"size={w}x{h}"))
            if mode != "RGB":
                invalid.append((str(path), f"mode={mode}"))
        except Exception as exc:
            invalid.append((str(path), repr(exc)))
    return {"folder": str(folder), "num_images": len(files), "sizes": sizes, "modes": modes, "invalid": invalid}


gen_report = inspect_folder(GENERATED_DIR, TARGET_SIZE)
ref_report = inspect_folder(reference_dir, TARGET_SIZE)

print("Generated report:", gen_report)
print("Reference report:", ref_report)

if gen_report["invalid"]:
    raise RuntimeError(f"Generated folder có ảnh lỗi/sai size/sai mode: {gen_report['invalid'][:5]}")
if ref_report["invalid"]:
    raise RuntimeError(f"Reference folder có ảnh lỗi/sai size/sai mode: {ref_report['invalid'][:5]}")
if gen_report["num_images"] != ref_report["num_images"]:
    raise RuntimeError(f"Số ảnh không khớp: generated={gen_report['num_images']}, reference={ref_report['num_images']}")
if gen_report["num_images"] != len(records):
    raise RuntimeError(f"Số ảnh generated={gen_report['num_images']} khác manifest records={len(records)}")
if len(records) % TTUR_BATCH_SIZE != 0:
    raise RuntimeError(
        f"TTUR_BATCH_SIZE={TTUR_BATCH_SIZE} không chia hết {len(records)}. "
        "Với TTUR fid.py gốc, hãy chọn batch size chia hết số ảnh để không bỏ batch cuối."
    )

print("[OK] Folder ảnh hợp lệ để chạy TTUR TensorFlow FID.")

# --- Source notebook cell 14 ---
def load_images_array(files: list[Path]) -> np.ndarray:
    # TTUR fid.py dùng imageio.imread và float32, pixel range 0..255.
    return np.array([imread(str(path)).astype(np.float32) for path in files])


def calculate_stats_from_folder_ttur(folder: Path, sess, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    files = list_images(folder)
    print(f"[TTUR] Loading {len(files)} images from {folder}")
    images = load_images_array(files)
    print("[TTUR] image array shape:", images.shape, "dtype:", images.dtype, "min:", float(images.min()), "max:", float(images.max()))
    mu, sigma = ttur_fid.calculate_activation_statistics(images, sess, batch_size=batch_size, verbose=True)
    del images
    return mu, sigma


inception_graph_path = ttur_fid.check_or_download_inception(str(INCEPTION_CACHE_DIR))
print("Inception graph:", inception_graph_path)

# Reset graph để tránh import graph trùng nếu chạy lại cell.
tf.compat.v1.reset_default_graph()
ttur_fid.create_inception_graph(str(inception_graph_path))

with ttur_fid.tf.Session() as sess:
    sess.run(ttur_fid.tf.global_variables_initializer())
    gen_mu, gen_sigma = calculate_stats_from_folder_ttur(GENERATED_DIR, sess, TTUR_BATCH_SIZE)
    ref_mu, ref_sigma = calculate_stats_from_folder_ttur(reference_dir, sess, TTUR_BATCH_SIZE)
    fid_score = float(ttur_fid.calculate_frechet_distance(gen_mu, gen_sigma, ref_mu, ref_sigma))

np.savez_compressed(OUTPUT_ROOT / "generated_ttur_stats.npz", mu=gen_mu, sigma=gen_sigma)
np.savez_compressed(OUTPUT_ROOT / "reference_ttur_stats.npz", mu=ref_mu, sigma=ref_sigma)

print("TTUR TensorFlow FID =", fid_score)

# --- Source notebook cell 16 ---
report = {
    "experiment": EXPERIMENT_NAME,
    "metric": "FID",
    "tool": "bioinf-jku/TTUR TensorFlow fid.py",
    "fid": fid_score,
    "dims": TTUR_DIMS,
    "batch_size": TTUR_BATCH_SIZE,
    "low_profile": TTUR_LOW_PROFILE,
    "gpu": TTUR_GPU,
    "target_size_hw": list(TARGET_SIZE),
    "num_generated": int(gen_report["num_images"]),
    "num_reference": int(ref_report["num_images"]),
    "export_dir": str(EXPORT_DIR),
    "generated_dir": str(GENERATED_DIR),
    "reference_dir": str(reference_dir),
    "manifest_path": str(MANIFEST_PATH),
    "ttur_fid_url": TTUR_RAW_FID_URL,
    "ttur_fid_sha256": ttur_sha256,
    "inception_graph_path": str(inception_graph_path),
    "note": "1073 samples is below TTUR README recommendation (>2048 and ideally 10000); use for controlled internal comparison.",
}

json_path = OUTPUT_ROOT / "ttur_tensorflow_fid_results.json"
csv_path = OUTPUT_ROOT / "ttur_tensorflow_fid_results.csv"
with json_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(report.keys()))
    writer.writeheader()
    writer.writerow(report)

zip_base = Path(str(WORK_DIR)) / f"{EXPERIMENT_NAME}__ttur_tensorflow_fid_eval"
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=OUTPUT_ROOT)

display(pd.DataFrame([{
    "Experiment": EXPERIMENT_NAME,
    "Metric": "FID↓",
    "Tool": "TTUR TensorFlow fid.py",
    "Value": fid_score,
    "Generated": gen_report["num_images"],
    "Reference": ref_report["num_images"],
    "Zip": zip_path,
}]))

print("Saved JSON:", json_path)
print("Saved CSV :", csv_path)
print("Saved ZIP :", zip_path)
