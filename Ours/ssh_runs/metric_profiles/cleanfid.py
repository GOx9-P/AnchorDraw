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

def display(value):
    if isinstance(value, str):
        print(value)

# Notebook: Ours/kaggle_metric_eval/cleanfid_fid_eval_kaggle.ipynb
# Notebook SHA256: 43f9c18fa11da3e7504757fa25bf5c35aad161a3e1c0f927850af4d28b7be713

# --- Source notebook cell 4 ---
from pathlib import Path
import json
import shutil

import pandas as pd
from PIL import Image
from cleanfid import fid

try:
    import torch
except Exception:
    torch = None


# Sửa path này nếu notebook không tự detect đúng dataset.
DATASET_ROOT = None
# DATASET_ROOT = Path("/kaggle/input/YOUR_DATASET_NAME")


# Empty list = đo tất cả experiment tìm được.
TARGET_EXPERIMENTS = []
# TARGET_EXPERIMENTS = ["semanticdraw_sd15_hypersd_full1073__metric_export"]
# TARGET_EXPERIMENTS = ["sdraw_sdxl_lightning4_euler_1024_full1073_b2_bt2_colab_24gb"]


OUTPUT_ROOT = Path(str(WORK_DIR / 'cleanfid_eval'))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

FID_MODE = "clean"

if torch is not None and torch.cuda.is_available():
    DEVICE = "cuda"
    BATCH_SIZE = 32
else:
    DEVICE = "cpu"
    BATCH_SIZE = 8


print("DATASET_ROOT      :", DATASET_ROOT)
print("TARGET_EXPERIMENTS:", TARGET_EXPERIMENTS if TARGET_EXPERIMENTS else "ALL")
print("OUTPUT_ROOT       :", OUTPUT_ROOT)
print("FID_MODE          :", FID_MODE)
print("DEVICE            :", DEVICE)
print("BATCH_SIZE        :", BATCH_SIZE)
DATASET_ROOT = Path(os.environ["METRIC_DATASET_ROOT"]).resolve()
TARGET_EXPERIMENTS = [item for item in os.environ.get("METRIC_EXPERIMENTS", "").split(",") if item]

# --- Source notebook cell 6 ---
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def has_reference_layout(path: Path) -> bool:
    return path.exists() and path.is_dir() and (path / "reference_images").exists()


def detect_dataset_root() -> Path:
    if DATASET_ROOT is not None:
        root = Path(DATASET_ROOT)
        if not has_reference_layout(root):
            raise FileNotFoundError(f"DATASET_ROOT không có reference_images/: {root}")
        return root

    kaggle_input = Path("/kaggle/input")
    candidates = []
    for child in sorted(kaggle_input.iterdir()):
        if child.is_dir() and has_reference_layout(child):
            candidates.append(child)
        if child.is_dir():
            for grandchild in sorted(child.iterdir()):
                if grandchild.is_dir() and has_reference_layout(grandchild):
                    candidates.append(grandchild)

    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy dataset root có reference_images/. "
            "Hãy set DATASET_ROOT thủ công trong cell config."
        )

    print("[AUTO] Candidate dataset roots:")
    for idx, c in enumerate(candidates):
        print(f"  {idx}: {c}")

    return candidates[0]


def find_experiment_pairs(dataset_root: Path):
    ref_parent = dataset_root / "reference_images"
    pairs = []

    for exp_dir in sorted(dataset_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name == "reference_images":
            continue

        gen_dir = exp_dir / "generated_images"
        ref_dir = ref_parent / exp_dir.name

        if gen_dir.exists() and ref_dir.exists():
            pairs.append(
                {
                    "experiment": exp_dir.name,
                    "generated_dir": gen_dir,
                    "reference_dir": ref_dir,
                    "num_generated": count_images(gen_dir),
                    "num_reference": count_images(ref_dir),
                }
            )

    if TARGET_EXPERIMENTS:
        target = set(TARGET_EXPERIMENTS)
        pairs = [p for p in pairs if p["experiment"] in target]

    return pairs


DATASET_ROOT_RESOLVED = detect_dataset_root()
pairs = find_experiment_pairs(DATASET_ROOT_RESOLVED)

df_pairs = pd.DataFrame(
    [
        {
            "experiment": p["experiment"],
            "num_generated": p["num_generated"],
            "num_reference": p["num_reference"],
            "generated_dir": str(p["generated_dir"]),
            "reference_dir": str(p["reference_dir"]),
        }
        for p in pairs
    ]
)

print("DATASET_ROOT_RESOLVED:", DATASET_ROOT_RESOLVED)
df_pairs

# --- Source notebook cell 8 ---
if len(pairs) == 0:
    raise RuntimeError(
        "Không tìm thấy cặp generated/reference nào. "
        "Hãy kiểm tra DATASET_ROOT hoặc TARGET_EXPERIMENTS."
    )

for p in pairs:
    print("=" * 100)
    print("Experiment :", p["experiment"])
    print("Generated  :", p["generated_dir"])
    print("Reference  :", p["reference_dir"])
    print("N generated:", p["num_generated"])
    print("N reference:", p["num_reference"])

    if p["num_generated"] == 0:
        raise RuntimeError(f"{p['experiment']} không có generated image.")
    if p["num_reference"] == 0:
        raise RuntimeError(f"{p['experiment']} không có reference image.")
    if p["num_generated"] != p["num_reference"]:
        raise RuntimeError(
            f"{p['experiment']} lệch số lượng: "
            f"generated={p['num_generated']}, reference={p['num_reference']}"
        )

print("[OK] Tất cả cặp generated/reference hợp lệ.")

# --- Source notebook cell 10 ---
def first_image(folder: Path) -> Path:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            return p
    raise FileNotFoundError(f"Không tìm thấy ảnh trong {folder}")


def verify_image_file(path: Path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return True, im.size, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def scan_image_folder(folder: Path):
    rows = []
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        ok, size, error = verify_image_file(p)
        rows.append(
            {
                "path": str(p),
                "name": p.name,
                "ok": ok,
                "size": f"{size[0]}x{size[1]}" if size else None,
                "error": error,
            }
        )
    return rows


size_rows = []
bad_rows = []

for p in pairs:
    gen_first = first_image(p["generated_dir"])
    ref_first = first_image(p["reference_dir"])

    gen_ok, gen_size, gen_error = verify_image_file(gen_first)
    ref_ok, ref_size, ref_error = verify_image_file(ref_first)

    size_rows.append(
        {
            "experiment": p["experiment"],
            "generated_first": gen_first.name,
            "reference_first": ref_first.name,
            "generated_size": f"{gen_size[0]}x{gen_size[1]}" if gen_size else None,
            "reference_size": f"{ref_size[0]}x{ref_size[1]}" if ref_size else None,
            "size_match": gen_size == ref_size,
            "first_generated_ok": gen_ok,
            "first_reference_ok": ref_ok,
            "first_generated_error": gen_error,
            "first_reference_error": ref_error,
        }
    )

    print("=" * 100)
    print("Scanning generated images:", p["experiment"])
    gen_scan = scan_image_folder(p["generated_dir"])
    print("Generated valid:", sum(r["ok"] for r in gen_scan), "/", len(gen_scan))

    print("Scanning reference images:", p["experiment"])
    ref_scan = scan_image_folder(p["reference_dir"])
    print("Reference valid:", sum(r["ok"] for r in ref_scan), "/", len(ref_scan))

    bad_rows.extend(
        {"experiment": p["experiment"], "kind": "generated", **r}
        for r in gen_scan
        if not r["ok"]
    )
    bad_rows.extend(
        {"experiment": p["experiment"], "kind": "reference", **r}
        for r in ref_scan
        if not r["ok"]
    )

df_sizes = pd.DataFrame(size_rows)
df_bad_images = pd.DataFrame(bad_rows)

if len(df_bad_images) > 0:
    bad_path = OUTPUT_ROOT / "bad_images.csv"
    df_bad_images.to_csv(bad_path, index=False, encoding="utf-8-sig")
    display(df_bad_images)
    raise RuntimeError(
        f"Có {len(df_bad_images)} ảnh bị lỗi/corrupt. "
        f"Danh sách đã lưu ở {bad_path}. "
        "Hãy sửa local reference/generated folder rồi upload lại Kaggle Dataset."
    )

print("[OK] Toàn bộ generated/reference images đều đọc được bằng PIL.")
df_sizes

# --- Source notebook cell 12 ---
results = []

for p in pairs:
    print("=" * 100)
    print("Computing clean-fid for:", p["experiment"])

    score = fid.compute_fid(
        fdir1=str(p["generated_dir"]),
        fdir2=str(p["reference_dir"]),
        mode=FID_MODE,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )

    row = {
        "experiment": p["experiment"],
        "metric": "FID",
        "fid_mode": FID_MODE,
        "score": float(score),
        "num_generated": int(p["num_generated"]),
        "num_reference": int(p["num_reference"]),
        "generated_dir": str(p["generated_dir"]),
        "reference_dir": str(p["reference_dir"]),
        "device": DEVICE,
        "batch_size": int(BATCH_SIZE),
    }

    results.append(row)
    print("FID:", score)

df_results = pd.DataFrame(results)
df_results

# --- Source notebook cell 14 ---
json_path = OUTPUT_ROOT / "cleanfid_results.json"
csv_path = OUTPUT_ROOT / "cleanfid_results.csv"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")

print("Saved JSON:", json_path)
print("Saved CSV :", csv_path)

df_results
