"""One-time extraction of benchmark code into standalone server programs.

This tool is for maintainers. The generated programs never read notebooks at
runtime. Every selected code cell is copied from its named source notebook;
only cloud paths, profile variables, and display imports are adapted.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


OURS = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "profiles"
DIAGNOSTIC_OUTPUT = Path(__file__).resolve().parent / "diagnostics"
METRIC_OUTPUT = Path(__file__).resolve().parent / "metric_profiles"

# Main benchmark rows in experiment_versions_checklist.md, excluding SD3.
# Exclude cloud dependency installation, repository cloning, COCO downloads,
# interactive previews, and cloud ZIP exports. Retain method gates and metrics.
PORTS = {
    "sem_sd15_lcm": ("kaggle_semanticdraw_smoke/semanticdraw_sd15_lcm_full1073_kaggle.ipynb", [4, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18]),
    "sem_sd15_hypersd": ("kaggle_semanticdraw_smoke/semanticdraw_sd15_hypersd_full1073_kaggle.ipynb", [4, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18]),
    "sem_sdxl_euler": ("kaggle_semanticdraw_smoke/semanticdraw_sdxl_euler_full1073_kaggle.ipynb", [4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21]),
    "md_sd15_lcm": ("kaggle_multidiffusion_experiments/multidiffusion_sd15_lcm_full1073_kaggle.ipynb", [4, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18]),
    "md_sd15_hypersd": ("kaggle_multidiffusion_experiments/multidiffusion_sd15_hypersd_full1073_kaggle.ipynb", [4, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18]),
    "md_sd15_ddim": ("colab_multidiffusion_experiments/multidiffusion_sd15_ddim_ref_colab.ipynb", [3, 5, 6, 7, 8, 9, 10, 11]),
    "md_sdxl_euler": ("colab_multidiffusion_experiments/multidiffusion_sdxl_euler_colab.ipynb", [4, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18]),
    "md_sdxl_ddim": ("colab_multidiffusion_experiments/multidiffusion_sdxl_ddim_ref_colab.ipynb", [3, 6, 7, 8, 9, 10, 11, 12, 13]),
    "ours_sd15_lcm": ("kaggle_semantic_anchor_experiments/semantic_anchor_sd15_lcm_w03_l04_sigmad4_full1073_kaggle.ipynb", [4, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18]),
    "ours_sd15_hypersd": ("kaggle_semantic_anchor_experiments/semantic_anchor_sd15_hypersd_w03_l04_sigmad4_full1073_kaggle.ipynb", [4, 6, 7, 8, 9, 10, 11, 12, 15, 16]),
    "ours_sd15_lcm_dpp": ("kaggle_semantic_anchor_experiments/anchordraw_sd15_lcm_w03_l04_dpp_l001_full1073_kaggle.ipynb", [4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]),
    "ours_sd15_hypersd_dpp": ("kaggle_semantic_anchor_experiments/anchordraw_sd15_hypersd_w03_l04_dpp_l001_full1073_kaggle.ipynb", [4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19]),
    "ours_sd15_hypersd_pladis": ("kaggle_semantic_anchor_experiments/anchordraw_sd15_hypersd_w03_l04_sigmad4_pladis_s05_full1073_kaggle.ipynb", [4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19]),
    "ours_sdxl_euler": ("kaggle_semantic_anchor_experiments/anchordraw_sdxl_euler_w03_l04_sigmad4_full1073_kaggle.ipynb", [2, 4, 5, 6, 7, 8, 10, 11, 13, 14]),
}

# Diagnostic and ablation notebooks use smaller or paired in-memory samples;
# their artifact contracts differ from the full1073 metric benchmark.
DIAGNOSTIC_PORTS = {
    "diag_sem_sd15_lcm_semanticfull": ("colab_semanticdraw_smoke/semanticdraw_sd15_lcm_semanticfull_colab.ipynb", [3, 5, 6, 7, 8, 9, 10, 11, 14]),
    "diag_sem_sdxl_euler_smoke": ("colab_semanticdraw_smoke/semanticdraw_sdxl_euler_smoke_bs2_colab.ipynb", [4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 20, 21]),
    "diag_sem_sd15_smoke": ("kaggle_semanticdraw_smoke/semanticdraw_sd15_smoke_kaggle.ipynb", [4, 6, 7, 9, 10, 11, 12, 13, 14, 17, 18]),
    "diag_sem_sdxl_original_smoke": ("kaggle_semanticdraw_smoke/semanticdraw_sdxl_euler_original_baseline_smoke_bs2_kaggle.ipynb", [3, 5, 6, 7, 8, 9, 10, 11]),
    "diag_ours_sd15_hypersd_anchor_smoke": ("semantic_anchor_experiments/semantic_anchor_sd15_hypersd_smoke8_colab.ipynb", [3, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    "diag_ours_sd15_hypersd_sigma_sweep": ("semantic_anchor_experiments/semantic_anchor_sd15_hypersd_spatial_sigma_sweep_colab.ipynb", [3, 4, 6, 7, 8, 9, 10]),
    "diag_ours_sd15_lcm_runtime_smoke": ("semantic_anchor_experiments/semantic_anchor_sd15_lcm_anchor_runtime_smoke8_colab.ipynb", [3, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    "diag_ours_sd15_lcm_anchor_smoke": ("semantic_anchor_experiments/semantic_anchor_sd15_lcm_smoke8_colab.ipynb", [3, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    "diag_ours_sd15_lcm_sigma_sweep": ("semantic_anchor_experiments/semantic_anchor_sd15_lcm_spatial_sigma_sweep_colab.ipynb", [3, 4, 6, 7, 8, 9, 10]),
    "diag_ours_sd15_lcm_layer_smoke": ("semantic_anchor_experiments/semantic_anchor_sd15_lcm_top3_layers_smoke4_colab.ipynb", [3, 4, 6, 7, 8, 9, 10]),
    "diag_ours_sd15_lcm_weighted_mask": ("semantic_anchor_experiments/semantic_anchor_sd15_lcm_weighted_mask_all_ids_colab.ipynb", [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),
    "diag_ours_sdxl_noop_parity": ("kaggle_semantic_anchor_experiments/anchordraw_sdxl_euler_noop_parity_smoke8_kaggle.ipynb", [2, 4, 5, 6, 7, 8]),
    "diag_ours_sdxl_smoke": ("kaggle_semantic_anchor_experiments/anchordraw_sdxl_euler_w03_l04_sigmad4_smoke8_kaggle.ipynb", [2, 4, 5, 6, 7, 8, 9, 10]),
    "diag_hypersd_dpp_isolation128": ("kaggle_semantic_anchor_experiments/hypersd_canonical_ddim_distillpp_l000_vs_l0005_isolation128_kaggle.ipynb", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
    "diag_pladis_capture_smoke": ("kaggle_semantic_anchor_experiments/pladis_anchordraw_hypersd_capture_ablation_smoke8_kaggle.ipynb", [3, 5, 6, 7, 8, 9, 10]),
    "diag_pladis_scale_sweep128": ("kaggle_semantic_anchor_experiments/pladis_anchordraw_hypersd_w03_l04_sigmad4_scale_sweep128_kaggle.ipynb", [3, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
    "diag_pladis_ours_smoke": ("kaggle_semantic_anchor_experiments/pladis_anchordraw_hypersd_w03_l04_sigmad4_smoke8_kaggle.ipynb", [3, 5, 6, 7, 8, 9, 10]),
    "diag_pladis_canonical_smoke": ("kaggle_semantic_anchor_experiments/pladis_hypersd15_canonical_smoke8_kaggle.ipynb", [2, 3, 4, 5, 6]),
    "diag_pladis_semanticdraw_smoke": ("kaggle_semantic_anchor_experiments/pladis_semanticdraw_hypersd_smoke8_kaggle.ipynb", [3, 5, 6, 7, 8, 9, 10, 11]),
    "diag_ours_sd15_hypersd_lambda128": ("kaggle_semantic_anchor_experiments/semantic_anchor_sd15_hypersd_w03_l04_distillationpp_lambda_sweep128_kaggle.ipynb", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
    "diag_ours_sd15_lcm_lambda128": ("kaggle_semantic_anchor_experiments/semantic_anchor_sd15_lcm_w03_l04_distillationpp_lambda_sweep_kaggle.ipynb", [2, 3, 4, 5, 6, 7, 8, 9, 10]),
}

METRIC_PORTS = {
    "cleanfid": ("kaggle_metric_eval/cleanfid_fid_eval_kaggle.ipynb", [4, 6, 8, 10, 12, 14]),
    "pytorch_fid": ("kaggle_metric_eval/pytorch_fid_semanticdraw_sd15_lcm_full1073_kaggle.ipynb", [4, 6, 8, 10, 12, 14]),
    "ttur_fid": ("kaggle_metric_eval/ttur_tensorflow_fid_semanticdraw_sd15_lcm_full1073_kaggle.ipynb", [4, 6, 8, 10, 12, 14, 16]),
}

PROFILE_VARS = {
    "RUN_PROFILE", "COLAB_GPU_MODE", "DDIM_EXPERIMENT_PROFILE",
    "LCM_EXPERIMENT_PROFILE", "PRECISION_PROFILE",
}
CLOUD_EXPORT_VARS = {"RUN_EXPORT_ZIP", "USE_GOOGLE_DRIVE_EXPORT", "SAVE_EXPORT_TO_GOOGLE_DRIVE"}

PREAMBLE = '''#!/usr/bin/env python3
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

'''


def _offset(lines: list[str], lineno: int, col: int) -> int:
    # AST column offsets count UTF-8 bytes; source slices count Unicode codepoints.
    line = lines[lineno - 1]
    return sum(len(previous) for previous in lines[: lineno - 1]) + len(line.encode("utf-8")[:col].decode("utf-8"))


SETUP_MARKERS = {
    "ours_sdxl_euler": "RUN_MANIFEST = REPO_ROOT / 'Ours/data_manifests/",
    "diag_ours_sdxl_noop_parity": "RUN_MANIFEST = REPO_ROOT / 'Ours/data_manifests/",
    "diag_ours_sdxl_smoke": "RUN_MANIFEST = REPO_ROOT / 'Ours/data_manifests/",
    "diag_hypersd_dpp_isolation128": "EXPERIMENT_ID = 'HYPSD-CANON-DDIM-DPP-",
    "diag_ours_sd15_hypersd_lambda128": "EXPERIMENT_ID = 'OURS-SD15-HYPSD-DPP-",
    "diag_ours_sd15_lcm_lambda128": "RUN_ID = 'semantic_anchor_sd15_lcm_w03_l04_sigmad4_distillpp_",
    "diag_pladis_canonical_smoke": "MANIFEST = REPO_ROOT / MANIFEST_REL",
}


def adapt_cell(source: str, *, setup_profile: str | None = None,
               profile: str | None = None, cell_id: int | None = None) -> str:
    """Apply narrowly scoped, auditable edits without changing algorithm code."""
    if setup_profile:
        marker = SETUP_MARKERS[setup_profile]
        if source.count(marker) != 1:
            raise ValueError(f"Setup cell changed: {setup_profile}; inspect before exporting")
        if setup_profile.startswith("ours_sdxl") or setup_profile.startswith("diag_ours_sdxl"):
            prefix = "from pathlib import Path\nimport os, json, importlib.util, types, zipfile\nimport torch\n\n"
        else:
            cloud_start = source.index("REPO_URL =")
            prefix = source[:cloud_start]
            if setup_profile == "diag_pladis_canonical_smoke":
                manifest_line = next(line for line in source.splitlines()
                                     if line.startswith("MANIFEST_REL = "))
                prefix += manifest_line + "\n"
        source = prefix + source[source.index(marker) :]
        if setup_profile == "diag_hypersd_dpp_isolation128":
            source = "OURS_SRC = REPO_ROOT / 'Ours/src'\nsys.path.insert(0, str(OURS_SRC))\n" + source
    if profile == "diag_hypersd_dpp_isolation128" and cell_id == 3:
        source = '''# Server adaptation of source cell 3: use the mounted COCO images.
import shutil
def valid_image(path):
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False
real_paths = []
for row in records:
    source_image = SSH_COCO_ROOT / 'val2017' / str(row['file_name'])
    destination = REAL_IMAGE_DIR / str(row['file_name'])
    if not valid_image(source_image):
        raise RuntimeError(f'Invalid source COCO image: {source_image}')
    shutil.copy2(source_image, destination)
    real_paths.append(destination)
assert len(real_paths) == RUN_SAMPLES and all(valid_image(path) for path in real_paths)
print('[OK] Real COCO subset ready:', len(real_paths))
'''
    if profile == "diag_hypersd_dpp_isolation128" and cell_id == 10:
        marker = "for path in real_paths:\n"
        if source.count(marker) != 1:
            raise ValueError("Isolation cleanup cell changed; inspect before exporting")
        source = source.replace(marker,
            "assert REAL_IMAGE_DIR.resolve().is_relative_to(WORK_DIR.resolve())\n"
            "assert all(path.resolve().is_relative_to(REAL_IMAGE_DIR.resolve()) for path in real_paths)\n"
            + marker)

    # An absolute prefix inside an f-string is a JoinedStr Constant. Rewriting
    # that Constant alone would insert text *inside* the literal path.
    source = re.sub(
        r"Path\(f([\"'])/(?:kaggle/working|content)/([^\"']+)\1\)",
        lambda match: f"(WORK_DIR / f{match.group(1)}{match.group(2)}{match.group(1)})",
        source,
    )

    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    changes: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if value in {"/content", "/kaggle/working"}:
                replacement = "str(WORK_DIR)"
            elif value.startswith("/content/"):
                replacement = f"str(WORK_DIR / {value[len('/content/'): ]!r})"
            elif value.startswith("/kaggle/working/"):
                replacement = f"str(WORK_DIR / {value[len('/kaggle/working/'): ]!r})"
            else:
                continue
            changes.append((_offset(lines, node.lineno, node.col_offset),
                            _offset(lines, node.end_lineno, node.end_col_offset), replacement))
        elif isinstance(node, ast.ImportFrom) and node.module == "IPython.display":
            changes.append((_offset(lines, node.lineno, node.col_offset),
                            _offset(lines, node.end_lineno, node.end_col_offset), "# terminal display provided above"))

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        name = targets[0].id
        value = node.value
        if name == "EXPERIMENT_NAME" and profile in {"pytorch_fid", "ttur_fid"}:
            changes.append((_offset(lines, value.lineno, value.col_offset),
                            _offset(lines, value.end_lineno, value.end_col_offset),
                            '(Path(os.environ["METRIC_EXPORT_DIR"]).parent.name if Path(os.environ["METRIC_EXPORT_DIR"]).name == "metric_export" else Path(os.environ["METRIC_EXPORT_DIR"]).name)'))
        elif name in PROFILE_VARS and isinstance(value, ast.Constant) and isinstance(value.value, str):
            changes.append((_offset(lines, value.lineno, value.col_offset),
                            _offset(lines, value.end_lineno, value.end_col_offset),
                            f"os.environ.get({name!r}, {value.value!r})"))
        elif name == "MAX_DISPLAY_RESULTS":
            changes.append((_offset(lines, value.lineno, value.col_offset),
                            _offset(lines, value.end_lineno, value.end_col_offset), "0"))
        elif name in CLOUD_EXPORT_VARS:
            changes.append((_offset(lines, value.lineno, value.col_offset),
                            _offset(lines, value.end_lineno, value.end_col_offset), "False"))
        elif name == "LOW_VRAM_KAGGLE_T4_MODE" and profile == "sem_sdxl_euler":
            if not isinstance(value, ast.Constant) or value.value is not True:
                raise ValueError("SemanticDraw SDXL VRAM setting changed; inspect before exporting")
            changes.append((_offset(lines, value.lineno, value.col_offset),
                            _offset(lines, value.end_lineno, value.end_col_offset),
                            'os.environ.get("COLAB_GPU_MODE", "high_vram_24gb") == "low_vram"'))

    for start, end, replacement in sorted(changes, reverse=True):
        source = source[:start] + replacement + source[end:]
    ast.parse(source)
    return source.rstrip() + "\n"


def render_one(profile: str, notebook_rel: str, cell_ids: list[int]) -> str:
    path = OURS / notebook_rel
    notebook_bytes = path.read_bytes()
    notebook = json.loads(notebook_bytes)
    chunks = [PREAMBLE, f"# Notebook: Ours/{notebook_rel}\n",
              f"# Notebook SHA256: {hashlib.sha256(notebook_bytes).hexdigest()}\n"]
    if profile == "ttur_fid":
        chunks.append("import tensorflow as tf\n")
    for cell_id in cell_ids:
        cell = notebook["cells"][cell_id]
        if cell["cell_type"] != "code":
            raise ValueError(f"Expected code in {path} cell {cell_id}")
        source = "".join(cell["source"])
        chunks += [f"\n# --- Source notebook cell {cell_id} ---\n",
                   adapt_cell(source, setup_profile=profile if cell_id == 2 and profile in SETUP_MARKERS else None,
                              profile=profile, cell_id=cell_id)]
        if cell_id == 4 and profile == "cleanfid":
            chunks.append('DATASET_ROOT = Path(os.environ["METRIC_DATASET_ROOT"]).resolve()\n')
            chunks.append('TARGET_EXPERIMENTS = [item for item in os.environ.get("METRIC_EXPERIMENTS", "").split(",") if item]\n')
        if cell_id == 4 and profile in {"pytorch_fid", "ttur_fid"}:
            chunks.append('EXPORT_DIR_OVERRIDE = os.environ["METRIC_EXPORT_DIR"]\n')
            chunks.append('REFERENCE_DIR_OVERRIDE = os.environ.get("METRIC_REFERENCE_DIR", "")\n')
            chunks.append('COCO_VAL2017_DIR_OVERRIDE = str(SSH_COCO_ROOT / "val2017")\n')
            if profile == "ttur_fid":
                chunks.append('TTUR_FID_FILE = Path(os.environ.get("TTUR_FID_FILE", str(TTUR_FID_FILE))).resolve()\n')
        if "COCO_ROOT =" in source:
            chunks.append("COCO_ROOT = SSH_COCO_ROOT\n")
    text = "".join(chunks)
    compile(text, str(path), "exec")
    return text


def export_one(profile: str, notebook_rel: str, cell_ids: list[int], *, directory: Path = OUTPUT) -> Path:
    text = render_one(profile, notebook_rel, cell_ids)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{profile}.py"
    target.write_text(text, encoding="utf-8")
    return target


def main() -> None:
    for profile, (notebook, cells) in PORTS.items():
        target = export_one(profile, notebook, cells)
        print(f"{profile}: {target}")
    for profile, (notebook, cells) in DIAGNOSTIC_PORTS.items():
        target = export_one(profile, notebook, cells, directory=DIAGNOSTIC_OUTPUT)
        print(f"{profile}: {target}")
    for profile, (notebook, cells) in METRIC_PORTS.items():
        target = export_one(profile, notebook, cells, directory=METRIC_OUTPUT)
        print(f"{profile}: {target}")


if __name__ == "__main__":
    main()
