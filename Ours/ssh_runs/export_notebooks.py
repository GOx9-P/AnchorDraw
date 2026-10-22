"""One-time extraction of benchmark code into standalone server programs.

This tool is for maintainers. The generated programs never read notebooks at
runtime. Every selected code cell is copied from its named source notebook;
cloud paths, profile variables, retry boundaries, and display imports are
adapted. SD3 SemanticDraw metrics are attached from the paired evaluator.
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

# Main benchmark rows in experiment_versions_checklist.md.
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
    "sem_sd3_ffm": ("colab_semanticdraw_smoke/semanticdraw_sd3_flashflowmatch_full1073_colab.ipynb", [6, 10, 12, 14, 16, 18, 20, 22, 24]),
    "ours_sd3_ffm": ("semantic_anchor_experiments/anchordraw_sd3_flashflowmatch_l04_sigmad4_colab.ipynb", [6, 10, 12, 14, 16, 18, 20, 22, 23, 25, 26]),
    "md_sd3_ffm": ("colab_multidiffusion_experiments/multidiffusion_sd3_flashflowmatch_naive_colab.ipynb", [4, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17]),
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
CLOUD_EXPORT_VARS = {"USE_GOOGLE_DRIVE_EXPORT", "SAVE_EXPORT_TO_GOOGLE_DRIVE"}

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

SD3_BASELINE_METRICS = '''
# Server metric stage for the SemanticDraw SD3 notebook, whose Colab metric
# evaluation lives in a separate notebook. Use the same evaluator as its pair.
import gc
from metrics import MetricEvaluationConfig, run_evaluation, write_metrics_report
checked_summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8"))
assert len(checked_summary) == EXPECTED_DATASET_SIZE
assert [row["index"] for row in checked_summary] == list(range(EXPECTED_DATASET_SIZE))
assert all(row["run_signature"] == run_config["run_signature"] for row in checked_summary)
for variable in ("smd", "sanity_image", "generated", "batch", "loader", "preview_batch"):
    if variable in globals():
        del globals()[variable]
gc.collect()
torch.cuda.empty_cache()
metrics_dir = OUTPUT_DIR / "metrics"
metric_config = MetricEvaluationConfig(
    manifest_path=RUN_MANIFEST, coco_root=COCO_ROOT,
    generated_dir=GENERATED_IMAGES_DIR, generation_summary=RUN_SUMMARY_PATH,
    output_dir=metrics_dir, model_family=MODEL_FAMILY, target_size=TARGET_SIZE,
    metrics=("fid", "is", "clip_fg", "clip_bg", "time"), batch_size=4,
    num_workers=0, pin_memory=False, device="cuda:0", clip_batch_size=8,
    is_splits=10,
)
metric_report = run_evaluation(metric_config)
assert metric_report["num_manifest_records"] == EXPECTED_DATASET_SIZE
assert metric_report["num_evaluated"] == EXPECTED_DATASET_SIZE
assert metric_report["num_missing_generated"] == 0
metric_json, metric_csv = write_metrics_report(
    metric_report, metrics_dir, prefix="sem_sd3_flashflowmatch_metrics"
)
print("[OK] SD3 SemanticDraw metrics:", metric_json, metric_csv)
'''


def adapt_sd3_generation(source: str, profile: str) -> str:
    """Keep notebook inference intact; retry each manifest index independently."""
    if profile == "md_sd3_ffm":
        start = source.index("summary = []\n")
        old_end = source.index("global_index = 0\n", start)
        signature = "run_signature"
    else:
        start = source.index('summary = json.loads(RUN_SUMMARY_PATH.read_text')
        old_end = source.index("global_index = 0\n", start)
        signature = 'run_config["run_signature"]'
    replacement = f'''summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8")) if RUN_SUMMARY_PATH.exists() else []
if not isinstance(summary, list):
    raise RuntimeError("Existing generation summary must be a list")
existing_by_index = {{}}
for old_row in summary:
    position = int(old_row.get("index", -1))
    if position < 0 or position >= len(expected_ids) or position in existing_by_index:
        raise RuntimeError(f"Invalid or duplicate existing summary index: {{position}}")
    expected_path = GENERATED_IMAGES_DIR / f"{{position:04d}}_{{expected_ids[position]}}_generated.png"
    if (old_row.get("sample_id") != expected_ids[position]
            or old_row.get("seed") != BASE_SEED + position
            or old_row.get("run_signature") != {signature}
            or Path(old_row.get("generated_path", "")).resolve() != expected_path.resolve()):
        raise RuntimeError(f"Existing summary row {{position}} does not match this run")
    if expected_path.is_file():
        try:
            with Image.open(expected_path) as check_image:
                check_image.verify()
            with Image.open(expected_path) as check_image:
                if check_image.size != TARGET_SIZE or check_image.mode != "RGB":
                    raise RuntimeError("Wrong image dimensions or mode")
        except Exception as exc:
            print(f"[RETRY] invalid existing image at index={{position}}: {{exc}}", flush=True)
            continue
        existing_by_index[position] = old_row
summary = sorted(existing_by_index.values(), key=lambda row: row["index"])
summary_tmp = RUN_SUMMARY_PATH.with_suffix(".tmp")
summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
summary_tmp.replace(RUN_SUMMARY_PATH)
print(f"[RESUME] Verified {{len(summary)}} existing summary rows")
failed_images = []
failure_log = OUTPUT_DIR / "failed_images.jsonl"
previous_retry_seconds = {{}}
if failure_log.exists():
    for failure_line in failure_log.read_text(encoding="utf-8").splitlines():
        if failure_line.strip():
            previous_failure = json.loads(failure_line)
            previous_index = int(previous_failure["index"])
            previous_retry_seconds[previous_index] = previous_retry_seconds.get(previous_index, 0.0) + float(previous_failure["attempt_elapsed_sec"])
'''
    if profile == "md_sd3_ffm":
        replacement += "summary = []  # Rebuild in manifest order, including verified resumed rows.\n"
    source = source[:start] + replacement + source[old_end:]
    if profile == "md_sd3_ffm":
        source = source.replace("existing_summary_by_index.get(global_index)", "existing_by_index.get(global_index)")
    else:
        source = source.replace("if global_index < len(summary):", "if global_index in existing_by_index:")
    inner_start = source.index("    for local_index, sample_id in enumerate(batch[")
    body_start = source.index("\n", inner_start) + 1
    body_end = source.index("\nassert len(summary)", body_start)
    inner = source[body_start:body_end]
    if inner.count("summary.append(row)") != 1:
        raise ValueError(f"SD3 generation cell changed: {profile} summary append")
    inner = inner.replace("summary.append(row)", 'summary.append(row)\n        summary.sort(key=lambda item: item["index"])')
    # These increments move outside the retry loop; skipped valid images also
    # consume one manifest index, never another random seed.
    inner = re.sub(r"^[ \t]*global_index \+= 1\n", "", inner, flags=re.MULTILINE)
    inner = inner.replace("            continue\n", "            break\n")
    if inner.count("elapsed = time.perf_counter() - tic") != 1:
        raise ValueError(f"SD3 generation timing changed: {profile}")
    inner = inner.replace("elapsed = time.perf_counter() - tic",
                          "elapsed = time.perf_counter() - tic + retry_overhead_sec")
    inner = inner.rstrip("\n") + "\n"
    wrapped = '''        for attempt in range(1, 4):
            attempt_started = time.perf_counter()
            try:
'''
    wrapped = "        retry_overhead_sec = previous_retry_seconds.get(global_index, 0.0)\n" + wrapped
    wrapped += "".join("        " + line if line.strip() else line for line in inner.splitlines(keepends=True))
    wrapped += '''            except Exception as exc:
                import traceback
                new_row = next((item for item in summary if item["index"] == global_index), None)
                if new_row is not None:
                    summary = [item for item in summary if item["index"] != global_index]
                    summary_tmp = RUN_SUMMARY_PATH.with_suffix(".tmp")
                    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                    summary_tmp.replace(RUN_SUMMARY_PATH)
                existing_by_index.pop(global_index, None)
                attempt_elapsed_sec = time.perf_counter() - attempt_started
                retry_overhead_sec += attempt_elapsed_sec
                failure = {"index": global_index, "sample_id": expected_ids[global_index],
                           "seed": BASE_SEED + global_index, "attempt": attempt,
                           "attempt_elapsed_sec": attempt_elapsed_sec,
                           "error": repr(exc), "traceback": traceback.format_exc()}
                with failure_log.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(failure, ensure_ascii=False) + "\\n")
                print(f"[IMAGE ERROR] index={global_index} attempt={attempt}/3: {exc}", flush=True)
                if attempt == 3:
                    failed_images.append(global_index)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                break
        global_index += 1

if failed_images:
    print(f"[INCOMPLETE] {len(failed_images)} indices exhausted three attempts: {failed_images}", flush=True)
'''
    source = source[:body_start] + wrapped + source[body_end:]
    ast.parse(source)
    return source


def adapt_image_failures(source: str, profile: str) -> str:
    """Isolate a failed image without changing its manifest index or seed."""
    tree = ast.parse(source)
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "BASE_SEED +" in segment and "generated_path" in segment and ".save(" in segment:
            candidates.append(node)
    if not candidates:
        return source
    target = min(candidates, key=lambda item: item.end_lineno - item.lineno)
    if (len(source.splitlines()[target.lineno - 1].splitlines()) != 1
            or not source.splitlines()[target.lineno - 1].lstrip().startswith("for ")):
        raise ValueError(f"Unexpected image loop header: {profile}")
    outer = next((node for node in tree.body if isinstance(node, ast.For)
                  and node.lineno <= target.lineno and node.end_lineno >= target.end_lineno), None)
    if outer is None:
        raise ValueError(f"No top-level image loop: {profile}")
    lines = source.splitlines(keepends=True)
    loop_indent = lines[target.lineno - 1][:len(lines[target.lineno - 1]) - len(lines[target.lineno - 1].lstrip())]
    original_body = "".join(lines[target.lineno:target.end_lineno])
    increments = len(re.findall(r"^[ \t]*global_index \+= 1[ \t]*$", original_body, re.MULTILINE))
    if profile not in {"md_sd15_lcm", "ours_sdxl_euler"} and increments < 1:
        raise ValueError(f"Missing global index increment: {profile}")
    body = re.sub(r"^[ \t]*global_index \+= 1[ \t]*\n", "", original_body, flags=re.MULTILINE)
    if profile in {"md_sd15_lcm", "md_sdxl_ddim"}:
        body = re.sub(r"^([ \t]*)continue([ \t]*)$", r"\1break\2", body, flags=re.MULTILINE)
    elif re.search(r"^[ \t]*continue[ \t]*$", body, re.MULTILINE):
        raise ValueError(f"Unexpected continue in image loop: {profile}")
    elapsed_pattern = r"(elapsed = time\.perf_counter\(\) - [A-Za-z_][A-Za-z_0-9]*)"
    body, elapsed_count = re.subn(elapsed_pattern, r"\1 + retry_overhead_sec", body)
    if elapsed_count == 0 and "generated, elapsed, records = generate_one(" in body:
        body = re.sub(r"^(?P<indent>[ \t]*)generated, elapsed, records = generate_one\([^\n]+\)$",
                      lambda match: match.group(0) + "\n" + match.group("indent") + "elapsed += retry_overhead_sec",
                      body, count=1, flags=re.MULTILINE)
        elapsed_count = body.count("elapsed += retry_overhead_sec")
    if elapsed_count != 1:
        raise ValueError(f"Cannot account for image retry time: {profile}")
    if profile == "ours_sdxl_euler":
        body = body.replace("method_wall_sec = time.perf_counter() - method_tic",
                            "method_wall_sec = time.perf_counter() - method_tic + retry_overhead_sec")
    body = body.rstrip("\n") + "\n"
    body_indent = loop_indent + "    "
    index_name = "index" if profile == "ours_sdxl_euler" else "global_index"
    tracked = ("summary_rows", "audit_rows") if profile == "ours_sdxl_euler" else (
        ("summary", "audit_rows") if profile in {"ours_sd15_lcm_dpp", "ours_sd15_hypersd_dpp"}
        else ("summary",))
    if profile == "ours_sdxl_euler":
        rollback_checkpoint = '''RUN_SUMMARY_PATH.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
AUDIT_PATH.write_text("\\n".join(json.dumps(item, ensure_ascii=False) for item in audit_rows) + "\\n", encoding="utf-8")'''
    elif profile == "md_sd15_lcm":
        rollback_checkpoint = "write_generation_checkpoint()"
    elif profile == "md_sdxl_ddim":
        rollback_checkpoint = 'RUN_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")'
    else:
        rollback_checkpoint = "pass"
    rollback_code = "\n".join(body_indent + "                " + line for line in rollback_checkpoint.splitlines())
    wrapped = body_indent + "for attempt in range(1, 4):\n"
    wrapped = body_indent + "retry_overhead_sec = 0.0\n" + wrapped
    wrapped += body_indent + "    attempt_started = time.perf_counter()\n"
    wrapped += body_indent + f"    tracked_lists = {tracked!r}\n"
    wrapped += body_indent + "    list_lengths_before = {name: len(globals()[name]) for name in tracked_lists}\n"
    if profile == "md_sd15_lcm":
        wrapped += body_indent + '    counters_before = {name: globals()[name] for name in ("new_generated_count", "resumed_count", "existing_image_only_count")}\n'
    wrapped += body_indent + "    try:\n"
    wrapped += "".join("        " + line if line.strip() else line for line in body.splitlines(keepends=True))
    wrapped += f'''{body_indent}    except Exception as exc:
{body_indent}        import traceback
{body_indent}        rolled_back = any(len(globals()[name]) > list_lengths_before[name] for name in tracked_lists)
{body_indent}        for name in tracked_lists:
{body_indent}            del globals()[name][list_lengths_before[name]:]
'''
    if profile == "md_sd15_lcm":
        wrapped += f'''{body_indent}        for name, value in counters_before.items():
{body_indent}            globals()[name] = value
'''
    wrapped += f'''{body_indent}        if rolled_back:
{body_indent}            try:
{rollback_code}
{body_indent}            except Exception as checkpoint_error:
{body_indent}                print(f"[CHECKPOINT ERROR] index={{{index_name}}}: {{checkpoint_error}}", flush=True)
{body_indent}        attempt_elapsed_sec = time.perf_counter() - attempt_started
{body_indent}        retry_overhead_sec += attempt_elapsed_sec
{body_indent}        failure = {{"index": {index_name}, "seed": BASE_SEED + {index_name},
{body_indent}                   "sample_id": str(locals().get("sample_id", "")),
{body_indent}                   "attempt": attempt, "attempt_elapsed_sec": attempt_elapsed_sec,
{body_indent}                   "error": repr(exc),
{body_indent}                   "traceback": traceback.format_exc()}}
{body_indent}        with (WORK_DIR / "failed_images.jsonl").open("a", encoding="utf-8") as output:
{body_indent}            output.write(json.dumps(failure, ensure_ascii=False) + "\\n")
{body_indent}        print(f"[IMAGE ERROR] index={{{index_name}}} attempt={{attempt}}/3: {{exc}}", flush=True)
{body_indent}        if attempt == 3:
{body_indent}            failed_images.append({index_name})
{body_indent}        if torch.cuda.is_available():
{body_indent}            torch.cuda.empty_cache()
{body_indent}    else:
{body_indent}        break
'''
    if increments:
        wrapped += body_indent + "global_index += 1\n"
    body_start = sum(len(line) for line in lines[:target.lineno])
    body_end = sum(len(line) for line in lines[:target.end_lineno])
    outer_end = sum(len(line) for line in lines[:outer.end_lineno])
    final_gate = '''\nif failed_images:
    raise RuntimeError(f"Run incomplete: {len(failed_images)} manifest indices failed after three attempts: {failed_images}")
'''
    source = source[:outer_end] + final_gate + source[outer_end:]
    source = source[:body_start] + wrapped + source[body_end:]
    source = "failed_images = []\n" + source
    ast.parse(source)
    return source


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
    """Apply narrowly scoped server edits without changing inference math."""
    if profile == "md_sd3_ffm" and cell_id == 10:
        # The notebook's Colab login stores HF_TOKEN and prints Colab-only
        # guidance. The SSH entrypoint checks gated access before model load;
        # Hugging Face Hub reads HF_TOKEN or the saved SSH login itself.
        start = source.index("# Login Hugging Face.")
        end = source.index("assert torch.cuda.is_available()", start)
        if source.count("maybe_login_to_huggingface()") != 2:
            raise ValueError("MultiDiffusion SD3 login cell changed; inspect before exporting")
        source = source[:start] + source[end:]
        source = source.replace("maybe_login_to_huggingface()\n", "")
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
    if profile == "ours_sdxl_euler" and "'seed_rule': '2024 + global_index'" in source:
        # Keep the provenance string aligned with the run-selected base seed.
        source = source.replace(
            "'seed_rule': '2024 + global_index'",
            "'seed_rule': f'{BASE_SEED} + global_index'",
        )
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
    if profile in PORTS:
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            assigned_names = {
                child.id for target in node.targets for child in ast.walk(target)
                if isinstance(child, ast.Name)
            }
            if "BASE_SEED" in assigned_names and not profile.endswith("sd3_ffm"):
                insertion = sum(len(line) for line in lines[:node.end_lineno])
                changes.append((insertion, insertion,
                    'BASE_SEED = int(os.environ.get("ANCHORDRAW_BASE_SEED", str(BASE_SEED)))\n'))
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
        elif name == "RUN_EXPORT_ZIP" and profile != "md_sd3_ffm":
            changes.append((_offset(lines, value.lineno, value.col_offset),
                            _offset(lines, value.end_lineno, value.end_col_offset), "False"))
        elif name == "SKIP_EXISTING_IMAGES_WITHOUT_SUMMARY" and profile == "md_sd15_lcm":
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
        if profile in {"sem_sd3_ffm", "ours_sd3_ffm", "md_sd3_ffm"} and cell_id == {"sem_sd3_ffm": 22, "ours_sd3_ffm": 23, "md_sd3_ffm": 13}[profile]:
            source = adapt_sd3_generation(source, profile)
        elif profile in PORTS and not profile.endswith("sd3_ffm"):
            source = adapt_image_failures(source, profile)
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
        if profile == "md_sd3_ffm" and cell_id == 4:
            chunks.append("MAX_DISPLAY_RESULTS = 0\n")
    if profile == "sem_sd3_ffm":
        chunks.append(SD3_BASELINE_METRICS)
    if profile in {"sem_sd3_ffm", "ours_sd3_ffm", "md_sd3_ffm"}:
        metric_var = {"sem_sd3_ffm": "metric_json", "ours_sd3_ffm": "metric_json", "md_sd3_ffm": "metrics_json"}[profile]
        zip_expr = "METRIC_EXPORT_ZIP_PATH" if profile == "md_sd3_ffm" else 'Path(str(METRIC_EXPORT_DIR) + ".zip")'
        chunks.append(f'''\n# Keep the verified report alongside the ZIP for three-seed aggregation.
sd3_export_zip = {zip_expr}
assert sd3_export_zip.is_file(), sd3_export_zip
sd3_sidecar = sd3_export_zip.with_name(sd3_export_zip.name.replace("__metric_export.zip", "__metrics.json"))
shutil.copy2({metric_var}, sd3_sidecar)
print("[OK] SD3 report sidecar:", sd3_sidecar)
''')
    text = "".join(chunks)
    if profile in PORTS and not profile.endswith("sd3_ffm") and text.count('BASE_SEED = int(os.environ.get("ANCHORDRAW_BASE_SEED", str(BASE_SEED)))') != 1:
        raise RuntimeError(f"Benchmark {profile} must have exactly one base-seed override")
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
