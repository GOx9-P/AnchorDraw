"""Aggregate complete SD3 Colab exports and their saved metric reports.

Each method directory contains three ``*seedN*__metric_export.zip`` files and
the corresponding ``*seedN*__metrics.json`` sidecars. This tool does not run
SD3 or treat the historical partial export as a valid result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from zipfile import ZipFile


SEEDS = (2024, 10002024, 20002024)
METRICS = ("fid", "is_mean", "clip_fg_x100", "clip_bg_x100", "time_mean_sec", "time_total_sec")
METHODS = ("semanticdraw", "anchordraw", "multidiffusion")


def _member(archive: ZipFile, filename: str) -> bytes:
    matches = [name for name in archive.namelist() if Path(name).name == filename]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {filename} in {archive.filename}; found {len(matches)}")
    return archive.read(matches[0])


def _number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"Invalid {label}: boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing or invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"Nonfinite {label}: {value!r}")
    return result


def _read_one(root: Path, method: str, seed: int) -> tuple[dict, dict]:
    matches = list(root.glob(f"*seed{seed}*__metric_export.zip"))
    if not matches:
        profile = {"semanticdraw": "sem_sd3_ffm", "anchordraw": "ours_sd3_ffm",
                   "multidiffusion": "md_sd3_ffm"}[method]
        matches = list(root.glob(f"seed_{seed}/{profile}/anchordraw_metric_exports/*seed{seed}*__metric_export.zip"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {method} ZIP for seed {seed} in {root}; found {matches}")
    archive_path = matches[0]
    stem = archive_path.name.removesuffix("__metric_export.zip")
    report_path = archive_path.with_name(f"{stem}__metrics.json")
    if not report_path.is_file():
        raise RuntimeError(f"Missing verified metric sidecar: {report_path}")
    with ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt SD3 ZIP member: {bad_member}")
        config = json.loads(_member(archive, "run_config.json"))
        summary = json.loads(_member(archive, "generation_summary.json"))
        export = json.loads(_member(archive, "export_summary.json"))
        manifest = [json.loads(line) for line in _member(archive, "metric_generated_manifest.jsonl").splitlines() if line.strip()]
        png_names = [Path(name).name for name in archive.namelist() if name.endswith("_generated.png")]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (config.get("base_seed") != seed or config.get("run_profile") != "full1073"
            or config.get("expected_samples") != 1073 or config.get("model_family") != "sd3"):
        raise RuntimeError(f"Wrong or incomplete SD3 configuration: {archive_path}")
    if method == "multidiffusion" and config.get("view_profile") != "native_full_v128":
        raise RuntimeError(f"MultiDiffusion SD3 is a diagnostic view, not the paper profile: {archive_path}")
    signature = config.get("run_signature")
    if not isinstance(signature, str) or len(signature) != 64 or export.get("run_signature") != signature:
        raise RuntimeError(f"Run signature mismatch: {archive_path}")
    if (not isinstance(summary, list) or len(summary) != 1073 or len(manifest) != 1073
            or len(png_names) != 1073 or len(set(png_names)) != 1073):
        raise RuntimeError(f"SD3 export is not a complete 1,073-image run: {archive_path}")
    if ([row.get("index") for row in summary] != list(range(1073))
            or [row.get("seed") for row in summary] != [seed + index for index in range(1073)]
            or any(row.get("run_signature") != signature for row in summary)):
        raise RuntimeError(f"SD3 summary has wrong order, seed, or signature: {archive_path}")
    summary_ids = [str(row.get("sample_id")) for row in summary]
    if len(set(summary_ids)) != 1073 or [str(row.get("sample_id")) for row in manifest] != summary_ids:
        raise RuntimeError(f"SD3 manifest differs from generated samples: {archive_path}")
    expected_source_names = [f"{index:04d}_{sample_id}_generated.png"
                             for index, sample_id in enumerate(summary_ids)]
    for index, (row, exported) in enumerate(zip(summary, manifest)):
        source_path = exported.get("source_generated_path", exported.get("generated_path"))
        relative_path = exported.get("generated_image_relative_path")
        if (exported.get("metric_index") != index
                or not isinstance(source_path, str)
                or Path(source_path).name != expected_source_names[index]
                or not isinstance(relative_path, str)
                or Path(relative_path).parent.as_posix() != "generated_images"
                or not Path(relative_path).name.endswith("_generated.png")
                or row.get("generated_path") != source_path):
            raise RuntimeError(f"SD3 metric export points to the wrong source image at {index}: {archive_path}")
    expected_export_names = {Path(row["generated_image_relative_path"]).name for row in manifest}
    if len(expected_export_names) != 1073 or set(png_names) != expected_export_names:
        raise RuntimeError(f"SD3 ZIP has missing or misnamed exported PNGs: {archive_path}")
    evaluated = report.get("evaluated", [])
    if (report.get("num_manifest_records") != 1073 or report.get("num_evaluated") != 1073
            or report.get("num_missing_generated") != 0
            or [str(row.get("sample_id")) for row in evaluated] != summary_ids):
        raise RuntimeError(f"SD3 metric report evaluated a different or incomplete set: {report_path}")
    for index, (source, generated, measured) in enumerate(zip(manifest, summary, evaluated)):
        if (measured.get("index") != index
                or measured.get("image_id") != source.get("image_id")
                or measured.get("file_name") != source.get("file_name")
                or measured.get("generated_path") != generated.get("generated_path")
                or measured.get("num_foreground_regions") != len(source.get("foreground_prompts", ()))):
            raise RuntimeError(f"SD3 metric report used the wrong image or region set at {index}: {report_path}")
    if (report.get("model_family") != "sd3" or tuple(report.get("target_size", ())) != (1024, 1024)
            or set(report.get("selected_metrics", ())) != {"fid", "is", "clip_fg", "clip_bg", "time"}
            or report.get("metric_seed") != 2024):
        raise RuntimeError(f"SD3 metric protocol differs from the paired evaluation: {report_path}")
    foreground_count = sum(len(row.get("foreground_prompts", ())) for row in manifest)
    if (foreground_count <= 0 or report.get("metrics", {}).get("clip_fg_count") != foreground_count
            or report.get("metrics", {}).get("clip_bg_count") != 1073):
        raise RuntimeError(f"SD3 CLIP metric omitted foreground or background regions: {report_path}")
    if (export.get("manifest_sha256") != config.get("manifest_sha256")
            or export.get("input_masks_sha256") != config.get("input_masks_sha256")
            or export.get("coco_annotations_sha256") != config.get("coco_annotations_sha256")):
        raise RuntimeError(f"SD3 export input hashes disagree: {archive_path}")
    for key in ("manifest_sha256", "input_masks_sha256", "coco_annotations_sha256", "metrics_source_sha256"):
        value = config.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"Missing {key} in {archive_path}")
    values = {key: _number(report.get("metrics", {}).get(key), f"{method}/{seed}/{key}") for key in METRICS}
    elapsed = math.fsum(_number(row.get("elapsed_sec"), f"{method}/{seed}/elapsed") for row in summary)
    if (report["metrics"].get("time_num_samples") != 1073
            or not math.isclose(values["time_total_sec"], elapsed, rel_tol=1e-6, abs_tol=1e-4)
            or not math.isclose(values["time_mean_sec"], elapsed / 1073, rel_tol=1e-6, abs_tol=1e-6)):
        raise RuntimeError(f"SD3 metric time differs from generated rows: {report_path}")
    return config, values


def aggregate(roots: dict[str, Path]) -> dict:
    if set(roots) != set(METHODS):
        raise RuntimeError(f"Provide exactly these methods: {METHODS}")
    profiles = {"semanticdraw": "sem_sd3_ffm", "anchordraw": "ours_sd3_ffm",
                "multidiffusion": "md_sd3_ffm"}
    suite_dirs = [roots[method] / f"seed_{seed}" / profiles[method]
                  for method in METHODS for seed in SEEDS]
    if any(path.exists() for path in suite_dirs):
        if not all(path.is_dir() for path in suite_dirs):
            raise RuntimeError("SD3 SSH suite layout is missing a method or seed")
        code_snapshots = set()
        environments = set()
        inputs = set()
        for run_dir in suite_dirs:
            code_snapshots.add((run_dir / "code_sha256.txt").read_bytes())
            environments.add((run_dir / "pip_freeze.txt").read_bytes())
            report = json.loads((run_dir / "input_verification.json").read_text(encoding="utf-8"))
            if report.get("family") != "sd3" or report.get("records") != 1073:
                raise RuntimeError(f"Invalid SD3 input verification: {run_dir}")
            inputs.add(tuple(report.get(key) for key in (
                "manifest_sha256", "instances_sha256", "captions_sha256", "manifest_images_sha256", "coco_root")))
        if len(code_snapshots) != 1 or len(environments) != 1 or len(inputs) != 1:
            raise RuntimeError("SD3 methods or seeds used different code, Python environment or COCO inputs")
    configs: dict[str, dict[int, dict]] = {}
    scores: dict[str, dict[int, dict]] = {}
    for method in METHODS:
        configs[method] = {}
        scores[method] = {}
        for seed in SEEDS:
            configs[method][seed], scores[method][seed] = _read_one(roots[method], method, seed)
        comparable = []
        for seed in SEEDS:
            config = dict(configs[method][seed])
            for key in ("base_seed", "white_bootstrap_seed", "white_latent_sha256",
                        "run_signature", "experiment_id", "google_drive_export_root"):
                config.pop(key, None)
            comparable.append(config)
        if comparable[1:] != comparable[:-1]:
            raise RuntimeError(f"Scientific configuration changed across {method} seeds")
    for seed in SEEDS:
        baseline = configs["semanticdraw"][seed]
        anchor = configs["anchordraw"][seed]
        for key in ("model_id", "model_revision", "flash_sd3_revision", "diffusers_commit",
                    "manifest_sha256", "input_masks_sha256", "coco_annotations_sha256",
                    "metrics_source_sha256", "data_source_sha256", "pipeline_sha256",
                    "dependency_versions", "bootstrap_steps", "t_index_list", "batch_size",
                    "guidance_scale", "mask_std", "mask_strength", "mask_type",
                    "preprocess_mask_cover_alpha", "num_inference_steps", "negative_prompt",
                    "dtype", "patch_safe_white_bootstrap_latent", "white_bootstrap_seed",
                    "white_latent_sha256"):
            if key not in baseline or key not in anchor or baseline[key] != anchor[key]:
                raise RuntimeError(f"SemanticDraw and AnchorDraw differ in {key} for seed {seed}")
    shared_metric_source = {configs[method][seed]["metrics_source_sha256"] for method in METHODS for seed in SEEDS}
    shared_manifests = {configs[method][seed]["manifest_sha256"] for method in METHODS for seed in SEEDS}
    if len(shared_metric_source) != 1 or len(shared_manifests) != 1:
        raise RuntimeError("SD3 methods used different metric source or sample manifest")
    for key in ("model_id", "model_revision", "flash_sd3_revision", "diffusers_commit",
                "dependency_versions", "coco_annotations_sha256"):
        values = [configs[method][seed].get(key) for method in METHODS for seed in SEEDS]
        if any(value is None for value in values) or any(value != values[0] for value in values[1:]):
            raise RuntimeError(f"SD3 methods or seeds differ in {key}")

    def stats(values: list[float]) -> dict:
        return {"per_seed": dict(zip((str(seed) for seed in SEEDS), values)),
                "mean": statistics.fmean(values), "std_across_seeds": statistics.stdev(values),
                "n_seeds": 3}

    return {
        "base_seeds": SEEDS,
        "samples_per_seed": 1073,
        "std_definition": "sample standard deviation across three seed-level metric scores, ddof=1",
        "methods": {method: {key: stats([scores[method][seed][key] for seed in SEEDS]) for key in METRICS}
                    for method in METHODS},
        "paired_anchor_minus_baseline": {
            key: stats([scores["anchordraw"][seed][key] - scores["semanticdraw"][seed][key] for seed in SEEDS])
            for key in METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for method in METHODS:
        parser.add_argument(f"--{method}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate({method: getattr(args, method) for method in METHODS})
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "sd3_three_seed_metrics.json"
    csv_path = args.output / "sd3_three_seed_metrics.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["method", "metric", *[f"seed_{seed}" for seed in SEEDS], "mean", "std_across_seeds"])
        for method, metrics in (*result["methods"].items(), ("anchor_minus_baseline", result["paired_anchor_minus_baseline"])):
            for key, row in metrics.items():
                writer.writerow([method, key, *[row["per_seed"][str(seed)] for seed in SEEDS],
                                 row["mean"], row["std_across_seeds"]])
    print(f"[PASS] {json_path} and {csv_path}", flush=True)


if __name__ == "__main__":
    main()
