"""Check the complete on-disk contract after a 1,073-image SSH benchmark."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from scientific_snapshot import requested_gpu_mode, same_shared_source


ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = {
    "sd15": ROOT / "Ours/data_manifests/coco_val2017_multidiffusion_coco_all_512x512_all.jsonl",
    "sdxl": ROOT / "Ours/data_manifests/coco_val2017_multidiffusion_coco_all_sdxl_1024x1024_all.jsonl",
    "sd3": ROOT / "Ours/data_manifests/coco_val2017_multidiffusion_coco_all_sd3_1024x1024_all.jsonl",
}
PAIRS = {
    "sem_sd15_lcm": "ours_sd15_lcm",
    "ours_sd15_lcm": "sem_sd15_lcm",
    "sem_sd15_hypersd": "ours_sd15_hypersd",
    "ours_sd15_hypersd": "sem_sd15_hypersd",
    "sem_sdxl_euler": "ours_sdxl_euler",
    "ours_sdxl_euler": "sem_sdxl_euler",
    "ours_sd15_lcm_dpp": "ours_sd15_lcm",
    "ours_sd15_hypersd_dpp": "ours_sd15_hypersd",
    "ours_sd15_hypersd_pladis": "ours_sd15_hypersd",
}


def _static_protocol(profile: str) -> dict:
    """Read literal protocol constants without importing GPU-heavy profiles."""
    source = ROOT / "Ours" / "ssh_runs" / "profiles" / f"{profile}.py"
    values = {}
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            values[target.id] = value
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple) and len(target.elts) == len(value):
            for name, item in zip(target.elts, value):
                if isinstance(name, ast.Name):
                    values[name.id] = item
    return values


def verify_main_pair_protocol(profile: str, counterpart: str) -> None:
    pairs = {
        frozenset(("sem_sd15_lcm", "ours_sd15_lcm")),
        frozenset(("sem_sd15_hypersd", "ours_sd15_hypersd")),
        frozenset(("sem_sdxl_euler", "ours_sdxl_euler")),
    }
    if frozenset((profile, counterpart)) not in pairs:
        return
    own = _static_protocol(profile)
    paired = _static_protocol(counterpart)
    for key in ("TARGET_SIZE", "BOOTSTRAP_STEPS", "MASK_STD", "MASK_STRENGTH",
                "PREPROCESS_MASK_COVER_ALPHA", "NEGATIVE_PROMPT"):
        if key not in own or key not in paired or own[key] != paired[key]:
            raise RuntimeError(f"Matched profiles differ in {key}: {profile} vs {counterpart}")
    if "sd15" in profile:
        for key in ("MODEL_ID", "MASK_TYPE"):
            if key not in own or key not in paired or own[key] != paired[key]:
                raise RuntimeError(f"Matched profiles differ in {key}: {profile} vs {counterpart}")
    else:
        for baseline_key, proposed_key in (("SEMANTICDRAW_T_INDEX_LIST", "T_INDEX_LIST"),
                                           ("EULER_GUIDANCE_SCALE", "GUIDANCE_SCALE")):
            baseline = own if profile == "sem_sdxl_euler" else paired
            proposed = paired if profile == "sem_sdxl_euler" else own
            if (baseline_key not in baseline or proposed_key not in proposed
                    or baseline[baseline_key] != proposed[proposed_key]):
                raise RuntimeError(f"Matched SDXL profiles differ in {baseline_key}: {profile} vs {counterpart}")


def verify_run(profile: str, run_dir: Path, *, check_pair: bool = True) -> dict:
    if profile in {"sem_sd3_ffm", "ours_sd3_ffm", "md_sd3_ffm"}:
        return verify_sd3_run(profile, run_dir)
    seed_file = run_dir / "base_seed.txt"
    if not seed_file.is_file():
        raise RuntimeError(f"Missing base-seed provenance: {seed_file}")
    base_seed = int(seed_file.read_text(encoding="utf-8").strip())
    if base_seed < 0 or base_seed + 1072 >= 2**32:
        raise RuntimeError(f"Invalid base seed for the 1073-image run: {base_seed}")
    if run_dir.parent.name != f"seed_{base_seed}":
        raise RuntimeError(f"Run directory and base seed disagree: {run_dir}")
    if "hypersd" in profile:
        console = (run_dir / "console.log").read_text(encoding="utf-8", errors="replace")
        for warning in ("fuse_lora skipped; running with active adapter instead",
                        "DDIMScheduler.from_pretrained skipped, falling back to from_config"):
            if warning in console:
                raise RuntimeError(f"Hyper-SD runtime took a different LoRA/scheduler path: {profile}: {warning}")
    family = "sdxl" if "sdxl" in profile else "sd15"
    manifest = [json.loads(line) for line in MANIFESTS[family].read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(manifest) != 1073:
        raise RuntimeError(f"Expected 1073 manifest rows, found {len(manifest)}")
    summaries = list(run_dir.rglob("generation_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one generation_summary.json, found {summaries}")
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    if not isinstance(summary, list) or len(summary) != len(manifest):
        raise RuntimeError(f"Generation summary has {len(summary) if isinstance(summary, list) else 'invalid'} rows")
    image_paths = [Path(row.get("generated_path", "")).resolve() for row in summary]
    image_dirs = {path.parent for path in image_paths}
    if len(image_dirs) != 1:
        raise RuntimeError(f"Generation summary spans several image directories: {image_dirs}")
    pngs = list(next(iter(image_dirs)).glob("*_generated.png"))
    if len(pngs) != len(manifest):
        raise RuntimeError(f"Expected 1073 generated PNGs in the source image directory, found {len(pngs)}")
    run_root = run_dir.resolve()
    expected_size = (1024, 1024) if family == "sdxl" else (512, 512)
    for index, (record, row) in enumerate(zip(manifest, summary)):
        if int(row.get("index", -1)) != index:
            raise RuntimeError(f"Summary index mismatch at {index}: {row.get('index')}")
        if str(row.get("sample_id")) != str(record["sample_id"]):
            raise RuntimeError(f"Sample ID mismatch at {index}: {row.get('sample_id')} != {record['sample_id']}")
        if int(row.get("image_id", -1)) != int(record["image_id"]) or row.get("file_name") != record["file_name"]:
            raise RuntimeError(f"COCO image identity mismatch at {index}")
        if int(row.get("seed", -1)) != base_seed + index:
            raise RuntimeError(f"Seed mismatch at {index}: {row.get('seed')}")
        elapsed = row.get("elapsed_sec")
        if elapsed is None or not math.isfinite(float(elapsed)) or float(elapsed) < 0:
            raise RuntimeError(f"Invalid generation time at {index}: {elapsed}")
        if profile in {"sem_sd15_lcm", "sem_sd15_hypersd"} and float(row.get("mask_std", -1)) != 1.0:
            raise RuntimeError(f"SD1.5 baseline mask_std differs from the matched pair at {index}")
        if profile in {"sem_sd15_lcm", "sem_sd15_hypersd"} and row.get("background_protocol") != "foreground_then_complement_v1":
            raise RuntimeError(f"SD1.5 baseline did not use the corrected background protocol at {index}")
        if profile in {"sem_sdxl_euler", "ours_sdxl_euler"} and int(row.get("bootstrap_steps", -1)) != 1:
            raise RuntimeError(f"SDXL bootstrap differs from the matched pair at {index}")
        if profile.startswith("ours_sd15_") and row.get("attention_coordinate_policy") != "inverse_roll_image_after_latent_shift_r1":
            raise RuntimeError(f"Unaligned SD1.5 attention coordinates at {index}")
        image = image_paths[index]
        if not image.is_relative_to(run_root) or not image.is_file():
            raise RuntimeError(f"Missing or out-of-run generated image at {index}: {image}")
        if image.name != f"{index:04d}_{record['sample_id']}_generated.png":
            raise RuntimeError(f"Generated filename mismatch at {index}: {image.name}")
        try:
            with Image.open(image) as opened:
                if opened.format != "PNG" or opened.size != expected_size or opened.mode != "RGB":
                    raise RuntimeError(f"Wrong PNG format, size, or mode at {index}: {image}")
                opened.verify()
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Corrupt generated image at {index}: {image}") from exc
    reports = []
    for candidate in run_dir.rglob("*.json"):
        if candidate == summaries[0]:
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("metrics"), dict) and "num_evaluated" in payload:
            reports.append((candidate, payload))
    if len(reports) != 1:
        raise RuntimeError(f"Expected one metric report, found {[str(path) for path, _ in reports]}")
    report_path, report = reports[0]
    if (report.get("num_manifest_records") != len(manifest)
            or report["num_evaluated"] != len(manifest)
            or report.get("num_missing_generated") != 0):
        raise RuntimeError(f"Incomplete metric report: {report_path}")
    if Path(report.get("manifest_path", "")).resolve() != MANIFESTS[family].resolve():
        raise RuntimeError(f"Metric report used a different manifest: {report_path}")
    if Path(report.get("generation_summary", "")).resolve() != summaries[0].resolve():
        raise RuntimeError(f"Metric report used a different generation summary: {report_path}")
    if Path(report.get("generated_dir", "")).resolve() != next(iter(image_dirs)):
        raise RuntimeError(f"Metric report used a different image directory: {report_path}")
    if report.get("model_family") != family or tuple(report.get("target_size", ())) != expected_size:
        raise RuntimeError(f"Metric report has wrong model family or target size: {report_path}")
    if set(report.get("selected_metrics", ())) != {"fid", "is", "clip_fg", "clip_bg", "time"}:
        raise RuntimeError(f"Metric report omitted or added a benchmark metric: {report_path}")
    if report.get("metric_seed") != 2024:
        raise RuntimeError(f"Metric report used an unexpected IS seed: {report_path}")
    if type(report.get("mask_cache_enabled")) is not bool:
        raise RuntimeError(f"Metric report does not declare its mask-cache policy: {report_path}")
    if report["mask_cache_enabled"]:
        if not report.get("mask_cache_dir"):
            raise RuntimeError(f"Metric report enabled mask caching without a directory: {report_path}")
        cache_dir = Path(report.get("mask_cache_dir", "")).resolve()
        if not cache_dir.is_relative_to(run_root):
            raise RuntimeError(f"Metric report used a shared or external mask cache: {report_path}")
    elif report.get("mask_cache_dir") is not None:
        raise RuntimeError(f"Metric report has an inconsistent disabled mask cache: {report_path}")
    input_report_path = run_dir / "input_verification.json"
    if not input_report_path.is_file():
        raise RuntimeError(f"Missing verified COCO input report: {input_report_path}")
    input_report = json.loads(input_report_path.read_text(encoding="utf-8"))
    if (input_report.get("profile") != profile or input_report.get("records") != len(manifest)
            or input_report.get("manifest_sha256") != hashlib.sha256(MANIFESTS[family].read_bytes()).hexdigest()
            or Path(report.get("coco_root", "")).resolve() != Path(input_report["coco_root"]).resolve()):
        raise RuntimeError(f"Metric report used different COCO input or manifest: {report_path}")
    for key in ("instances_sha256", "captions_sha256", "manifest_images_sha256"):
        digest = input_report.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"Missing verified input hash {key}: {input_report_path}")
    coco_root = Path(input_report["coco_root"])
    for filename, key in (("instances_val2017.json", "instances_sha256"),
                          ("captions_val2017.json", "captions_sha256")):
        path = coco_root / "annotations" / filename
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != input_report[key]:
            raise RuntimeError(f"COCO annotation changed after input preflight: {path}")
    current_images_hash = hashlib.sha256()
    for filename in sorted({row["file_name"] for row in manifest}):
        path = coco_root / "val2017" / filename
        if not path.is_file():
            raise RuntimeError(f"COCO image disappeared after input preflight: {path}")
        current_images_hash.update(filename.encode("utf-8"))
        current_images_hash.update(hashlib.sha256(path.read_bytes()).digest())
    if current_images_hash.hexdigest() != input_report["manifest_images_sha256"]:
        raise RuntimeError(f"COCO images changed after input preflight: {coco_root}")
    for filename in ("code_sha256.txt", "pip_freeze.txt"):
        snapshot = run_dir / filename
        after = run_dir / filename.replace(".txt", "_after.txt")
        if (not snapshot.is_file() or not after.is_file() or not snapshot.read_bytes()
                or snapshot.read_bytes() != after.read_bytes()):
            raise RuntimeError(f"Missing or changed code/Python environment snapshot: {snapshot}")
    counterpart = PAIRS.get(profile)
    if counterpart and check_pair:
        counterpart_dir = run_dir.parent / counterpart
        counterpart_input = counterpart_dir / "input_verification.json"
        counterpart_complete = counterpart_dir / "benchmark_complete.txt"
        if counterpart_dir.exists() and not counterpart_input.is_file():
            raise RuntimeError(f"Matched run lacks verified input report: {counterpart_dir}")
        if counterpart_input.is_file():
            verify_main_pair_protocol(profile, counterpart)
            counterpart_seed_file = counterpart_dir / "base_seed.txt"
            if (not counterpart_seed_file.is_file()
                    or counterpart_seed_file.read_text(encoding="utf-8").strip() != str(base_seed)):
                raise RuntimeError(f"Matched runs use different base seeds: {profile} vs {counterpart}")
            if (not counterpart_complete.is_file()
                    or counterpart_complete.read_text(encoding="utf-8").strip() != counterpart):
                raise RuntimeError(f"Matched run has not completed verification: {counterpart_dir}")
            paired = json.loads(counterpart_input.read_text(encoding="utf-8"))
            for key in ("manifest_sha256", "instances_sha256", "captions_sha256", "manifest_images_sha256"):
                if not input_report.get(key) or input_report.get(key) != paired.get(key):
                    raise RuntimeError(f"Matched runs use different {key}: {profile} vs {counterpart}")
            if not same_shared_source(run_dir / "code_sha256.txt", counterpart_dir / "code_sha256.txt"):
                raise RuntimeError(f"Matched runs use different shared generation/metric source: {profile} vs {counterpart}")
            own_gpu = requested_gpu_mode(run_dir)
            paired_gpu = requested_gpu_mode(counterpart_dir)
            if not own_gpu or own_gpu != paired_gpu:
                raise RuntimeError(f"Matched runs use different or unrecorded GPU presets: {profile} vs {counterpart}")
            own_environment = run_dir / "pip_freeze.txt"
            paired_environment = counterpart_dir / "pip_freeze.txt"
            if (not own_environment.is_file() or not paired_environment.is_file()
                    or not own_environment.read_bytes()
                    or own_environment.read_bytes() != paired_environment.read_bytes()):
                raise RuntimeError(f"Matched runs use different Python dependencies: {profile} vs {counterpart}")
    evaluated = report.get("evaluated", [])
    if len(evaluated) != len(manifest) or [str(row.get("sample_id")) for row in evaluated] != [str(row["sample_id"]) for row in manifest]:
        raise RuntimeError(f"Metric report sample order differs from manifest: {report_path}")
    for index, (record, row) in enumerate(zip(manifest, evaluated)):
        if (int(row.get("index", -1)) != index or
                int(row.get("image_id", -1)) != int(record["image_id"]) or
                row.get("file_name") != record["file_name"] or
                int(row.get("num_foreground_regions", -1)) != len(record["foreground_prompts"]) or
                Path(row.get("generated_path", "")).resolve() != image_paths[index]):
            raise RuntimeError(f"Metric evaluated a different image at {index}: {report_path}")
    for key in ("fid", "is_mean", "is_std", "clip_fg_x100", "clip_bg_x100", "time_mean_sec", "time_total_sec"):
        value = report["metrics"].get(key)
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"Missing or non-finite {key} in {report_path}")
    expected_foreground_regions = sum(len(row["foreground_prompts"]) for row in manifest)
    if (int(report["metrics"].get("clip_fg_count", -1)) != expected_foreground_regions
            or int(report["metrics"].get("clip_bg_count", -1)) != len(manifest)):
        raise RuntimeError(f"CLIP did not evaluate every foreground region and background: {report_path}")
    total_elapsed = math.fsum(float(row["elapsed_sec"]) for row in summary)
    if (int(report["metrics"].get("time_num_samples", -1)) != len(summary)
            or not math.isclose(float(report["metrics"]["time_total_sec"]), total_elapsed, rel_tol=1e-6, abs_tol=1e-4)
            or not math.isclose(float(report["metrics"]["time_mean_sec"]), total_elapsed / len(summary), rel_tol=1e-6, abs_tol=1e-6)):
        raise RuntimeError(f"Metric timing does not match generated samples: {report_path}")
    return {"profile": profile, "base_seed": base_seed, "images": len(pngs),
            "summary": str(summaries[0]), "metrics": str(report_path)}


def verify_sd3_run(profile: str, run_dir: Path) -> dict:
    from aggregate_sd3_seeds import _read_one

    method = {"sem_sd3_ffm": "semanticdraw", "ours_sd3_ffm": "anchordraw",
              "md_sd3_ffm": "multidiffusion"}[profile]
    seed = int((run_dir / "base_seed.txt").read_text(encoding="utf-8").strip())
    if run_dir.parent.name != f"seed_{seed}":
        raise RuntimeError("SD3 run directory and base seed differ")
    config, metrics = _read_one(run_dir / "anchordraw_metric_exports", method, seed)
    input_report = json.loads((run_dir / "input_verification.json").read_text(encoding="utf-8"))
    if (input_report.get("profile") != profile or input_report.get("family") != "sd3"
            or input_report.get("records") != 1073
            or input_report.get("manifest_sha256") != config.get("manifest_sha256")
            or input_report.get("manifest_sha256") != hashlib.sha256(MANIFESTS["sd3"].read_bytes()).hexdigest()):
        raise RuntimeError("SD3 input preflight differs from exported manifest")
    coco_root = Path(input_report["coco_root"])
    for filename, key in (("instances_val2017.json", "instances_sha256"),
                          ("captions_val2017.json", "captions_sha256")):
        annotation = coco_root / "annotations" / filename
        if not annotation.is_file() or hashlib.sha256(annotation.read_bytes()).hexdigest() != input_report[key]:
            raise RuntimeError(f"SD3 COCO annotation changed during run: {annotation}")
    summaries = list((run_dir / "anchordraw_runs").glob("*/generation_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one local SD3 generation summary, found {summaries}")
    rows = json.loads(summaries[0].read_text(encoding="utf-8"))
    manifest = [json.loads(line) for line in MANIFESTS["sd3"].read_text(encoding="utf-8").splitlines() if line.strip()]
    local_config = json.loads((summaries[0].parent / "run_config.json").read_text(encoding="utf-8"))
    if local_config != config:
        raise RuntimeError("Local SD3 configuration differs from the metric ZIP")
    input_images_hash = hashlib.sha256()
    for filename in sorted({record["file_name"] for record in manifest}):
        source_image = coco_root / "val2017" / filename
        if not source_image.is_file():
            raise RuntimeError(f"SD3 COCO input image disappeared: {source_image}")
        input_images_hash.update(filename.encode("utf-8"))
        input_images_hash.update(hashlib.sha256(source_image.read_bytes()).digest())
    if input_images_hash.hexdigest() != input_report.get("manifest_images_sha256"):
        raise RuntimeError("SD3 COCO input images changed during run")
    if len(rows) != len(manifest) or [row.get("index") for row in rows] != list(range(1073)):
        raise RuntimeError("Local SD3 generation summary is incomplete or unordered")
    image_dir = summaries[0].parent / "generated_images"
    if len(list(image_dir.glob("*_generated.png"))) != 1073:
        raise RuntimeError("Local SD3 image directory does not contain exactly 1073 images")
    archive_paths = list((run_dir / "anchordraw_metric_exports").glob(f"*seed{seed}*__metric_export.zip"))
    if len(archive_paths) != 1:
        raise RuntimeError(f"Expected one SD3 metric ZIP, found {archive_paths}")
    with ZipFile(archive_paths[0]) as archive:
        members = {Path(name).name: name for name in archive.namelist() if name.endswith("_generated.png")}
        summary_members = [name for name in archive.namelist() if Path(name).name == "generation_summary.json"]
        manifest_members = [name for name in archive.namelist() if Path(name).name == "metric_generated_manifest.jsonl"]
        if len(summary_members) != 1 or json.loads(archive.read(summary_members[0])) != rows:
            raise RuntimeError("Local SD3 summary differs from the metric ZIP")
        if len(manifest_members) != 1:
            raise RuntimeError("SD3 metric ZIP lacks a unique generated manifest")
        metric_rows = [json.loads(line) for line in archive.read(manifest_members[0]).splitlines() if line.strip()]
        if len(metric_rows) != len(manifest):
            raise RuntimeError("SD3 metric manifest has the wrong number of samples")
        for index, (expected, measured) in enumerate(zip(manifest, metric_rows)):
            for key in ("sample_id", "image_id", "file_name", "annotation_ids", "foreground_prompts"):
                if measured.get(key) != expected.get(key):
                    raise RuntimeError(f"SD3 metric manifest differs from official COCO manifest at {index}: {key}")
        if len(members) != 1073:
            raise RuntimeError("SD3 metric ZIP has missing or duplicate image names")
        exported_names = []
        for index, (record, metric_row) in enumerate(zip(manifest, metric_rows)):
            source_name = f"{index:04d}_{record['sample_id']}_generated.png"
            local_image = image_dir / source_name
            source_path = metric_row.get("source_generated_path", metric_row.get("generated_path"))
            relative_path = metric_row.get("generated_image_relative_path")
            if (metric_row.get("metric_index") != index or not isinstance(source_path, str)
                    or Path(source_path).name != source_name or not isinstance(relative_path, str)
                    or Path(relative_path).parent.as_posix() != "generated_images"):
                raise RuntimeError(f"SD3 metric export points to the wrong local image at index {index}")
            export_name = Path(relative_path).name
            exported_names.append(export_name)
            if (export_name not in members
                    or hashlib.sha256(archive.read(members[export_name])).digest()
                    != hashlib.sha256(local_image.read_bytes()).digest()):
                raise RuntimeError(f"SD3 metric ZIP image differs from the local output at index {index}")
        if len(set(exported_names)) != 1073 or set(exported_names) != set(members):
            raise RuntimeError("SD3 metric ZIP contains extra or duplicate exported images")
    for index, (row, record) in enumerate(zip(rows, manifest)):
        path = image_dir / f"{index:04d}_{record['sample_id']}_generated.png"
        if (row.get("sample_id") != record["sample_id"] or row.get("seed") != seed + index
                or row.get("image_id") != record["image_id"] or row.get("file_name") != record["file_name"]
                or Path(row.get("generated_path", "")).resolve() != path.resolve()):
            raise RuntimeError(f"Local SD3 sample identity differs at index {index}")
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (1024, 1024):
                raise RuntimeError(f"Invalid SD3 PNG at index {index}: {path}")
            image.verify()
    for filename in ("code_sha256.txt", "pip_freeze.txt"):
        before = run_dir / filename
        after = run_dir / filename.replace(".txt", "_after.txt")
        if not before.is_file() or not after.is_file() or before.read_bytes() != after.read_bytes():
            raise RuntimeError(f"SD3 provenance changed during generation: {filename}")
    return {"profile": profile, "base_seed": seed, "images": 1073,
            "summary": str(summaries[0]), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_run(args.profile, args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
