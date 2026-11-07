"""Measure validated images already present in an SSH benchmark run.

This report is deliberately separate from the verified 1073-image report. It
never creates images, never fills missing indices, and never marks a benchmark
complete. The saved subset manifest makes the FID reference set match exactly
the generated images that were actually measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from verify_run import MANIFESTS


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED = 1073


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _source_dir(run_dir: Path) -> tuple[Path, Path | None]:
    summaries = [path for path in run_dir.rglob("generation_summary.json")
                 if not any("metric_export" in part for part in path.relative_to(run_dir).parts)]
    directories: dict[Path, Path | None] = {}
    for summary in summaries:
        for candidate in (summary.parent / "generated_images", summary.parent):
            if any(candidate.glob("*_generated.png")):
                directories[candidate] = summary
    if not directories:
        for image in run_dir.rglob("*_generated.png"):
            relative = image.relative_to(run_dir)
            if any("metric_export" in part for part in relative.parts):
                continue
            directories.setdefault(image.parent, None)
    if len(directories) != 1:
        raise RuntimeError(f"Expected one source image directory, found {list(directories)}")
    directory, summary = next(iter(directories.items()))
    if not directory.resolve().is_relative_to(run_dir.resolve()):
        raise RuntimeError(f"Image directory is outside the run: {directory}")
    return directory, summary


def _validated_input(profile: str, run_dir: Path, coco_root: Path) -> tuple[list[dict], str, int]:
    from verify_inputs import verify_inputs

    family = "sd3" if profile.endswith("_sd3_ffm") else ("sdxl" if "sdxl" in profile else "sd15")
    manifest_path = MANIFESTS[family]
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if len(records) != EXPECTED:
        raise RuntimeError(f"Official {family} manifest contains {len(records)} instead of {EXPECTED} records")
    seed = int((run_dir / "base_seed.txt").read_text(encoding="utf-8").strip())
    if run_dir.parent.name != f"seed_{seed}":
        raise RuntimeError("Run directory and base seed disagree")
    saved = json.loads((run_dir / "input_verification.json").read_text(encoding="utf-8"))
    current = verify_inputs(profile, coco_root)
    for key in ("profile", "family", "records", "coco_root", "manifest_sha256",
                "instances_sha256", "captions_sha256", "manifest_images_sha256"):
        if saved.get(key) != current.get(key):
            raise RuntimeError(f"Saved COCO input differs from the current input: {key}")
    return records, family, seed


def _scan(profile: str, run_dir: Path, coco_root: Path) -> dict:
    records, family, seed = _validated_input(profile, run_dir, coco_root)
    image_dir, summary_path = _source_dir(run_dir)
    summary_rows = (json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary_path is not None else [])
    if not isinstance(summary_rows, list):
        raise RuntimeError("Existing generation summary is not a list")
    by_index = {}
    for row in summary_rows:
        index = row.get("index")
        if not isinstance(index, int) or index in by_index or not 0 <= index < EXPECTED:
            raise RuntimeError(f"Invalid or duplicate summary index: {index}")
        by_index[index] = row

    available = []
    missing = []
    invalid = []
    derived_summary = []
    selected_records = []
    image_hashes = []
    expected_names = set()
    target_size = (1024, 1024) if family in {"sdxl", "sd3"} else (512, 512)
    for index, record in enumerate(records):
        filename = f"{index:04d}_{record['sample_id']}_generated.png"
        expected_names.add(filename)
        image = image_dir / filename
        if not image.is_file():
            missing.append(index)
            continue
        saved_row = by_index.get(index)
        if saved_row is not None and (saved_row.get("sample_id") != record["sample_id"]
                                      or saved_row.get("seed") != seed + index
                                      or saved_row.get("image_id") != record["image_id"]
                                      or saved_row.get("file_name") != record["file_name"]
                                      or Path(str(saved_row.get("generated_path", ""))).name != filename):
            invalid.append({"index": index, "reason": "generation summary identity mismatch"})
            continue
        try:
            if not image.resolve().is_relative_to(run_dir.resolve()):
                raise ValueError("Generated PNG resolves outside the run")
            with Image.open(image) as opened:
                if opened.format != "PNG" or opened.mode != "RGB" or opened.size != target_size:
                    raise ValueError(f"Wrong PNG format/mode/size: {opened.format}, {opened.mode}, {opened.size}")
                opened.verify()
        except (OSError, ValueError) as exc:
            invalid.append({"index": index, "reason": str(exc)})
            continue
        available.append(index)
        selected_records.append(record)
        derived_summary.append({**(saved_row or {}), "index": index,
                                "sample_id": record["sample_id"],
                                "generated_path": str(image)})
        image_hashes.append((index, _digest(image)))

    source_pngs = list(image_dir.glob("*_generated.png"))
    extra_names = sorted(path.name for path in source_pngs if path.name not in expected_names)
    fingerprint_source = {
        "profile": profile, "seed": seed, "family": family,
        "input_verification_sha256": _digest(run_dir / "input_verification.json"),
        "manifest_sha256": _digest(MANIFESTS[family]),
        "source_summary_sha256": None if summary_path is None else _digest(summary_path),
        "images": image_hashes,
        "source_png_names": sorted(path.name for path in source_pngs),
        "invalid_images": invalid,
        "measurement_source_sha256": _digest(Path(__file__)),
        "metrics_source": sorted((path.name, _digest(path))
                                 for path in (REPO_ROOT / "Ours" / "src" / "metrics").glob("*.py")),
        "metric_python_freeze": hashlib.sha256(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"])).hexdigest(),
    }
    profile_file = Path(__file__).resolve().parent / "profiles" / f"{profile}.py"
    current_profile_sha = _digest(profile_file)
    saved_profile_file = run_dir / "profile_sha256.txt"
    saved_profile_tokens = (saved_profile_file.read_text(encoding="utf-8").split()
                            if saved_profile_file.is_file() else [])
    saved_profile_sha = saved_profile_tokens[0] if saved_profile_tokens else None
    generation_freeze_file = run_dir / "pip_freeze.txt"
    generation_freeze_sha = _digest(generation_freeze_file) if generation_freeze_file.is_file() else None
    fingerprint_source["generation_profile_sha256"] = saved_profile_sha
    fingerprint_source["current_profile_sha256"] = current_profile_sha
    fingerprint_source["generation_python_freeze_sha256"] = generation_freeze_sha
    fingerprint = hashlib.sha256(json.dumps(fingerprint_source, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "profile": profile, "base_seed": seed, "family": family,
        "source_image_dir": str(image_dir),
        "source_summary": None if summary_path is None else str(summary_path),
        "available_indices": available, "missing_indices": missing,
        "invalid_images": invalid, "extra_image_names": extra_names,
        "summary_missing_indices": [index for index in available if index not in by_index],
        "source_png_count": len(source_pngs), "invalid_count": len(invalid),
        "generated_count": len(available), "expected_count": EXPECTED,
        "coverage_fraction": len(available) / EXPECTED,
        "metric_python_freeze_sha256": fingerprint_source["metric_python_freeze"],
        "metric_source_files_sha256": fingerprint_source["metrics_source"],
        "generation_profile_sha256": saved_profile_sha,
        "current_profile_sha256": current_profile_sha,
        "generation_profile_matches_current": saved_profile_sha == current_profile_sha,
        "generation_python_freeze_sha256": generation_freeze_sha,
        "generation_python_matches_metric": generation_freeze_sha == fingerprint_source["metric_python_freeze"],
        "fingerprint": fingerprint,
        "selected_records": selected_records,
        "selected_summary": derived_summary,
    }


def measure(profile: str, run_dir: Path, coco_root: Path, *,
            common_indices: set[int] | None = None, label: str | None = None) -> dict:
    scan = _scan(profile, run_dir, coco_root)
    output = run_dir / "available_metrics"
    if label is not None:
        if not label or not all(char.isalnum() or char == "_" for char in label):
            raise ValueError(f"Invalid metric label: {label!r}")
        output = output / label
    output.mkdir(exist_ok=True)
    source_indices = list(scan["available_indices"])
    if common_indices is not None:
        if not common_indices.issubset(set(source_indices)):
            raise RuntimeError(f"Common index set includes missing images for {profile}")
        selected = [(record, row) for record, row in zip(scan["selected_records"],
                    scan["selected_summary"]) if row["index"] in common_indices]
        scan["selected_records"] = [record for record, _ in selected]
        scan["selected_summary"] = [row for _, row in selected]
        scan["available_indices"] = sorted(common_indices)
        scan["excluded_for_comparison_indices"] = sorted(set(source_indices) - common_indices)
        scan["fingerprint"] = hashlib.sha256((scan["fingerprint"] + ":" +
            json.dumps(sorted(common_indices))).encode("utf-8")).hexdigest()
    scan["source_generated_count"] = len(source_indices)
    scan["evaluated_count"] = len(scan["available_indices"])
    scan["evaluated_fraction"] = scan["evaluated_count"] / EXPECTED
    report_path = output / "report.json"
    subset_manifest = output / "measured_manifest.jsonl"
    subset_summary = output / "measured_generation_summary.json"
    if report_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if not isinstance(previous, dict):
            previous = {}
        if (previous.get("fingerprint") == scan["fingerprint"]
                and previous.get("status") == "measured"
                and subset_manifest.is_file() and subset_summary.is_file()
                and previous.get("subset_manifest_sha256") == _digest(subset_manifest)
                and previous.get("subset_summary_sha256") == _digest(subset_summary)):
            print(f"[METRIC REUSE] {profile}: {scan['evaluated_count']}/{EXPECTED} measured; "
                  f"{scan['generated_count']}/{EXPECTED} generated", flush=True)
            return previous

    selected_records = scan.pop("selected_records")
    selected_summary = scan.pop("selected_summary")
    report = {**scan, "status": "no_valid_images" if not selected_records else "metric_pending",
              "metric_protocol": "available_manifest_subset_v1",
              "comparison_note": "Scores use only listed indices; compare methods on identical indices.",
              "metric_report": None}
    if not selected_records:
        _write_json(report_path, report)
        return report

    subset_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                       for row in selected_records), encoding="utf-8")
    _write_json(subset_summary, selected_summary)
    report["subset_manifest_sha256"] = _digest(subset_manifest)
    report["subset_summary_sha256"] = _digest(subset_summary)

    metrics = ["clip_fg", "clip_bg", "time"]
    if len(selected_records) >= 2:
        metrics.insert(0, "fid")
    if len(selected_records) >= 10:
        metrics.insert(1, "is")
    metric_batch = 1 if scan["family"] in {"sdxl", "sd3"} else 8
    clip_batch = 4 if scan["family"] in {"sdxl", "sd3"} else 16
    try:
        metrics_import_root = str(REPO_ROOT / "Ours" / "src")
        if metrics_import_root not in sys.path:
            sys.path.insert(0, metrics_import_root)
        from metrics import MetricEvaluationConfig, run_evaluation

        config = MetricEvaluationConfig(
            manifest_path=subset_manifest, coco_root=coco_root,
            generated_dir=Path(scan["source_image_dir"]), generation_summary=subset_summary,
            output_dir=output, model_family=scan["family"],
            target_size=(1024, 1024) if scan["family"] in {"sdxl", "sd3"} else (512, 512),
            metrics=metrics, batch_size=metric_batch, num_workers=0, pin_memory=False,
            device="cuda:0", clip_batch_size=clip_batch, is_splits=10,
        )
        result = run_evaluation(config)
        if (result["num_manifest_records"] != len(selected_records)
                or result["num_evaluated"] != len(selected_records)
                or result["num_missing_generated"] != 0):
            raise RuntimeError("Measured subset count differs from validated images")
    except Exception as exc:
        report["status"] = "metric_failed"
        report["metric_error"] = f"{type(exc).__name__}: {exc}"
        _write_json(report_path, report)
        raise
    report["status"] = "measured"
    report["metric_report"] = _json_safe(result)
    report["unavailable_metrics"] = [name for name in ("fid", "is") if name not in metrics]
    _write_json(report_path, report)
    print(f"[METRIC AVAILABLE] {profile}: {len(selected_records)}/{EXPECTED} measured; "
          f"{scan['generated_count']}/{EXPECTED} generated | {report_path}",
          flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("coco_root", type=Path)
    args = parser.parse_args()
    result = measure(args.profile, args.run_dir, args.coco_root)
    if result["status"] != "measured":
        raise SystemExit(f"No valid generated images in {args.run_dir}")


if __name__ == "__main__":
    main()
