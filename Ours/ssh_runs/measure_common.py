"""Measure paired SSH methods on exactly the same existing COCO indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from measure_existing import EXPECTED, _scan, _write_json, measure


GROUPS = {
    "sd15_lcm": ("sem_sd15_lcm", "ours_sd15_lcm", "ours_sd15_lcm_dpp", "md_sd15_lcm"),
    "sd15_hypersd": ("sem_sd15_hypersd", "ours_sd15_hypersd", "ours_sd15_hypersd_dpp",
                      "ours_sd15_hypersd_pladis", "md_sd15_hypersd"),
    "sdxl_euler": ("sem_sdxl_euler", "ours_sdxl_euler", "md_sdxl_euler"),
    "sd3_ffm": ("sem_sd3_ffm", "ours_sd3_ffm", "md_sd3_ffm"),
}


def measure_group(group: str, seed_root: Path, coco_root: Path) -> dict:
    profiles = GROUPS[group]
    scans = {}
    unavailable = {}
    for profile in profiles:
        run_dir = seed_root / profile
        if not run_dir.is_dir():
            unavailable[profile] = "run directory missing"
            continue
        try:
            scan = _scan(profile, run_dir, coco_root)
        except Exception as exc:
            unavailable[profile] = f"{type(exc).__name__}: {exc}"
            continue
        if scan["generated_count"] == 0:
            unavailable[profile] = "no valid generated images"
            continue
        scans[profile] = scan

    common = (set.intersection(*(set(scan["available_indices"]) for scan in scans.values()))
              if len(scans) >= 2 else set())
    report = {
        "group": group, "base_seed": seed_root.name.removeprefix("seed_"),
        "expected_count": EXPECTED, "common_count": len(common),
        "common_indices": sorted(common),
        "source_generated_counts": {profile: scan["generated_count"] for profile, scan in scans.items()},
        "provenance_warnings": {
            profile: [message for condition, message in (
                (scan.get("generation_profile_matches_current", True), "generation profile differs from current source"),
                (scan.get("generation_python_matches_metric", True), "metric Python differs from generation Python"),
            ) if not condition]
            + ([f"{len(set(scan.get('summary_missing_indices', [])) & common)} compared image(s) have no saved generation-summary row; seed and runtime cannot be independently verified"]
               if set(scan.get("summary_missing_indices", [])) & common else [])
            for profile, scan in scans.items()
        },
        "unavailable_profiles": unavailable, "metric_paths": {},
        "status": "no_comparable_pair" if len(scans) < 2 or not common else "metric_pending",
        "comparison_note": "Only listed methods and identical COCO indices are compared.",
    }
    output = seed_root / "common_metrics"
    output.mkdir(exist_ok=True)
    report_path = output / f"{group}.json"
    if report["status"] == "metric_pending":
        runner_log = seed_root.parent / "runner_logs" / report["base_seed"]
        all_verified = len(common) == EXPECTED and all(
            (runner_log / f"{profile}.log.ok").is_file()
            and (seed_root / profile / "benchmark_complete.txt").is_file()
            for profile in scans
        )
        if all_verified:
            report["status"] = "reuse_verified_full_metrics"
            report["verified_run_dirs"] = {
                profile: str(seed_root / profile) for profile in scans
            }
        else:
            errors = {}
            for profile in scans:
                try:
                    measured = measure(profile, seed_root / profile, coco_root,
                                       common_indices=common, label=f"common_{group}")
                    if measured["status"] != "measured":
                        raise RuntimeError(f"Metric status: {measured['status']}")
                    report["metric_paths"][profile] = str(
                        seed_root / profile / "available_metrics" / f"common_{group}" / "report.json")
                except Exception as exc:
                    errors[profile] = f"{type(exc).__name__}: {exc}"
            report["metric_errors"] = errors
            report["status"] = "measured" if not errors else "metric_failed"
    _write_json(report_path, report)
    print(f"[COMMON METRIC] {group}: {len(common)}/{EXPECTED} shared indices; "
          f"{len(scans)} methods; status={report['status']} | {report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("seed_root", type=Path)
    parser.add_argument("coco_root", type=Path)
    args = parser.parse_args()
    report = measure_group(args.group, args.seed_root, args.coco_root)
    if report["status"] == "metric_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
