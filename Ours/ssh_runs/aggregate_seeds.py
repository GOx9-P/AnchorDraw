"""Aggregate verified, paired full1073 SSH runs across three base seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from scientific_snapshot import requested_gpu_mode, shared_source_snapshot


BENCHMARKS = (
    "sem_sd15_lcm", "ours_sd15_lcm", "ours_sd15_lcm_dpp",
    "sem_sd15_hypersd", "ours_sd15_hypersd", "ours_sd15_hypersd_dpp",
    "ours_sd15_hypersd_pladis", "sem_sdxl_euler", "ours_sdxl_euler",
    "md_sd15_lcm", "md_sd15_hypersd", "md_sd15_ddim",
    "md_sdxl_euler", "md_sdxl_ddim",
)
PAIRS = (
    ("sem_sd15_lcm", "ours_sd15_lcm"),
    ("sem_sd15_hypersd", "ours_sd15_hypersd"),
    ("sem_sdxl_euler", "ours_sdxl_euler"),
    ("ours_sd15_lcm", "ours_sd15_lcm_dpp"),
    ("ours_sd15_hypersd", "ours_sd15_hypersd_dpp"),
    ("ours_sd15_hypersd", "ours_sd15_hypersd_pladis"),
)
MAIN_METRICS = (
    "fid", "is_mean", "clip_fg_x100", "clip_bg_x100",
    "time_mean_sec", "time_total_sec",
)
INPUT_HASHES = (
    "manifest_sha256", "instances_sha256", "captions_sha256",
    "manifest_images_sha256",
)


def unique_report(run_dir: Path, filename: str) -> Path:
    matches = list(run_dir.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename} in {run_dir}, found {matches}")
    return matches[0]


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"Expected a number for {label}, got bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Missing or invalid {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"Nonfinite {label}: {value!r}")
    return result


def summarize(values: list[float], seeds: list[int]) -> dict:
    if len(values) != len(seeds) or len(values) != 3:
        raise RuntimeError("A three-seed aggregate requires exactly three values")
    return {
        "per_seed": {str(seed): value for seed, value in zip(seeds, values)},
        "mean": statistics.fmean(values),
        "std_across_seeds": statistics.stdev(values),
        "std_definition": "sample standard deviation, ddof=1",
        "n_seeds": 3,
    }


def aggregate(suite_root: Path, seeds: list[int]) -> dict:
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(seed < 0 for seed in seeds):
        raise RuntimeError("Pass exactly three distinct nonnegative base seeds")
    # Distillation++ uses a private teacher stream at sample_seed + 1_000_003.
    # Leave the full 1,073-image teacher range clear of the next seed group.
    if any(b - a <= 1_000_003 + 1072 for a, b in zip(sorted(seeds), sorted(seeds)[1:])):
        raise RuntimeError("Student or Distillation++ teacher seed ranges may overlap")

    all_reports: dict[str, dict[int, dict]] = {profile: {} for profile in BENCHMARKS}
    reference_inputs: dict[str, dict] = {}
    reference_environment: dict[str, tuple[tuple[tuple[str, str], ...], bytes, bytes, str]] = {}
    family_inputs: dict[str, dict] = {}
    suite_environment: tuple[tuple[tuple[str, str], ...], bytes, str] | None = None
    cleanfid_by_seed: dict[int, dict[str, float]] = {}
    auxiliary_by_seed: dict[str, dict[int, float]] = {"pytorch_fid": {}, "ttur_fid": {}}
    ttur_source_hash: str | None = None

    for seed in seeds:
        seed_root = suite_root / f"seed_{seed}"
        complete = seed_root / "seed_complete.txt"
        if not complete.is_file() or complete.read_text(encoding="utf-8").splitlines() != [
            *BENCHMARKS, "pytorch_fid", "cleanfid", "ttur_fid"
        ]:
            raise RuntimeError(f"Missing or incomplete three-seed source group: {seed_root}")
        for profile in BENCHMARKS:
            run_dir = seed_root / profile
            if (run_dir / "benchmark_complete.txt").read_text(encoding="utf-8").strip() != profile:
                raise RuntimeError(f"Unverified benchmark: {run_dir}")
            if (run_dir / "base_seed.txt").read_text(encoding="utf-8").strip() != str(seed):
                raise RuntimeError(f"Wrong base seed in {run_dir}")
            input_report = json.loads((run_dir / "input_verification.json").read_text(encoding="utf-8"))
            if input_report.get("profile") != profile or input_report.get("records") != 1073:
                raise RuntimeError(f"Incomplete input verification: {run_dir}")
            hashes = {key: input_report.get(key) for key in INPUT_HASHES}
            if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
                raise RuntimeError(f"Missing input hashes in {run_dir}")
            if profile in reference_inputs and hashes != reference_inputs[profile]:
                raise RuntimeError(f"Input data changed across seeds for {profile}")
            reference_inputs[profile] = hashes
            family = "sdxl" if "sdxl" in profile else "sd15"
            if family in family_inputs and hashes != family_inputs[family]:
                raise RuntimeError(f"Input data changed between {family} profiles: {profile}")
            family_inputs[family] = hashes
            shared_source = shared_source_snapshot(run_dir / "code_sha256.txt")
            dependencies = (run_dir / "pip_freeze.txt").read_bytes()
            profile_source = (run_dir / "profile_sha256.txt").read_bytes()
            gpu_mode = requested_gpu_mode(run_dir)
            if not dependencies or not profile_source or not gpu_mode:
                raise RuntimeError(f"Missing code or environment snapshot in {run_dir}")
            environment = (shared_source, dependencies, profile_source, gpu_mode)
            if profile in reference_environment and environment != reference_environment[profile]:
                raise RuntimeError(f"Code or Python dependencies changed across seeds for {profile}")
            reference_environment[profile] = environment
            shared_environment = (shared_source, dependencies, gpu_mode)
            if suite_environment is not None and shared_environment != suite_environment:
                raise RuntimeError(f"Shared source or Python dependencies changed between suite profiles: {profile}")
            suite_environment = shared_environment
            candidates = []
            for path in run_dir.rglob("*.json"):
                try:
                    report = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if isinstance(report, dict) and isinstance(report.get("metrics"), dict) and "num_evaluated" in report:
                    candidates.append(report)
            if len(candidates) != 1:
                raise RuntimeError(f"Expected exactly one metric report in {run_dir}")
            report = candidates[0]
            if (report.get("num_manifest_records") != 1073 or report.get("num_evaluated") != 1073
                    or report.get("num_missing_generated") != 0 or report.get("metric_seed") != 2024):
                raise RuntimeError(f"Incomplete or incompatible metric report in {run_dir}")
            summary_path = unique_report(run_dir, "generation_summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (len(summary) != 1073 or [row.get("index") for row in summary] != list(range(1073))
                    or [row.get("seed") for row in summary] != [seed + index for index in range(1073)]):
                raise RuntimeError(f"Generation did not use the expected per-image seeds: {run_dir}")
            all_reports[profile][seed] = report

        clean_rows = json.loads(unique_report(seed_root / "cleanfid", "cleanfid_results.json").read_text(encoding="utf-8"))
        if not isinstance(clean_rows, list) or len(clean_rows) != len(BENCHMARKS):
            raise RuntimeError(f"Incomplete CleanFID report for seed {seed}")
        clean_scores = {}
        for row in clean_rows:
            profile = row.get("experiment")
            if (profile not in BENCHMARKS or profile in clean_scores
                    or row.get("num_generated") != 1073 or row.get("num_reference") != 1073):
                raise RuntimeError(f"Wrong CleanFID row for seed {seed}: {row}")
            clean_scores[profile] = finite_number(row.get("score"), f"CleanFID {seed}/{profile}")
        cleanfid_by_seed[seed] = clean_scores
        for tool, filename in (("pytorch_fid", "pytorch_fid_results.json"),
                               ("ttur_fid", "ttur_tensorflow_fid_results.json")):
            row = json.loads(unique_report(seed_root / tool, filename).read_text(encoding="utf-8"))
            if (row.get("experiment") != "sem_sd15_lcm"
                    or row.get("num_generated") != 1073 or row.get("num_reference") != 1073):
                raise RuntimeError(f"Incomplete {tool} report for seed {seed}")
            if tool == "ttur_fid":
                current_hash = row.get("ttur_fid_sha256")
                if not isinstance(current_hash, str) or len(current_hash) != 64:
                    raise RuntimeError(f"Missing TTUR source hash for seed {seed}")
                if ttur_source_hash is not None and current_hash != ttur_source_hash:
                    raise RuntimeError("TTUR fid.py changed between seeds")
                ttur_source_hash = current_hash
            auxiliary_by_seed[tool][seed] = finite_number(row.get("fid"), f"{tool} seed {seed}")

    main = {
        profile: {
            metric: summarize([
                finite_number(all_reports[profile][seed]["metrics"].get(metric), f"{profile}/{seed}/{metric}")
                for seed in seeds
            ], seeds)
            for metric in MAIN_METRICS
        }
        for profile in BENCHMARKS
    }
    paired_deltas = {
        f"{baseline}__to__{proposed}": {
            metric: summarize([
                finite_number(all_reports[proposed][seed]["metrics"].get(metric), f"{proposed}/{seed}/{metric}")
                - finite_number(all_reports[baseline][seed]["metrics"].get(metric), f"{baseline}/{seed}/{metric}")
                for seed in seeds
            ], seeds)
            for metric in MAIN_METRICS
        }
        for baseline, proposed in PAIRS
    }
    auxiliary = {
        "cleanfid": {
            profile: summarize([cleanfid_by_seed[seed][profile] for seed in seeds], seeds)
            for profile in BENCHMARKS
        },
        **{
            tool: summarize([scores[seed] for seed in seeds], seeds)
            for tool, scores in auxiliary_by_seed.items()
        },
    }
    return {
        "base_seeds": seeds,
        "per_image_seed_rule": "base_seed + zero_based_manifest_index",
        "samples_per_seed": 1073,
        "std_definition": "sample standard deviation across three independent seed-level scores, ddof=1",
        "main_metrics": main,
        "paired_deltas_proposed_minus_baseline": paired_deltas,
        "auxiliary_fid": auxiliary,
    }


def write_outputs(result: dict, suite_root: Path) -> tuple[Path, Path]:
    json_path = suite_root / "three_seed_metrics.json"
    csv_path = suite_root / "three_seed_metrics.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for section, groups in (("main", result["main_metrics"]),
                            ("paired_delta", result["paired_deltas_proposed_minus_baseline"]),
                            ("cleanfid", result["auxiliary_fid"]["cleanfid"])):
        for profile, metrics in groups.items():
            items = {"fid": metrics} if section == "cleanfid" else metrics
            for metric, stats in items.items():
                rows.append({
                    "section": section, "profile": profile, "metric": metric,
                    **{f"seed_{seed}": stats["per_seed"][str(seed)] for seed in result["base_seeds"]},
                    "mean": stats["mean"], "std_across_seeds": stats["std_across_seeds"],
                    "n_seeds": stats["n_seeds"],
                })
    for tool in ("pytorch_fid", "ttur_fid"):
        stats = result["auxiliary_fid"][tool]
        rows.append({
            "section": tool, "profile": "sem_sd15_lcm", "metric": "fid",
            **{f"seed_{seed}": stats["per_seed"][str(seed)] for seed in result["base_seeds"]},
            "mean": stats["mean"], "std_across_seeds": stats["std_across_seeds"],
            "n_seeds": stats["n_seeds"],
        })
    fields = ["section", "profile", "metric", *[f"seed_{seed}" for seed in result["base_seeds"]],
              "mean", "std_across_seeds", "n_seeds"]
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_root", type=Path)
    parser.add_argument("seeds", type=int, nargs=3)
    args = parser.parse_args()
    result = aggregate(args.suite_root, args.seeds)
    json_path, csv_path = write_outputs(result, args.suite_root)
    print(f"[PASS] Three-seed metric aggregate: {json_path} and {csv_path}", flush=True)


if __name__ == "__main__":
    main()
