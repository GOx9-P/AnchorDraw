"""Small artifact fixtures for the scientific three-seed reporting gate."""

import csv
import json

import pytest

from aggregate_seeds import BENCHMARKS, aggregate, write_outputs


SEEDS = [2024, 10002024, 20002024]
SHARED_SNAPSHOT = (
    f"{'a' * 64}  /repo/Baseline/semantic-draw-main/src/model/pipeline.py\n"
    f"{'b' * 64}  /repo/Ours/src/metrics/evaluate.py\n"
    f"{'c' * 64}  /repo/Ours/ssh_runs/unrelated_diagnostic.py\n"
).encode()


def make_suite(root):
    for seed_index, seed in enumerate(SEEDS):
        group = root / f"seed_{seed}"
        group.mkdir()
        (group / "seed_complete.txt").write_text(
            "\n".join([*BENCHMARKS, "pytorch_fid", "cleanfid", "ttur_fid"]) + "\n",
            encoding="utf-8",
        )
        for profile in BENCHMARKS:
            run = group / profile
            run.mkdir()
            (run / "benchmark_complete.txt").write_text(profile + "\n", encoding="utf-8")
            (run / "base_seed.txt").write_text(str(seed) + "\n", encoding="utf-8")
            (run / "input_verification.json").write_text(json.dumps({
                "profile": profile, "records": 1073,
                "manifest_sha256": "a" * 64, "instances_sha256": "b" * 64,
                "captions_sha256": "c" * 64, "manifest_images_sha256": "d" * 64,
            }), encoding="utf-8")
            (run / "code_sha256.txt").write_bytes(SHARED_SNAPSHOT)
            (run / "profile_sha256.txt").write_text(f"{'d' * 64}  /repo/Ours/ssh_runs/profiles/{profile}.py\n", encoding="utf-8")
            (run / "gpu_mode.txt").write_text("a100_80gb\n", encoding="utf-8")
            (run / "pip_freeze.txt").write_bytes(b"same environment\n")
            (run / "generation_summary.json").write_text(json.dumps([
                {"index": index, "seed": seed + index} for index in range(1073)
            ]), encoding="utf-8")
            value = float(10 + 2 * seed_index)
            (run / "metrics.json").write_text(json.dumps({
                "num_manifest_records": 1073, "num_evaluated": 1073,
                "num_missing_generated": 0, "metric_seed": 2024,
                "metrics": {
                    "fid": value, "is_mean": value, "clip_fg_x100": value,
                    "clip_bg_x100": value, "time_mean_sec": value,
                    "time_total_sec": value * 1073,
                },
            }), encoding="utf-8")
        for tool, filename in (("pytorch_fid", "pytorch_fid_results.json"),
                               ("ttur_fid", "ttur_tensorflow_fid_results.json")):
            directory = group / tool
            directory.mkdir()
            (directory / filename).write_text(json.dumps({
                "experiment": "sem_sd15_lcm",
                "fid": float(10 + 2 * seed_index),
                "num_generated": 1073, "num_reference": 1073,
                "ttur_fid_sha256": "f" * 64,
            }), encoding="utf-8")
        clean_dir = group / "cleanfid"
        clean_dir.mkdir()
        (clean_dir / "cleanfid_results.json").write_text(json.dumps([
            {"experiment": profile, "score": float(10 + 2 * seed_index),
             "num_generated": 1073, "num_reference": 1073}
            for profile in BENCHMARKS
        ]), encoding="utf-8")


def test_three_seed_report_and_incomplete_seed_rejection(tmp_path):
    make_suite(tmp_path)
    result = aggregate(tmp_path, SEEDS)
    fid = result["main_metrics"]["sem_sd15_lcm"]["fid"]
    assert fid["per_seed"] == {"2024": 10.0, "10002024": 12.0, "20002024": 14.0}
    assert fid["mean"] == 12.0
    assert fid["std_across_seeds"] == 2.0
    assert result["paired_deltas_proposed_minus_baseline"][
        "sem_sd15_lcm__to__ours_sd15_lcm"]["fid"]["mean"] == 0.0
    json_path, csv_path = write_outputs(result, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["base_seeds"] == SEEDS
    with csv_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert any(row["profile"] == "sem_sd15_lcm" and row["metric"] == "fid"
               and float(row["std_across_seeds"]) == 2.0 for row in rows)

    bad_summary = tmp_path / "seed_10002024" / "sem_sd15_lcm" / "generation_summary.json"
    rows = json.loads(bad_summary.read_text(encoding="utf-8"))
    rows[0]["seed"] = 2024
    bad_summary.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected per-image seeds"):
        aggregate(tmp_path, SEEDS)

    rows[0]["seed"] = SEEDS[1]
    bad_summary.write_text(json.dumps(rows), encoding="utf-8")
    changed_inputs = tmp_path / "seed_20002024" / "sem_sd15_lcm" / "input_verification.json"
    input_report = json.loads(changed_inputs.read_text(encoding="utf-8"))
    input_report["instances_sha256"] = "e" * 64
    changed_inputs.write_text(json.dumps(input_report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Input data changed across seeds"):
        aggregate(tmp_path, SEEDS)


def test_rejects_drift_between_different_profiles_in_same_suite(tmp_path):
    make_suite(tmp_path)
    md = tmp_path / "seed_2024" / "md_sd15_lcm"
    (md / "code_sha256.txt").write_bytes(SHARED_SNAPSHOT.replace(b"b" * 64, b"e" * 64))
    with pytest.raises(RuntimeError, match="between suite profiles"):
        aggregate(tmp_path, SEEDS)

    (md / "code_sha256.txt").write_bytes(SHARED_SNAPSHOT)
    report_path = md / "input_verification.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["manifest_images_sha256"] = "e" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="between sd15 profiles"):
        aggregate(tmp_path, SEEDS)
