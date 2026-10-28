"""A suite restart must reuse only verified, compatible on-disk work."""

import hashlib
import json
from pathlib import Path

import pytest

from scientific_snapshot import shared_source_snapshot
from suite_checkpoint import SSH_ROOT, check_completed, select_suite
from verify_diagnostic import RUNNER_FILES, verify_diagnostic


def _snapshot(runner_digest="c"):
    return (
        f"{'a' * 64}  /repo/Baseline/semantic-draw-main/src/model/pipeline.py\n"
        f"{'b' * 64}  /repo/Ours/src/metrics/evaluate.py\n"
        f"{runner_digest * 64}  /repo/Ours/ssh_runs/diagnostics/unrelated.py\n"
    )


def test_shared_source_ignores_unrelated_runner_changes(tmp_path):
    before = tmp_path / "before.txt"
    after = tmp_path / "after.txt"
    before.write_text(_snapshot("c"), encoding="utf-8")
    after.write_text(_snapshot("d"), encoding="utf-8")
    assert shared_source_snapshot(before) == shared_source_snapshot(after)
    after.write_text(_snapshot("d").replace("b" * 64, "e" * 64), encoding="utf-8")
    assert shared_source_snapshot(before) != shared_source_snapshot(after)


def test_diagnostic_provenance_files_are_not_result_artifacts(tmp_path):
    for filename in RUNNER_FILES:
        (tmp_path / filename).write_text("metadata\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no result artifact"):
        verify_diagnostic("diag_sem_sd15_smoke", tmp_path)


def test_select_latest_matching_suite_and_legacy_run(tmp_path):
    coco = tmp_path / "coco"
    first = tmp_path / "20261020T000000Z"
    first.mkdir()
    (first / "suite_failures.tsv").write_text("seed\tprofile\n", encoding="utf-8")
    (first / "suite_metadata.json").write_text(json.dumps({
        "gpu_mode": "a100_80gb", "coco_root": str(coco),
    }), encoding="utf-8")
    wrong = tmp_path / "20261021T000000Z"
    wrong.mkdir()
    (wrong / "suite_failures.tsv").write_text("seed\tprofile\n", encoding="utf-8")
    (wrong / "suite_metadata.json").write_text(json.dumps({
        "gpu_mode": "high_vram_24gb", "coco_root": str(coco),
    }), encoding="utf-8")
    assert select_suite(tmp_path, "a100_80gb", coco) == first.name

    legacy = tmp_path / "20261022T000000Z"
    run = legacy / "seed_2024" / "sem_sd15_lcm"
    log = legacy / "runner_logs" / "2024"
    run.mkdir(parents=True)
    log.mkdir(parents=True)
    (legacy / "suite_failures.tsv").write_text("seed\tprofile\n", encoding="utf-8")
    (log / "sem_sd15_lcm.log.ok").touch()
    (log / "sem_sd15_lcm.log").write_text(
        "[RUN] sem_sd15_lcm | requested_gpu_mode=a100_80gb | output=/tmp/run\n",
        encoding="utf-8",
    )
    (run / "input_verification.json").write_text(json.dumps({"coco_root": str(coco)}), encoding="utf-8")
    assert select_suite(tmp_path, "a100_80gb", coco) == legacy.name


def test_diagnostic_reuse_requires_profile_core_environment_and_inputs(tmp_path):
    profile = "diag_sem_sd15_smoke"
    run = tmp_path / profile
    run.mkdir()
    (run / "verification.log").write_text(json.dumps({"profile": profile}) + "\n", encoding="utf-8")
    (run / "artifact.txt").write_text("result\n", encoding="utf-8")
    runner_log = tmp_path / "runner_logs" / "diagnostic"
    runner_log.mkdir(parents=True)
    (runner_log / f"{profile}.log").write_text("[RUN] requested_gpu_mode=a100_80gb\n", encoding="utf-8")
    (run / "code_sha256.txt").write_text(_snapshot(), encoding="utf-8")
    source = SSH_ROOT / "diagnostics" / f"{profile}.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "profile_sha256.txt").write_text(f"{digest}  {source}\n", encoding="utf-8")
    dependencies = b"package==1\n"
    (run / "pip_freeze.txt").write_bytes(dependencies)
    coco = tmp_path / "coco"
    hashes = {key: "f" * 64 for key in (
        "manifest_sha256", "instances_sha256", "captions_sha256", "manifest_images_sha256")}
    (run / "input_verification.json").write_text(json.dumps({"coco_root": str(coco), **hashes}), encoding="utf-8")
    context = tmp_path / "context.json"
    context.write_text(json.dumps({
        "gpu_mode": "a100_80gb", "coco_root": str(coco),
        "shared_source": shared_source_snapshot(run / "code_sha256.txt"),
        "dependencies_sha256": hashlib.sha256(dependencies).hexdigest(),
        "inputs": {"sd15": hashes},
    }), encoding="utf-8")

    check_completed(profile, "diagnostic", run, "diagnostic", context)
    (run / "profile_sha256.txt").write_text(f"{'e' * 64}  {source}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Profile source changed"):
        check_completed(profile, "diagnostic", run, "diagnostic", context)
    (run / "profile_sha256.txt").write_text(f"{digest}  {source}\n", encoding="utf-8")
    (run / "code_sha256.txt").write_text(_snapshot().replace("b" * 64, "e" * 64), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Shared generation or metric source changed"):
        check_completed(profile, "diagnostic", run, "diagnostic", context)


def test_benchmark_reuse_requires_complete_metric_export(tmp_path, monkeypatch):
    import verify_run

    profile = "sem_sd15_lcm"
    run = tmp_path / "seed_2024" / profile
    run.mkdir(parents=True)
    (run / "verification.log").write_text(json.dumps({"profile": profile}), encoding="utf-8")
    (run / "benchmark_complete.txt").write_text(profile + "\n", encoding="utf-8")
    (run / "base_seed.txt").write_text("2024\n", encoding="utf-8")
    runner_log = tmp_path / "runner_logs" / "2024"
    runner_log.mkdir(parents=True)
    (runner_log / f"{profile}.log").write_text("[RUN] requested_gpu_mode=a100_80gb\n", encoding="utf-8")
    (run / "code_sha256.txt").write_text(_snapshot(), encoding="utf-8")
    source = SSH_ROOT / "profiles" / f"{profile}.py"
    (run / "profile_sha256.txt").write_text(
        f"{hashlib.sha256(source.read_bytes()).hexdigest()}  {source}\n", encoding="utf-8")
    dependencies = b"package==1\n"
    (run / "pip_freeze.txt").write_bytes(dependencies)
    coco = tmp_path / "coco"
    hashes = {key: "f" * 64 for key in (
        "manifest_sha256", "instances_sha256", "captions_sha256", "manifest_images_sha256")}
    (run / "input_verification.json").write_text(json.dumps({"coco_root": str(coco), **hashes}), encoding="utf-8")
    context = tmp_path / "context.json"
    context.write_text(json.dumps({
        "gpu_mode": "a100_80gb", "coco_root": str(coco),
        "shared_source": shared_source_snapshot(run / "code_sha256.txt"),
        "dependencies_sha256": hashlib.sha256(dependencies).hexdigest(),
        "inputs": {"sd15": hashes},
    }), encoding="utf-8")
    called = []
    monkeypatch.setattr(verify_run, "verify_run", lambda *args, **kwargs: called.append((args, kwargs)))
    export = run / "metric_export"
    images = export / "generated_images"
    images.mkdir(parents=True)
    (export / "metric_export_summary.json").write_text(json.dumps({
        "experiment_id": profile, "num_generated_images": 1073,
    }), encoding="utf-8")
    (export / "metric_generated_manifest.jsonl").write_text("{}\n" * 1073, encoding="utf-8")
    for index in range(1073):
        (images / f"{index:04d}.png").touch()

    check_completed(profile, "full1073", run, "2024", context)
    assert called == [((profile, run), {"check_pair": False})]
    (images / "0000.png").unlink()
    with pytest.raises(RuntimeError, match="Incomplete metric export"):
        check_completed(profile, "full1073", run, "2024", context)
