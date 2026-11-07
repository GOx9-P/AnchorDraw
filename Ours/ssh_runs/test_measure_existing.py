"""Existing-image metrics must preserve the exact manifest subset and count."""

import json
import sys
import types

from PIL import Image

import measure_common
import measure_existing


def test_existing_images_use_only_matching_manifest_indices(tmp_path, monkeypatch):
    import verify_inputs

    monkeypatch.setattr(measure_existing, "EXPECTED", 3)
    repo = tmp_path / "repo"
    (repo / "Ours" / "src" / "metrics").mkdir(parents=True)
    monkeypatch.setattr(measure_existing, "REPO_ROOT", repo)
    manifest = tmp_path / "manifest.jsonl"
    records = [{"sample_id": f"sample_{index}", "image_id": index,
                "file_name": f"{index:012d}.jpg", "target_size": [512, 512]}
               for index in range(3)]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    monkeypatch.setitem(measure_existing.MANIFESTS, "sd15", manifest)
    run = tmp_path / "seed_2024" / "sem_sd15_lcm"
    image_dir = run / "source" / "generated_images"
    image_dir.mkdir(parents=True)
    for index in (0, 2):
        Image.new("RGB", (512, 512), (index, 0, 0)).save(
            image_dir / f"{index:04d}_sample_{index}_generated.png")
    (run / "base_seed.txt").write_text("2024\n", encoding="utf-8")
    coco = tmp_path / "coco"
    verified = {"profile": "sem_sd15_lcm", "family": "sd15", "records": 3,
                "coco_root": str(coco), "manifest_sha256": "a", "instances_sha256": "b",
                "captions_sha256": "c", "manifest_images_sha256": "d"}
    (run / "input_verification.json").write_text(json.dumps(verified), encoding="utf-8")
    monkeypatch.setattr(verify_inputs, "verify_inputs", lambda *_: verified)
    monkeypatch.setattr(measure_existing.subprocess, "check_output", lambda *_: b"metric-env\n")

    calls = []

    def fake_evaluation(config):
        subset = [json.loads(line) for line in config.manifest_path.read_text(encoding="utf-8").splitlines()]
        calls.append([row["sample_id"] for row in subset])
        return {"num_manifest_records": len(subset), "num_evaluated": len(subset),
                "num_missing_generated": 0, "metrics": {"fid": 1.0}}

    fake_metrics = types.SimpleNamespace(MetricEvaluationConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
                                         run_evaluation=fake_evaluation)
    monkeypatch.setitem(sys.modules, "metrics", fake_metrics)
    result = measure_existing.measure("sem_sd15_lcm", run, coco)
    assert result["status"] == "measured"
    assert result["generated_count"] == result["evaluated_count"] == 2
    assert result["source_png_count"] == 2
    assert result["expected_count"] == 3
    assert result["available_indices"] == [0, 2]
    assert result["missing_indices"] == [1]
    assert result["summary_missing_indices"] == [0, 2]
    assert calls == [["sample_0", "sample_2"]]
    assert measure_existing.measure("sem_sd15_lcm", run, coco)["status"] == "measured"
    assert len(calls) == 1  # Exact same image/input/environment snapshot reuses the metric.
    Image.new("RGB", (512, 512), (9, 0, 0)).save(image_dir / "0002_sample_2_generated.png")
    assert measure_existing.measure("sem_sd15_lcm", run, coco)["status"] == "measured"
    assert len(calls) == 2  # An image changed, so the saved metric is stale.
    summary = [{"index": index, "sample_id": f"sample_{index}", "image_id": index,
                "file_name": f"{index:012d}.jpg", "seed": 2024 + index,
                "generated_path": str(image_dir / f"{index:04d}_sample_{index}_generated.png"),
                "elapsed_sec": 1.0} for index in (0, 2)]
    (image_dir.parent / "generation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    assert measure_existing.measure("sem_sd15_lcm", run, coco)["status"] == "measured"
    assert len(calls) == 3  # Timing provenance changed, so the metric is recomputed.


def test_common_group_uses_intersection_and_reports_each_source_count(tmp_path, monkeypatch):
    monkeypatch.setitem(measure_common.GROUPS, "test_pair", ("first", "second"))
    seed_root = tmp_path / "seed_2024"
    (seed_root / "first").mkdir(parents=True)
    (seed_root / "second").mkdir()
    selections = {"first": [0, 2], "second": [1, 2]}
    monkeypatch.setattr(measure_common, "_scan", lambda profile, *_: {
        "available_indices": selections[profile], "generated_count": 2})
    measured = []

    def fake_measure(profile, *_args, **kwargs):
        measured.append((profile, kwargs["common_indices"]))
        return {"status": "measured"}

    monkeypatch.setattr(measure_common, "measure", fake_measure)
    report = measure_common.measure_group("test_pair", seed_root, tmp_path / "coco")
    assert report["common_indices"] == [2]
    assert report["common_count"] == 1
    assert report["source_generated_counts"] == {"first": 2, "second": 2}
    assert measured == [("first", {2}), ("second", {2})]
