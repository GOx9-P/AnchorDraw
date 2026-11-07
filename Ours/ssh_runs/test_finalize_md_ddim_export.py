"""The MD DDIM export can be completed from verified generation artifacts."""

import json
from pathlib import Path

import pytest

import finalize_md_sd15_ddim_export as adapter
import prepare_metric_export as export


def _raw_run(tmp_path):
    run = tmp_path / "seed_2024" / "md_sd15_ddim"
    images = run / "anchordraw_runs" / "experiment" / "generated_images"
    images.mkdir(parents=True)
    summary = [
        {"index": index, "sample_id": str(index), "image_id": index,
         "file_name": f"{index:012d}.jpg", "seed": 2024 + index,
         "elapsed_sec": 1.0, "generated_path": str(images / f"{index:04d}_generated.png")}
        for index in range(2)
    ]
    for row in summary:
        Path(row["generated_path"]).write_bytes(b"png")
    (images.parent / "generation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    target = images.parent / "metric_generated_manifest.jsonl"
    target.write_text("".join(json.dumps(row) + "\n" for row in summary), encoding="utf-8")
    return run, target


def test_finalize_preserves_raw_rows_before_preparing_enriched_manifest(tmp_path, monkeypatch):
    run, target = _raw_run(tmp_path)
    verified = []
    monkeypatch.setattr(adapter, "verify_run", lambda *args: verified.append(args))
    monkeypatch.setattr(export, "verify_run", lambda *args: verified.append(args))
    manifest = tmp_path / "source.jsonl"
    manifest.write_text("".join(json.dumps({
        "sample_id": str(index), "image_id": index, "file_name": f"{index:012d}.jpg",
        "target_size": [512, 512], "caption": "background", "foreground_prompts": ["object"],
        "category_names": ["object"], "annotation_ids": [index],
    }) + "\n" for index in range(2)), encoding="utf-8")
    monkeypatch.setattr(export, "MANIFESTS", {"sd15": manifest})
    monkeypatch.setenv("COCO_ROOT", str(tmp_path / "coco"))
    assert adapter.finalize(run) == target
    assert target.with_name("generation_rows_manifest.jsonl").is_file()
    enriched = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [row["metric_index"] for row in enriched] == [0, 1]
    assert [row["seed"] for row in enriched] == [2024, 2025]
    assert all(Path(row["generated_image_relative_path"]).parts[0] == "generated_images" for row in enriched)
    assert adapter.finalize(run) == target  # a retry preserves the enriched manifest
    assert verified == [("md_sd15_ddim", run)] * 4


def test_finalize_rejects_unrecognized_existing_manifest(tmp_path, monkeypatch):
    run, target = _raw_run(tmp_path)
    target.write_text('{"unexpected": true}\n', encoding="utf-8")
    monkeypatch.setattr(adapter, "verify_run", lambda *args: None)
    with pytest.raises(RuntimeError, match="Unrecognized existing metric manifest"):
        adapter.finalize(run)
