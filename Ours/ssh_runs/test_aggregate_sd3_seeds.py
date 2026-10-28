"""Exercise full SD3 ZIP/report acceptance without downloading model weights."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_sd3_seeds import METHODS, SEEDS, aggregate, main  # noqa: E402


def test_complete_sd3_seed_aggregate_and_reject_wrong_seed(tmp_path: Path, monkeypatch) -> None:
    roots = {method: tmp_path / method for method in METHODS}
    for method, root in roots.items():
        root.mkdir()
        for offset, seed in enumerate(SEEDS):
            stem = f"{method}_full1073_seed{seed}_colab"
            signature = f"{offset + 1:064x}"
            config = {
                "base_seed": seed, "run_profile": "full1073", "expected_samples": 1073,
                "model_family": "sd3", "experiment_id": stem, "run_signature": signature,
                    "gpu_mode": "a100_80gb", "num_workers": 2,
                "view_profile": "native_full_v128" if method == "multidiffusion" else None,
                "manifest_sha256": "a" * 64, "input_masks_sha256": "b" * 64,
                "coco_annotations_sha256": "c" * 64, "metrics_source_sha256": "d" * 64,
                "model_id": "model", "model_revision": "model_rev",
                "flash_sd3_revision": "flash_rev", "diffusers_commit": "diff_rev",
                "dependency_versions": {"torch": "test"}, "bootstrap_steps": 2,
                "t_index_list": [0, 4, 12, 25, 37], "guidance_scale": 0.0,
                "mask_std": 0.0, "preprocess_mask_cover_alpha": 0.3,
                "data_source_sha256": "e" * 64, "pipeline_sha256": "f" * 64,
                "batch_size": 1, "mask_strength": 1.0, "mask_type": "discrete",
                "num_inference_steps": None, "negative_prompt": "", "dtype": "float16",
                "patch_safe_white_bootstrap_latent": True,
                "white_latent_sha256": f"{offset + 1:064x}",
                "white_bootstrap_seed": seed,
                "google_drive_export_root": f"/unused/{method}/{seed}/drive",
            }
            summary = [{"index": index, "sample_id": str(index), "seed": seed + index,
                        "run_signature": signature, "elapsed_sec": 1.0,
                        "generated_path": f"/verified/{method}/{seed}/{index:04d}_{index}_generated.png"}
                       for index in range(1073)]
            export = {"run_signature": signature, "manifest_sha256": "a" * 64,
                      "input_masks_sha256": "b" * 64, "coco_annotations_sha256": "c" * 64}
            def export_name(index: int) -> str:
                if method == "multidiffusion":
                    return f"{index:06d}__coco_{index:012d}__{index}__generated.png"
                return f"{index:04d}_{index}_generated.png"

            metric_manifest = "".join(json.dumps({
                "metric_index": index, "sample_id": str(index), "image_id": index,
                "file_name": f"{index:012d}.jpg", "foreground_prompts": ["object"],
                "generated_image_relative_path": f"generated_images/{export_name(index)}",
                **({"source_generated_path": summary[index]["generated_path"]}
                   if method == "multidiffusion" else {"generated_path": summary[index]["generated_path"]}),
            }) + "\n" for index in range(1073))
            zip_path = root / f"{stem}__metric_export.zip"
            with ZipFile(zip_path, "w") as archive:
                for name, value in (("run_config.json", config), ("generation_summary.json", summary),
                                    ("export_summary.json", export)):
                    archive.writestr(name, json.dumps(value))
                archive.writestr("metric_generated_manifest.jsonl", metric_manifest)
                for index in range(1073):
                    archive.writestr(f"generated_images/{export_name(index)}", b"png")
            score = 10.0 + 2.0 * offset + (1.0 if method == "anchordraw" else 0.0)
            report = {
                "num_manifest_records": 1073, "num_evaluated": 1073, "num_missing_generated": 0,
                "model_family": "sd3", "target_size": [1024, 1024],
                "selected_metrics": ["fid", "is", "clip_fg", "clip_bg", "time"], "metric_seed": 2024,
                "evaluated": [{"index": index, "sample_id": str(index), "image_id": index,
                               "file_name": f"{index:012d}.jpg",
                               "generated_path": summary[index]["generated_path"],
                               "num_foreground_regions": 1} for index in range(1073)],
                "metrics": {"fid": score, "is_mean": score, "clip_fg_x100": score,
                            "clip_bg_x100": score, "time_mean_sec": 1.0, "time_total_sec": 1073.0,
                            "clip_fg_count": 1073, "clip_bg_count": 1073,
                            "time_num_samples": 1073},
            }
            (root / f"{stem}__metrics.json").write_text(json.dumps(report), encoding="utf-8")

    result = aggregate(roots)
    fid = result["methods"]["semanticdraw"]["fid"]
    assert fid["mean"] == 12.0 and fid["std_across_seeds"] == 2.0
    assert result["paired_anchor_minus_baseline"]["fid"]["mean"] == 1.0
    suite_root = tmp_path / "suite"
    for method, profile in (("semanticdraw", "sem_sd3_ffm"),
                            ("anchordraw", "ours_sd3_ffm"),
                            ("multidiffusion", "md_sd3_ffm")):
        for seed in SEEDS:
            destination = suite_root / f"seed_{seed}" / profile / "anchordraw_metric_exports"
            destination.mkdir(parents=True)
            run_dir = destination.parent
            (run_dir / "code_sha256.txt").write_text(
                f"{'a' * 64}  /repo/Baseline/semantic-draw-main/src/model/pipeline.py\n"
                f"{'b' * 64}  /repo/Ours/src/metrics/evaluate.py\n", encoding="utf-8")
            (run_dir / "profile_sha256.txt").write_text(
                f"{'c' * 64}  /repo/Ours/ssh_runs/profiles/{profile}.py\n", encoding="utf-8")
            (run_dir / "gpu_mode.txt").write_text("a100_80gb\n", encoding="utf-8")
            (run_dir / "pip_freeze.txt").write_text("same verified environment\n", encoding="utf-8")
            (run_dir / "input_verification.json").write_text(json.dumps({
                "family": "sd3", "records": 1073, "manifest_sha256": "a" * 64,
                "instances_sha256": "b" * 64, "captions_sha256": "c" * 64,
                "manifest_images_sha256": "d" * 64, "coco_root": "/same/coco",
            }), encoding="utf-8")
            for source in roots[method].glob(f"*seed{seed}*__metric_export.zip"):
                shutil.copy2(source, destination / source.name)
                sidecar = source.with_name(source.name.replace("__metric_export.zip", "__metrics.json"))
                shutil.copy2(sidecar, destination / sidecar.name)
    assert aggregate({method: suite_root for method in METHODS})["methods"]["semanticdraw"]["fid"]["mean"] == 12.0
    changed_code = suite_root / f"seed_{SEEDS[0]}" / "ours_sd3_ffm" / "code_sha256.txt"
    changed_code.write_text(
        f"{'d' * 64}  /repo/Baseline/semantic-draw-main/src/model/pipeline.py\n"
        f"{'b' * 64}  /repo/Ours/src/metrics/evaluate.py\n", encoding="utf-8")
    try:
        aggregate({method: suite_root for method in METHODS})
    except RuntimeError as error:
        assert "different code" in str(error)
    else:
        raise AssertionError("SD3 suite accepted different source snapshots")
    changed_code.write_text(
        f"{'a' * 64}  /repo/Baseline/semantic-draw-main/src/model/pipeline.py\n"
        f"{'b' * 64}  /repo/Ours/src/metrics/evaluate.py\n", encoding="utf-8")
    output = tmp_path / "summary"
    monkeypatch.setattr(sys, "argv", ["aggregate_sd3_seeds.py", "--semanticdraw", str(roots["semanticdraw"]),
                                      "--anchordraw", str(roots["anchordraw"]),
                                      "--multidiffusion", str(roots["multidiffusion"]),
                                      "--output", str(output)])
    main()
    assert json.loads((output / "sd3_three_seed_metrics.json").read_text(encoding="utf-8"))["methods"]["semanticdraw"]["fid"]["mean"] == 12.0
    assert "std_across_seeds" in (output / "sd3_three_seed_metrics.csv").read_text(encoding="utf-8")

    first_report = roots["semanticdraw"] / f"semanticdraw_full1073_seed{SEEDS[0]}_colab__metrics.json"
    changed = json.loads(first_report.read_text(encoding="utf-8"))
    changed["num_evaluated"] = 1072
    first_report.write_text(json.dumps(changed), encoding="utf-8")
    try:
        aggregate(roots)
    except RuntimeError as error:
        assert "incomplete set" in str(error)
    else:
        raise AssertionError("Partial SD3 metric report was accepted")
    changed["num_evaluated"] = 1073
    changed["evaluated"][0]["generated_path"] = "/another_method/0000_0_generated.png"
    first_report.write_text(json.dumps(changed), encoding="utf-8")
    try:
        aggregate(roots)
    except RuntimeError as error:
        assert "wrong image or region set" in str(error)
    else:
        raise AssertionError("SD3 metric report accepted an image from another run")
