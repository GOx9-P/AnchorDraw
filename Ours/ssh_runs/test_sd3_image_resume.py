"""Check SD3 sparse summary recovery without model weights."""

from __future__ import annotations

import json
import tempfile
import time
import types
import unittest
from pathlib import Path

from PIL import Image

from export_notebooks import adapt_sd3_generation


SOURCE = '''expected_ids = ["a", "b", "c"]
summary = json.loads(RUN_SUMMARY_PATH.read_text(encoding="utf-8")) if RUN_SUMMARY_PATH.exists() else []
for position, old_row in enumerate(summary):
    if old_row["index"] != position:
        raise RuntimeError("old summary mismatch")
global_index = 0
for batch_index, batch in enumerate(loader):
    for local_index, sample_id in enumerate(batch["sample_ids"]):
        seed = BASE_SEED + global_index
        generated_path = GENERATED_IMAGES_DIR / f"{global_index:04d}_{sample_id}_generated.png"
        if global_index < len(summary):
            global_index += 1
            continue
        seed_everything(seed)
        tic = time.perf_counter()
        generated = make_generated(sample_id)
        elapsed = time.perf_counter() - tic
        generated.save(generated_path)
        row = {"index": global_index, "sample_id": sample_id, "seed": seed,
               "run_signature": run_config["run_signature"],
               "generated_path": str(generated_path), "elapsed_sec": elapsed}
        summary.append(row)
        RUN_SUMMARY_PATH.write_text(json.dumps(summary), encoding="utf-8")
        global_index += 1
        torch.cuda.empty_cache()
assert len(summary) == EXPECTED_DATASET_SIZE
assert [row["index"] for row in summary] == list(range(EXPECTED_DATASET_SIZE))
'''


class Sd3ImageResumeTest(unittest.TestCase):
    def test_post_save_failure_does_not_commit_or_duplicate_index(self) -> None:
        source = SOURCE.replace(
            '        global_index += 1\n        torch.cuda.empty_cache()\n',
            '        if sample_id == "b" and attempt == 1:\n'
            '            raise RuntimeError("synthetic post-save failure")\n'
            '        global_index += 1\n        torch.cuda.empty_cache()\n',
        )
        program = adapt_sd3_generation(source, "sem_sd3_ffm")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_dir = root / "generated_images"
            image_dir.mkdir()
            attempts: dict[str, int] = {}

            def make_generated(sample_id: str) -> Image.Image:
                attempts[sample_id] = attempts.get(sample_id, 0) + 1
                return Image.new("RGB", (2, 2), "red")

            state = {
                "RUN_SUMMARY_PATH": root / "generation_summary.json",
                "GENERATED_IMAGES_DIR": image_dir, "OUTPUT_DIR": root,
                "TARGET_SIZE": (2, 2), "Image": Image, "Path": Path,
                "BASE_SEED": 2024, "EXPECTED_DATASET_SIZE": 3,
                "run_config": {"run_signature": "synthetic"},
                "loader": [{"sample_ids": ["a", "b", "c"]}],
                "seed_everything": lambda seed: None,
                "make_generated": make_generated, "time": time, "json": json,
                "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(
                    is_available=lambda: False, empty_cache=lambda: None)),
            }
            exec(compile(program, "sd3_post_save.py", "exec"), state)
            saved = json.loads(state["RUN_SUMMARY_PATH"].read_text(encoding="utf-8"))
            self.assertEqual([row["index"] for row in saved], [0, 1, 2])
            self.assertEqual(attempts, {"a": 1, "b": 2, "c": 1})
            failures = [json.loads(line) for line in (root / "failed_images.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([(row["index"], row["attempt"]) for row in failures], [(1, 1)])

    def test_sparse_summary_can_fill_only_missing_manifest_index(self) -> None:
        program = adapt_sd3_generation(SOURCE, "sem_sd3_ffm")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_dir = root / "generated_images"
            image_dir.mkdir()
            attempts: dict[str, int] = {}
            fail_b = True

            def make_generated(sample_id: str) -> Image.Image:
                attempts[sample_id] = attempts.get(sample_id, 0) + 1
                if sample_id == "b" and fail_b:
                    raise RuntimeError("temporary image failure")
                return Image.new("RGB", (2, 2), "red")

            common = {
                "RUN_SUMMARY_PATH": root / "generation_summary.json",
                "GENERATED_IMAGES_DIR": image_dir, "OUTPUT_DIR": root,
                "TARGET_SIZE": (2, 2), "Image": Image, "Path": Path,
                "BASE_SEED": 2024, "EXPECTED_DATASET_SIZE": 3,
                "run_config": {"run_signature": "synthetic"},
                "loader": [{"sample_ids": ["a", "b", "c"]}],
                "seed_everything": lambda seed: None,
                "make_generated": make_generated, "time": time, "json": json,
                "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(
                    is_available=lambda: False, empty_cache=lambda: None)),
            }
            with self.assertRaises(AssertionError):
                exec(compile(program, "sd3_synthetic.py", "exec"), dict(common))
            first_rows = json.loads(common["RUN_SUMMARY_PATH"].read_text(encoding="utf-8"))
            self.assertEqual([row["index"] for row in first_rows], [0, 2])
            failed_attempts = [json.loads(line) for line in (root / "failed_images.jsonl").read_text(encoding="utf-8").splitlines()]
            failed_seconds = sum(row["attempt_elapsed_sec"] for row in failed_attempts)
            fail_b = False
            resumed = dict(common)
            exec(compile(program, "sd3_synthetic.py", "exec"), resumed)
            self.assertEqual([row["index"] for row in resumed["summary"]], [0, 1, 2])
            self.assertEqual([row["seed"] for row in resumed["summary"]], [2024, 2025, 2026])
            self.assertGreaterEqual(resumed["summary"][1]["elapsed_sec"], failed_seconds)
            self.assertEqual(attempts, {"a": 1, "b": 4, "c": 1})


if __name__ == "__main__":
    unittest.main()
