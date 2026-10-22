"""Exercise the exported per-image isolation without loading diffusion models."""

from __future__ import annotations

import json
import tempfile
import time
import types
import unittest
from pathlib import Path

from export_notebooks import adapt_image_failures


SOURCE = '''summary = []
global_index = 0
for batch_index, batch in enumerate(loader):
    for local_index, sample_id in enumerate(batch):
        seed = BASE_SEED + global_index
        tic = time.perf_counter()
        generated_path = WORK_DIR / f"{global_index:04d}_{sample_id}_generated.png"
        generated = make_fake(sample_id, seed)
        elapsed = time.perf_counter() - tic
        generated.save(generated_path)
        summary.append({"index": global_index, "sample_id": sample_id,
                        "seed": seed, "elapsed_sec": elapsed,
                        "generated_path": str(generated_path)})
        global_index += 1
assert len(summary) == 3
'''


class FakeImage:
    def save(self, path: Path) -> None:
        path.write_bytes(b"fake")


class ImageFailureHandlingTest(unittest.TestCase):
    def run_case(self, fail_indices: set[str], transient: set[str],
                 post_save_failure: bool = False) -> tuple[dict, list[dict]]:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attempts: dict[str, int] = {}

            def make_fake(sample_id: str, seed: int) -> FakeImage:
                attempts[sample_id] = attempts.get(sample_id, 0) + 1
                if sample_id in fail_indices or (sample_id in transient and attempts[sample_id] == 1):
                    raise RuntimeError(f"synthetic failure for {sample_id}")
                return FakeImage()

            namespace = {
                "loader": [["a", "b", "c"]], "BASE_SEED": 2024,
                "WORK_DIR": root, "make_fake": make_fake, "Path": Path,
                "json": json, "torch": types.SimpleNamespace(
                    cuda=types.SimpleNamespace(is_available=lambda: False)
                ), "time": time,
            }
            source = SOURCE
            if post_save_failure:
                source = source.replace(
                    "        global_index += 1\n",
                    '        if sample_id == "b" and attempt == 1:\n'
                    '            raise RuntimeError("synthetic post-save failure")\n'
                    '        global_index += 1\n',
                )
            program = adapt_image_failures(source, "sem_sd15_lcm")
            try:
                exec(compile(program, "synthetic_profile.py", "exec"), namespace)
            except RuntimeError as exc:
                namespace["raised_error"] = str(exc)
            log_path = root / "failed_images.jsonl"
            failures = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
            namespace["attempts"] = attempts
            return namespace, failures

    def test_transient_failure_retries_same_seed(self) -> None:
        state, failures = self.run_case(set(), {"a"})
        self.assertNotIn("raised_error", state)
        self.assertEqual(state["attempts"], {"a": 2, "b": 1, "c": 1})
        self.assertEqual([row["seed"] for row in state["summary"]], [2024, 2025, 2026])
        self.assertEqual([(row["index"], row["seed"], row["attempt"]) for row in failures], [(0, 2024, 1)])

    def test_permanent_failure_does_not_stop_later_images_or_pass_gate(self) -> None:
        state, failures = self.run_case({"b"}, set())
        self.assertIn("Run incomplete", state["raised_error"])
        self.assertEqual([row["index"] for row in state["summary"]], [0, 2])
        self.assertEqual(state["attempts"], {"a": 1, "b": 3, "c": 1})
        self.assertEqual([row["seed"] for row in failures], [2025, 2025, 2025])

    def test_post_save_exception_rolls_back_summary_and_retries(self) -> None:
        state, failures = self.run_case(set(), set(), post_save_failure=True)
        self.assertNotIn("raised_error", state)
        self.assertEqual(state["attempts"], {"a": 1, "b": 2, "c": 1})
        self.assertEqual([row["index"] for row in state["summary"]], [0, 1, 2])
        self.assertEqual([(row["index"], row["attempt"]) for row in failures], [(1, 1)])


if __name__ == "__main__":
    unittest.main()
