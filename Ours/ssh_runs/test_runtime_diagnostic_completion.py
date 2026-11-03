"""Regression checks for the four-branch runtime diagnostic's completion gate."""

from pathlib import Path

import pytest


PROFILE = Path(__file__).parent / "diagnostics" / "diag_ours_sd15_lcm_runtime_smoke.py"


def _completion_check():
    lines = PROFILE.read_text(encoding="utf-8").splitlines()
    start = lines.index("assert global_index == EXPECTED_SAMPLES, (global_index, EXPECTED_SAMPLES)")
    end = lines.index("assert not missing_images, missing_images")
    return compile("\n".join(lines[start:end + 1]), str(PROFILE), "exec")


def _complete_run(tmp_path):
    modes = ("baseline", "bbox_control", "semantic_anchor", "semantic_topk_anchor")
    rows = []
    for sample_index in range(8):
        for mode in modes:
            image = tmp_path / f"{sample_index}_{mode}.png"
            image.write_bytes(b"image")
            rows.append({"sample_index": sample_index, "branch": mode, "generated_path": str(image)})
    return {
        "global_index": 8,
        "EXPECTED_SAMPLES": 8,
        "RUNTIME_MODES": modes,
        "generation_rows": rows,
        "metrics_rows": [None] * 32,
        "expected_metric_rows": 32,
        "Path": Path,
    }


def test_runtime_diagnostic_accepts_eight_samples_in_four_branches(tmp_path):
    exec(_completion_check(), _complete_run(tmp_path))


@pytest.mark.parametrize("fault", ["duplicate_pair", "missing_image", "missing_metric"])
def test_runtime_diagnostic_rejects_incomplete_output(tmp_path, fault):
    run = _complete_run(tmp_path)
    if fault == "duplicate_pair":
        run["generation_rows"][-1] = dict(run["generation_rows"][0])
    elif fault == "missing_image":
        Path(run["generation_rows"][-1]["generated_path"]).unlink()
    else:
        run["metrics_rows"].pop()
    with pytest.raises(AssertionError):
        exec(_completion_check(), run)
