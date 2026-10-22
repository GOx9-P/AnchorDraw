"""Guard the baseline/AnchorDraw protocol comparisons used after generation."""

from __future__ import annotations

import pytest

import verify_run


@pytest.mark.parametrize("baseline,proposed", [
    ("sem_sd15_lcm", "ours_sd15_lcm"),
    ("sem_sd15_hypersd", "ours_sd15_hypersd"),
    ("sem_sdxl_euler", "ours_sdxl_euler"),
])
def test_current_main_pairs_match_literal_protocol(baseline: str, proposed: str) -> None:
    verify_run.verify_main_pair_protocol(baseline, proposed)


def test_pair_gate_rejects_changed_mask_std(monkeypatch) -> None:
    original = verify_run._static_protocol

    def changed(profile: str) -> dict:
        values = original(profile)
        if profile == "ours_sd15_lcm":
            values["MASK_STD"] = 0.0
        return values

    monkeypatch.setattr(verify_run, "_static_protocol", changed)
    with pytest.raises(RuntimeError, match="MASK_STD"):
        verify_run.verify_main_pair_protocol("sem_sd15_lcm", "ours_sd15_lcm")


def test_hypersd_fallback_cannot_be_accepted_as_matched_benchmark(tmp_path) -> None:
    run_dir = tmp_path / "seed_2024" / "ours_sd15_hypersd"
    run_dir.mkdir(parents=True)
    (run_dir / "base_seed.txt").write_text("2024\n", encoding="utf-8")
    (run_dir / "console.log").write_text(
        "[WARN] fuse_lora skipped; running with active adapter instead: synthetic error\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="different LoRA/scheduler path"):
        verify_run.verify_run("ours_sd15_hypersd", run_dir)
