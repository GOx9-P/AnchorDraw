"""The paired SD3 profiles must accept and record the suite's A100 preset."""

import ast
import os
from pathlib import Path

import pytest


PROFILES = Path(__file__).parent / "profiles"


@pytest.mark.parametrize("name", ["sem_sd3_ffm", "ours_sd3_ffm"])
def test_paired_sd3_a100_configuration(name, monkeypatch):
    monkeypatch.setenv("COLAB_GPU_MODE", "a100_80gb")
    tree = ast.parse((PROFILES / f"{name}.py").read_text(encoding="utf-8"))
    selected = []
    wanted = {"COLAB_GPU_MODE", "LOW_VRAM_MODE", "BATCH_SIZE", "NUM_WORKERS"}
    config_keys = None
    signature_keys = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if target in wanted:
                selected.append(node)
            elif target == "run_config" and isinstance(node.value, ast.Dict):
                config_keys = {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
            elif target == "immutable_keys":
                signature_keys = set(ast.literal_eval(node.value))
        elif isinstance(node, ast.Assert) and "COLAB_GPU_MODE" in ast.unparse(node.test):
            selected.append(node)

    assert config_keys is not None and signature_keys is not None
    assert {"gpu_mode", "batch_size", "num_workers"} <= config_keys & signature_keys
    namespace = {"os": os, "RUN_PROFILE": "full1073"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), name, "exec"), namespace)
    assert namespace["COLAB_GPU_MODE"] == "a100_80gb"
    assert namespace["LOW_VRAM_MODE"] is False
    assert namespace["BATCH_SIZE"] == 1  # generation remains sequential
    assert namespace["NUM_WORKERS"] == 2
