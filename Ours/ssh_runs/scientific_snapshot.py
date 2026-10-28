"""Compare the shared generation and metric source saved by SSH runs.

``code_sha256.txt`` still records every runner file. For comparability we use
only Baseline and Ours/src, because changing an unrelated SSH diagnostic must
not invalidate an already verified benchmark. Each profile is checked against
its own ``profile_sha256.txt`` separately by the suite checkpoint.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


_ROW = re.compile(r"^([0-9a-f]{64}) [ *](.+)$")
_PREFIXES = ("Baseline/", "Ours/src/")
_SHARED_FILES = ("Ours/ssh_runs/prepare_metric_export.py",)
_GPU_PATTERN = re.compile(r"\brequested_gpu_mode=([a-z0-9_]+)\b")


def requested_gpu_mode(run_dir: Path) -> str | None:
    marker = run_dir / "gpu_mode.txt"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    # The runner prints [RUN] before tee starts. For pre-checkpoint runs this
    # line lives in runner_logs, not in the profile's console.log.
    suite = run_dir.parent.parent if run_dir.parent.name.startswith("seed_") else run_dir.parent
    label = "diagnostic" if run_dir.parent == suite else run_dir.parent.name.removeprefix("seed_")
    log = suite / "runner_logs" / label / f"{run_dir.name}.log"
    if log.is_file():
        for line in log.open(encoding="utf-8", errors="replace"):
            if line.startswith("[RUN]"):
                match = _GPU_PATTERN.search(line)
                return match.group(1) if match else None
    return None


def shared_source_snapshot(path: Path) -> tuple[tuple[str, str], ...]:
    """Return sorted (repository-relative path, SHA-256) pairs from a run."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ROW.fullmatch(line)
        if not match:
            raise RuntimeError(f"Malformed source snapshot row: {path}: {line!r}")
        digest, source = match.groups()
        source = source.replace("\\", "/")
        for prefix in _PREFIXES:
            marker = "/" + prefix
            if marker in source:
                relative = prefix + source.rsplit(marker, 1)[1]
            elif source.startswith(prefix):
                relative = source
            else:
                continue
            if relative in entries:
                raise RuntimeError(f"Duplicate source in snapshot: {relative}")
            entries[relative] = digest
            break
        else:
            for relative in _SHARED_FILES:
                if source.endswith("/" + relative) or source == relative:
                    if relative in entries:
                        raise RuntimeError(f"Duplicate source in snapshot: {relative}")
                    entries[relative] = digest
                    break
    if not any(name.startswith("Baseline/") for name in entries) or not any(
        name.startswith("Ours/src/") for name in entries
    ):
        raise RuntimeError(f"Incomplete shared source snapshot: {path}")
    return tuple(sorted(entries.items()))


def current_shared_source(repo_root: Path) -> tuple[tuple[str, str], ...]:
    entries = []
    for prefix in _PREFIXES:
        root = repo_root / prefix
        if not root.is_dir():
            raise RuntimeError(f"Missing shared source directory: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                entries.append((path.relative_to(repo_root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    for relative in _SHARED_FILES:
        path = repo_root / relative
        entries.append((relative, hashlib.sha256(path.read_bytes()).hexdigest()))
    if not entries:
        raise RuntimeError(f"No shared source files under {repo_root}")
    return tuple(sorted(entries))


def same_shared_source(first: Path, second: Path) -> bool:
    return shared_source_snapshot(first) == shared_source_snapshot(second)
