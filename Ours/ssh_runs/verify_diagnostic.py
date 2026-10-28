"""Check that a diagnostic completed and wrote its own result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RUNNER_FILES = {
    "console.log", "verification.log", "input_verification.json", "repo_commit.txt",
    "repo_status.txt", "code_sha256.txt", "profile_sha256.txt", "pip_freeze.txt", "nvidia_smi.txt",
    "gpu_mode.txt", "base_seed.txt",
}


def verify_diagnostic(profile: str, run_dir: Path) -> dict:
    artifacts = [path for path in run_dir.rglob("*")
                 if path.is_file() and path.name not in RUNNER_FILES and path.stat().st_size > 0]
    if not artifacts:
        raise RuntimeError(f"Diagnostic {profile} produced no result artifact under {run_dir}")
    for path in artifacts:
        if path.suffix != ".json":
            continue
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        if isinstance(content, dict) and content.get("status") == "FAIL":
            raise RuntimeError(f"Diagnostic declared FAIL: {path}")
    return {"profile": profile, "artifact_count": len(artifacts),
            "note": "Source diagnostic assertions completed; benchmark metrics are not asserted here."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_diagnostic(args.profile, args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
