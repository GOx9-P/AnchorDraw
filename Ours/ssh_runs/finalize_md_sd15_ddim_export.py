"""Finish the verified MD SD1.5 DDIM metric export without regenerating images.

The source notebook writes generation rows to a file named
``metric_generated_manifest.jsonl``. The SSH metric adapter uses that same name
for an enriched COCO manifest. Preserve the source rows before replacing them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prepare_metric_export import prepare_metric_export
from verify_run import verify_run


def finalize(run_dir: Path) -> Path:
    profile = "md_sd15_ddim"
    verify_run(profile, run_dir)
    summaries = list(run_dir.rglob("generation_summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"Expected one verified generation summary: {summaries}")
    summary_path = summaries[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_dirs = {Path(row["generated_path"]).resolve().parent for row in summary}
    if len(source_dirs) != 1 or next(iter(source_dirs)).name != "generated_images":
        raise RuntimeError(f"Unexpected MD DDIM image directory: {source_dirs}")
    target = next(iter(source_dirs)).parent / "metric_generated_manifest.jsonl"
    if not target.resolve().is_relative_to(run_dir.resolve()):
        raise RuntimeError(f"Metric manifest is outside the run: {target}")
    if target.is_file():
        rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows == summary:
            backup = target.with_name("generation_rows_manifest.jsonl")
            if backup.exists():
                if backup.read_bytes() != target.read_bytes():
                    raise RuntimeError(f"Generation-row backup differs from source: {backup}")
                target.unlink()
            else:
                os.replace(target, backup)
            print(f"[EXPORT] Preserved source generation rows: {backup}", flush=True)
        elif len(rows) != len(summary) or any(
            not isinstance(row, dict) or row.get("metric_index") != index
            for index, row in enumerate(rows)
        ):
            raise RuntimeError(f"Unrecognized existing metric manifest: {target}")
    return prepare_metric_export(profile, run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(finalize(args.run_dir))


if __name__ == "__main__":
    main()
