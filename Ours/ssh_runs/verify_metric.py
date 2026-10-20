"""Check result reports written by the three auxiliary FID programs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REPORTS = {
    "cleanfid": "cleanfid_results.json",
    "pytorch_fid": "pytorch_fid_results.json",
    "ttur_fid": "ttur_tensorflow_fid_results.json",
}


def verify_metric(tool: str, run_dir: Path) -> dict:
    expected = REPORTS[tool]
    found = list(run_dir.rglob(expected))
    if len(found) != 1:
        raise RuntimeError(f"Expected one {expected}, found {found}")
    data = json.loads(found[0].read_text(encoding="utf-8"))
    rows = data if tool == "cleanfid" else [data]
    if not rows:
        raise RuntimeError("Metric report contains no experiments")
    for row in rows:
        score = row.get("score") if tool == "cleanfid" else row.get("fid")
        if score is None or not math.isfinite(float(score)):
            raise RuntimeError(f"Non-finite FID in {found[0]}")
        generated = int(row.get("num_generated", -1))
        reference = int(row.get("num_reference", -1))
        if generated <= 0 or generated != reference:
            raise RuntimeError(f"Generated/reference count mismatch in {found[0]}")
        if tool != "cleanfid" and generated != 1073:
            raise RuntimeError(f"Legacy full1073 FID expected 1073 pairs, got {generated}")
    return {"tool": tool, "experiments": len(rows), "report": str(found[0])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", choices=REPORTS)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_metric(args.tool, args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
