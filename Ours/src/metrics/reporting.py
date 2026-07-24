from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping


def flatten_metrics_report(report: Mapping[str, object]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for key in ("manifest_path", "generated_dir", "generation_summary", "num_manifest_records", "num_evaluated"):
        flat[key] = report.get(key)

    metrics = report.get("metrics", {})
    if isinstance(metrics, Mapping):
        for key, value in metrics.items():
            flat[key] = value
    return flat


def write_metrics_report(report: Mapping[str, object], output_dir: Path, prefix: str = "metrics") -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{prefix}.json"
    csv_path = output_dir / f"{prefix}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    flat = flatten_metrics_report(report)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)

    return json_path, csv_path
