from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Optional

from PIL import Image


JsonDict = dict[str, object]


def load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def load_generation_summary(path: Optional[Path]) -> list[JsonDict]:
    if path is None or not Path(path).exists():
        return []

    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("samples", "records", "results", "generations"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    raise TypeError(f"Unsupported generation summary format: {type(data)!r}")


def build_summary_index(summary_rows: Iterable[JsonDict]) -> dict[str, JsonDict]:
    index: dict[str, JsonDict] = {}
    for row in summary_rows:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            index[str(sample_id)] = row
    return index


def candidate_generated_paths(
    record: Mapping[str, object],
    generated_dir: Optional[Path],
    summary_index: Mapping[str, JsonDict],
    global_index: Optional[int] = None,
) -> list[Path]:
    sample_id = str(record["sample_id"])
    image_id = int(record["image_id"])
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    summary_row = summary_index.get(sample_id)
    if summary_row is not None:
        generated_path = summary_row.get("generated_path")
        if generated_path:
            path = Path(str(generated_path))
            add(path)
            if generated_dir is not None:
                add(generated_dir / path.name)

    if generated_dir is not None:
        if global_index is not None:
            add(generated_dir / f"{global_index:04d}_{sample_id}_generated.png")
            add(generated_dir / f"{global_index:02d}_{sample_id}_generated.png")

        for pattern in (
            f"*{sample_id}*_generated.png",
            f"*{sample_id}*.png",
            f"*{image_id:012d}*_generated.png",
            f"*{image_id:012d}*.png",
        ):
            for path in sorted(generated_dir.glob(pattern)):
                add(path)

    return candidates


def resolve_generated_image_path(
    record: Mapping[str, object],
    generated_dir: Optional[Path],
    summary_index: Mapping[str, JsonDict],
    global_index: Optional[int] = None,
) -> Optional[Path]:
    for path in candidate_generated_paths(record, generated_dir, summary_index, global_index):
        if path.exists():
            return path
    return None


def load_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")
