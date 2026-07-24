from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Iterable, Mapping


def summarize_generation_time(summary_rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    values: list[float] = []
    for row in summary_rows:
        value = row.get("elapsed_sec")
        if value is None:
            value = row.get("time_sec")
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue

    if not values:
        return {
            "time_num_samples": 0,
            "time_mean_sec": math.nan,
            "time_std_sec": math.nan,
            "time_total_sec": 0.0,
            "time_min_sec": math.nan,
            "time_max_sec": math.nan,
        }

    return {
        "time_num_samples": len(values),
        "time_mean_sec": mean(values),
        "time_std_sec": pstdev(values) if len(values) > 1 else 0.0,
        "time_total_sec": sum(values),
        "time_min_sec": min(values),
        "time_max_sec": max(values),
    }
