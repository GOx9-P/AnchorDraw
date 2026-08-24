"""Evaluate exported per-layer attention maps without loading the diffusion model.

Usage on Windows:
    python evaluate_per_layer_export_offline.py \
      --export-root "D:\\File of Phuc\\Research\\AnchorDraw\\semantic_anchor_weighted_mask_sd15_lcm_smoke2_all_artifacts_shortpaths__export"

The export must contain ``attention_and_weight_arrays``, ``mask_cache`` and
``weighted_mask_per_layer_metrics_*.csv``.  This script only reads those
artifacts and writes overlays, contact sheets and layer rankings.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


LAYER_KEY_RE = re.compile(
    r"^r(?P<region>\d+)_s(?P<step>\d+)_t(?P<timestep>-?\d+)_"
    r"layer(?P<layer>\d+)_.*_attention$"
)
SAMPLE_INDEX_RE = re.compile(r"^(?P<sample>\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output directory; defaults to <export-root>/per_layer_anchor_evaluation.",
    )
    parser.add_argument("--sample-count", type=int, default=2)
    parser.add_argument("--topk-percent", type=float, default=10.0)
    parser.add_argument(
        "--no-contact-sheets",
        action="store_true",
        help="Skip contact sheets and only write individual overlays/metrics.",
    )
    return parser.parse_args()


def load_cached_mask(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.detach().float().cpu().squeeze()


def compute_anchor_measurements(
    attention: torch.Tensor,
    mask: torch.Tensor,
    topk_percent: float,
) -> dict[str, Any]:
    attention = attention.detach().float().cpu()
    mask = mask.detach().float().cpu().squeeze().bool()
    if attention.ndim != 2:
        raise ValueError(f"Attention map must be 2D, got {tuple(attention.shape)}")
    if attention.shape != mask.shape:
        mask = F.interpolate(
            mask[None, None].float(), size=attention.shape, mode="nearest"
        )[0, 0].bool()
    if not mask.any():
        raise ValueError("Mask is empty after resize.")

    masked = attention.masked_fill(~mask, float("-inf"))
    width = int(attention.shape[1])
    anchor_flat = int(masked.argmax())
    anchor_y, anchor_x = divmod(anchor_flat, width)

    ys, xs = torch.where(mask)
    in_values = attention[mask]
    topk_count = max(1, int(np.ceil(in_values.numel() * topk_percent / 100.0)))
    topk_values, topk_indices = torch.topk(in_values, k=topk_count)
    topk_xs = xs[topk_indices].float()
    topk_ys = ys[topk_indices].float()
    topk_center_x = float(topk_xs.mean())
    topk_center_y = float(topk_ys.mean())
    nearest = int(
        ((topk_xs - topk_center_x).pow(2) + (topk_ys - topk_center_y).pow(2)).argmin()
    )
    topk_anchor_x = float(topk_xs[nearest])
    topk_anchor_y = float(topk_ys[nearest])

    global_flat = int(attention.argmax())
    global_y, global_x = divmod(global_flat, width)
    mask_ys, mask_xs = torch.where(mask)
    bbox_x0, bbox_x1 = int(mask_xs.min()), int(mask_xs.max())
    bbox_y0, bbox_y1 = int(mask_ys.min()), int(mask_ys.max())
    centroid_x, centroid_y = float(mask_xs.float().mean()), float(mask_ys.float().mean())
    bbox_center_x = (bbox_x0 + bbox_x1) / 2.0
    bbox_center_y = (bbox_y0 + bbox_y1) / 2.0
    diagonal = float((attention.shape[0] ** 2 + attention.shape[1] ** 2) ** 0.5)

    flat = attention.flatten()
    global_topk_count = max(1, int(np.ceil(flat.numel() * topk_percent / 100.0)))
    global_topk_indices = torch.topk(flat, k=global_topk_count).indices
    global_topk = torch.zeros_like(flat, dtype=torch.bool)
    global_topk[global_topk_indices] = True
    global_topk = global_topk.reshape(attention.shape)
    intersection = (global_topk & mask).sum().float()
    union = (global_topk | mask).sum().float().clamp_min(1)
    total = attention.sum().clamp_min(1e-8)
    inside_mean = attention[mask].mean()
    outside_mean = attention[~mask].mean() if (~mask).any() else torch.tensor(0.0)

    return {
        "anchor_x": float(anchor_x),
        "anchor_y": float(anchor_y),
        "anchor_attention": float(attention[anchor_y, anchor_x]),
        "topk_percent": float(topk_percent),
        "topk_pixel_count": int(topk_count),
        "topk_attention_threshold": float(topk_values.min()),
        "topk_center_x": topk_center_x,
        "topk_center_y": topk_center_y,
        "topk_anchor_x": topk_anchor_x,
        "topk_anchor_y": topk_anchor_y,
        "topk_anchor_inside_mask": bool(mask[int(topk_anchor_y), int(topk_anchor_x)]),
        "topk_anchor_attention": float(attention[int(topk_anchor_y), int(topk_anchor_x)]),
        "topk_anchor_attention_mean": float(topk_values.mean()),
        "global_peak_x": float(global_x),
        "global_peak_y": float(global_y),
        "global_peak_inside_mask": bool(mask[global_y, global_x]),
        "distance_to_centroid_px": float(
            ((anchor_x - centroid_x) ** 2 + (anchor_y - centroid_y) ** 2) ** 0.5
        ),
        "distance_to_bbox_center_px": float(
            ((anchor_x - bbox_center_x) ** 2 + (anchor_y - bbox_center_y) ** 2) ** 0.5
        ),
        "distance_to_centroid_norm": float(
            ((anchor_x - centroid_x) ** 2 + (anchor_y - centroid_y) ** 2) ** 0.5
            / diagonal
        ),
        "distance_to_bbox_center_norm": float(
            ((anchor_x - bbox_center_x) ** 2 + (anchor_y - bbox_center_y) ** 2) ** 0.5
            / diagonal
        ),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bbox_center_x": bbox_center_x,
        "bbox_center_y": bbox_center_y,
        "global_topk_iou": float(intersection / union),
        "global_topk_precision": float(intersection / global_topk_count),
        "global_topk_recall": float(intersection / mask.sum().float().clamp_min(1)),
        "attention_mass_inside": float(attention[mask].sum() / total),
        "attention_inside_mean": float(inside_mean),
        "attention_outside_mean": float(outside_mean),
        "attention_inside_outside_gap": float(inside_mean - outside_mean),
    }


def draw_overlay(
    attention: np.ndarray,
    mask: np.ndarray | None,
    measurement: dict[str, Any],
    title: str,
    destination: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.imshow(attention, cmap="magma", vmin=0, vmax=1)
    if mask is not None:
        ax.contour(mask, levels=[0.5], colors="cyan", linewidths=0.8)
    ax.scatter(
        measurement["anchor_x"], measurement["anchor_y"],
        c="lime", marker="x", s=100, linewidths=2.5, label="masked argmax",
    )
    ax.scatter(
        measurement["topk_center_x"], measurement["topk_center_y"],
        facecolors="none", edgecolors="white", marker="o", s=70,
        linewidths=1.5, label="top-k centroid",
    )
    ax.scatter(
        measurement["topk_anchor_x"], measurement["topk_anchor_y"],
        c="orange", marker="D", s=48, label="top-k projected",
    )
    ax.scatter(
        measurement["bbox_center_x"], measurement["bbox_center_y"],
        c="yellow", marker="+", s=90, linewidths=2, label="bbox center",
    )
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=6)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=130, bbox_inches="tight")
    plt.close(fig)


def minmax(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    low, high = float(series.min()), float(series.max())
    if abs(high - low) < 1e-8:
        return pd.Series(0.5, index=series.index)
    scaled = (series - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def stored_measurements(row: pd.Series) -> dict[str, Any]:
    """Recover anchor coordinates from the exported per-layer CSV."""
    float_columns = [
        "anchor_x", "anchor_y", "anchor_attention", "topk_center_x", "topk_center_y",
        "topk_anchor_x", "topk_anchor_y", "bbox_center_x", "bbox_center_y",
        "distance_to_centroid_px", "distance_to_bbox_center_px", "distance_to_centroid_norm",
        "distance_to_bbox_center_norm", "topk_anchor_attention", "topk_anchor_attention_mean",
    ]
    values: dict[str, Any] = {}
    for column in float_columns:
        value = row.get(column, np.nan)
        values[column] = float(value) if not pd.isna(value) else np.nan
    return values


def main() -> None:
    args = parse_args()
    export_root = args.export_root.expanduser().resolve()
    if not export_root.is_dir():
        raise FileNotFoundError(export_root)
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if not 0.0 < args.topk_percent <= 100.0:
        raise ValueError("--topk-percent must be in (0, 100]")

    output_root = (args.output_root or (export_root / "per_layer_anchor_evaluation")).resolve()
    overlay_root = output_root / "overlays"
    sheet_root = output_root / "contact_sheets"
    overlay_root.mkdir(parents=True, exist_ok=True)
    if not args.no_contact_sheets:
        sheet_root.mkdir(parents=True, exist_ok=True)

    metrics_candidates = sorted(export_root.glob("weighted_mask_per_layer_metrics_*.csv"))
    if not metrics_candidates:
        raise FileNotFoundError("weighted_mask_per_layer_metrics_*.csv")
    metrics_df = pd.read_csv(metrics_candidates[0])
    metric_lookup: dict[tuple[str, int, int, int, int, int], tuple[str, int]] = {}
    for item in metrics_df.itertuples(index=False):
        key = (
            str(item.experiment_id), int(item.sample_index), int(item.region_index),
            int(item.step_index), int(item.timestep), int(item.layer_index),
        )
        metric_lookup[key] = (
            str(item.layer_name), int(getattr(item, "native_spatial_size", 0)),
        )

    mask_root = export_root / "mask_cache"
    mask_cache_available = any(mask_root.glob("*.pt"))
    if not mask_cache_available:
        print("[WARN] mask_cache is empty; using stored per-layer CSV metrics.")
        print("[WARN] Global top-k IoU/mass metrics will be NaN, but anchors and ranking remain available.")
    npz_root = export_root / "attention_and_weight_arrays"
    npz_paths = sorted(npz_root.glob("*/*.npz"))
    npz_paths = [
        path for path in npz_paths
        if (match := SAMPLE_INDEX_RE.match(path.name))
        and int(match.group("sample")) < args.sample_count
    ]
    if not npz_paths:
        raise FileNotFoundError("No matching per-layer NPZ files found.")

    rows: list[dict[str, Any]] = []
    print(f"[INFO] Export: {export_root}")
    print(f"[INFO] NPZ files: {len(npz_paths)} | metrics: {metrics_candidates[0].name}")

    for npz_path in npz_paths:
        experiment_id = npz_path.parent.name
        sample_match = SAMPLE_INDEX_RE.match(npz_path.name)
        assert sample_match is not None
        sample_index = int(sample_match.group("sample"))
        with np.load(npz_path, allow_pickle=False) as arrays:
            grouped: dict[tuple[int, int, int], list[tuple[int, int, np.ndarray, dict[str, Any]]]] = defaultdict(list)
            for array_key in arrays.files:
                match = LAYER_KEY_RE.match(array_key)
                if match is None:
                    continue
                region_index = int(match.group("region"))
                step_index = int(match.group("step"))
                timestep = int(match.group("timestep"))
                layer_index = int(match.group("layer"))

                lookup_key = (
                    experiment_id, sample_index, region_index,
                    step_index, timestep, layer_index,
                )
                metric_row = metric_lookup.get(lookup_key)
                if metric_row is None:
                    raise KeyError(f"Missing layer metadata for {lookup_key}")
                layer_name, native_size = metric_row
                metric_rows = metrics_df[
                    (metrics_df["experiment_id"].astype(str) == experiment_id)
                    & (metrics_df["sample_index"] == sample_index)
                    & (metrics_df["region_index"] == region_index)
                    & (metrics_df["step_index"] == step_index)
                    & (metrics_df["timestep"] == timestep)
                    & (metrics_df["layer_index"] == layer_index)
                ]
                if metric_rows.empty:
                    raise KeyError(f"Missing annotation metadata for {lookup_key}")
                metric_row = metric_rows.iloc[0]
                annotation_id = int(metric_row["annotation_id"])
                mask_path = mask_root / (
                    f"image_{int(metric_row['image_id']):012d}_ann_{annotation_id}.pt"
                )
                attention = torch.from_numpy(np.asarray(arrays[array_key])).float()
                mask = load_cached_mask(mask_path).numpy() if mask_path.exists() else None
                if mask is not None:
                    measurement = compute_anchor_measurements(
                        attention, torch.from_numpy(mask), args.topk_percent
                    )
                else:
                    measurement = stored_measurements(metric_row)
                    measurement.update({
                        "topk_anchor_inside_mask": metric_row.get("topk_anchor_inside_mask", np.nan),
                        "global_peak_inside_mask": metric_row.get("global_peak_inside_mask", np.nan),
                        "global_topk_iou": np.nan,
                        "global_topk_precision": np.nan,
                        "global_topk_recall": np.nan,
                        "attention_mass_inside": np.nan,
                        "attention_inside_mean": np.nan,
                        "attention_outside_mean": np.nan,
                        "attention_inside_outside_gap": np.nan,
                    })
                row = {
                    **measurement,
                    "experiment_id": experiment_id,
                    "sample_index": sample_index,
                    "sample_id": str(metric_row["sample_id"]),
                    "image_id": int(metric_row["image_id"]),
                    "region_index": region_index,
                    "annotation_id": annotation_id,
                    "category": str(metric_row["category"]),
                    "step_index": step_index,
                    "timestep": timestep,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "native_spatial_size": native_size,
                    "array_key": array_key,
                    "mask_available": mask is not None,
                }
                rows.append(row)

                overlay_path = overlay_root / experiment_id / f"{sample_index:04d}" / (
                    f"r{region_index:02d}_s{step_index:02d}_t{timestep}_L{layer_index:02d}.png"
                )
                draw_overlay(
                    attention.numpy(), mask, measurement,
                    f"{experiment_id} | sample {sample_index} | region {region_index} | "
                    f"step {step_index} | L{layer_index:02d} ({native_size}x{native_size})",
                    overlay_path,
                )
                grouped[(region_index, step_index, timestep)].append(
                    (layer_index, native_size, attention.numpy(), measurement, mask)
                )

            if not args.no_contact_sheets:
                for (region_index, step_index, timestep), items in grouped.items():
                    items.sort(key=lambda item: item[0])
                    columns = 4
                    sheet_rows = max(1, int(np.ceil(len(items) / columns)))
                    fig, axes = plt.subplots(
                        sheet_rows, columns,
                        figsize=(4.2 * columns, 4.5 * sheet_rows),
                    )
                    axes = np.atleast_1d(axes).ravel()
                    for axis, (layer_index, native_size, attention_np, measurement, item_mask) in zip(axes, items):
                        axis.imshow(attention_np, cmap="magma", vmin=0, vmax=1)
                        if item_mask is not None:
                            axis.contour(item_mask, levels=[0.5], colors="cyan", linewidths=0.6)
                        axis.scatter(measurement["anchor_x"], measurement["anchor_y"], c="lime", marker="x", s=65, linewidths=2)
                        axis.scatter(measurement["topk_anchor_x"], measurement["topk_anchor_y"], c="orange", marker="D", s=28)
                        axis.set_title(f"L{layer_index:02d} | {native_size}x{native_size}", fontsize=8)
                        axis.axis("off")
                    for axis in axes[len(items):]:
                        axis.axis("off")
                    sheet_path = sheet_root / experiment_id / f"{sample_index:04d}" / (
                        f"r{region_index:02d}_s{step_index:02d}_t{timestep}_layers.png"
                    )
                    sheet_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.suptitle(
                        f"{experiment_id} | sample {sample_index} | region {region_index} | step {step_index}",
                        fontsize=12,
                    )
                    fig.tight_layout()
                    fig.savefig(sheet_path, dpi=120, bbox_inches="tight")
                    plt.close(fig)

    evaluation_df = pd.DataFrame(rows)
    evaluation_df["temporal_anchor_std_px"] = np.nan
    evaluation_df["mean_adjacent_anchor_jump_px"] = np.nan
    evaluation_df["max_adjacent_anchor_jump_px"] = np.nan
    for _, group in evaluation_df.groupby(
        ["experiment_id", "sample_index", "region_index", "layer_index"]
    ):
        indices = group.sort_values("step_index").index
        points = evaluation_df.loc[indices, ["anchor_x", "anchor_y"]].to_numpy(float)
        center = points.mean(axis=0, keepdims=True)
        jumps = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.asarray([0.0])
        evaluation_df.loc[indices, "temporal_anchor_std_px"] = np.linalg.norm(points - center, axis=1).mean()
        evaluation_df.loc[indices, "mean_adjacent_anchor_jump_px"] = jumps.mean()
        evaluation_df.loc[indices, "max_adjacent_anchor_jump_px"] = jumps.max()

    ranking_df = evaluation_df.groupby(
        ["experiment_id", "layer_index", "layer_name", "native_spatial_size"],
        as_index=False,
    ).agg(
        measurements=("array_key", "size"),
        global_topk_iou=("global_topk_iou", "mean"),
        global_topk_precision=("global_topk_precision", "mean"),
        global_topk_recall=("global_topk_recall", "mean"),
        attention_mass_inside=("attention_mass_inside", "mean"),
        attention_inside_outside_gap=("attention_inside_outside_gap", "mean"),
        distance_to_bbox_center_norm=("distance_to_bbox_center_norm", "mean"),
        temporal_anchor_std_px=("temporal_anchor_std_px", "mean"),
        mean_adjacent_anchor_jump_px=("mean_adjacent_anchor_jump_px", "mean"),
        anchor_attention=("anchor_attention", "mean"),
        topk_anchor_attention_mean=("topk_anchor_attention_mean", "mean"),
        global_peak_inside_mask=("global_peak_inside_mask", "mean"),
    )
    ranking_mode = "mask-aware" if ranking_df["global_topk_iou"].notna().any() else "stored-csv-fallback"

    def grouped_score(primary: str, fallback: str | None, higher: bool) -> pd.Series:
        source = ranking_df[primary].copy()
        if not source.notna().any() and fallback is not None:
            source = ranking_df[fallback].copy()
        source = source.fillna(source.median() if source.notna().any() else 0.0)
        result = pd.Series(index=source.index, dtype=float)
        for _, indices in ranking_df.groupby("experiment_id").groups.items():
            result.loc[indices] = minmax(source.loc[indices], higher)
        return result

    ranking_df["score_localization"] = grouped_score(
        "global_topk_iou", "global_peak_inside_mask", True
    )
    ranking_df["score_attention"] = grouped_score(
        "attention_mass_inside", "topk_anchor_attention_mean", True
    )
    ranking_df["score_separation"] = grouped_score(
        "attention_inside_outside_gap", "anchor_attention", True
    )
    ranking_df["score_stability"] = grouped_score(
        "temporal_anchor_std_px", None, False
    )
    ranking_df["score_geometry"] = grouped_score(
        "distance_to_bbox_center_norm", None, False
    )
    ranking_df["layer_score"] = (
        0.30 * ranking_df["score_localization"]
        + 0.25 * ranking_df["score_attention"]
        + 0.20 * ranking_df["score_separation"]
        + 0.15 * ranking_df["score_stability"]
        + 0.10 * ranking_df["score_geometry"]
    )
    ranking_df = ranking_df.sort_values(
        ["experiment_id", "layer_score"], ascending=[True, False]
    ).reset_index(drop=True)

    evaluation_csv = output_root / "per_layer_anchor_evaluation.csv"
    ranking_csv = output_root / "per_layer_anchor_ranking.csv"
    evaluation_jsonl = output_root / "per_layer_anchor_evaluation.jsonl"
    evaluation_df.to_csv(evaluation_csv, index=False, encoding="utf-8-sig")
    ranking_df.to_csv(ranking_csv, index=False, encoding="utf-8-sig")
    with evaluation_jsonl.open("w", encoding="utf-8") as handle:
        for row in evaluation_df.to_dict(orient="records"):
            handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
    (output_root / "config.json").write_text(
        json.dumps(
            {
                "export_root": str(export_root),
                "sample_count": args.sample_count,
                "topk_percent": args.topk_percent,
                "generation_rerun": False,
                "ranking_mode": ranking_mode,
                "score_weights": {
                    "global_topk_iou": 0.30,
                    "attention_mass_inside": 0.25,
                    "attention_inside_outside_gap": 0.20,
                    "temporal_stability": 0.15,
                    "bbox_geometry": 0.10,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Evaluated layer maps: {len(evaluation_df)}")
    print(f"[OK] Evaluation CSV: {evaluation_csv}")
    print(f"[OK] Ranking CSV: {ranking_csv}")
    for experiment_id, group in ranking_df.groupby("experiment_id"):
        print(f"\n[{experiment_id}] top 5 layers")
        print(group.head(5)[["layer_index", "layer_name", "native_spatial_size", "layer_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
