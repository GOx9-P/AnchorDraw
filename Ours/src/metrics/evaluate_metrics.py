from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .config import MetricEvaluationConfig
    from .evaluator import run_evaluation
    from .reporting import write_metrics_report
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metrics.config import MetricEvaluationConfig
    from metrics.evaluator import run_evaluation
    from metrics.reporting import write_metrics_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated SemanticDraw/AnchorDraw images with FID, IS, CLIP(fg), CLIP(bg), and Time(s)."
    )
    parser.add_argument("--manifest-path", required=True, help="Path to the JSONL manifest used for generation.")
    parser.add_argument("--coco-root", required=True, help="COCO root containing val2017/ and annotations/.")
    parser.add_argument("--generated-dir", required=True, help="Directory containing *_generated.png outputs.")
    parser.add_argument("--generation-summary", default=None, help="Optional generation_summary.json path.")
    parser.add_argument("--output-dir", default="Ours/eval_outputs", help="Directory for metrics JSON/CSV reports.")
    parser.add_argument("--report-prefix", default="metrics", help="Output file prefix.")

    parser.add_argument("--model-family", default="sd15", choices=["sd15", "sdxl", "sd3"])
    parser.add_argument("--target-size", default=None, help="Optional HxW override, for example 512x512 or 1024x1024.")
    parser.add_argument("--metrics", default="fid,is,clip_fg,clip_bg,time")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--clip-model-name", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--fid-feature", type=int, default=2048)
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--allow-missing-generated", action="store_true")
    parser.add_argument("--no-mask-foreground-for-clip", action="store_true")
    parser.add_argument("--no-mask-background-for-clip", action="store_true")
    return parser.parse_args(argv)


def parse_target_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid target size: {value}. Use HxW, for example 512x512.")
    return (int(parts[0]), int(parts[1]))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = tuple(metric.strip() for metric in args.metrics.split(",") if metric.strip())

    config = MetricEvaluationConfig(
        manifest_path=args.manifest_path,
        coco_root=args.coco_root,
        generated_dir=args.generated_dir,
        generation_summary=args.generation_summary,
        output_dir=args.output_dir,
        model_family=args.model_family,
        target_size=parse_target_size(args.target_size),
        metrics=metrics,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        clip_model_name=args.clip_model_name,
        clip_pretrained=args.clip_pretrained,
        clip_batch_size=args.clip_batch_size,
        fid_feature=args.fid_feature,
        is_splits=args.is_splits,
        error_on_missing_generated=not args.allow_missing_generated,
        mask_foreground_for_clip=not args.no_mask_foreground_for_clip,
        mask_background_for_clip=not args.no_mask_background_for_clip,
    )

    report = run_evaluation(config)
    json_path, csv_path = write_metrics_report(report, Path(args.output_dir), prefix=args.report_prefix)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"[OK] Metrics JSON: {json_path}")
    print(f"[OK] Metrics CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
