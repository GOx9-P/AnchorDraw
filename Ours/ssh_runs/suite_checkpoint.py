"""Select a previous SSH suite and revalidate completed profiles before reuse.

This is an internal helper for the single Bash entrypoint. A completed run is
reused only when its own profile, shared generation/metric source, Python
environment, input data, seed, and requested GPU preset still match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from scientific_snapshot import current_shared_source, requested_gpu_mode, shared_source_snapshot


SSH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SSH_ROOT.parents[1]
INPUT_KEYS = ("manifest_sha256", "instances_sha256", "captions_sha256", "manifest_images_sha256")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def select_suite(output_root: Path, gpu_mode: str, coco_root: Path) -> str:
    """Choose the latest matching suite; ignore unrelated directories."""
    if not output_root.is_dir():
        return ""
    for suite in sorted(output_root.iterdir(), key=lambda path: path.name, reverse=True):
        if not suite.is_dir() or not (suite / "suite_failures.tsv").is_file():
            continue
        metadata_path = suite / "suite_metadata.json"
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if metadata.get("gpu_mode") != gpu_mode or metadata.get("coco_root") != str(coco_root):
                continue
        else:
            # Suites made before checkpointing have no metadata. Infer the
            # requested preset and COCO location from an actual successful run.
            completed = sorted(suite.glob("runner_logs/**/*.log.ok"))
            if not completed:
                continue
            sample = completed[0]
            relative = sample.relative_to(suite / "runner_logs")
            label, filename = relative.parts
            profile = filename.removesuffix(".log.ok")
            run_dir = (suite / profile if label == "diagnostic"
                       else suite / f"seed_{label}" / profile)
            if requested_gpu_mode(run_dir) != gpu_mode:
                continue
            report = run_dir / "input_verification.json"
            if not report.is_file():
                continue
            try:
                saved_coco = json.loads(report.read_text(encoding="utf-8")).get("coco_root")
            except (OSError, ValueError):
                continue
            if saved_coco != str(coco_root):
                continue
        return suite.name
    return ""


def write_context(path: Path, coco_root: Path, gpu_mode: str) -> None:
    from verify_inputs import verify_inputs

    families = {"sd15": "sem_sd15_lcm", "sdxl": "sem_sdxl_euler", "sd3": "sem_sd3_ffm"}
    reports = {family: verify_inputs(profile, coco_root) for family, profile in families.items()}
    dependencies = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    payload = {
        "gpu_mode": gpu_mode,
        "coco_root": str(coco_root),
        "shared_source": current_shared_source(REPO_ROOT),
        "dependencies_sha256": _sha256(dependencies),
        "inputs": {family: {key: report[key] for key in INPUT_KEYS} for family, report in reports.items()},
    }
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def check_completed(profile: str, kind: str, run_dir: Path, seed: str,
                    context_path: Path) -> None:
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if kind == "diagnostic":
        expected_marker = None
        profile_file = SSH_ROOT / "diagnostics" / f"{profile}.py"
    elif kind == "metric":
        expected_marker = None
        profile_file = SSH_ROOT / "metric_profiles" / f"{profile}.py"
    elif kind == "full1073":
        expected_marker = profile
        profile_file = SSH_ROOT / "profiles" / f"{profile}.py"
    else:
        raise RuntimeError(f"Unknown profile kind: {kind}")
    if not run_dir.is_dir() or not (run_dir / "verification.log").is_file():
        raise RuntimeError("Run directory or verification log is missing")
    if not (run_dir / "verification.log").stat().st_size:
        raise RuntimeError("Verification log is empty")
    verification = json.loads((run_dir / "verification.log").read_text(encoding="utf-8"))
    if verification.get("profile" if kind != "metric" else "tool") != profile:
        raise RuntimeError("Verification log belongs to another profile")
    if expected_marker and (run_dir / "benchmark_complete.txt").read_text(encoding="utf-8").strip() != expected_marker:
        raise RuntimeError("Benchmark completion marker is missing or wrong")
    if kind != "diagnostic" and (run_dir / "base_seed.txt").read_text(encoding="utf-8").strip() != seed:
        raise RuntimeError("Base seed changed")
    if requested_gpu_mode(run_dir) != context["gpu_mode"]:
        raise RuntimeError("Requested GPU preset changed or is not recorded")
    recorded_profile = (run_dir / "profile_sha256.txt").read_text(encoding="utf-8").split()[0]
    if recorded_profile != _sha256(profile_file.read_bytes()):
        raise RuntimeError("Profile source changed")
    stored_shared = shared_source_snapshot(run_dir / "code_sha256.txt")
    current_shared = tuple(tuple(item) for item in context["shared_source"])
    if stored_shared != current_shared:
        raise RuntimeError("Shared generation or metric source changed")
    if _sha256((run_dir / "pip_freeze.txt").read_bytes()) != context["dependencies_sha256"]:
        raise RuntimeError("Python environment changed")
    if kind != "metric":
        saved = json.loads((run_dir / "input_verification.json").read_text(encoding="utf-8"))
        family = "sd3" if "sd3" in profile else ("sdxl" if "sdxl" in profile else "sd15")
        if saved.get("coco_root") != context["coco_root"] or any(
            saved.get(key) != context["inputs"][family][key] for key in INPUT_KEYS
        ):
            raise RuntimeError("COCO inputs or manifest changed")
        if profile == "diag_sem_sd15_lcm_semanticfull" and saved.get("checked_images") != 5000:
            raise RuntimeError("Full COCO diagnostic did not check all 5000 images")
        if profile == "diag_sem_sd15_lcm_semanticfull":
            from verify_inputs import verify_inputs
            verify_inputs(profile, Path(context["coco_root"]))

    # Recheck artifacts on disk. The benchmark verifier omits the paired-run
    # gate here because the counterpart might still be queued for this suite.
    if kind == "full1073":
        from verify_run import verify_run
        verify_run(profile, run_dir, check_pair=False)
        if not profile.endswith("_sd3_ffm"):
            summaries = list(run_dir.rglob("metric_export_summary.json"))
            if len(summaries) != 1:
                raise RuntimeError("Missing or duplicate metric export summary")
            export = summaries[0].parent
            summary = json.loads(summaries[0].read_text(encoding="utf-8"))
            manifest = export / "metric_generated_manifest.jsonl"
            images = export / "generated_images"
            if (summary.get("experiment_id") != profile or summary.get("num_generated_images") != 1073
                    or not manifest.is_file() or not images.is_dir()
                    or sum(1 for line in manifest.open(encoding="utf-8") if line.strip()) != 1073
                    or sum(1 for _ in images.glob("*.png")) != 1073):
                raise RuntimeError("Incomplete metric export for verified benchmark")
    elif kind == "diagnostic":
        from verify_diagnostic import verify_diagnostic
        verify_diagnostic(profile, run_dir)
    else:
        from verify_metric import verify_metric
        verify_metric(profile, run_dir)


def check_resume(run_dir: Path, seed: str, context_path: Path) -> None:
    """Keep a partial SD3 run only if the original strict resume gate can pass."""
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if (run_dir / "base_seed.txt").read_text(encoding="utf-8").strip() != seed:
        raise RuntimeError("Partial run used another seed")
    if requested_gpu_mode(run_dir) != context["gpu_mode"]:
        raise RuntimeError("Partial run used another GPU preset")
    if _sha256((run_dir / "pip_freeze.txt").read_bytes()) != context["dependencies_sha256"]:
        raise RuntimeError("Partial run used another Python environment")
    profile_file = SSH_ROOT / "profiles" / f"{run_dir.name}.py"
    recorded_profile = (run_dir / "profile_sha256.txt").read_text(encoding="utf-8").split()[0]
    if recorded_profile != _sha256(profile_file.read_bytes()):
        raise RuntimeError("Partial run used another profile source")
    # The original Bash --resume gate compares the complete code snapshot.
    # Do not claim compatibility if a runner file was added, removed or edited.
    saved = {}
    for line in (run_dir / "code_sha256.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if not match:
            raise RuntimeError("Malformed partial-run code snapshot")
        saved[match.group(2)] = match.group(1)
    current = {}
    for root in (REPO_ROOT / "Baseline", REPO_ROOT / "Ours/src", SSH_ROOT):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh"}:
                current[str(path)] = _sha256(path.read_bytes())
    if saved != current:
        raise RuntimeError("Partial run predates a code change; strict SD3 resume cannot continue it")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    select = sub.add_parser("select")
    select.add_argument("output_root", type=Path)
    select.add_argument("gpu_mode")
    select.add_argument("coco_root", type=Path)
    context = sub.add_parser("context")
    context.add_argument("path", type=Path)
    context.add_argument("coco_root", type=Path)
    context.add_argument("gpu_mode")
    check = sub.add_parser("check")
    check.add_argument("profile")
    check.add_argument("kind", choices=("full1073", "diagnostic", "metric"))
    check.add_argument("run_dir", type=Path)
    check.add_argument("seed")
    check.add_argument("context_path", type=Path)
    resume = sub.add_parser("resume")
    resume.add_argument("run_dir", type=Path)
    resume.add_argument("seed")
    resume.add_argument("context_path", type=Path)
    args = parser.parse_args()
    if args.action == "select":
        print(select_suite(args.output_root, args.gpu_mode, args.coco_root))
    elif args.action == "context":
        write_context(args.path, args.coco_root, args.gpu_mode)
    elif args.action == "check":
        try:
            check_completed(args.profile, args.kind, args.run_dir, args.seed, args.context_path)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"[CHECKPOINT] {args.profile}: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
    else:
        try:
            check_resume(args.run_dir, args.seed, args.context_path)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"[CHECKPOINT] {args.run_dir.name}: {exc}", file=sys.stderr)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
