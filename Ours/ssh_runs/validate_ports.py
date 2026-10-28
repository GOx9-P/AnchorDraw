"""Static regression check for every standalone benchmark profile."""

from __future__ import annotations

import ast
import builtins
import symtable

from export_notebooks import DIAGNOSTIC_OUTPUT, DIAGNOSTIC_PORTS, METRIC_OUTPUT, METRIC_PORTS, OURS, OUTPUT, PORTS, render_one


def main() -> None:
    baseline_source = OURS.parent / "Baseline/semantic-draw-main/src/model/pipeline_semantic_draw.py"
    baseline_text = baseline_source.read_text(encoding="utf-8")
    foreground_background_count_fix = (
        "num_masks = len(fg_masks)\n"
        "                num_prompts = len(prompts)\n"
        "                num_nprompts = len(negative_prompts)"
    )
    if (baseline_text.count(foreground_background_count_fix) != 1
            or baseline_text.index("has_background = False # Regard that background does not exist.")
            > baseline_text.index(foreground_background_count_fix)):
        raise RuntimeError("SemanticDraw SD1.5 background region counts are not patched")
    pair_parameters = {
        ("sem_sd15_lcm", "ours_sd15_lcm"): (
            "MODEL_ID", "BASE_SEED", "BOOTSTRAP_STEPS", "MASK_STD", "MASK_STRENGTH",
            "PREPROCESS_MASK_COVER_ALPHA", "MASK_TYPE", "NEGATIVE_PROMPT", "METRIC_NAMES", "IS_SPLITS"),
        ("sem_sd15_hypersd", "ours_sd15_hypersd"): (
            "MODEL_ID", "BASE_SEED", "BOOTSTRAP_STEPS", "MASK_STD", "MASK_STRENGTH",
            "PREPROCESS_MASK_COVER_ALPHA", "MASK_TYPE", "NEGATIVE_PROMPT", "METRIC_NAMES", "IS_SPLITS",
            "HYPER_SD_REPO_ID", "HYPER_SD_WEIGHT_NAME", "HYPER_SD_LORA_SCALE",
            "HYPER_SD_NUM_INFERENCE_STEPS", "SEMANTICDRAW_GUIDANCE_SCALE"),
        ("ours_sd15_lcm", "ours_sd15_lcm_dpp"): (
            "MODEL_ID", "BASE_SEED", "BOOTSTRAP_STEPS", "MASK_STD", "MASK_STRENGTH",
            "PREPROCESS_MASK_COVER_ALPHA", "MASK_TYPE", "NEGATIVE_PROMPT", "METRIC_NAMES", "IS_SPLITS"),
        ("ours_sd15_hypersd", "ours_sd15_hypersd_dpp"): (
            "MODEL_ID", "BASE_SEED", "BOOTSTRAP_STEPS", "MASK_STD", "MASK_STRENGTH",
            "PREPROCESS_MASK_COVER_ALPHA", "MASK_TYPE", "NEGATIVE_PROMPT", "METRIC_NAMES", "IS_SPLITS",
            "HYPER_SD_REPO_ID", "HYPER_SD_WEIGHT_NAME", "HYPER_SD_LORA_SCALE",
            "HYPER_SD_NUM_INFERENCE_STEPS", "SEMANTICDRAW_GUIDANCE_SCALE"),
        ("ours_sd15_hypersd", "ours_sd15_hypersd_pladis"): (
            "MODEL_ID", "BASE_SEED", "BOOTSTRAP_STEPS", "MASK_STD", "MASK_STRENGTH",
            "PREPROCESS_MASK_COVER_ALPHA", "MASK_TYPE", "NEGATIVE_PROMPT", "METRIC_NAMES", "IS_SPLITS",
            "HYPER_SD_REPO_ID", "HYPER_SD_WEIGHT_NAME", "HYPER_SD_LORA_SCALE",
            "HYPER_SD_NUM_INFERENCE_STEPS", "SEMANTICDRAW_GUIDANCE_SCALE"),
        ("sem_sdxl_euler", "ours_sdxl_euler"): (
            "BASE_SEED", "BOOTSTRAP_STEPS", "MASK_STD", "MASK_STRENGTH",
            "PREPROCESS_MASK_COVER_ALPHA", "NEGATIVE_PROMPT", "METRIC_NAMES", "IS_SPLITS"),
    }
    for (baseline, proposed), names in pair_parameters.items():
        base_values = literal_constants(OUTPUT / f"{baseline}.py")
        proposed_values = literal_constants(OUTPUT / f"{proposed}.py")
        for name in names:
            if name not in base_values or name not in proposed_values or base_values[name] != proposed_values[name]:
                raise RuntimeError(
                    f"Matched-pair parameter differs: {baseline}/{proposed} {name}: "
                    f"{base_values.get(name, '<missing>')!r} != {proposed_values.get(name, '<missing>')!r}"
                )
    sdxl_baseline = literal_constants(OUTPUT / "sem_sdxl_euler.py")
    sdxl_proposed = literal_constants(OUTPUT / "ours_sdxl_euler.py")
    for baseline_name, proposed_name in (("SEMANTICDRAW_T_INDEX_LIST", "T_INDEX_LIST"),
                                         ("EULER_GUIDANCE_SCALE", "GUIDANCE_SCALE")):
        if (baseline_name not in sdxl_baseline or proposed_name not in sdxl_proposed
                or sdxl_baseline[baseline_name] != sdxl_proposed[proposed_name]):
            raise RuntimeError(f"Matched SDXL schedule/guidance differs: {baseline_name} vs {proposed_name}")
    available = {path.relative_to(OURS).as_posix() for path in OURS.rglob("*.ipynb")
                 if "sd3" not in path.name.lower()}
    mapped = {notebook for notebook, _ in (*PORTS.values(), *DIAGNOSTIC_PORTS.values(), *METRIC_PORTS.values())
              if "sd3" not in notebook.lower()}
    if available != mapped:
        raise RuntimeError(f"Non-SD3 notebook coverage mismatch: missing={available-mapped}, stale={mapped-available}")
    for directory, inventory in ((OUTPUT, PORTS), (DIAGNOSTIC_OUTPUT, DIAGNOSTIC_PORTS),
                                 (METRIC_OUTPUT, METRIC_PORTS)):
        expected = {f"{name}.py" for name in inventory}
        actual = {path.name for path in directory.glob("*.py")}
        if actual != expected:
            raise RuntimeError(f"Profile inventory mismatch in {directory}: missing={expected-actual}, extra={actual-expected}")
        for name, (notebook, cells) in inventory.items():
            check_one(directory, name, notebook, cells, full1073=(directory == OUTPUT))


def literal_constants(path):
    values = {}
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, KeyError, SyntaxError, MemoryError, RecursionError):
            continue
        if isinstance(target, ast.Name):
            values[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)):
            if len(target.elts) == len(value):
                for name, item in zip(target.elts, value):
                    if isinstance(name, ast.Name):
                        values[name.id] = item
    return values


def check_terminal_cloud_imports(tree: ast.AST, path) -> None:
    """Reject cloud APIs that the server could reach without a safe guard."""
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    disabled_flags = {
        target.id for node in getattr(tree, "body", ())
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is False
        for target in node.targets if isinstance(target, ast.Name)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"get_ipython", "input"}:
            raise RuntimeError(f"Interactive notebook call remains in terminal profile: {path}:{node.lineno}")
        if not isinstance(node, ast.ImportFrom) or node.module not in {"google.colab", "kaggle_secrets"}:
            continue
        child = node
        protected = False
        while child in parents:
            ancestor = parents[child]
            if isinstance(ancestor, ast.Try) and child in ancestor.body:
                if any(isinstance(handler.type, ast.Name) and handler.type.id in {
                    "Exception", "ImportError", "ModuleNotFoundError"
                } for handler in ancestor.handlers):
                    protected = True
                    break
            if (isinstance(ancestor, ast.If) and child in ancestor.body
                    and isinstance(ancestor.test, ast.Name) and ancestor.test.id in disabled_flags):
                protected = True
                break
            child = ancestor
        if not protected:
            raise RuntimeError(f"Reachable cloud-only import in terminal profile: {path}:{node.lineno}")


def check_one(directory, name, notebook, cells, *, full1073):
        path = directory / f"{name}.py"
        text = path.read_text(encoding="utf-8")
        if text != render_one(name, notebook, cells):
            raise RuntimeError(f"Generated port is stale or hand-edited: {path}")
        if ('"-m", "pip", "install"' in text or "'-m', 'pip', 'install'" in text):
            raise RuntimeError(f"Profile installs packages during a run: {path}")
        tree = ast.parse(text, filename=str(path))
        check_terminal_cloud_imports(tree, path)
        display_shims = [node for node in tree.body
                         if isinstance(node, ast.FunctionDef) and node.name == "display"]
        if len(display_shims) != 1 or display_shims[0].args.vararg is None:
            raise RuntimeError(f"Terminal display shim cannot accept multiple notebook outputs: {path}")
        lora_cleanup = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "clear_existing_lora_adapters"), None
        )
        if lora_cleanup is not None and not isinstance(lora_cleanup.body[-1], ast.Raise):
            raise RuntimeError(f"LoRA cleanup may silently continue with LCM loaded: {path}")
        image_directories = {
            node.value.left.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "generated_path" for target in node.targets)
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Div)
            and isinstance(node.value.left, ast.Name)
        }
        metric_directories = {
            keyword.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "MetricEvaluationConfig"
            for keyword in node.keywords
            if keyword.arg == "generated_dir" and isinstance(keyword.value, ast.Name)
        }
        if metric_directories and image_directories and metric_directories != image_directories:
            raise RuntimeError(
                f"Metric/image folder mismatch in {path}: generated={image_directories}, metric={metric_directories}"
            )
        if name == "sem_sdxl_euler" and 'os.environ.get("COLAB_GPU_MODE", "high_vram_24gb") == "low_vram"' not in text:
            raise RuntimeError("SemanticDraw SDXL server VRAM mode is not tied to the Bash GPU mode")
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(("/content", "/kaggle/working")):
                raise RuntimeError(f"Cloud absolute path remains in executable code: {path}: {node.value}")
            if isinstance(node, ast.ImportFrom) and node.module == "IPython.display":
                raise RuntimeError(f"Notebook display import remains: {path}")
        table = symtable.symtable(text, str(path), "exec")
        defined = {symbol.get_name() for symbol in table.get_symbols()
                   if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}
        missing: set[str] = set()

        def check_scope(scope) -> None:
            for symbol in scope.get_symbols():
                candidate = symbol.get_name()
                if (symbol.is_referenced() and symbol.is_global() and candidate not in defined
                        and candidate not in dir(builtins) and candidate != "__file__"):
                    missing.add(candidate)
            for child in scope.get_children():
                check_scope(child)

        check_scope(table)
        if missing:
            raise RuntimeError(f"Unbound global names in {path}: {sorted(missing)}")
        if full1073 and ("generation_summary.json" not in text or "_generated.png" not in text):
            raise RuntimeError(f"Missing image or summary contract in {path}")
        if name == "diag_hypersd_dpp_isolation128":
            if "shutil.copy2(source_image, destination)" not in text or "path.resolve().is_relative_to(REAL_IMAGE_DIR.resolve())" not in text:
                raise RuntimeError("Isolation diagnostic must delete only copied, run-owned COCO images")
        print(f"PASS {name}: {len(cells)} source cells, no unresolved globals")


if __name__ == "__main__":
    main()
