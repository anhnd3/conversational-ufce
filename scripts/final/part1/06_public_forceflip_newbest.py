#!/usr/bin/env python3
"""
Public UFCE force-flip vs new-best hyper-tuned evidence pack.

This wrapper delegates all UFCE execution to 01b_reproduce_ufce_only.py,
then builds direction-aware comparison tables:

- published Table 7 values
- public/final-freeze with force_flip=1
- new_best_params with domain_actionable constraints
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RUNNER_01B = ROOT / "scripts" / "final" / "part1" / "01b_reproduce_ufce_only.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS = ["UFCE1", "UFCE2", "UFCE3"]
METRICS = ["Prox-Jac", "Prox-Euc", "Sparsity", "Actionability", "Plausibility", "Feasibility"]
ALL_DATASETS = ["bank", "bupa", "grad", "wine", "movie"]

LOWER_IS_BETTER = {"Prox-Jac", "Prox-Euc", "Sparsity"}
HIGHER_IS_BETTER = {"Actionability", "Plausibility", "Feasibility"}
METRIC_DIRECTIONS: Dict[str, str] = {
    **{metric: "lower_is_better" for metric in LOWER_IS_BETTER},
    **{metric: "higher_is_better" for metric in HIGHER_IS_BETTER},
}

DEFAULT_OUT_PARENT = ROOT / "outputs" / "final" / "part1"
EPS = 1e-9


def extract_01b_constant(name: str) -> Any:
    tree = ast.parse(RUNNER_01B.read_text(encoding="utf-8"), filename=str(RUNNER_01B))
    for node in tree.body:
        target_names: List[str] = []
        value = None
        if isinstance(node, ast.Assign):
            target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_names = [node.target.id]
            value = node.value
        if name in target_names and value is not None:
            expression = ast.Expression(value)
            ast.fix_missing_locations(expression)
            return eval(compile(expression, str(RUNNER_01B), "eval"), {"__builtins__": {}, "float": float}, {})
    raise KeyError(f"Constant not found in 01b runner: {name}")


def load_runner_constants() -> Dict[str, Any]:
    return {
        "AUTHOR_TABLE7": extract_01b_constant("AUTHOR_TABLE7"),
        "NEW_BEST_PARAMS": extract_01b_constant("NEW_BEST_PARAMS"),
    }


def local_run_id() -> str:
    return datetime.now().strftime("06_public_forceflip_newbest_%Y%m%d_%H%M%S")


def finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def raw_delta(old_value: Any, new_value: Any) -> Optional[float]:
    old_f = finite_float(old_value)
    new_f = finite_float(new_value)
    if old_f is None or new_f is None:
        return None
    return float(new_f - old_f)


def directional_delta(old_value: Any, new_value: Any, metric_direction: str) -> Optional[float]:
    old_f = finite_float(old_value)
    new_f = finite_float(new_value)
    if old_f is None or new_f is None:
        return None
    if metric_direction == "lower_is_better":
        return float(old_f - new_f)
    if metric_direction == "higher_is_better":
        return float(new_f - old_f)
    raise ValueError(f"Unsupported metric_direction: {metric_direction}")


def comparison_verdict(delta: Optional[float], *, positive: str, negative: str) -> str:
    if delta is None:
        return "na"
    if delta > EPS:
        return positive
    if delta < -EPS:
        return negative
    return "same"


def expand_datasets(value: str) -> List[str]:
    text = value.strip().lower()
    if text == "all":
        return list(ALL_DATASETS)
    datasets = [item.strip().lower() for item in text.split(",") if item.strip()]
    unknown = [item for item in datasets if item not in ALL_DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {', '.join(unknown)}")
    if not datasets:
        raise ValueError("At least one dataset is required.")
    return datasets


def render_command(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def tail(text: str, limit: int = 40) -> List[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()][-limit:]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_file(paths: Iterable[Path]) -> Optional[Path]:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def find_run_manifest(run_dir: Path) -> Optional[Path]:
    return latest_file(run_dir.glob("run_manifest_*.json"))


def find_summary_csv(run_dir: Path, dataset: str) -> Optional[Path]:
    return latest_file(run_dir.glob(f"summary_{dataset}_*.csv"))


def load_summary_records(run_dir: Path, datasets: Sequence[str], label: str) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    for dataset in datasets:
        summary_path = find_summary_csv(run_dir, dataset)
        if summary_path is None:
            warnings.append(f"{label}: missing summary CSV for dataset={dataset}")
            continue
        df = pd.read_csv(summary_path, index_col=0)
        df.index = [str(item).strip() for item in df.index]
        for method in METHODS:
            for metric in METRICS:
                value = None
                if metric in df.index and method in df.columns:
                    value = finite_float(df.loc[metric, method])
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "source": label,
                        "summary_csv": str(summary_path),
                    }
                )
    return pd.DataFrame(rows), warnings


def load_published_records(author_table7: Dict[str, Dict[str, Dict[str, float]]], datasets: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for method in METHODS:
            for metric in METRICS:
                value = author_table7.get(dataset, {}).get(method, {}).get(metric)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "value": finite_float(value),
                        "source": "published",
                        "summary_csv": "",
                    }
                )
    return pd.DataFrame(rows)


def value_lookup(df: pd.DataFrame, source_column: str) -> Dict[Tuple[str, str, str], Optional[float]]:
    out: Dict[Tuple[str, str, str], Optional[float]] = {}
    if df.empty:
        return out
    for row in df.to_dict("records"):
        key = (str(row["dataset"]), str(row["method"]), str(row["metric"]))
        out[key] = finite_float(row[source_column])
    return out


def build_final_comparison(
    *,
    published_df: pd.DataFrame,
    public_raw_df: pd.DataFrame,
    public_forceflip_df: pd.DataFrame,
    new_best_df: pd.DataFrame,
    datasets: Sequence[str],
) -> pd.DataFrame:
    published = value_lookup(published_df, "value")
    public_raw = value_lookup(public_raw_df, "value")
    public_forceflip = value_lookup(public_forceflip_df, "value")
    new_best = value_lookup(new_best_df, "value")

    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for method in METHODS:
            for metric in METRICS:
                metric_direction = METRIC_DIRECTIONS[metric]
                key = (dataset, method, metric)
                published_value = published.get(key)
                public_raw_value = public_raw.get(key)
                public_forceflip_value = public_forceflip.get(key)
                new_best_value = new_best.get(key)

                ff_raw_delta = raw_delta(published_value, public_forceflip_value)
                ff_directional_delta = directional_delta(
                    published_value,
                    public_forceflip_value,
                    metric_direction,
                )
                nb_raw_delta = raw_delta(public_forceflip_value, new_best_value)
                nb_directional_delta = directional_delta(
                    public_forceflip_value,
                    new_best_value,
                    metric_direction,
                )

                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "metric_direction": metric_direction,
                        "published_value": published_value,
                        "public_raw_value": public_raw_value,
                        "public_forceflip_value": public_forceflip_value,
                        "new_best_value": new_best_value,
                        "forceflip_vs_published_raw_delta": ff_raw_delta,
                        "forceflip_vs_published_directional_delta": ff_directional_delta,
                        "forceflip_vs_published_verdict": comparison_verdict(
                            ff_directional_delta,
                            positive="better",
                            negative="worse",
                        ),
                        "new_best_vs_forceflip_raw_delta": nb_raw_delta,
                        "new_best_vs_forceflip_directional_delta": nb_directional_delta,
                        "new_best_vs_forceflip_verdict": comparison_verdict(
                            nb_directional_delta,
                            positive="improved",
                            negative="regressed",
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_pairwise_tables(final_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    published_vs_forceflip = final_df[
        [
            "dataset",
            "method",
            "metric",
            "metric_direction",
            "published_value",
            "public_forceflip_value",
            "forceflip_vs_published_raw_delta",
            "forceflip_vs_published_directional_delta",
            "forceflip_vs_published_verdict",
        ]
    ].copy()
    forceflip_vs_new_best = final_df[
        [
            "dataset",
            "method",
            "metric",
            "metric_direction",
            "public_forceflip_value",
            "new_best_value",
            "new_best_vs_forceflip_raw_delta",
            "new_best_vs_forceflip_directional_delta",
            "new_best_vs_forceflip_verdict",
        ]
    ].copy()
    return published_vs_forceflip, forceflip_vs_new_best


def verdict_counts(df: pd.DataFrame, column: str) -> Dict[str, int]:
    counts = df[column].value_counts(dropna=False).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def metric_breakdown(df: pd.DataFrame, column: str) -> Dict[str, Dict[str, int]]:
    breakdown: Dict[str, Dict[str, int]] = {}
    for metric, group in df.groupby("metric"):
        breakdown[str(metric)] = verdict_counts(group, column)
    return breakdown


def rows_for_verdict(df: pd.DataFrame, column: str, verdict: str, limit: int = 20) -> List[Dict[str, object]]:
    rows = df[df[column] == verdict].head(limit)
    keep_cols = [
        "dataset",
        "method",
        "metric",
        "metric_direction",
        "published_value",
        "public_forceflip_value",
        "new_best_value",
        "forceflip_vs_published_directional_delta",
        "new_best_vs_forceflip_directional_delta",
    ]
    return rows[[col for col in keep_cols if col in rows.columns]].to_dict("records")


def build_claim_check(final_df: pd.DataFrame, child_runs: Sequence[Dict[str, Any]], manifest_validation: Dict[str, Any]) -> Dict[str, Any]:
    force_counts = verdict_counts(final_df, "forceflip_vs_published_verdict")
    new_best_counts = verdict_counts(final_df, "new_best_vs_forceflip_verdict")
    comparable_force = int(len(final_df[final_df["forceflip_vs_published_verdict"] != "na"]))
    comparable_new_best = int(len(final_df[final_df["new_best_vs_forceflip_verdict"] != "na"]))
    higher_df = final_df[final_df["metric_direction"] == "higher_is_better"].copy()

    return {
        "public_forceflip_degrades": {
            "passed": int(force_counts.get("worse", 0)) > 0,
            "basis": "directional_delta; worse means force_flip=1 moved opposite to metric direction relative to published",
            "comparable_rows": comparable_force,
            "verdict_counts": force_counts,
            "metric_breakdown": metric_breakdown(final_df, "forceflip_vs_published_verdict"),
            "higher_is_better_worse_rows": int((higher_df["forceflip_vs_published_verdict"] == "worse").sum()),
            "sample_worse_rows": rows_for_verdict(final_df, "forceflip_vs_published_verdict", "worse"),
        },
        "new_best_improves": {
            "passed": int(new_best_counts.get("improved", 0)) > 0,
            "basis": "directional_delta; improved means new_best moved in the beneficial metric direction relative to public force_flip=1",
            "comparable_rows": comparable_new_best,
            "verdict_counts": new_best_counts,
            "metric_breakdown": metric_breakdown(final_df, "new_best_vs_forceflip_verdict"),
            "sample_improved_rows": rows_for_verdict(final_df, "new_best_vs_forceflip_verdict", "improved"),
            "sample_regressed_rows": rows_for_verdict(final_df, "new_best_vs_forceflip_verdict", "regressed"),
        },
        "manifest_validation": manifest_validation,
        "selected_candidate_flip_validation": {
            "passed": bool(
                manifest_validation.get("run_b_public_forceflip", {}).get("passed")
                and manifest_validation.get("run_c_new_best", {}).get("passed")
            ),
            "basis": (
                "01b applies the UFCE flip filter before strict metric evaluation; "
                "this wrapper verifies effective_ufce_flip_filter=1 in Run B and Run C manifests."
            ),
        },
        "child_runs_passed": all(bool(run.get("passed")) for run in child_runs),
        "child_runs": child_runs,
    }


def build_child_command(
    *,
    dataset_arg: str,
    run_dir: Path,
    runtime_profile: str,
    bundle_mode: str,
    no_cf: int,
    max_folds: int,
    contprox_metric: str,
    fold_file: Optional[str],
    ufce_flip_filter: Optional[int],
) -> List[str]:
    command = [
        sys.executable,
        str(RUNNER_01B),
        "--dataset",
        dataset_arg,
        "--runtime_profile",
        runtime_profile,
        "--bundle_mode",
        bundle_mode,
        "--out_dir",
        str(run_dir),
        "--no_cf",
        str(no_cf),
        "--max_folds",
        str(max_folds),
        "--contprox_metric",
        contprox_metric,
    ]
    if fold_file:
        command.extend(["--fold_file", fold_file])
    if ufce_flip_filter is not None:
        command.extend(["--ufce_flip_filter", str(ufce_flip_filter)])
    return command


def run_child(command: Sequence[str], run_dir: Path, log_prefix: str) -> Dict[str, Any]:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path = logs_dir / f"{log_prefix}.stdout.log"
    stderr_path = logs_dir / f"{log_prefix}.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
    return {
        "name": log_prefix,
        "command": render_command(command),
        "exit_code": int(completed.returncode),
        "passed": completed.returncode == 0,
        "runtime_sec": float(time.time() - started),
        "run_dir": str(run_dir),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_tail": tail(completed.stdout),
        "stderr_tail": tail(completed.stderr),
    }


def run_or_reuse_child(
    *,
    name: str,
    run_dir: Path,
    reuse_dir: Optional[Path],
    command: Sequence[str],
) -> Dict[str, Any]:
    if reuse_dir is not None:
        return {
            "name": name,
            "command": "reuse-existing-run",
            "exit_code": 0,
            "passed": True,
            "runtime_sec": 0.0,
            "run_dir": str(reuse_dir),
            "stdout_path": "",
            "stderr_path": "",
            "stdout_tail": [],
            "stderr_tail": [],
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_child(command, run_dir, name)


def get_effective_dataset_cfg(manifest: Dict[str, Any], dataset: str) -> Dict[str, Any]:
    configs = manifest.get("effective_config_by_dataset", {})
    cfg = configs.get(dataset)
    if isinstance(cfg, dict):
        return cfg
    return {}


def validate_manifest_profile(
    *,
    manifest_path: Optional[Path],
    expected_runtime_profile: str,
    expected_bundle_mode: str,
    expected_flip: int,
    datasets: Sequence[str],
    expected_params: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict[str, Any]:
    errors: List[str] = []
    if manifest_path is None:
        return {"passed": False, "errors": ["missing run_manifest"], "manifest_path": None}
    manifest = read_json(manifest_path)
    if manifest.get("runtime_profile") != expected_runtime_profile:
        errors.append(
            f"runtime_profile expected={expected_runtime_profile} actual={manifest.get('runtime_profile')}"
        )
    if manifest.get("effective_bundle_mode") != expected_bundle_mode:
        errors.append(
            f"effective_bundle_mode expected={expected_bundle_mode} actual={manifest.get('effective_bundle_mode')}"
        )
    for dataset in datasets:
        cfg = get_effective_dataset_cfg(manifest, dataset)
        if int(cfg.get("effective_ufce_flip_filter", -1)) != int(expected_flip):
            errors.append(f"{dataset}: effective_ufce_flip_filter expected={expected_flip} actual={cfg.get('effective_ufce_flip_filter')}")
        if expected_params is not None:
            expected = expected_params.get(dataset, {})
            field_map = {
                "radius": "effective_radius",
                "n_neighbors": "effective_n_neighbors",
                "min_act": "effective_min_act",
                "min_feas": "effective_min_feas",
                "ufce_flip_filter": "effective_ufce_flip_filter",
            }
            for expected_key, actual_key in field_map.items():
                if expected_key not in expected:
                    continue
                if int(cfg.get(actual_key, -999999)) != int(expected[expected_key]):
                    errors.append(
                        f"{dataset}: {actual_key} expected={expected[expected_key]} actual={cfg.get(actual_key)}"
                    )
    return {"passed": not errors, "errors": errors, "manifest_path": str(manifest_path)}


def build_manifest_validation(
    *,
    run_a_dir: Path,
    run_b_dir: Path,
    run_c_dir: Path,
    datasets: Sequence[str],
    new_best_params: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    run_a = validate_manifest_profile(
        manifest_path=find_run_manifest(run_a_dir),
        expected_runtime_profile="final_freeze",
        expected_bundle_mode="table7_author_public",
        expected_flip=0,
        datasets=datasets,
    )
    run_b = validate_manifest_profile(
        manifest_path=find_run_manifest(run_b_dir),
        expected_runtime_profile="final_freeze",
        expected_bundle_mode="table7_author_public",
        expected_flip=1,
        datasets=datasets,
    )
    run_c = validate_manifest_profile(
        manifest_path=find_run_manifest(run_c_dir),
        expected_runtime_profile="new_best_params",
        expected_bundle_mode="domain_actionable",
        expected_flip=1,
        datasets=datasets,
        expected_params=new_best_params,
    )
    return {
        "passed": bool(run_a["passed"] and run_b["passed"] and run_c["passed"]),
        "run_a_public_raw": run_a,
        "run_b_public_forceflip": run_b,
        "run_c_new_best": run_c,
    }


def format_count_dict(counts: Dict[str, int]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"


def render_markdown_summary(
    *,
    output_root: Path,
    child_runs: Sequence[Dict[str, Any]],
    claim_check: Dict[str, Any],
    final_df: pd.DataFrame,
    warnings: Sequence[str],
) -> str:
    force_counts = claim_check["public_forceflip_degrades"]["verdict_counts"]
    new_counts = claim_check["new_best_improves"]["verdict_counts"]
    lines = [
        "# Public UFCE Force-Flip vs New-Best Hyper-Tuned Evidence",
        "",
        "## Status",
        f"- child_runs_passed: `{claim_check['child_runs_passed']}`",
        f"- manifest_validation_passed: `{claim_check['manifest_validation']['passed']}`",
        f"- selected_candidate_flip_validation_passed: `{claim_check['selected_candidate_flip_validation']['passed']}`",
        f"- output_root: `{output_root}`",
        "",
        "## Metric Direction Contract",
        "- `Prox-Jac`, `Prox-Euc`, `Sparsity`: lower is better.",
        "- `Actionability`, `Plausibility`, `Feasibility`: higher is better.",
        "- Directional delta > 0 means improvement in the metric's beneficial direction.",
        "",
        "## Claim Check",
        f"- public_forceflip_degrades: `{claim_check['public_forceflip_degrades']['passed']}` ({format_count_dict(force_counts)})",
        f"- new_best_improves: `{claim_check['new_best_improves']['passed']}` ({format_count_dict(new_counts)})",
        "",
        "## Child Runs",
    ]
    for run in child_runs:
        lines.append(
            f"- `{run['name']}`: passed=`{run['passed']}`, exit_code=`{run['exit_code']}`, run_dir=`{run['run_dir']}`"
        )
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Direction-Aware Comparison Preview",
            "",
            "| dataset | method | metric | direction | published | force_flip | new_best | force_flip verdict | new_best verdict |",
            "|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    preview = final_df.head(60)
    for row in preview.to_dict("records"):
        lines.append(
            "| {dataset} | {method} | {metric} | {metric_direction} | {published_value} | {public_forceflip_value} | {new_best_value} | {forceflip_vs_published_verdict} | {new_best_vs_forceflip_verdict} |".format(
                **{key: "" if value is None else value for key, value in row.items()}
            )
        )
    if len(final_df) > len(preview):
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | ... | ... |")
    lines.extend(
        [
            "",
            "Full comparison is in `final_metric_direction_comparison.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build public force-flip vs new-best UFCE evidence pack.")
    parser.add_argument("--dataset", default="all", help="Dataset name, comma list, or all.")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="Exact output folder. Defaults to outputs/final/part1/<run_id>.")
    parser.add_argument("--no_cf", type=int, default=10)
    parser.add_argument("--max_folds", type=int, default=0)
    parser.add_argument("--fold_file", default=None)
    parser.add_argument("--contprox_metric", default="euclidean")
    parser.add_argument("--reuse-run-a", dest="reuse_run_a", default=None, help="Existing run_a_public_raw directory.")
    parser.add_argument("--reuse-run-b", dest="reuse_run_b", default=None, help="Existing run_b_public_forceflip directory.")
    parser.add_argument("--reuse-run-c", dest="reuse_run_c", default=None, help="Existing run_c_new_best directory.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    runner_constants = load_runner_constants()
    datasets = expand_datasets(args.dataset)

    output_root = Path(args.out_dir).resolve() if args.out_dir else (DEFAULT_OUT_PARENT / local_run_id()).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_a_dir = Path(args.reuse_run_a).resolve() if args.reuse_run_a else output_root / "run_a_public_raw"
    run_b_dir = Path(args.reuse_run_b).resolve() if args.reuse_run_b else output_root / "run_b_public_forceflip"
    run_c_dir = Path(args.reuse_run_c).resolve() if args.reuse_run_c else output_root / "run_c_new_best"

    dataset_arg = args.dataset if len(datasets) == 1 or args.dataset.strip().lower() == "all" else "all"
    child_specs = [
        (
            "run_a_public_raw",
            run_a_dir,
            Path(args.reuse_run_a).resolve() if args.reuse_run_a else None,
            build_child_command(
                dataset_arg=dataset_arg,
                run_dir=run_a_dir,
                runtime_profile="final_freeze",
                bundle_mode="table7_author_public",
                no_cf=args.no_cf,
                max_folds=args.max_folds,
                contprox_metric=args.contprox_metric,
                fold_file=args.fold_file,
                ufce_flip_filter=0,
            ),
        ),
        (
            "run_b_public_forceflip",
            run_b_dir,
            Path(args.reuse_run_b).resolve() if args.reuse_run_b else None,
            build_child_command(
                dataset_arg=dataset_arg,
                run_dir=run_b_dir,
                runtime_profile="final_freeze",
                bundle_mode="table7_author_public",
                no_cf=args.no_cf,
                max_folds=args.max_folds,
                contprox_metric=args.contprox_metric,
                fold_file=args.fold_file,
                ufce_flip_filter=1,
            ),
        ),
        (
            "run_c_new_best",
            run_c_dir,
            Path(args.reuse_run_c).resolve() if args.reuse_run_c else None,
            build_child_command(
                dataset_arg=dataset_arg,
                run_dir=run_c_dir,
                runtime_profile="new_best_params",
                bundle_mode="domain_actionable",
                no_cf=args.no_cf,
                max_folds=args.max_folds,
                contprox_metric=args.contprox_metric,
                fold_file=args.fold_file,
                ufce_flip_filter=None,
            ),
        ),
    ]

    child_runs = [
        run_or_reuse_child(name=name, run_dir=run_dir, reuse_dir=reuse_dir, command=command)
        for name, run_dir, reuse_dir, command in child_specs
    ]

    warnings: List[str] = []
    public_raw_df, run_warnings = load_summary_records(run_a_dir, datasets, "public_raw")
    warnings.extend(run_warnings)
    public_forceflip_df, run_warnings = load_summary_records(run_b_dir, datasets, "public_forceflip")
    warnings.extend(run_warnings)
    new_best_df, run_warnings = load_summary_records(run_c_dir, datasets, "new_best")
    warnings.extend(run_warnings)
    published_df = load_published_records(runner_constants["AUTHOR_TABLE7"], datasets)

    final_df = build_final_comparison(
        published_df=published_df,
        public_raw_df=public_raw_df,
        public_forceflip_df=public_forceflip_df,
        new_best_df=new_best_df,
        datasets=datasets,
    )
    published_vs_forceflip_df, forceflip_vs_new_best_df = build_pairwise_tables(final_df)

    manifest_validation = build_manifest_validation(
        run_a_dir=run_a_dir,
        run_b_dir=run_b_dir,
        run_c_dir=run_c_dir,
        datasets=datasets,
        new_best_params=runner_constants["NEW_BEST_PARAMS"],
    )
    claim_check = build_claim_check(final_df, child_runs, manifest_validation)

    published_vs_forceflip_df.to_csv(output_root / "published_vs_public_forceflip.csv", index=False)
    forceflip_vs_new_best_df.to_csv(output_root / "public_forceflip_vs_new_best.csv", index=False)
    final_df.to_csv(output_root / "final_metric_direction_comparison.csv", index=False)
    write_json(output_root / "claim_check.json", claim_check)

    summary_md = render_markdown_summary(
        output_root=output_root,
        child_runs=child_runs,
        claim_check=claim_check,
        final_df=final_df,
        warnings=warnings,
    )
    (output_root / "summary.md").write_text(summary_md, encoding="utf-8")

    run_summary = {
        "status": "passed" if claim_check["child_runs_passed"] and manifest_validation["passed"] else "failed",
        "output_root": str(output_root),
        "comparison_csv": str(output_root / "final_metric_direction_comparison.csv"),
        "claim_check_json": str(output_root / "claim_check.json"),
        "summary_md": str(output_root / "summary.md"),
    }
    write_json(output_root / "summary.json", run_summary)
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))
    return 0 if run_summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
