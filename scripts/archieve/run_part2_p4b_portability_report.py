#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.part2_eval.common import add_summary_output_args, build_runner_command, write_optional_summary_outputs
from llm.src.part2_eval.p4_portability import (
    P4B_PRIMARY_MATRIX,
    P4B_REFINEMENT_MATRIX,
    build_portability_summary,
    compute_enabled_combinations,
    default_benchmark_path,
    default_product_config,
    execute_primary_matrix,
    execute_refinement_matrix,
    parse_csv_list,
    render_portability_markdown,
    summarize_backend_conformance,
    summarize_dataset_conformance,
)
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "part2_p4b_portability"


def parse_args() -> argparse.Namespace:
    product_config = default_product_config()
    parser = argparse.ArgumentParser(description="Generate the P4b portability report.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark", type=Path, default=default_benchmark_path())
    parser.add_argument("--lm-studio-api-base", default=product_config.lm_studio_api_base)
    parser.add_argument("--model-alias", default=product_config.model_alias)
    parser.add_argument("--product-mode", default=product_config.product_mode)
    parser.add_argument("--api-version", default=product_config.api_version)
    parser.add_argument("--app-version", default=product_config.app_version)
    parser.add_argument("--datasets", default="bank,grad")
    parser.add_argument("--backends", default="ufce,dice,ar")
    add_summary_output_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_runner_command(Path(__file__).resolve())
    summary = run_p4b_portability_report(args=args, command=command)
    markdown = render_portability_markdown(summary)
    write_optional_summary_outputs(
        summary=summary,
        summary_json_path=args.summary_json,
        summary_markdown_path=args.summary_md,
        markdown_text=markdown,
    )
    if args.summary_json is None:
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    else:
        print(f"summary_json_path={Path(args.summary_json).resolve()}")
    return 0


def run_p4b_portability_report(*, args: argparse.Namespace, command: str) -> dict:
    datasets = parse_csv_list(args.datasets)
    backends = parse_csv_list(args.backends)
    run_root = Path(args.out_dir) / f"p4b_portability_{local_now_compact()}"
    run_root.mkdir(parents=True, exist_ok=False)

    dataset_conformance = summarize_dataset_conformance(datasets)
    backend_conformance = summarize_backend_conformance(backends)
    enabled_combinations = compute_enabled_combinations(datasets, backends)
    primary_results = execute_primary_matrix(
        run_root=run_root / "primary",
        dataset_ids=datasets,
        backend_ids=backends,
        benchmark_path=Path(args.benchmark),
        lm_studio_api_base=args.lm_studio_api_base,
        model_alias=args.model_alias,
        product_mode=args.product_mode,
        api_version=args.api_version,
        app_version=args.app_version,
    )
    refinement_results = execute_refinement_matrix(
        run_root=run_root / "refinement",
        dataset_ids=datasets,
        backend_ids=backends,
        benchmark_path=Path(args.benchmark),
        lm_studio_api_base=args.lm_studio_api_base,
        model_alias=args.model_alias,
        product_mode=args.product_mode,
        api_version=args.api_version,
        app_version=args.app_version,
    )
    summary = build_portability_summary(
        milestone="P4b",
        run_root=run_root,
        command=command,
        datasets=datasets,
        backends=backends,
        required_primary_matrix=list(P4B_PRIMARY_MATRIX),
        optional_primary_matrix=[],
        required_refinement_matrix=list(P4B_REFINEMENT_MATRIX),
        dataset_conformance=dataset_conformance,
        backend_conformance=backend_conformance,
        enabled_combinations=enabled_combinations,
        primary_results=primary_results,
        refinement_results=refinement_results,
    )
    write_json(run_root / "p4b_portability_report.json", summary)
    (run_root / "p4b_portability_report.md").write_text(
        render_portability_markdown(summary),
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
