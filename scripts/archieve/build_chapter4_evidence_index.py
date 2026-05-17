#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = Path("chapter4_evidence_index.csv")
PART2_RUN_ID = "part2_thesis_metrics_20260404_125616_618657"

RELEVANT_FILE_NAMES = {
    "part2_tier_b_bank_sessions_v1.json",
    "part2_session_outcomes_normalized.csv",
    "part2_failure_taxonomy.csv",
    "part2_conditional_quality_denominators.csv",
    "part2_core_evidence_summary.md",
    "part2_expected_case_distribution.csv",
    "part2_expected_cases_normalized.csv",
    "part2_trace_cases_selected.csv",
    "part2_trace_cases_selected.md",
    "thesis_metrics_report.json",
    "thesis_metrics_report.md",
    "final_table7_delta_vs_author_raw.csv",
    "final_table7_delta_vs_author_strict_validity.csv",
    "final_table7_raw_long.csv",
    "final_table7_strict_validity_long.csv",
    "part1_table7_main_long.csv",
    "part1_table7_dataset_bank.csv",
    "part1_table7_dataset_grad.csv",
    "part1_table7_dataset_wine.csv",
    "part1_table7_dataset_bupa.csv",
    "part1_table7_dataset_movie.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_relevant(path: Path) -> bool:
    if path.name in RELEVANT_FILE_NAMES:
        return True
    return path.name.startswith("part1_table7_") and path.suffix == ".csv"


def iter_relevant_files(root: Path) -> list[Path]:
    if not root.exists():
        print(f"WARNING: root does not exist, skipping: {root}")
        return []
    if root.is_file():
        return [root] if is_relevant(root) else []

    matches: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        dirnames[:] = [
            item
            for item in dirnames
            if item
            not in {
                ".git",
                ".venv",
                "__pycache__",
                "node_modules",
                ".pytest_cache",
            }
        ]
        for filename in filenames:
            path = current / filename
            if is_relevant(path):
                matches.append(path)
    return matches


def infer_artifact_group(path: Path) -> str:
    name = path.name
    text = str(path)
    if name == "part2_tier_b_bank_sessions_v1.json":
        return "part2_corpus"
    if name in {
        "part2_session_outcomes_normalized.csv",
        "part2_core_evidence_summary.md",
        "thesis_metrics_report.json",
        "thesis_metrics_report.md",
    }:
        return "part2_metrics_report"
    if name == "part2_failure_taxonomy.csv":
        return "part2_failure_taxonomy"
    if name == "part2_conditional_quality_denominators.csv":
        return "part2_conditional_quality"
    if name.startswith("part2_expected_"):
        return "part2_expected_distribution"
    if name.startswith("part2_trace_cases_selected"):
        return "part2_trace_cases"
    if name == "turn_result.json" and "isolated_product_artifacts" in text:
        return "part2_trace_source"
    if name.startswith("final_table7_") or name.startswith("part1_table7_"):
        return "part1_table7"
    return "chapter4_evidence"


def infer_purpose(path: Path) -> str:
    group = infer_artifact_group(path)
    return {
        "part2_corpus": "Frozen pre-run G1/G2 bank session corpus.",
        "part2_metrics_report": "Source for 200-session Part II outcome metrics.",
        "part2_failure_taxonomy": "Terminal outcome taxonomy for Part II.",
        "part2_conditional_quality": "Conditional G2 exposed-counterfactual quality denominators.",
        "part2_expected_distribution": "Expected pre-run design distribution for main 200 sessions.",
        "part2_trace_cases": "Selected trace case summaries for Section 4.4.",
        "part2_trace_source": "Terminal product trace source for selected Section 4.4 case.",
        "part1_table7": "Part I Table 7 reproduction and formatted per-dataset tables.",
    }.get(group, "")


def infer_section(path: Path) -> str:
    group = infer_artifact_group(path)
    return {
        "part2_corpus": "4.3",
        "part2_metrics_report": "4.3",
        "part2_failure_taxonomy": "4.3.3",
        "part2_conditional_quality": "4.3",
        "part2_expected_distribution": "4.3",
        "part2_trace_cases": "4.4",
        "part2_trace_source": "4.4",
        "part1_table7": "4.2.1",
    }.get(group, "")


def infer_run_id(path: Path) -> str:
    text = str(path)
    if PART2_RUN_ID in text:
        return PART2_RUN_ID
    if infer_artifact_group(path).startswith("part2_"):
        return PART2_RUN_ID
    return ""


def trace_source_files(trace_csv: Path) -> list[Path]:
    if not trace_csv.exists():
        return []
    try:
        df = pd.read_csv(trace_csv)
    except Exception as exc:
        print(f"WARNING: could not read trace CSV {trace_csv}: {exc}")
        return []
    if "source_file" not in df.columns:
        return []
    return [Path(value) for value in df["source_file"].dropna().astype(str).tolist() if value]


def row_for_path(path: Path) -> dict[str, str]:
    return {
        "artifact_group": infer_artifact_group(path),
        "path": str(path),
        "purpose": infer_purpose(path),
        "run_id": infer_run_id(path),
        "sha256": sha256_file(path),
        "used_in_section": infer_section(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Chapter 4 evidence artifact index with SHA-256 hashes.")
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths: dict[str, Path] = {}
    for root in args.root:
        for path in iter_relevant_files(root):
            paths[str(path)] = path

    for trace_csv_name in ("part2_trace_cases_selected.csv",):
        trace_csv = Path(trace_csv_name)
        for source in trace_source_files(trace_csv):
            if source.exists():
                paths[str(source)] = source

    rows = [row_for_path(path) for path in sorted(paths.values(), key=lambda item: str(item))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["artifact_group", "path", "purpose", "run_id", "sha256", "used_in_section"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"output = {args.output}")
    print(f"files = {len(rows)}")
    if rows:
        print(pd.DataFrame(rows).head(30).to_string(index=False))


if __name__ == "__main__":
    main()
