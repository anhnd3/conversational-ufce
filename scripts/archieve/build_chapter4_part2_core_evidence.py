#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_REPORT = (
    ROOT
    / "outputs"
    / "part2_thesis_metrics"
    / "part2_thesis_metrics_20260404_125616_618657"
    / "thesis_metrics_report.json"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "chapter4_part2_evidence"

EXPECTED_TOTAL_SESSIONS = 200
EXPECTED_GROUP_COUNTS = {"G1": 100, "G2": 100}
EXPECTED_SUCCESSFUL_RESOLUTION = 59
EXPECTED_NON_SUCCESS = 141
EXPECTED_CONDITIONAL_DENOMINATOR = 28

SESSION_OUTCOME_COLUMNS = [
    "case_id",
    "session_id",
    "group",
    "final_public_state",
    "summary_type",
    "reject_class",
    "is_counterfactual_found",
    "is_no_recourse_needed",
    "is_successful_resolution",
    "primary_reason",
]
FAILURE_TAXONOMY_COLUMNS = [
    "primary_reason",
    "n_sessions",
    "pct_total",
    "pct_non_success",
    "definition",
]
CONDITIONAL_METRIC_COLUMNS = [
    "metric",
    "value",
    "n_pass",
    "n_denominator",
    "scope",
    "definition",
]

FAILURE_REASONS = [
    "parser_schema_failure",
    "unsupported",
    "conflict",
    "clarification_exhausted",
    "no_valid_cf",
    "constraint_blocked",
    "other_runtime_reject",
]

FAILURE_DEFINITIONS = {
    "parser_schema_failure": "Parser output did not pass schema or canonical validation after bounded repair.",
    "unsupported": "The request was outside the supported Bank counterfactual task.",
    "conflict": "The request or constraints were contradictory and could not be merged safely.",
    "clarification_exhausted": "The session reached the clarification limit before a complete valid request.",
    "no_valid_cf": "The backend did not find a valid counterfactual under the current request and constraints.",
    "constraint_blocked": "A hard request constraint blocked exposing a candidate recommendation.",
    "other_runtime_reject": "Runtime rejection not mapped to the standard thesis taxonomy.",
}

CONDITIONAL_METRICS = [
    (
        "validity",
        "M8_validity_success_rate",
        "G2 exposed counterfactuals that flipped to the desired label.",
    ),
    (
        "actionability",
        "M11_actionability",
        "G2 exposed counterfactuals whose changed fields are policy-actionable and obey active constraints.",
    ),
    (
        "plausibility",
        "M12_plausibility",
        "G2 exposed counterfactuals with passed invariant validation; this is not a density estimate.",
    ),
    (
        "feasibility",
        "M13_feasibility",
        "G2 exposed counterfactuals emitted after runtime execution and post-runtime validation.",
    ),
    (
        "constraint_satisfaction",
        "M14_constraint_satisfaction",
        "G2 exposed counterfactuals that satisfy blocked-field, cardinality, and numeric-bound constraints.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chapter 4 Part II core evidence CSV/Markdown files.")
    parser.add_argument("--source-report", type=Path, default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = load_report(args.source_report)
    outputs = build_evidence_outputs(report, source_report=args.source_report)
    validate_evidence(outputs)
    write_outputs(outputs, out_dir=args.out_dir)
    print(json.dumps(build_stdout_summary(outputs, args.out_dir), ensure_ascii=True, indent=2))
    return 0


def load_report(path: Path) -> dict[str, Any]:
    source_path = Path(path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source report not found: {source_path}")
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Source report is not valid JSON: {source_path}; {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Source report must be a JSON object: {source_path}")
    return payload


def build_evidence_outputs(report: dict[str, Any], *, source_report: Path) -> dict[str, Any]:
    validate_source_report(report)
    per_case_results = get_per_case_results(report)
    session_rows = [normalize_session_outcome(row) for row in per_case_results]
    failure_rows = build_failure_taxonomy(session_rows)
    conditional_rows = build_conditional_denominators(report)
    summary = build_markdown_summary(
        report=report,
        source_report=source_report,
        session_rows=session_rows,
        failure_rows=failure_rows,
        conditional_rows=conditional_rows,
    )
    return {
        "source_report": str(Path(source_report).resolve()),
        "session_rows": session_rows,
        "failure_rows": failure_rows,
        "conditional_rows": conditional_rows,
        "summary_markdown": summary,
    }


def validate_source_report(report: dict[str, Any]) -> None:
    aggregate_validation = report.get("aggregate_validation")
    if not isinstance(aggregate_validation, dict) or aggregate_validation.get("ok") is not True:
        raise ValueError("Source report aggregate_validation.ok must be True")
    if "per_case_results" not in report:
        raise ValueError("Source report is missing per_case_results")
    if "system_metrics" not in report:
        raise ValueError("Source report is missing system_metrics")
    if "g2_metrics" not in report:
        raise ValueError("Source report is missing g2_metrics")


def get_per_case_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("per_case_results")
    if not isinstance(rows, list):
        raise ValueError("per_case_results must be a list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Every per_case_results item must be an object")
    return rows


def normalize_session_outcome(row: dict[str, Any]) -> dict[str, Any]:
    summary_type = str(row.get("summary_type") or "")
    is_counterfactual_found = summary_type == "counterfactual_found"
    is_no_recourse_needed = summary_type == "no_recourse_needed"
    is_successful_resolution = is_counterfactual_found or is_no_recourse_needed
    if is_counterfactual_found:
        primary_reason = "counterfactual_found"
    elif is_no_recourse_needed:
        primary_reason = "no_recourse_needed"
    else:
        primary_reason = normalize_failure_reason(row)
    return {
        "case_id": row.get("case_id") or "",
        "session_id": row.get("session_id") or "",
        "group": row.get("group") or "",
        "final_public_state": row.get("final_public_state") or "",
        "summary_type": summary_type,
        "reject_class": row.get("reject_class") or "",
        "is_counterfactual_found": is_counterfactual_found,
        "is_no_recourse_needed": is_no_recourse_needed,
        "is_successful_resolution": is_successful_resolution,
        "primary_reason": primary_reason,
    }


def normalize_failure_reason(row: dict[str, Any]) -> str:
    reject_class = str(row.get("reject_class") or "")
    case_completion_reason = str(row.get("case_completion_reason") or "")
    final_public_state = str(row.get("final_public_state") or "")

    if reject_class == "no_feasible_cf":
        return "no_valid_cf"
    if reject_class == "request_constraints_blocked":
        return "constraint_blocked"
    if reject_class == "parser_failure" or case_completion_reason == "parser_failure" or final_public_state == "PARSER_FAILURE":
        return "parser_schema_failure"
    if (
        reject_class == "unsupported_request"
        or case_completion_reason == "unsupported_request"
        or final_public_state == "UNSUPPORTED_REQUEST"
    ):
        return "unsupported"
    if reject_class == "conflict" or case_completion_reason == "conflict" or final_public_state == "CONFLICT":
        return "conflict"
    if reject_class == "clarification_limit_reached" or case_completion_reason == "clarification_limit_reached":
        return "clarification_exhausted"
    return "other_runtime_reject"


def build_failure_taxonomy(session_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_sessions = len(session_rows)
    non_success_rows = [row for row in session_rows if not row["is_successful_resolution"]]
    non_success_count = len(non_success_rows)
    counts = Counter(row["primary_reason"] for row in non_success_rows)
    rows: list[dict[str, Any]] = []
    for reason in FAILURE_REASONS:
        n_sessions = int(counts.get(reason, 0))
        rows.append(
            {
                "primary_reason": reason,
                "n_sessions": n_sessions,
                "pct_total": ratio(n_sessions, total_sessions),
                "pct_non_success": ratio(n_sessions, non_success_count),
                "definition": FAILURE_DEFINITIONS[reason],
            }
        )
    return rows


def build_conditional_denominators(report: dict[str, Any]) -> list[dict[str, Any]]:
    g2_metrics = report.get("g2_metrics")
    if not isinstance(g2_metrics, dict):
        raise ValueError("g2_metrics must be an object")
    rows: list[dict[str, Any]] = []
    for metric_name, source_key, default_definition in CONDITIONAL_METRICS:
        metric_block = g2_metrics.get(source_key)
        if not isinstance(metric_block, dict):
            raise ValueError(f"g2_metrics.{source_key} must be an object")
        rows.append(
            {
                "metric": metric_name,
                "value": metric_block.get("mean"),
                "n_pass": int(metric_block.get("numerator", 0)),
                "n_denominator": int(metric_block.get("denominator", 0)),
                "scope": "G2_exposed_counterfactuals",
                "definition": str(metric_block.get("formula") or default_definition),
            }
        )
    return rows


def validate_evidence(outputs: dict[str, Any]) -> None:
    session_rows = outputs["session_rows"]
    failure_rows = outputs["failure_rows"]
    conditional_rows = outputs["conditional_rows"]

    total_sessions = len(session_rows)
    if total_sessions != EXPECTED_TOTAL_SESSIONS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_SESSIONS} sessions, found {total_sessions}")

    group_counts = Counter(row["group"] for row in session_rows)
    for group, expected in EXPECTED_GROUP_COUNTS.items():
        actual = int(group_counts.get(group, 0))
        if actual != expected:
            raise ValueError(f"Expected group {group} count {expected}, found {actual}")
    unexpected_groups = sorted(group for group in group_counts if group not in EXPECTED_GROUP_COUNTS)
    if unexpected_groups:
        raise ValueError(f"Unexpected groups in main_200 scope: {unexpected_groups}")

    successful_resolution = sum(1 for row in session_rows if row["is_successful_resolution"])
    if successful_resolution != EXPECTED_SUCCESSFUL_RESOLUTION:
        raise ValueError(
            f"Expected successful_resolution {EXPECTED_SUCCESSFUL_RESOLUTION}, found {successful_resolution}"
        )

    non_success = total_sessions - successful_resolution
    if non_success != EXPECTED_NON_SUCCESS:
        raise ValueError(f"Expected non_success {EXPECTED_NON_SUCCESS}, found {non_success}")

    failure_total = sum(int(row["n_sessions"]) for row in failure_rows)
    if failure_total != EXPECTED_NON_SUCCESS:
        raise ValueError(f"Expected failure taxonomy sum {EXPECTED_NON_SUCCESS}, found {failure_total}")

    other_runtime_reject = next(
        int(row["n_sessions"]) for row in failure_rows if row["primary_reason"] == "other_runtime_reject"
    )
    if other_runtime_reject != 0:
        raise ValueError(f"Expected other_runtime_reject 0, found {other_runtime_reject}")

    for row in conditional_rows:
        denominator = int(row["n_denominator"])
        if denominator != EXPECTED_CONDITIONAL_DENOMINATOR:
            raise ValueError(
                f"Expected conditional denominator {EXPECTED_CONDITIONAL_DENOMINATOR} "
                f"for {row['metric']}, found {denominator}"
            )


def build_markdown_summary(
    *,
    report: dict[str, Any],
    source_report: Path,
    session_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    conditional_rows: list[dict[str, Any]],
) -> str:
    total_sessions = len(session_rows)
    counterfactual_found = sum(1 for row in session_rows if row["is_counterfactual_found"])
    no_recourse_needed = sum(1 for row in session_rows if row["is_no_recourse_needed"])
    successful_resolution = sum(1 for row in session_rows if row["is_successful_resolution"])
    non_success = total_sessions - successful_resolution

    lines = [
        "# Part II Core Evidence For Chapter 4",
        "",
        "## Source",
        "",
        f"- Source report: `{Path(source_report).resolve()}`",
        f"- Run ID: `{report.get('run_id', '')}`",
        f"- Runner scope: `{report.get('runner_scope', '')}`",
        f"- Corpus version: `{report.get('corpus_version', '')}`",
        f"- Corpus SHA-256: `{report.get('corpus_sha256', '')}`",
        "- Thesis scope: `main_200 = G1 + G2`; refinement sessions are excluded from this denominator.",
        "",
        "## Outcome Context",
        "",
        f"- Total sessions: `{total_sessions}`",
        f"- Successful resolution: `{successful_resolution}/{total_sessions}`",
        f"- Counterfactual found: `{counterfactual_found}/{total_sessions}`",
        f"- No recourse needed: `{no_recourse_needed}/{total_sessions}`",
        f"- Non-success: `{non_success}/{total_sessions}`",
        "",
        "## Failure Taxonomy",
        "",
        "| Primary reason | n | pct total | pct non-success |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in failure_rows:
        lines.append(
            f"| `{row['primary_reason']}` | {row['n_sessions']} | "
            f"{row['pct_total']:.6f} | {row['pct_non_success']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Conditional Quality Denominators",
            "",
            "| Metric | Value | Pass / Denominator | Scope |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in conditional_rows:
        lines.append(
            f"| `{row['metric']}` | {float(row['value']):.6f} | "
            f"{row['n_pass']}/{row['n_denominator']} | `{row['scope']}` |"
        )
    lines.extend(
        [
            "",
            "## Wording Guardrails",
            "",
            "- Do not write that system quality is 1.0 over all 200 sessions.",
            "- Write that conditional quality metrics are 1.0 over G2 exposed counterfactuals only.",
            "- Do not merge `no_recourse_needed` with `counterfactual_found`; both count as successful resolution, but only the latter exposes a counterfactual.",
            "- Do not claim plausibility is density, probability, LOF, or Mahalanobis unless the implementation changes.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(outputs: dict[str, Any], *, out_dir: Path) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "part2_session_outcomes_normalized.csv", SESSION_OUTCOME_COLUMNS, outputs["session_rows"])
    write_csv(output_dir / "part2_failure_taxonomy.csv", FAILURE_TAXONOMY_COLUMNS, outputs["failure_rows"])
    write_csv(
        output_dir / "part2_conditional_quality_denominators.csv",
        CONDITIONAL_METRIC_COLUMNS,
        outputs["conditional_rows"],
    )
    (output_dir / "part2_core_evidence_summary.md").write_text(outputs["summary_markdown"], encoding="utf-8")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_stdout_summary(outputs: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    session_rows = outputs["session_rows"]
    successful_resolution = sum(1 for row in session_rows if row["is_successful_resolution"])
    return {
        "out_dir": str(Path(out_dir).resolve()),
        "source_report": outputs["source_report"],
        "total_sessions": len(session_rows),
        "successful_resolution": successful_resolution,
        "non_success": len(session_rows) - successful_resolution,
        "failure_taxonomy": {row["primary_reason"]: row["n_sessions"] for row in outputs["failure_rows"]},
        "conditional_denominators": {
            row["metric"]: row["n_denominator"] for row in outputs["conditional_rows"]
        },
    }


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
