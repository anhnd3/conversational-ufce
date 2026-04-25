from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from llm.src.utils.io import write_case_scores_csv, write_json, write_jsonl


def build_summary(
    rows: list[dict[str, Any]],
    *,
    benchmark_name: str,
    model_alias: str,
    run_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    groups = sorted({row["group"] for row in rows})
    summary = {
        "benchmark_name": benchmark_name,
        "model_alias": model_alias,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "overall": summarize_row_set(rows),
        "groups": {group: summarize_row_set([row for row in rows if row["group"] == group]) for group in groups},
        "failure_counts": {
            "api_error_count": sum(1 for row in rows if row.get("api_error")),
            "invalid_json_count": sum(1 for row in rows if not row["valid_json"]),
            "schema_invalid_count": sum(1 for row in rows if not row["schema_valid"]),
            "exact_match_fail_count": sum(1 for row in rows if not row["exact_match"]),
        },
    }
    return summary


def summarize_row_set(rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_stability = collapse_case_stability(rows)
    request_latencies = numeric_values(rows, "request_latency_ms")
    tokens_per_second = numeric_values(rows, "tokens_per_second")
    return {
        "run_count": len(rows),
        "case_count": len({row["case_id"] for row in rows}),
        "valid_json_rate": mean_bool(rows, "valid_json"),
        "schema_valid_rate": mean_bool(rows, "schema_valid"),
        "exact_match_rate": mean_bool(rows, "exact_match"),
        "field_accuracy_mean": mean_numeric(rows, "field_accuracy"),
        "status_accuracy": mean_bool(rows, "status_correct"),
        "missing_fields_accuracy": mean_bool(rows, "missing_fields_correct"),
        "conflict_accuracy": mean_bool(rows, "conflicts_correct"),
        "avg_hallucination_count": mean_numeric(rows, "hallucination_count"),
        "stability": mean(case_stability.values()) if case_stability else 0.0,
        "avg_request_latency_ms": mean(request_latencies) if request_latencies else None,
        "avg_tokens_per_second": mean(tokens_per_second) if tokens_per_second else None,
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Summary: {summary['model_alias']}",
        "",
        f"- Benchmark: `{summary['benchmark_name']}`",
        f"- Run id: `{summary['run_id']}`",
        f"- Run dir: `{summary['run_dir']}`",
        "",
        "## Overall",
        "",
        render_metric_table(summary["overall"]),
        "",
        "## By Group",
        "",
    ]
    for group, metrics in summary["groups"].items():
        lines.append(f"### Group {group}")
        lines.append("")
        lines.append(render_metric_table(metrics))
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_metric_table(metrics: dict[str, Any]) -> str:
    rows = [
        ("run_count", metrics["run_count"]),
        ("case_count", metrics["case_count"]),
        ("valid_json_rate", format_metric(metrics["valid_json_rate"])),
        ("schema_valid_rate", format_metric(metrics["schema_valid_rate"])),
        ("exact_match_rate", format_metric(metrics["exact_match_rate"])),
        ("field_accuracy_mean", format_metric(metrics["field_accuracy_mean"])),
        ("status_accuracy", format_metric(metrics["status_accuracy"])),
        ("missing_fields_accuracy", format_metric(metrics["missing_fields_accuracy"])),
        ("conflict_accuracy", format_metric(metrics["conflict_accuracy"])),
        ("avg_hallucination_count", format_metric(metrics["avg_hallucination_count"])),
        ("stability", format_metric(metrics["stability"])),
        ("avg_request_latency_ms", format_metric(metrics["avg_request_latency_ms"])),
        ("avg_tokens_per_second", format_metric(metrics["avg_tokens_per_second"])),
    ]
    lines = ["| metric | value |", "| --- | --- |"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    return "\n".join(lines)


def render_errors_markdown(rows: list[dict[str, Any]]) -> str:
    api_errors = Counter(row["api_error"] for row in rows if row.get("api_error"))
    parse_errors = Counter(row["parse_error"] for row in rows if row.get("parse_error"))
    validation_errors = Counter()
    for row in rows:
        for error in row.get("validation_error_list", []):
            validation_errors[error] += 1

    lines = [
        "# Error Summary",
        "",
        f"- API errors: {sum(api_errors.values())}",
        f"- Invalid JSON runs: {sum(1 for row in rows if not row['valid_json'])}",
        f"- Schema-invalid runs: {sum(1 for row in rows if not row['schema_valid'])}",
        f"- Exact-match failures: {sum(1 for row in rows if not row['exact_match'])}",
        "",
    ]

    lines.extend(render_counter_section("Top API errors", api_errors))
    lines.extend(render_counter_section("Top parse errors", parse_errors))
    lines.extend(render_counter_section("Top validation errors", validation_errors))

    representative_rows = [
        row
        for row in rows
        if row.get("api_error") or row.get("parse_error") or not row.get("schema_valid") or not row.get("exact_match")
    ][:10]
    if representative_rows:
        lines.extend(["## Representative failed runs", ""])
        for row in representative_rows:
            lines.append(f"### {row['case_id']} repeat {row['repeat_id']}")
            lines.append(f"- api_error: {row.get('api_error') or 'none'}")
            lines.append(f"- parse_error: {row.get('parse_error') or 'none'}")
            lines.append(
                "- validation_errors: "
                + (", ".join(row.get("validation_error_list", [])) or "none")
            )
            lines.append(f"- exact_match: {row['exact_match']}")
            api_response_excerpt = build_api_response_excerpt(row)
            if api_response_excerpt:
                lines.append(f"- api_response_excerpt: {api_response_excerpt[:220]}")
            excerpt = (row.get("final_message_text") or "").replace("\n", " ").strip()
            lines.append(f"- final_message_excerpt: {excerpt[:220] or '(empty)'}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_counter_section(title: str, counter: Counter[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not counter:
        lines.extend(["- none", ""])
        return lines
    for value, count in counter.most_common(10):
        lines.append(f"- {count}x {value}")
    lines.append("")
    return lines


def build_api_response_excerpt(row: dict[str, Any]) -> str:
    response_json = row.get("full_api_response")
    if isinstance(response_json, dict):
        return json.dumps(response_json, ensure_ascii=True, sort_keys=True)
    response_text = row.get("raw_response_text")
    if isinstance(response_text, str):
        return response_text.replace("\n", " ").strip()
    return ""


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key)) / len(rows)


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float:
    values = numeric_values(rows, key)
    return mean(values) if values else 0.0


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def collapse_case_stability(rows: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get("stability_case"), (int, float)):
            grouped[row["case_id"]].append(float(row["stability_case"]))
    return {case_id: values[0] for case_id, values in grouped.items() if values}


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
