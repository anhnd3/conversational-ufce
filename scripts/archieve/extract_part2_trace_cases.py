#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTCOMES = Path("outputs/chapter4_part2_evidence/part2_session_outcomes_normalized.csv")
DEFAULT_LOG_ROOT = Path(
    "outputs/part2_thesis_metrics/part2_thesis_metrics_20260404_125616_618657/isolated_product_artifacts"
)
DEFAULT_OUTPUT_MD = Path("part2_trace_cases_selected.md")
DEFAULT_OUTPUT_CSV = Path("part2_trace_cases_selected.csv")

SELECTIONS = [
    ("counterfactual_found", "TIERB-G1-004"),
    ("constraint_blocked", "TIERB-G2-038"),
    ("no_recourse_needed", "TIERB-G1-010"),
]


def compact(value: Any, max_len: int = 1600) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def turn_number(path: Path) -> int:
    match = re.search(r"_turn_(\d+)$", path.parent.name)
    return int(match.group(1)) if match else 0


def load_json(path: Path) -> dict[str, Any]:
    return json.load(path.open("r", encoding="utf-8"))


def first_nonempty_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def find_turn_results(log_roots: list[Path], session_id: str) -> list[tuple[Path, dict[str, Any]]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for root in log_roots:
        if not root.exists():
            continue
        candidate_files: list[Path]
        if root.is_file():
            candidate_files = [root] if root.name == "turn_result.json" else []
        else:
            candidate_files = [
                path
                for path in root.rglob("turn_result.json")
                if session_id in str(path.parent) or session_id in str(path)
            ]
        for path in candidate_files:
            try:
                matches.append((path, load_json(path)))
            except Exception as exc:
                print(f"WARNING: could not read {path}: {exc}")
    matches.sort(key=lambda item: (turn_number(item[0]), str(item[0])))
    return matches


def summarize_trace(
    *,
    label: str,
    outcome_row: pd.Series,
    turn_results: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    terminal_path, terminal = turn_results[-1]
    explanation = terminal.get("explanation_payload") or {}
    normalized_parse = terminal.get("normalized_parse") or {}
    builder_result = terminal.get("builder_result") or {}
    runtime_result = terminal.get("runtime_result") or {}
    counterfactual = explanation.get("counterfactual_summary") or {}
    reason_codes = explanation.get("reason_codes") or []
    user_inputs = [record.get("user_input", "") for _, record in turn_results if record.get("user_input")]
    active_constraint_spec = first_nonempty_dict(
        terminal.get("active_constraint_spec"),
        normalized_parse.get("constraint_spec"),
        (builder_result.get("runtime_request") or {}).get("constraint_spec"),
        (runtime_result.get("canonical_request") or {}).get("hard_constraints"),
        ((terminal.get("runtime_debug_trace") or {}).get("constraint_filter") or {}).get("constraint_spec"),
    )
    solution = {
        "changed_fields": explanation.get("changed_fields") or [],
        "counterfactual_summary": counterfactual,
        "next_step_suggestions": explanation.get("next_step_suggestions") or [],
    }
    backend_decision = {
        "final_public_state": outcome_row.get("final_public_state", ""),
        "case_completion_reason": terminal.get("case_completion_reason", ""),
        "summary_type": explanation.get("summary_type", outcome_row.get("summary_type", "")),
        "reason_codes": reason_codes,
    }

    return {
        "case_label": label,
        "case_id": str(outcome_row.get("case_id", "")),
        "session_id": str(outcome_row.get("session_id", "")),
        "group": str(outcome_row.get("group", "")),
        "final_public_state": str(outcome_row.get("final_public_state", "")),
        "summary_type": str(outcome_row.get("summary_type", "")),
        "primary_reason": str(outcome_row.get("primary_reason", "")),
        "reject_class": "" if pd.isna(outcome_row.get("reject_class", "")) else str(outcome_row.get("reject_class", "")),
        "source_file": str(terminal_path),
        "turn_count": len(turn_results),
        "user_inputs": compact(user_inputs),
        "terminal_user_input": compact(terminal.get("user_input")),
        "parsed_request": compact(normalized_parse.get("cf_request")),
        "parser_status": compact(normalized_parse.get("status")),
        "missing_fields": compact(normalized_parse.get("missing_fields")),
        "active_constraint_spec": compact(active_constraint_spec),
        "case_completion_reason": compact(terminal.get("case_completion_reason")),
        "backend_decision": compact(backend_decision),
        "reason_codes": compact(reason_codes),
        "solution": compact(solution),
        "response": compact(terminal.get("response_text"), max_len=2200),
    }


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    lines = ["# Selected Part II Trace Cases\n\n"]
    for row in rows:
        lines.append(f"## {row['case_label']} - {row['case_id']}\n\n")
        lines.append(f"- Session ID: `{row['session_id']}`\n")
        lines.append(f"- Group: `{row['group']}`\n")
        lines.append(f"- Final state: `{row['final_public_state']}`\n")
        lines.append(f"- Summary type: `{row['summary_type']}`\n")
        lines.append(f"- Primary reason: `{row['primary_reason']}`\n")
        lines.append(f"- Source file: `{row['source_file']}`\n\n")
        lines.append("| Step | Evidence |\n")
        lines.append("|---|---|\n")
        for key, label in [
            ("user_inputs", "input"),
            ("parsed_request", "parsed request"),
            ("active_constraint_spec", "state"),
            ("backend_decision", "backend decision"),
            ("reason_codes", "reason code"),
            ("solution", "solution"),
            ("response", "response"),
        ]:
            value = str(row.get(key, "")).replace("|", "\\|")
            lines.append(f"| {label} | {value} |\n")
        lines.append("\n")
    output.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract selected real trace cases for Chapter 4 Section 4.4.")
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--log-root", action="append", type=Path)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    log_roots = args.log_root or [DEFAULT_LOG_ROOT]
    outcomes = pd.read_csv(args.outcomes)
    rows: list[dict[str, Any]] = []

    for label, case_id in SELECTIONS:
        matches = outcomes[outcomes["case_id"].eq(case_id)]
        if matches.empty:
            raise ValueError(f"Could not find selected case_id in outcomes: {case_id}")
        outcome_row = matches.iloc[0]
        session_id = str(outcome_row["session_id"])
        turn_results = find_turn_results(log_roots, session_id)
        if not turn_results:
            raise FileNotFoundError(f"Could not find turn_result.json files for session {session_id}")
        rows.append(summarize_trace(label=label, outcome_row=outcome_row, turn_results=turn_results))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    write_markdown(rows, args.output_md)

    print(f"outcomes = {args.outcomes}")
    print("log_roots =", [str(item) for item in log_roots])
    print(f"output_csv = {args.output_csv}")
    print(f"output_md = {args.output_md}")
    print(pd.DataFrame(rows)[["case_label", "case_id", "session_id", "source_file"]].to_string(index=False))


if __name__ == "__main__":
    main()
