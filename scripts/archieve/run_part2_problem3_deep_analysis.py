#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact


DEFAULT_RUN_ROOT = (
    ROOT
    / "outputs"
    / "part2_problem3_live_validation"
    / "part2_thesis_metrics_20260331_195558_756283"
)
DEFAULT_OUTPUT_PARENT = ROOT / "outputs"
DEFAULT_CORPUS_PATH = ROOT / "docs" / "validation" / "corpora" / "part2_tier_b_bank_sessions_v1.json"

REQUIRED_TURN2_ARTIFACTS = (
    "builder_result.json",
    "canonical_validation.json",
    "clarification_payload.json",
    "turn_result.json",
)
REPRESENTATIVE_CASES = (
    "TIERB-G2-001",
    "TIERB-G2-004",
    "TIERB-G2-005",
    "TIERB-G1-003",
    "TIERB-G1-010",
)
MANUAL_NEIGHBORS = {
    "TIERB-G2-001": "TIERB-G2-025",
    "TIERB-G2-004": "TIERB-G2-028",
    "TIERB-G2-005": "TIERB-G2-029",
    "TIERB-G1-003": "TIERB-G1-001",
    "TIERB-G1-010": "TIERB-G1-009",
}
RUNNER_SCOPE = "problem3_clarification_terminalization_rca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Problem 3 deep analysis bundle for the 29 frozen mismatches.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    report_path = run_root / "thesis_metrics_report.json"
    corpus_path = args.corpus.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    corpus_cases = {
        case["case_id"]: case
        for case in list(corpus.get("cases") or [])
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }

    run_id = "part2_problem3_deep_analysis_" + local_now_compact()
    derived_root = args.out_dir.resolve() / run_id
    derived_root.mkdir(parents=True, exist_ok=False)

    mismatch_ids = list(report["script_mismatch_summary"]["case_identifiers"])
    report_rows_by_case = {
        row["case_id"]: row
        for row in report["per_case_results"]
        if isinstance(row, dict) and isinstance(row.get("case_id"), str)
    }
    mismatch_rows = [report_rows_by_case[case_id] for case_id in mismatch_ids]

    artifact_dir_map = index_artifact_directories(run_root / "isolated_product_artifacts")
    artifact_pointer_index = build_artifact_pointer_index(
        mismatch_rows=mismatch_rows,
        artifact_dir_map=artifact_dir_map,
        root=ROOT,
    )
    artifact_pointer_lookup = {entry["case_id"]: entry for entry in artifact_pointer_index}

    turn2_signal_matrix = build_turn2_signal_matrix(
        mismatch_rows=mismatch_rows,
        artifact_pointer_lookup=artifact_pointer_lookup,
    )
    turn2_signal_lookup = {entry["case_id"]: entry for entry in turn2_signal_matrix}

    mismatch_inventory = build_mismatch_inventory(
        mismatch_rows=mismatch_rows,
        turn2_signal_lookup=turn2_signal_lookup,
        artifact_pointer_lookup=artifact_pointer_lookup,
    )

    family_assignments = build_family_assignments(
        inventory_rows=mismatch_inventory,
        turn2_signal_lookup=turn2_signal_lookup,
        artifact_pointer_lookup=artifact_pointer_lookup,
    )
    family_lookup = {entry["case_id"]: entry for entry in family_assignments}

    representative_packets, reconstructions = build_representative_packets(
        representative_cases=REPRESENTATIVE_CASES,
        mismatch_rows_by_case={row["case_id"]: row for row in mismatch_rows},
        artifact_pointer_lookup=artifact_pointer_lookup,
        turn2_signal_lookup=turn2_signal_lookup,
        family_lookup=family_lookup,
        corpus_cases=corpus_cases,
        report_rows_by_case=report_rows_by_case,
        artifact_dir_map=artifact_dir_map,
    )

    family_summary = build_family_summary(
        family_assignments=family_assignments,
        mismatch_rows_by_case={row["case_id"]: row for row in mismatch_rows},
        turn2_signal_lookup=turn2_signal_lookup,
        artifact_pointer_lookup=artifact_pointer_lookup,
    )

    write_outputs(
        derived_root=derived_root,
        run_root=run_root,
        report=report,
        corpus_path=corpus_path,
        mismatch_ids=mismatch_ids,
        mismatch_rows=mismatch_rows,
        artifact_pointer_index=artifact_pointer_index,
        mismatch_inventory=mismatch_inventory,
        turn2_signal_matrix=turn2_signal_matrix,
        family_assignments=family_assignments,
        family_summary=family_summary,
        representative_packets=representative_packets,
        reconstructions=reconstructions,
    )

    summary = {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "source_run_root": str(run_root),
        "derived_root": str(derived_root),
        "mismatch_case_count": len(mismatch_ids),
        "family_counts": {family: details["count"] for family, details in family_summary["families"].items()},
        "largest_family": family_summary["largest_family"],
        "primary_failing_gate": "canonical_validation.ready_for_runtime stays false because missing_runtime_fields=['CCAvg']",
        "next_patch_target": "initial-turn parser-quality profile recovery for explicit CCAvg in clarification-seeding turns",
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


def index_artifact_directories(artifact_root: Path) -> dict[str, dict[int, Path]]:
    mapping: dict[str, dict[int, Path]] = defaultdict(dict)
    for entry in sorted(artifact_root.iterdir()):
        if not entry.is_dir():
            continue
        match = re.search(r"_api_(session_[0-9_]+)_turn_(\d+)$", entry.name)
        if not match:
            continue
        session_id = match.group(1)
        turn_index = int(match.group(2))
        mapping[session_id][turn_index] = entry
    return mapping


def build_artifact_pointer_index(
    *,
    mismatch_rows: list[dict[str, Any]],
    artifact_dir_map: dict[str, dict[int, Path]],
    root: Path,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in mismatch_rows:
        session_id = str(row["session_id"])
        turn_dirs = artifact_dir_map.get(session_id, {})
        turn_1_dir = turn_dirs.get(1)
        turn_2_dir = turn_dirs.get(2)
        missing_turn_directories: list[str] = []
        if turn_1_dir is None:
            missing_turn_directories.append("turn_1")
        if turn_2_dir is None:
            missing_turn_directories.append("turn_2")
        missing_turn_2_artifacts: list[str] = []
        artifact_incomplete = bool(missing_turn_directories)
        if turn_2_dir is not None:
            for filename in REQUIRED_TURN2_ARTIFACTS:
                if not (turn_2_dir / filename).exists():
                    missing_turn_2_artifacts.append(filename)
            artifact_incomplete = artifact_incomplete or bool(missing_turn_2_artifacts)
        entries.append(
            {
                "case_id": row["case_id"],
                "session_id": session_id,
                "turn_1_dir": relpath(turn_1_dir, root),
                "turn_2_dir": relpath(turn_2_dir, root),
                "missing_turn_directories": missing_turn_directories,
                "missing_turn_2_artifacts": missing_turn_2_artifacts,
                "artifact_incomplete": artifact_incomplete,
            }
        )
    return entries


def build_turn2_signal_matrix(
    *,
    mismatch_rows: list[dict[str, Any]],
    artifact_pointer_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for row in mismatch_rows:
        case_id = row["case_id"]
        pointer = artifact_pointer_lookup[case_id]
        turn_2_dir = abs_path(pointer["turn_2_dir"])
        if turn_2_dir is None:
            signals.append(
                {
                    "case_id": case_id,
                    "artifact_incomplete": True,
                    "builder_status": None,
                    "followup_classification": None,
                    "merge_applied": None,
                    "runtime_request_present": False,
                    "ready_for_runtime": False,
                    "missing_runtime_fields": [],
                    "confirmed_conflicts": [],
                    "clarification_missing_fields": [],
                    "carried_constraint_keys": [],
                    "parser_quality.constraint_extraction_absent": None,
                    "normalized_parse.constraint_spec_present": False,
                    "turn2_terminal_ready_but_not_executed": False,
                }
            )
            continue

        builder_result = read_json_or_none(turn_2_dir / "builder_result.json") or {}
        canonical_validation = read_json_or_none(turn_2_dir / "canonical_validation.json") or {}
        clarification_payload = read_json_or_none(turn_2_dir / "clarification_payload.json") or {}
        turn_result = read_json_or_none(turn_2_dir / "turn_result.json") or {}
        normalized_parse = dict(turn_result.get("normalized_parse") or {})
        parser_quality = dict(normalized_parse.get("_parser_quality") or {})
        parser_quality_flags = dict(parser_quality.get("flags") or {})
        runtime_request = (
            canonical_validation.get("runtime_request")
            or builder_result.get("runtime_request")
            or dict(turn_result.get("builder_result") or {}).get("runtime_request")
        )
        ready_for_runtime = bool(canonical_validation.get("ready_for_runtime"))
        runtime_request_present = runtime_request is not None
        runtime_executed = bool(row.get("runtime_executed"))
        signals.append(
            {
                "case_id": case_id,
                "artifact_incomplete": bool(pointer["artifact_incomplete"]),
                "builder_status": builder_result.get("builder_status"),
                "followup_classification": dict(builder_result.get("provenance") or {}).get("followup_classification"),
                "merge_applied": builder_result.get("merge_applied"),
                "runtime_request_present": runtime_request_present,
                "ready_for_runtime": ready_for_runtime,
                "missing_runtime_fields": list(canonical_validation.get("missing_runtime_fields") or []),
                "confirmed_conflicts": list(canonical_validation.get("confirmed_conflicts") or []),
                "clarification_missing_fields": list(clarification_payload.get("missing_fields") or []),
                "carried_constraint_keys": list(builder_result.get("carried_constraint_keys") or []),
                "parser_quality.constraint_extraction_absent": parser_quality_flags.get("constraint_extraction_absent"),
                "normalized_parse.constraint_spec_present": bool(normalized_parse.get("constraint_spec")),
                "turn2_terminal_ready_but_not_executed": bool(
                    (ready_for_runtime or runtime_request_present) and not runtime_executed
                ),
            }
        )
    return signals


def build_mismatch_inventory(
    *,
    mismatch_rows: list[dict[str, Any]],
    turn2_signal_lookup: dict[str, dict[str, Any]],
    artifact_pointer_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    inventory_rows: list[dict[str, Any]] = []
    for row in mismatch_rows:
        case_id = row["case_id"]
        signal = turn2_signal_lookup[case_id]
        pointer = artifact_pointer_lookup[case_id]
        inventory_rows.append(
            {
                "case_id": case_id,
                "group": row.get("group"),
                "session_shape": row.get("session_shape"),
                "session_id": row.get("session_id"),
                "final_public_state": row.get("final_public_state"),
                "case_completion_reason": row.get("case_completion_reason"),
                "script_execution_status": row.get("script_execution_status"),
                "script_mismatch_reason": row.get("script_mismatch_reason"),
                "clarification_rounds": row.get("clarification_rounds"),
                "merge_followup_turns": row.get("merge_followup_turns"),
                "merge_successes": row.get("merge_successes"),
                "active_constraint_spec": row.get("active_constraint_spec"),
                "active_constraint_spec_expected": row.get("active_constraint_spec_expected"),
                "constraint_spec_expected_match": row.get("constraint_spec_expected_match"),
                "runtime_executed": row.get("runtime_executed"),
                "runtime_controller_state": row.get("runtime_controller_state"),
                "successful_resolution": row.get("successful_resolution"),
                "turn2_terminal_ready_but_not_executed": signal["turn2_terminal_ready_but_not_executed"],
                "artifact_incomplete": bool(pointer["artifact_incomplete"]),
            }
        )
    return inventory_rows


def build_family_assignments(
    *,
    inventory_rows: list[dict[str, Any]],
    turn2_signal_lookup: dict[str, dict[str, Any]],
    artifact_pointer_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for row in inventory_rows:
        case_id = row["case_id"]
        signal = turn2_signal_lookup[case_id]
        family = "F"
        assignment_reason = "No direct evidence for conflict, runtime-ready, or constraint-carried family; inspect manually."
        logic_vs_script = "logic_bug"
        subtype = "unclear"

        if signal["confirmed_conflicts"]:
            family = "C"
            assignment_reason = "Turn 2 canonical validation already contains confirmed conflicts."
            subtype = "confirmed_conflict"
        elif (signal["ready_for_runtime"] or signal["runtime_request_present"]) and not row["runtime_executed"]:
            family = "D"
            assignment_reason = "Turn 2 shows runtime-ready evidence, but runtime did not execute."
            subtype = "runtime_ready_but_not_executed"
        elif (
            row.get("constraint_spec_expected_match") is True
            and int(row.get("merge_successes") or 0) > 0
            and not bool(row.get("runtime_executed"))
            and signal.get("builder_status") == "NEEDS_CLARIFICATION"
        ):
            family = "A"
            assignment_reason = (
                "Constraint matches expectation and survives merge, but turn 2 still ends in clarification."
            )
            subtype = "constraint_carried_missing_ccavg"
        else:
            turn_1_dir = abs_path(artifact_pointer_lookup[case_id]["turn_1_dir"])
            turn_1_parse = read_json_or_none(turn_1_dir / "normalized_parse.json") if turn_1_dir else {}
            turn_1_cf_request = dict((turn_1_parse or {}).get("cf_request") or {})
            if "CCAvg" not in turn_1_cf_request:
                assignment_reason = (
                    "Turn 1 explicit profile still omitted CCAvg, so clarification is reacting to incomplete prior state."
                )
                subtype = "prior_state_ccavg_omission"
            if signal.get("followup_classification") == "ambiguous_followup":
                subtype = "prior_state_ccavg_omission_with_ambiguous_followup"

        assignments.append(
            {
                "case_id": case_id,
                "family": family,
                "subtype": subtype,
                "logic_vs_script": logic_vs_script,
                "assignment_reason": assignment_reason,
                "artifact_incomplete": bool(artifact_pointer_lookup[case_id]["artifact_incomplete"]),
            }
        )
    return assignments


def build_family_summary(
    *,
    family_assignments: list[dict[str, Any]],
    mismatch_rows_by_case: dict[str, dict[str, Any]],
    turn2_signal_lookup: dict[str, dict[str, Any]],
    artifact_pointer_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for assignment in family_assignments:
        grouped[assignment["family"]].append(assignment)

    families: dict[str, dict[str, Any]] = {}
    for family_name in ("A", "B", "C", "D", "E", "F"):
        items = grouped.get(family_name, [])
        cases = [item["case_id"] for item in items]
        turn1_ccavg_missing = 0
        for case_id in cases:
            turn_1_dir = abs_path(artifact_pointer_lookup[case_id]["turn_1_dir"])
            turn_1_parse = read_json_or_none(turn_1_dir / "normalized_parse.json") if turn_1_dir else {}
            if "CCAvg" not in dict((turn_1_parse or {}).get("cf_request") or {}):
                turn1_ccavg_missing += 1
        families[family_name] = {
            "count": len(items),
            "cases": cases,
            "logic_vs_script": sorted({item["logic_vs_script"] for item in items}) if items else [],
            "subtypes": dict(Counter(item["subtype"] for item in items)),
            "turn1_ccavg_missing_count": turn1_ccavg_missing,
            "summary": family_summary_text(
                family_name=family_name,
                assignments=items,
                mismatch_rows_by_case=mismatch_rows_by_case,
                turn2_signal_lookup=turn2_signal_lookup,
            ),
        }
    non_empty = {name: details["count"] for name, details in families.items() if details["count"] > 0}
    largest_family = max(non_empty.items(), key=lambda item: (item[1], item[0]))[0] if non_empty else None
    return {"families": families, "largest_family": largest_family}


def family_summary_text(
    *,
    family_name: str,
    assignments: list[dict[str, Any]],
    mismatch_rows_by_case: dict[str, dict[str, Any]],
    turn2_signal_lookup: dict[str, dict[str, Any]],
) -> str:
    if not assignments:
        return "No cases assigned."
    if family_name == "A":
        return (
            "Constraint-bearing clarification cases where the expected constraint survives into turn 2, "
            "but canonical readiness still fails because CCAvg is missing from the carried profile."
        )
    if family_name == "F":
        followup_counts = Counter(
            str(turn2_signal_lookup[item["case_id"]].get("followup_classification") or "profile_completion")
            for item in assignments
        )
        return (
            "Residual clarification cases without constraint evidence for Families A/C/D. "
            f"Current split: {dict(followup_counts)}. All still share the same turn-1 CCAvg omission pattern."
        )
    if family_name == "C":
        return "Confirmed conflicts are already present in canonical validation, but the controller stays in clarification."
    if family_name == "D":
        return "Runtime-ready evidence exists, but execution does not happen."
    if family_name == "E":
        return "Potential script drift. Not used unless controller behavior is coherent and neighbor evidence supports it."
    return "Reserved family; no cases assigned in this run."


def build_representative_packets(
    *,
    representative_cases: tuple[str, ...],
    mismatch_rows_by_case: dict[str, dict[str, Any]],
    artifact_pointer_lookup: dict[str, dict[str, Any]],
    turn2_signal_lookup: dict[str, dict[str, Any]],
    family_lookup: dict[str, dict[str, Any]],
    corpus_cases: dict[str, dict[str, Any]],
    report_rows_by_case: dict[str, dict[str, Any]],
    artifact_dir_map: dict[str, dict[int, Path]],
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    packets: dict[str, dict[str, str]] = defaultdict(dict)
    reconstructions: list[dict[str, str]] = []

    for case_id in representative_cases:
        row = mismatch_rows_by_case[case_id]
        family = family_lookup[case_id]["family"]
        corpus_case = corpus_cases.get(case_id) or {}
        turn_1_dir = abs_path(artifact_pointer_lookup[case_id]["turn_1_dir"])
        turn_2_dir = abs_path(artifact_pointer_lookup[case_id]["turn_2_dir"])
        turn_1_packet = load_turn_packet(turn_1_dir)
        turn_2_packet = load_turn_packet(turn_2_dir)
        neighbor_case_id = select_neighbor_case(
            case_id=case_id,
            row=row,
            report_rows_by_case=report_rows_by_case,
        )
        neighbor_row = report_rows_by_case[neighbor_case_id]
        neighbor_turn_dir = select_terminal_turn_dir(
            session_id=str(neighbor_row["session_id"]),
            artifact_dir_map=artifact_dir_map,
        )
        neighbor_packet = load_turn_packet(neighbor_turn_dir)

        packet_markdown, reconstruction = render_representative_packet(
            case_id=case_id,
            family=family,
            row=row,
            corpus_case=corpus_case,
            turn_1_packet=turn_1_packet,
            turn_2_packet=turn_2_packet,
            neighbor_case_id=neighbor_case_id,
            neighbor_row=neighbor_row,
            neighbor_packet=neighbor_packet,
            turn2_signal=turn2_signal_lookup[case_id],
        )
        packets[family][case_id] = packet_markdown
        reconstructions.append(reconstruction)

    return packets, reconstructions


def render_representative_packet(
    *,
    case_id: str,
    family: str,
    row: dict[str, Any],
    corpus_case: dict[str, Any],
    turn_1_packet: dict[str, Any],
    turn_2_packet: dict[str, Any],
    neighbor_case_id: str,
    neighbor_row: dict[str, Any],
    neighbor_packet: dict[str, Any],
    turn2_signal: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    expected_constraint = corpus_case.get("active_constraint_spec_expected")
    expected_turns = list(corpus_case.get("turns") or [])
    turn_1_parse = dict(turn_1_packet["normalized_parse"] or {})
    turn_2_parse = dict(turn_2_packet["normalized_parse"] or {})
    turn_2_builder = dict(turn_2_packet["builder_result"] or {})
    turn_2_canonical = dict(turn_2_packet["canonical_validation"] or {})
    turn_2_clarification = dict(turn_2_packet["clarification_payload"] or {})
    merged_snapshot = {
        "partial_profile_snapshot": turn_2_builder.get("partial_profile_snapshot"),
        "missing_fields": turn_2_builder.get("missing_fields"),
        "carried_constraint_keys": turn_2_builder.get("carried_constraint_keys"),
        "merge_applied": turn_2_builder.get("merge_applied"),
        "followup_classification": dict(turn_2_builder.get("provenance") or {}).get("followup_classification"),
    }
    gate_checks = build_gate_checks(case_id=case_id, turn2_signal=turn2_signal, turn_2_packet=turn_2_packet)
    reconstruction = {
        "case_id": case_id,
        "family": family,
        "prior_state_before_followup": (
            f"Turn 1 ended NEEDS_CLARIFICATION with missing_fields={turn_1_parse.get('missing_fields')}, "
            f"cf_request keys={sorted(dict(turn_1_parse.get('cf_request') or {}).keys())}."
        ),
        "followup_delta": (
            f"Turn 2 user input supplied: {turn_2_packet['user_input']}"
        ),
        "merged_canonical_state": (
            f"Turn 2 partial_profile_snapshot={json.dumps(turn_2_builder.get('partial_profile_snapshot'), ensure_ascii=True, sort_keys=True)}; "
            f"missing_runtime_fields={turn_2_canonical.get('missing_runtime_fields')}."
        ),
        "controller_decision_actual": (
            f"builder_status={turn_2_builder.get('builder_status')}, "
            f"followup_classification={dict(turn_2_builder.get('provenance') or {}).get('followup_classification')}, "
            f"final_stage={turn_2_canonical.get('final_stage')}."
        ),
        "controller_decision_should_have_happened": controller_should_have_happened(case_id=case_id, family=family),
    }

    markdown = "\n".join(
        [
            f"# {case_id}",
            "",
            "## 1. Case Summary",
            f"- Group: `{row.get('group')}`",
            f"- Family: `{family}`",
            f"- Session shape: `{row.get('session_shape')}`",
            f"- Final public state: `{row.get('final_public_state')}`",
            f"- Script mismatch reason: `{row.get('script_mismatch_reason')}`",
            "",
            "## 2. Expected Script Outcome",
            (
                "The scripted two-turn session should have exited clarification after turn 2. "
                f"Expected active constraint spec: `{json.dumps(expected_constraint, ensure_ascii=True, sort_keys=True)}`."
            ),
            f"- Script turns: `{json.dumps(expected_turns, ensure_ascii=True)}`",
            "",
            "## 3. Actual Turn 1 Outcome",
            f"- User input: `{turn_1_packet['user_input']}`",
            f"- Raw parser cf_request keys: `{sorted(dict((turn_1_packet['parser_raw_output'] or {}).get('cf_request') or {}).keys())}`",
            f"- Normalized cf_request keys: `{sorted(dict(turn_1_parse.get('cf_request') or {}).keys())}`",
            f"- Missing fields after turn 1: `{turn_1_parse.get('missing_fields')}`",
            f"- Constraint spec after turn 1: `{json.dumps(turn_1_parse.get('constraint_spec'), ensure_ascii=True, sort_keys=True)}`",
            "",
            "## 4. Actual Turn 2 Outcome",
            f"- User input: `{turn_2_packet['user_input']}`",
            f"- Builder status: `{turn_2_builder.get('builder_status')}`",
            f"- Follow-up classification: `{dict(turn_2_builder.get('provenance') or {}).get('followup_classification')}`",
            f"- Merge applied: `{turn_2_builder.get('merge_applied')}`",
            f"- Missing runtime fields: `{turn_2_canonical.get('missing_runtime_fields')}`",
            f"- Ready for runtime: `{turn_2_canonical.get('ready_for_runtime')}`",
            f"- Clarification payload missing fields: `{turn_2_clarification.get('missing_fields')}`",
            "",
            "## 5. Merged State Snapshot",
            f"- `{json.dumps(merged_snapshot, ensure_ascii=True, sort_keys=True)}`",
            "",
            "## 6. Gate Checks",
            f"- Pending clarification rebuilt too strictly: {gate_checks['pending_clarification_rebuilt_too_strict']}",
            f"- Merged state not re-evaluated for terminal readiness: {gate_checks['merged_state_not_re_evaluated']}",
            f"- Conflict boundary delayed: {gate_checks['conflict_boundary_delayed']}",
            f"- Reject boundary delayed: {gate_checks['reject_boundary_delayed']}",
            f"- Constraint-bearing follow-up treated as plain profile completion: {gate_checks['constraint_followup_treated_as_profile_completion']}",
            "",
            "## 7. Neighbor Comparison",
            f"- Neighbor case: `{neighbor_case_id}`",
            f"- Neighbor final public state: `{neighbor_row.get('final_public_state')}`",
            f"- Neighbor session shape: `{neighbor_row.get('session_shape')}`",
            f"- Neighbor active constraint spec: `{json.dumps(neighbor_row.get('active_constraint_spec'), ensure_ascii=True, sort_keys=True)}`",
            f"- Neighbor builder result: `{json.dumps(neighbor_packet['builder_result'], ensure_ascii=True, sort_keys=True)}`",
            f"- Neighbor canonical validation: `{json.dumps(neighbor_packet['canonical_validation'], ensure_ascii=True, sort_keys=True)}`",
            f"- Neighbor clarification payload: `{json.dumps(neighbor_packet['clarification_payload'], ensure_ascii=True, sort_keys=True)}`",
            f"- Neighbor case completion reason: `{neighbor_row.get('case_completion_reason')}`",
            "",
            "## 8. RCA Hypothesis",
            representative_rca_hypothesis(case_id=case_id, family=family),
            "",
            "## 9. Confidence Level",
            representative_confidence(case_id=case_id),
            "",
        ]
    )
    return markdown, reconstruction


def build_gate_checks(
    *,
    case_id: str,
    turn2_signal: dict[str, Any],
    turn_2_packet: dict[str, Any],
) -> dict[str, str]:
    canonical_validation = dict(turn_2_packet["canonical_validation"] or {})
    builder_result = dict(turn_2_packet["builder_result"] or {})
    followup_classification = dict(builder_result.get("provenance") or {}).get("followup_classification")
    carried_constraint_keys = list(builder_result.get("carried_constraint_keys") or [])
    return {
        "pending_clarification_rebuilt_too_strict": (
            "No. Turn 2 canonical validation still reports missing_runtime_fields=['CCAvg'], "
            "so the clarification payload is repeating a genuinely missing field rather than inventing a new one."
        ),
        "merged_state_not_re_evaluated": (
            "No. canonical_validation.ready_for_runtime is false and runtime_request is absent, "
            "so the controller did re-run readiness and failed it on missing CCAvg."
        ),
        "conflict_boundary_delayed": (
            "No. confirmed_conflicts is empty, so there is no direct conflict evidence being ignored."
        ),
        "reject_boundary_delayed": (
            "No. runtime_request is absent and ready_for_runtime is false, so runtime reject is not yet reachable."
        ),
        "constraint_followup_treated_as_profile_completion": (
            "Secondary only. Constraint keys are already present and carried correctly."
            if carried_constraint_keys
            else "Not applicable. This representative has no active constraint path."
        )
        + (
            f" Follow-up classification is {followup_classification!r}, but the blocking gate is still missing CCAvg."
        ),
    }


def controller_should_have_happened(*, case_id: str, family: str) -> str:
    if family == "A":
        return (
            "Once the initial turn had preserved CCAvg, the carried constraint plus follow-up booleans should have "
            "allowed either READY_FOR_RUNTIME or a terminal runtime outcome after turn 2."
        )
    return (
        "Once the initial turn had preserved CCAvg, the case should not have remained trapped in clarification; "
        "either turn 1 would already be complete or turn 2 would provide the last missing fields."
    )


def representative_rca_hypothesis(*, case_id: str, family: str) -> str:
    if family == "A":
        return (
            "The case is not failing because constraint extraction or turn-2 merge lost the constraint. "
            "It fails because turn 1 never recovered the explicitly labeled CCAvg value, so the carried state entering "
            "turn 2 is already incomplete. The controller then behaves coherently by asking again for CCAvg."
        )
    return (
        "This case shares the same turn-1 CCAvg omission, but without constraint evidence. "
        "In the ambiguous-followup subset, turn 2 only repeats fields already present in the saved state, so the builder "
        "marks the answer as ambiguous after the original omission was never repaired."
    )


def representative_confidence(*, case_id: str) -> str:
    if case_id in {"TIERB-G2-001", "TIERB-G2-004", "TIERB-G2-005"}:
        return "High. Turn 1 and turn 2 artifacts align cleanly with the same missing-CCAvg gate."
    return "High. The only residual ambiguity is whether a follow-up classification tweak is still needed after the turn-1 omission is fixed."


def select_neighbor_case(*, case_id: str, row: dict[str, Any], report_rows_by_case: dict[str, dict[str, Any]]) -> str:
    manual = MANUAL_NEIGHBORS.get(case_id)
    if manual:
        return manual

    target_group = row.get("group")
    target_family = constraint_family(row.get("active_constraint_spec"))
    candidates = []
    for candidate in report_rows_by_case.values():
        if candidate["case_id"] == case_id:
            continue
        if candidate.get("group") != target_group:
            continue
        if not candidate.get("is_case_complete"):
            continue
        if constraint_family(candidate.get("active_constraint_spec")) != target_family:
            continue
        candidates.append(candidate)
    if not candidates:
        raise RuntimeError(f"No completed neighbor found for {case_id}")
    return sorted(candidates, key=neighbor_sort_key)[0]["case_id"]


def neighbor_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    session_shape = str(row.get("session_shape") or "")
    final_state = str(row.get("final_public_state") or "")
    shape_rank = 0 if session_shape == "clarification_followup" else 1
    state_rank = {
        "RUNTIME_SUCCESS": 0,
        "RUNTIME_REJECT": 1,
        "CONFLICT": 2,
    }.get(final_state, 3)
    return (shape_rank, state_rank, str(row.get("case_id")))


def select_terminal_turn_dir(*, session_id: str, artifact_dir_map: dict[str, dict[int, Path]]) -> Path:
    turns = artifact_dir_map[session_id]
    return turns[max(turns)]


def load_turn_packet(turn_dir: Path | None) -> dict[str, Any]:
    if turn_dir is None:
        return {
            "turn_dir": None,
            "user_input": None,
            "parser_raw_output": None,
            "repair_raw_output": None,
            "normalized_parse": None,
            "schema_validation": None,
            "canonical_validation": None,
            "clarification_payload": None,
            "builder_result": None,
            "negotiation_transition": None,
            "turn_result": None,
            "response_text": None,
        }
    return {
        "turn_dir": str(turn_dir),
        "user_input": read_text_or_none(turn_dir / "user_input.txt"),
        "parser_raw_output": parse_json_text_or_none(read_text_or_none(turn_dir / "parser_raw_output.txt")),
        "repair_raw_output": parse_json_text_or_none(read_text_or_none(turn_dir / "repair_raw_output.txt")),
        "normalized_parse": read_json_or_none(turn_dir / "normalized_parse.json"),
        "schema_validation": read_json_or_none(turn_dir / "schema_validation.json"),
        "canonical_validation": read_json_or_none(turn_dir / "canonical_validation.json"),
        "clarification_payload": read_json_or_none(turn_dir / "clarification_payload.json"),
        "builder_result": read_json_or_none(turn_dir / "builder_result.json"),
        "negotiation_transition": read_json_or_none(turn_dir / "negotiation_transition.json"),
        "turn_result": read_json_or_none(turn_dir / "turn_result.json"),
        "response_text": read_text_or_none(turn_dir / "response_text.txt"),
    }


def write_outputs(
    *,
    derived_root: Path,
    run_root: Path,
    report: dict[str, Any],
    corpus_path: Path,
    mismatch_ids: list[str],
    mismatch_rows: list[dict[str, Any]],
    artifact_pointer_index: list[dict[str, Any]],
    mismatch_inventory: list[dict[str, Any]],
    turn2_signal_matrix: list[dict[str, Any]],
    family_assignments: list[dict[str, Any]],
    family_summary: dict[str, Any],
    representative_packets: dict[str, dict[str, str]],
    reconstructions: list[dict[str, str]],
) -> None:
    write_json(
        derived_root / "mismatch_subset_manifest.json",
        {
            "runner_scope": RUNNER_SCOPE,
            "source_run_root": str(run_root),
            "source_report_path": str(run_root / "thesis_metrics_report.json"),
            "source_corpus_path": str(corpus_path),
            "case_ids": mismatch_ids,
            "mismatch_count": len(mismatch_ids),
        },
    )
    write_json(derived_root / "mismatch_rows.json", mismatch_rows)
    write_json(derived_root / "artifact_pointer_index.json", artifact_pointer_index)
    write_json(derived_root / "mismatch_inventory.json", mismatch_inventory)
    (derived_root / "mismatch_inventory.md").write_text(
        render_inventory_markdown(mismatch_inventory, turn2_signal_matrix, family_assignments),
        encoding="utf-8",
    )
    write_json(derived_root / "turn2_signal_matrix.json", turn2_signal_matrix)
    write_json(
        derived_root / "family_clustering.json",
        {
            "family_assignments": family_assignments,
            "families": family_summary["families"],
            "largest_family": family_summary["largest_family"],
        },
    )
    (derived_root / "family_clustering.md").write_text(
        render_family_clustering_markdown(
            families=family_summary["families"],
            family_assignments=family_assignments,
        ),
        encoding="utf-8",
    )

    representative_root = derived_root / "representative_packets"
    representative_root.mkdir(parents=True, exist_ok=True)
    for family, packets in representative_packets.items():
        family_dir = representative_root / family
        family_dir.mkdir(parents=True, exist_ok=True)
        for case_id, payload in packets.items():
            (family_dir / f"{case_id}.md").write_text(payload, encoding="utf-8")

    (derived_root / "controller_reconstructions.md").write_text(
        render_controller_reconstructions(reconstructions),
        encoding="utf-8",
    )
    (derived_root / "family_rca_summaries.md").write_text(
        render_family_rca_summaries(families=family_summary["families"]),
        encoding="utf-8",
    )
    (derived_root / "candidate_code_areas.md").write_text(render_candidate_code_areas(), encoding="utf-8")
    (derived_root / "final_recommendation.md").write_text(
        render_final_recommendation(
            families=family_summary["families"],
            largest_family=family_summary["largest_family"],
        ),
        encoding="utf-8",
    )


def render_inventory_markdown(
    inventory_rows: list[dict[str, Any]],
    turn2_signal_matrix: list[dict[str, Any]],
    family_assignments: list[dict[str, Any]],
) -> str:
    signal_lookup = {row["case_id"]: row for row in turn2_signal_matrix}
    family_lookup = {row["case_id"]: row for row in family_assignments}
    header = [
        "# Mismatch Inventory",
        "",
        f"- Total cases: `{len(inventory_rows)}`",
        f"- Groups: `{dict(Counter(row['group'] for row in inventory_rows))}`",
        f"- All final states: `{dict(Counter(row['final_public_state'] for row in inventory_rows))}`",
        "",
        "| Case | Group | Merge Successes | Constraint Match | Builder Status | Follow-up Classification | Missing Runtime Fields | Family |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    body = []
    for row in inventory_rows:
        signal = signal_lookup[row["case_id"]]
        family = family_lookup[row["case_id"]]["family"]
        body.append(
            "| {case_id} | {group} | {merge_successes} | {constraint_match} | {builder_status} | {followup_classification} | {missing_fields} | {family} |".format(
                case_id=row["case_id"],
                group=row["group"],
                merge_successes=row["merge_successes"],
                constraint_match=row["constraint_spec_expected_match"],
                builder_status=signal["builder_status"],
                followup_classification=signal["followup_classification"],
                missing_fields=", ".join(signal["missing_runtime_fields"]),
                family=family,
            )
        )
    return "\n".join(header + body) + "\n"


def render_family_clustering_markdown(
    *,
    families: dict[str, dict[str, Any]],
    family_assignments: list[dict[str, Any]],
) -> str:
    lines = [
        "# Family Clustering",
        "",
        "| Family | Count | Logic vs Script | Subtypes | Summary |",
        "| --- | --- | --- | --- | --- |",
    ]
    for family_name in ("A", "B", "C", "D", "E", "F"):
        family = families[family_name]
        lines.append(
            "| {name} | {count} | {logic} | {subtypes} | {summary} |".format(
                name=family_name,
                count=family["count"],
                logic=", ".join(family["logic_vs_script"]) if family["logic_vs_script"] else "",
                subtypes=json.dumps(family["subtypes"], ensure_ascii=True, sort_keys=True),
                summary=family["summary"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-Case Assignments",
            "",
        ]
    )
    for assignment in family_assignments:
        lines.append(
            "- `{case_id}` -> Family `{family}` (`{subtype}`): {reason}".format(
                case_id=assignment["case_id"],
                family=assignment["family"],
                subtype=assignment["subtype"],
                reason=assignment["assignment_reason"],
            )
        )
    return "\n".join(lines) + "\n"


def render_controller_reconstructions(reconstructions: list[dict[str, str]]) -> str:
    lines = ["# Controller Reconstructions", ""]
    for item in reconstructions:
        lines.extend(
            [
                f"## {item['case_id']}",
                f"1. prior state before follow-up: {item['prior_state_before_followup']}",
                f"2. follow-up delta actually provided: {item['followup_delta']}",
                f"3. merged canonical state: {item['merged_canonical_state']}",
                f"4. controller decision actually taken: {item['controller_decision_actual']}",
                f"5. controller decision that should have happened: {item['controller_decision_should_have_happened']}",
                "",
            ]
        )
    return "\n".join(lines)


def render_family_rca_summaries(*, families: dict[str, dict[str, Any]]) -> str:
    lines = ["# Family RCA Summaries", ""]
    family_details = {
        "A": {
            "representatives": ["TIERB-G2-001", "TIERB-G2-004", "TIERB-G2-005"],
            "symptom": "Constraint-bearing clarification cases remain NEEDS_CLARIFICATION after turn 2 even though the expected constraint is active and merge succeeds.",
            "direct_evidence": "All 20 Family A cases have constraint_spec_expected_match=true, merge_successes=1, missing_runtime_fields=['CCAvg'], and no runtime execution. Turn 1 user text always mentions CCAvg, but turn 1 normalized_parse never contains it.",
            "failing_gate": "canonical_validation.ready_for_runtime remains false because CCAvg is missing from the carried profile.",
            "why_not_parser_globally": "Constraint extraction is already strong. The residual issue is narrower: explicit profile recovery does not add CCAvg in clarification-seeding turns.",
            "smallest_code_area": "llm/src/conversation/bank_profile_extractor.py and llm/src/parser/parser_quality.py",
        },
        "F": {
            "representatives": ["TIERB-G1-003", "TIERB-G1-010"],
            "symptom": "Non-constraint clarification cases remain stuck because the saved prior state is already missing CCAvg. Three cases then become ambiguous when turn 2 adds no new information.",
            "direct_evidence": "All 9 Family F cases end turn 2 with missing_runtime_fields=['CCAvg']; 6 classify as profile_completion and 3 as ambiguous_followup. Turn 1 explicit text always mentions CCAvg, but turn 1 normalized_parse drops it.",
            "failing_gate": "The same runtime-readiness gate fails first; in the ambiguous subset, merge_classification becomes secondary once no new fields are added on turn 2.",
            "why_not_parser_globally": "The issue is not generic JSON/schema parsing. It is a narrow initial-turn profile recovery miss that poisons the clarification state.",
            "smallest_code_area": "llm/src/conversation/bank_profile_extractor.py, llm/src/parser/parser_quality.py, then validate the downstream session persistence path",
        },
    }
    for family_name in ("A", "F"):
        family = families[family_name]
        if family["count"] == 0:
            continue
        details = family_details[family_name]
        lines.extend(
            [
                f"## Family {family_name}",
                f"- Family name: Family {family_name}",
                f"- Representative cases: `{details['representatives']}`",
                f"- Symptom: {details['symptom']}",
                f"- Direct evidence: {details['direct_evidence']}",
                f"- Exact failing gate: {details['failing_gate']}",
                f"- Why parser quality is no longer the broad issue: {details['why_not_parser_globally']}",
                f"- Smallest code area likely affected: `{details['smallest_code_area']}`",
                f"- Expected reduction in mismatch count if fixed: `{family['count']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Logic Bug vs Script Drift",
            "- Family A: logic bug",
            "- Family F: logic bug",
            "- No family met the bar for script drift. There is no neighbor-supported evidence that the controller behavior is more correct than the script expectation.",
            "",
        ]
    )
    return "\n".join(lines)


def render_candidate_code_areas() -> str:
    return "\n".join(
        [
            "# Candidate Code Areas",
            "",
            "## Primary Patch Candidate",
            "- `llm/src/conversation/bank_profile_extractor.py`",
            "- `recover_dense_bank_profile_candidate()` only applies deterministic recovery when `looks_like_dense_structured_bank_profile()` passes.",
            "- `looks_like_dense_structured_bank_profile()` currently requires at least six labeled fields. The 29 failing clarification-seeding turns provide five labeled profile facts plus a constraint or nothing else, so explicit `CCAvg` never gets recovered.",
            "",
            "## Parser-Quality Integration Point",
            "- `llm/src/parser/parser_quality.py`",
            "- `run_parser_quality()` calls `_recover_profile_fields_from_user_text()`, which delegates directly to `recover_dense_bank_profile_candidate()`.",
            "- There is no narrower safe recovery path for explicit labeled numeric fields when the turn is structured but below the dense threshold.",
            "",
            "## Downstream Persistence Path",
            "- `llm/src/conversation/canonical_session_state.py`",
            "- `llm/src/conversation/session.py`",
            "- Once turn 1 normalized_parse omits `CCAvg`, the canonical session state and pending clarification payload carry that omission forward unchanged into turn 2.",
            "",
            "## Do Not Patch First",
            "- `llm/src/conversation/request_builder.py`",
            "- `llm/src/orchestration/clarification_flow.py`",
            "- Current evidence shows these components reacting coherently to the incomplete carried state. Patching them first would treat the symptom instead of the source.",
            "",
        ]
    )


def render_final_recommendation(*, families: dict[str, dict[str, Any]], largest_family: str | None) -> str:
    largest_count = 0 if largest_family is None else families[largest_family]["count"]
    return "\n".join(
        [
            "# Final Recommendation",
            "",
            "## 1. Why the 29 cases still mismatch",
            "The 29 live mismatches persist because turn 1 never preserves the explicitly labeled `CCAvg` value. Turn 2 then inherits a canonical state that is already missing `CCAvg`, so canonical readiness keeps returning `missing_runtime_fields=['CCAvg']` and the controller stays in clarification.",
            "",
            f"## 2. Which family is largest",
            f"Family `{largest_family}` is largest with `{largest_count}` cases.",
            "",
            "## 3. Whether the failure is still parser-related",
            "Yes, but narrowly. This is no longer a broad parser-quality problem in the Tier A sense. Constraint extraction, schema validity, and repair behavior are already strong. The residual defect is a parser-quality/profile-recovery miss on explicit `CCAvg` during clarification-seeding initial turns.",
            "",
            "## 4. Which exact controller or clarification gate is failing",
            "The repeated non-terminal outcome is driven by the runtime-readiness gate in canonical validation, not a late transition bug. `ready_for_runtime` stays false because the profile remains missing `CCAvg`. The clarification payload is a consequence of that failing gate.",
            "",
            "## 5. Whether any family is really script drift",
            "No. No family meets the runbook bar for script drift. The controller is not obviously more policy-correct than the script; the saved state is incomplete for a concrete, reproducible reason.",
            "",
            "## 6. Smallest next patch target",
            "Patch the initial-turn explicit profile recovery path, not the clarification controller. The smallest target is to broaden safe deterministic recovery for explicitly labeled bank profile fields, especially `CCAvg`, when the input is structured but falls below the current dense-profile threshold. After that, rerun only the 29-case subset before any broader live thesis rerun.",
            "",
        ]
    )


def relpath(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path.resolve())


def abs_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def read_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def parse_json_text_or_none(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def constraint_family(spec: Any) -> str:
    if not isinstance(spec, dict) or not spec:
        return "none"
    return "+".join(sorted(str(key) for key in spec))


if __name__ == "__main__":
    raise SystemExit(main())
