#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.src.conversation.parser_adapter import DEFAULT_BENCHMARK_PATH, LiveLmStudioParserAdapter
from llm.src.part2_eval.annotation_scoring import score_initial_constraint_cases
from llm.src.part2_eval.common import (
    add_summary_output_args,
    apply_script_mismatch_validation,
    build_script_mismatch_summary,
    build_in_process_service,
    build_runner_command,
    build_session_detail_payload,
    call_with_legacy_stdout_redirect,
    counter_dict,
    get_dataset_entry,
    prepare_run_layout,
    progress_iter,
    replay_scripted_session_case,
    recompute_and_validate_aggregates,
    safe_mean,
    sha256_json_payload,
    summarize_latency_ms,
    write_optional_summary_outputs,
)
from llm.src.part2_eval.corpora import (
    TIER_A_ANNOTATION_SCHEMA_PATH,
    TIER_A_SCORER_OUTPUT_SCHEMA_PATH,
    TIER_A_CORPUS_PATH,
    TIER_B_CORPUS_PATH,
    load_tier_a_annotation_corpus,
    load_tier_b_bank_corpus,
)
from llm.src.phase3.phase3_2_catalog import DEFAULT_CATALOG_PATH, load_catalog
from llm.src.product.config import ProductConfig
from llm.src.runtime.constraint_spec import effective_blocked_fields
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.utils.hashing import sha256_file
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "part2_thesis_metrics"
RUNNER_SCOPE = "part2_g1_g2_system_eval"
SCORER_VERSION = "part2_thesis_metrics_report_v1"
SCOPE_NOTE = (
    "This runner is the authoritative system-facing G1/G2 report. Parser-internal low-level metrics remain "
    "separately derived from Tier A annotation scoring and parser artifacts."
)


def parse_args() -> argparse.Namespace:
    product_config = ProductConfig.load()
    parser = argparse.ArgumentParser(description="Generate the Part II G1/G2 thesis metrics report.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--lm-studio-api-base", default=product_config.lm_studio_api_base)
    parser.add_argument("--model-alias", default=product_config.model_alias)
    parser.add_argument("--product-mode", default=product_config.product_mode)
    parser.add_argument("--api-version", default=product_config.api_version)
    parser.add_argument("--app-version", default=product_config.app_version)
    parser.add_argument("--tier-a-corpus", type=Path, default=TIER_A_CORPUS_PATH)
    parser.add_argument("--tier-b-corpus", type=Path, default=TIER_B_CORPUS_PATH)
    parser.add_argument("--no-progress", action="store_true")
    add_summary_output_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_runner_command(Path(__file__).resolve())
    summary = run_thesis_metrics_report(args=args, command=command)
    markdown = render_markdown(summary)
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


def run_thesis_metrics_report(*, args: argparse.Namespace, command: str) -> dict[str, Any]:
    baseline_catalog = load_catalog(args.baseline_catalog)
    model_registry = ModelRegistry()
    policy_registry = PolicyRegistry(model_registry)
    bank_bundle = model_registry.get_bundle("bank")
    bank_policy = policy_registry.get_policy("bank")

    tier_a_corpus = load_tier_a_annotation_corpus(args.tier_a_corpus)
    initial_cases = [case for case in tier_a_corpus["cases"] if case["annotation_type"] == "initial_constraint_spec"]
    parser_adapter = LiveLmStudioParserAdapter(
        model_alias=args.model_alias,
        api_base=args.lm_studio_api_base,
        benchmark_path=args.benchmark,
    )
    benchmark = parser_adapter.load_benchmark()
    tier_a_summary = score_initial_constraint_cases(
        parser_adapter=parser_adapter,
        benchmark=benchmark,
        cases=initial_cases,
        progress_enabled=not args.no_progress,
    )

    del bank_bundle
    del bank_policy
    full_tier_b_corpus = load_tier_b_bank_corpus(args.tier_b_corpus)
    g1g2_cases = [case for case in full_tier_b_corpus["cases"] if case["group"] in {"G1", "G2"}]
    g1g2_scope = {
        "scope": "tier_b_groups:G1,G2",
        "evaluated_case_count": len(g1g2_cases),
        "evaluated_group_counts": {
            "G1": sum(1 for case in g1g2_cases if case["group"] == "G1"),
            "G2": sum(1 for case in g1g2_cases if case["group"] == "G2"),
        },
    }
    g1g2_scope["scope_sha256"] = sha256_json_payload(g1g2_scope)

    run_id = "part2_thesis_metrics_" + local_now_compact()
    layout = prepare_run_layout(out_dir=args.out_dir, run_id=run_id)
    handle = build_in_process_service(
        layout=layout,
        lm_studio_api_base=args.lm_studio_api_base,
        model_alias=args.model_alias,
        product_mode=args.product_mode,
        api_version=args.api_version,
        app_version=args.app_version,
        benchmark_path=args.benchmark,
    )

    version_payload, dataset_entry, session_results = collect_thesis_sessions(
        handle=handle,
        cases=g1g2_cases,
        dataset_key="bank",
        feature_ranges=build_feature_ranges(model_registry.get_bundle("bank").dataset_df, model_registry.get_bundle("bank").feature_order),
        progress_enabled=not args.no_progress,
    )
    summary = build_thesis_summary(
        run_id=run_id,
        command=command,
        run_root=layout["run_root"],
        handle=handle,
        baseline_catalog=baseline_catalog,
        version_payload=version_payload,
        dataset_entry=dataset_entry,
        tier_a_corpus=tier_a_corpus,
        tier_a_corpus_path=args.tier_a_corpus,
        tier_a_summary=tier_a_summary,
        full_tier_b_corpus=full_tier_b_corpus,
        tier_b_corpus_path=args.tier_b_corpus,
        g1g2_scope=g1g2_scope,
        session_results=session_results,
        benchmark_path=args.benchmark,
    )
    write_json(layout["run_root"] / "thesis_metrics_report.json", summary)
    (layout["run_root"] / "thesis_metrics_report.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def collect_thesis_sessions(
    *,
    handle,
    cases: list[dict[str, Any]],
    dataset_key: str,
    feature_ranges: dict[str, tuple[float, float]],
    progress_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    version_payload = call_with_legacy_stdout_redirect(handle.service.version)
    dataset_catalog = call_with_legacy_stdout_redirect(handle.service.list_dataset_catalog)
    dataset_entry = get_dataset_entry(dataset_catalog, dataset_key=dataset_key)

    results: list[dict[str, Any]] = []
    for case in progress_iter(cases, enabled=progress_enabled, desc="G1/G2 sessions", unit="session"):
        replay = replay_scripted_session_case(handle=handle, case=case)
        row = build_session_result(
            case=case,
            session_id=replay["session_id"],
            turn_payloads=replay["turn_payloads"],
            session_detail=replay["session_detail"],
            dataset_entry=dataset_entry,
            feature_ranges=feature_ranges,
        )
        row.update(
            {
                "scripted_turn_count": replay["scripted_turn_count"],
                "executed_turn_count": replay["executed_turn_count"],
                "script_execution_status": replay["script_execution_status"],
                "failed_turn_index": replay["failed_turn_index"],
                "script_mismatch_reason": replay["script_mismatch_reason"],
                "premature_terminal_state": replay["premature_terminal_state"],
                "premature_case_completion_reason": replay["premature_case_completion_reason"],
            }
        )
        results.append(row)
    return version_payload, dataset_entry, results


def build_session_result(
    *,
    case: dict[str, Any],
    session_id: str,
    turn_payloads: list[dict[str, Any]],
    session_detail: dict[str, Any],
    dataset_entry: dict[str, Any],
    feature_ranges: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    final_turn = turn_payloads[-1] if turn_payloads else {
        "public_state": session_detail.get("current_public_state"),
        "is_case_complete": session_detail.get("is_case_complete"),
        "case_completion_reason": session_detail.get("case_completion_reason"),
        "explanation_payload": {},
        "debug_summary": {},
    }
    public_state = str(final_turn["public_state"])
    case_completion_reason = final_turn.get("case_completion_reason")
    explanation_payload = final_turn.get("explanation_payload") or {}
    debug_summary = final_turn.get("debug_summary") or {}
    runtime_summary = debug_summary.get("runtime_summary") or {}
    summary_type = explanation_payload.get("summary_type")
    active_constraint_spec = session_detail.get("active_constraint_spec") or {}
    counterfactual_summary = explanation_payload.get("counterfactual_summary") or {}
    feature_order = [str(item) for item in dataset_entry.get("full_feature_list", []) if isinstance(item, str)]
    if not feature_order:
        feature_order = list(case["seed_profile"].keys())
    changed_fields = [str(item) for item in counterfactual_summary.get("changed_fields", []) if isinstance(item, str)]
    merge_followup_turns = [
        turn_payloads[index]
        for index in range(1, len(turn_payloads))
        if turn_payloads[index - 1]["public_state"] == "NEEDS_CLARIFICATION"
    ]
    reject_class = classify_reject_class(
        public_state=public_state,
        case_completion_reason=case_completion_reason,
        explanation_payload=explanation_payload,
        debug_summary=debug_summary,
    )
    g2_applicable = (
        case["group"] == "G2"
        and public_state == "RUNTIME_SUCCESS"
        and summary_type == "counterfactual_found"
    )
    proximity = None
    sparsity = None
    if g2_applicable and counterfactual_summary.get("profile"):
        proximity = compute_normalized_proximity(
            factual_profile=case["seed_profile"],
            counterfactual_profile=counterfactual_summary["profile"],
            feature_ranges=feature_ranges,
            feature_order=feature_order,
        )
        sparsity = len(changed_fields)
    return {
        "case_id": case["case_id"],
        "group": case["group"],
        "session_shape": case["session_shape"],
        "session_id": session_id,
        "turn_count": len(turn_payloads),
        "final_public_state": public_state,
        "is_case_complete": bool(final_turn["is_case_complete"]),
        "case_completion_reason": case_completion_reason,
        "summary_type": summary_type,
        "reject_class": reject_class,
        "active_constraint_spec": active_constraint_spec,
        "active_constraint_spec_expected": case.get("active_constraint_spec_expected"),
        "constraint_spec_expected_match": case.get("active_constraint_spec_expected") == active_constraint_spec
        if case["group"] == "G2"
        else None,
        "clarification_rounds": sum(1 for turn in turn_payloads if turn["public_state"] == "NEEDS_CLARIFICATION"),
        "had_clarification": any(turn["public_state"] == "NEEDS_CLARIFICATION" for turn in turn_payloads),
        "merge_followup_turns": len(merge_followup_turns),
        "merge_successes": sum(
            1 for turn in merge_followup_turns if (turn.get("debug_summary") or {}).get("merge_applied") is True
        ),
        "successful_resolution": summary_type in {"no_recourse_needed", "counterfactual_found"},
        "final_latency_ms": ((debug_summary.get("timing_metrics") or {}).get("end_to_end_latency_ms")),
        "runtime_latency_ms": ((debug_summary.get("timing_metrics") or {}).get("runtime_latency_ms")),
        "runtime_executed": bool(runtime_summary.get("executed")),
        "runtime_controller_state": runtime_summary.get("controller_state"),
        "invariant_validation_status": debug_summary.get("invariant_validation_status"),
        "g2_applicable": g2_applicable,
        "M8_validity": int(g2_applicable),
        "M9_proximity": proximity,
        "M10_sparsity": sparsity,
        "M11_actionability": None
        if not g2_applicable
        else compute_actionability(
            counterfactual_summary=counterfactual_summary,
            active_constraint_spec=active_constraint_spec,
            policy_f2change=[str(item) for item in dataset_entry.get("f2change", []) if isinstance(item, str)],
            feature_order=feature_order,
        ),
        "M12_plausibility": None if not g2_applicable else compute_plausibility(debug_summary=debug_summary),
        "M13_feasibility": None
        if not g2_applicable
        else compute_final_feasibility(
            public_state=public_state,
            summary_type=summary_type,
            debug_summary=debug_summary,
        ),
        "M14_constraint_satisfaction": None
        if not g2_applicable
        else compute_constraint_satisfaction(
            counterfactual_summary=counterfactual_summary,
            active_constraint_spec=active_constraint_spec,
            feature_order=feature_order,
        ),
        "M15_constraint_blocked": int(reject_class == "request_constraints_blocked") if case["group"] == "G2" else None,
    }


def build_feature_ranges(dataset_df, feature_order: list[str]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for feature_name in feature_order:
        series = dataset_df[feature_name]
        ranges[feature_name] = (float(series.min()), float(series.max()))
    return ranges


def compute_normalized_proximity(
    *,
    factual_profile: dict[str, Any],
    counterfactual_profile: dict[str, Any],
    feature_ranges: dict[str, tuple[float, float]],
    feature_order: list[str],
) -> float:
    squared_sum = 0.0
    for feature_name in feature_order:
        factual_value = float(factual_profile[feature_name])
        counterfactual_value = float(counterfactual_profile[feature_name])
        minimum, maximum = feature_ranges[feature_name]
        span = maximum - minimum
        if span <= 0.0:
            span = 1.0
        scaled_delta = (counterfactual_value - factual_value) / span
        squared_sum += scaled_delta * scaled_delta
    return round(squared_sum ** 0.5, 6)


def compute_actionability(
    *,
    counterfactual_summary: dict[str, Any],
    active_constraint_spec: dict[str, Any],
    policy_f2change: list[str],
    feature_order: list[str],
) -> int:
    changed_fields = [str(item) for item in counterfactual_summary.get("changed_fields", []) if isinstance(item, str)]
    if any(field not in policy_f2change for field in changed_fields):
        return 0
    return compute_constraint_satisfaction(
        counterfactual_summary=counterfactual_summary,
        active_constraint_spec=active_constraint_spec,
        feature_order=feature_order,
    )


def compute_constraint_satisfaction(
    *,
    counterfactual_summary: dict[str, Any],
    active_constraint_spec: dict[str, Any],
    feature_order: list[str],
) -> int:
    changed_fields = [str(item) for item in counterfactual_summary.get("changed_fields", []) if isinstance(item, str)]
    blocked_fields = set(effective_blocked_fields(active_constraint_spec, feature_order=feature_order))
    if any(field in blocked_fields for field in changed_fields):
        return 0
    max_changed_features = active_constraint_spec.get("max_changed_features")
    if isinstance(max_changed_features, int) and len(changed_fields) > max_changed_features:
        return 0
    numeric_bounds = active_constraint_spec.get("numeric_bounds") or {}
    profile = counterfactual_summary.get("profile") or {}
    for field_name, bounds in numeric_bounds.items():
        value = profile.get(field_name)
        if not isinstance(value, (int, float)):
            return 0
        if "min" in bounds and float(value) < float(bounds["min"]):
            return 0
        if "max" in bounds and float(value) > float(bounds["max"]):
            return 0
    return 1


def compute_plausibility(*, debug_summary: dict[str, Any]) -> int:
    return 1 if debug_summary.get("invariant_validation_status") == "passed" else 0


def compute_final_feasibility(*, public_state: str, summary_type: str | None, debug_summary: dict[str, Any]) -> int:
    runtime_summary = debug_summary.get("runtime_summary") or {}
    return int(
        public_state == "RUNTIME_SUCCESS"
        and summary_type == "counterfactual_found"
        and runtime_summary.get("executed") is True
        and runtime_summary.get("controller_state") == "TERMINAL_SUCCESS"
        and debug_summary.get("invariant_validation_status") == "passed"
    )


def classify_reject_class(
    *,
    public_state: str,
    case_completion_reason: str | None,
    explanation_payload: dict[str, Any],
    debug_summary: dict[str, Any],
) -> str | None:
    if public_state == "RUNTIME_REJECT":
        runtime_summary = debug_summary.get("runtime_summary") or {}
        reason_codes = list(explanation_payload.get("reason_codes") or runtime_summary.get("reason_codes") or [])
        if "REQUEST_CONSTRAINTS_BLOCKED" in reason_codes:
            return "request_constraints_blocked"
        if "NO_FEASIBLE_CF_FOUND" in reason_codes:
            return "no_feasible_cf"
        if "INVALID_COUNTERFACTUAL_BLOCKED" in reason_codes:
            return "invariant_blocked"
        return "system_error"
    if case_completion_reason == "clarification_limit_reached":
        return "clarification_limit_reached"
    if public_state == "CONFLICT" or case_completion_reason == "conflict":
        return "conflict"
    if public_state == "UNSUPPORTED_REQUEST" or case_completion_reason == "unsupported_request":
        return "unsupported_request"
    if public_state == "PARSER_FAILURE" or case_completion_reason == "parser_failure":
        return "parser_failure"
    return None


def compute_thesis_aggregate_blocks(session_results: list[dict[str, Any]]) -> dict[str, Any]:
    g1_cases = [item for item in session_results if item["group"] == "G1"]
    g2_cases = [item for item in session_results if item["group"] == "G2"]
    total_sessions = len(session_results)
    completed_sessions = sum(1 for item in session_results if item["is_case_complete"])
    successful_sessions = sum(1 for item in session_results if item["successful_resolution"])
    clarification_sessions = [item for item in session_results if item["had_clarification"]]
    conflict_sessions = sum(1 for item in session_results if item["reject_class"] == "conflict")
    unsupported_sessions = sum(1 for item in session_results if item["reject_class"] == "unsupported_request")
    clarification_exhausted = sum(
        1 for item in session_results if item["case_completion_reason"] == "clarification_limit_reached"
    )
    clarification_rounds = sum(item["clarification_rounds"] for item in clarification_sessions)
    merge_followup_turns = sum(item["merge_followup_turns"] for item in session_results)
    merge_successes = sum(item["merge_successes"] for item in session_results)
    g2_applicable = [item for item in g2_cases if item["g2_applicable"]]
    g2_applicable_denominator = len(g2_applicable)
    g2_constraint_total = len(g2_cases)
    return {
        "g1_metrics": {
            "session_count": len(g1_cases),
            "public_state_counts": counter_dict(item["final_public_state"] for item in g1_cases),
            "case_completion_reason_counts": counter_dict(
                item["case_completion_reason"] for item in g1_cases if item["case_completion_reason"] is not None
            ),
        },
        "g2_metrics": {
            "session_count": len(g2_cases),
            "public_state_counts": counter_dict(item["final_public_state"] for item in g2_cases),
            "case_completion_reason_counts": counter_dict(
                item["case_completion_reason"] for item in g2_cases if item["case_completion_reason"] is not None
            ),
            "M8_validity_success_rate": {
                "numerator": sum(int(item["M8_validity"]) for item in g2_applicable),
                "denominator": g2_applicable_denominator,
                "mean": safe_mean(sum(int(item["M8_validity"]) for item in g2_applicable), g2_applicable_denominator),
            },
            "M9_proximity": summarize_latency_ms([item["M9_proximity"] for item in g2_applicable]),
            "M10_sparsity": summarize_latency_ms([item["M10_sparsity"] for item in g2_applicable]),
            "M11_actionability": {
                "numerator": sum(int(item["M11_actionability"]) for item in g2_applicable),
                "denominator": g2_applicable_denominator,
                "mean": safe_mean(
                    sum(int(item["M11_actionability"]) for item in g2_applicable),
                    g2_applicable_denominator,
                ),
                "formula": (
                    "Applicable G2 exposed counterfactuals whose changed fields stay within policy-allowed changeable "
                    "features and obey active blocked fields, max_changed_features, and numeric_bounds, divided by "
                    "applicable G2 exposed counterfactuals."
                ),
            },
            "M12_plausibility": {
                "numerator": sum(int(item["M12_plausibility"]) for item in g2_applicable),
                "denominator": g2_applicable_denominator,
                "mean": safe_mean(
                    sum(int(item["M12_plausibility"]) for item in g2_applicable),
                    g2_applicable_denominator,
                ),
                "formula": (
                    "Applicable G2 exposed counterfactuals with passed invariant validation, divided by applicable G2 "
                    "exposed counterfactuals."
                ),
            },
            "M13_feasibility": {
                "numerator": sum(int(item["M13_feasibility"]) for item in g2_applicable),
                "denominator": g2_applicable_denominator,
                "mean": safe_mean(
                    sum(int(item["M13_feasibility"]) for item in g2_applicable),
                    g2_applicable_denominator,
                ),
                "formula": (
                    "Applicable G2 exposed counterfactuals that survive the full system pipeline and are emitted as "
                    "the final accepted recommendation after runtime execution and post-runtime validation, divided by "
                    "applicable G2 exposed counterfactuals."
                ),
            },
            "M14_constraint_satisfaction": {
                "numerator": sum(int(item["M14_constraint_satisfaction"]) for item in g2_applicable),
                "denominator": g2_applicable_denominator,
                "mean": safe_mean(
                    sum(int(item["M14_constraint_satisfaction"]) for item in g2_applicable),
                    g2_applicable_denominator,
                ),
            },
            "M15_constraint_blocked_rate": {
                "numerator": sum(
                    int(item["M15_constraint_blocked"]) for item in g2_cases if item["M15_constraint_blocked"] is not None
                ),
                "denominator": g2_constraint_total,
                "mean": safe_mean(
                    sum(
                        int(item["M15_constraint_blocked"])
                        for item in g2_cases
                        if item["M15_constraint_blocked"] is not None
                    ),
                    g2_constraint_total,
                ),
            },
        },
        "system_metrics": {
            "M16_end_to_end_completion_rate": {
                "numerator": completed_sessions,
                "denominator": total_sessions,
                "mean": safe_mean(completed_sessions, total_sessions),
            },
            "M17_successful_recourse_resolution_rate": {
                "numerator": successful_sessions,
                "denominator": total_sessions,
                "mean": safe_mean(successful_sessions, total_sessions),
            },
            "M18_clarification_rate": {
                "numerator": len(clarification_sessions),
                "denominator": total_sessions,
                "mean": safe_mean(len(clarification_sessions), total_sessions),
            },
            "M19_average_clarification_rounds": {
                "numerator": clarification_rounds,
                "denominator": len(clarification_sessions),
                "mean": safe_mean(clarification_rounds, len(clarification_sessions)),
            },
            "M20_merge_success_rate": {
                "numerator": merge_successes,
                "denominator": merge_followup_turns,
                "mean": safe_mean(merge_successes, merge_followup_turns),
            },
            "M21_conflict_rate": {
                "numerator": conflict_sessions,
                "denominator": total_sessions,
                "mean": safe_mean(conflict_sessions, total_sessions),
            },
            "M22_unsupported_request_rate": {
                "numerator": unsupported_sessions,
                "denominator": total_sessions,
                "mean": safe_mean(unsupported_sessions, total_sessions),
            },
            "M23_clarification_exhaustion_rate": {
                "numerator": clarification_exhausted,
                "denominator": total_sessions,
                "mean": safe_mean(clarification_exhausted, total_sessions),
            },
            "M24_average_turns_to_terminal_state": {
                "numerator": sum(item["turn_count"] for item in session_results),
                "denominator": total_sessions,
                "mean": safe_mean(sum(item["turn_count"] for item in session_results), total_sessions),
            },
            "M25_end_to_end_latency_ms": summarize_latency_ms([item["final_latency_ms"] for item in session_results]),
            "public_state_counts": counter_dict(item["final_public_state"] for item in session_results),
            "case_completion_reason_counts": counter_dict(
                item["case_completion_reason"] for item in session_results if item["case_completion_reason"] is not None
            ),
            "reject_class_counts": counter_dict(
                item["reject_class"] for item in session_results if item["reject_class"] is not None
            ),
        },
    }


def build_thesis_summary(
    *,
    run_id: str,
    command: str,
    run_root: Path,
    handle,
    baseline_catalog,
    version_payload: dict[str, Any],
    dataset_entry: dict[str, Any],
    tier_a_corpus: dict[str, Any],
    tier_a_corpus_path: Path,
    tier_a_summary: dict[str, Any],
    full_tier_b_corpus: dict[str, Any],
    tier_b_corpus_path: Path,
    g1g2_scope: dict[str, Any],
    session_results: list[dict[str, Any]],
    benchmark_path: Path,
) -> dict[str, Any]:
    baseline_catalog_sha256 = sha256_file(baseline_catalog.source_path)
    aggregate_blocks = compute_thesis_aggregate_blocks(session_results)
    summary = {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "scorer_version": SCORER_VERSION,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "command": command,
        "execution_mode": handle.execution_mode,
        "service_command": handle.service_command,
        "service_base_url": handle.base_url,
        "catalog_version": baseline_catalog.catalog_version,
        "catalog_path": str(baseline_catalog.source_path),
        "catalog_sha256": baseline_catalog_sha256,
        "git_commit": version_payload.get("git_commit"),
        "corpus_path": str(Path(tier_b_corpus_path).resolve()),
        "corpus_version": full_tier_b_corpus["corpus_version"],
        "corpus_sha256": full_tier_b_corpus["corpus_sha256"],
        "report_json_path": str((run_root / "thesis_metrics_report.json").resolve()),
        "report_markdown_path": str((run_root / "thesis_metrics_report.md").resolve()),
        "isolated_run": False,
        "sqlite_path": str(handle.sqlite_path),
        "artifact_root": str(handle.artifact_root),
        "loaded_corpora": {
            "tier_a": {
                "corpus_path": str(Path(tier_a_corpus_path).resolve()),
                "corpus_version": tier_a_corpus["corpus_version"],
                "corpus_sha256": tier_a_corpus["corpus_sha256"],
            },
            "tier_b": {
                "corpus_path": str(Path(tier_b_corpus_path).resolve()),
                "corpus_version": full_tier_b_corpus["corpus_version"],
                "corpus_sha256": full_tier_b_corpus["corpus_sha256"],
            },
        },
        "provenance": {
            "runner_scope": RUNNER_SCOPE,
            "scope_note": SCOPE_NOTE,
            "scorer_version": SCORER_VERSION,
            "timestamp_local": local_now_iso(),
            "timezone": "UTC+07:00",
            "command": command,
            "execution_mode": handle.execution_mode,
            "catalog_version": baseline_catalog.catalog_version,
            "catalog_sha256": baseline_catalog_sha256,
            "catalog_created_timestamp_utc": baseline_catalog.created_timestamp_utc,
            "corpus_path": str(Path(tier_b_corpus_path).resolve()),
            "corpus_version": full_tier_b_corpus["corpus_version"],
            "corpus_sha256": full_tier_b_corpus["corpus_sha256"],
            "tier_a_corpus_version": tier_a_corpus["corpus_version"],
            "tier_a_corpus_sha256": tier_a_corpus["corpus_sha256"],
            "tier_a_corpus_path": str(Path(tier_a_corpus_path).resolve()),
            "dataset_key": dataset_entry["dataset_key"],
            "api_version": version_payload.get("api_version"),
            "app_version": version_payload.get("app_version"),
            "model_alias": version_payload.get("model_alias"),
            "runtime_mode": version_payload.get("runtime_mode"),
            "git_commit": version_payload.get("git_commit"),
            "lm_studio_api_base": handle.config.lm_studio_api_base,
            "benchmark_path": str(Path(benchmark_path).resolve()),
        },
        "frozen_inputs": {
            "tier_a_annotation_schema_version": tier_a_corpus["annotation_schema_version"],
            "tier_a_annotation_schema_sha256": sha256_file(TIER_A_ANNOTATION_SCHEMA_PATH),
            "tier_a_scorer_output_schema_version": tier_a_corpus["scorer_output_schema_version"],
            "tier_a_scorer_output_schema_sha256": sha256_file(TIER_A_SCORER_OUTPUT_SCHEMA_PATH),
            "tier_a_case_count": tier_a_corpus["case_count"],
            "tier_b_available_corpus_version": full_tier_b_corpus["corpus_version"],
            "tier_b_available_corpus_sha256": full_tier_b_corpus["corpus_sha256"],
            "tier_b_available_corpus_path": str(Path(tier_b_corpus_path).resolve()),
            "tier_b_available_group_counts": dict(full_tier_b_corpus["group_counts"]),
            "g1g2_scope": g1g2_scope,
        },
        "tier_a_parser_fidelity": tier_a_summary,
        "g1_metrics": aggregate_blocks["g1_metrics"],
        "g2_metrics": aggregate_blocks["g2_metrics"],
        "system_metrics": aggregate_blocks["system_metrics"],
        "per_case_results": session_results,
    }
    summary["aggregate_validation"] = recompute_and_validate_aggregates(
        expected_blocks={
            "g1_metrics": summary["g1_metrics"],
            "g2_metrics": summary["g2_metrics"],
            "system_metrics": summary["system_metrics"],
        },
        recomputed_blocks=aggregate_blocks,
    )
    summary["script_mismatch_summary"] = build_script_mismatch_summary(session_results)
    summary["aggregate_validation"] = apply_script_mismatch_validation(
        summary["aggregate_validation"],
        script_mismatch_summary=summary["script_mismatch_summary"],
    )
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Part II Thesis Metrics Report",
        "",
        "## Provenance",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- runner_scope: `{summary['runner_scope']}`",
        f"- scorer_version: `{summary['scorer_version']}`",
        f"- timestamp_local: `{summary['timestamp_local']}`",
        f"- timezone: `{summary['timezone']}`",
        f"- execution_mode: `{summary['execution_mode']}`",
        f"- corpus_version: `{summary['corpus_version']}`",
        f"- corpus_sha256: `{summary['corpus_sha256']}`",
        f"- catalog_version: `{summary['catalog_version']}`",
        f"- catalog_sha256: `{summary['catalog_sha256']}`",
        f"- git_commit: `{summary['git_commit']}`",
        "",
        "## Scope",
        "",
        SCOPE_NOTE,
        "",
        "## Frozen Inputs",
        "",
        f"- Tier A annotation schema: `{summary['frozen_inputs']['tier_a_annotation_schema_version']}`",
        f"- Tier A annotation schema sha256: `{summary['frozen_inputs']['tier_a_annotation_schema_sha256']}`",
        f"- Tier A scorer output schema: `{summary['frozen_inputs']['tier_a_scorer_output_schema_version']}`",
        f"- Tier A scorer output schema sha256: `{summary['frozen_inputs']['tier_a_scorer_output_schema_sha256']}`",
        f"- Tier A case count: `{summary['frozen_inputs']['tier_a_case_count']}`",
        f"- Tier B available group counts: `{summary['frozen_inputs']['tier_b_available_group_counts']}`",
        f"- Tier B evaluated G1/G2 scope: `{summary['frozen_inputs']['g1g2_scope']}`",
        "",
        "## Tier A",
        "",
        f"- M6 exact_match_accuracy: `{summary['tier_a_parser_fidelity']['M6_constraint_extraction_fidelity']['mean']}`",
        f"- M6 component_mean: `{summary['tier_a_parser_fidelity']['M6_constraint_extraction_fidelity']['component_mean']}`",
        f"- M1 JSON validity: `{summary['tier_a_parser_fidelity']['M1_json_validity_rate']['mean']}`",
        f"- M2 schema compliance: `{summary['tier_a_parser_fidelity']['M2_schema_compliance_rate']['mean']}`",
        f"- M3 canonical validation pass: `{summary['tier_a_parser_fidelity']['M3_canonical_validation_pass_rate']['mean']}`",
        f"- M4 repair rate: `{summary['tier_a_parser_fidelity']['M4_repair_rate']['mean']}`",
        f"- M5 final parser failure rate: `{summary['tier_a_parser_fidelity']['M5_final_parser_failure_rate']['mean']}`",
        f"- M7 parser latency ms: `{summary['tier_a_parser_fidelity']['M7_parser_latency_ms']}`",
        "",
        "## G1",
        "",
        f"- session_count: `{summary['g1_metrics']['session_count']}`",
        f"- public_state_counts: `{summary['g1_metrics']['public_state_counts']}`",
        f"- case_completion_reason_counts: `{summary['g1_metrics']['case_completion_reason_counts']}`",
        "",
        "## G2",
        "",
        "- `M11 Actionability`: applicable G2 exposed counterfactuals whose changed fields stay within policy-allowed changeable features and obey active blocked fields, `max_changed_features`, and `numeric_bounds`, divided by applicable G2 exposed counterfactuals.",
        "- `M12 Plausibility`: applicable G2 exposed counterfactuals with passed invariant validation, divided by applicable G2 exposed counterfactuals.",
        "- `M13 Feasibility`: applicable G2 exposed counterfactuals that survive the full system pipeline and are emitted as the final accepted recommendation after runtime execution and post-runtime validation, divided by applicable G2 exposed counterfactuals.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| `M8 validity/success` | `{summary['g2_metrics']['M8_validity_success_rate']['mean']}` |",
        f"| `M9 proximity` | `{summary['g2_metrics']['M9_proximity']}` |",
        f"| `M10 sparsity` | `{summary['g2_metrics']['M10_sparsity']}` |",
        f"| `M11 actionability` | `{summary['g2_metrics']['M11_actionability']['mean']}` |",
        f"| `M12 plausibility` | `{summary['g2_metrics']['M12_plausibility']['mean']}` |",
        f"| `M13 feasibility` | `{summary['g2_metrics']['M13_feasibility']['mean']}` |",
        f"| `M14 constraint satisfaction` | `{summary['g2_metrics']['M14_constraint_satisfaction']['mean']}` |",
        f"| `M15 constraint blocked rate` | `{summary['g2_metrics']['M15_constraint_blocked_rate']['mean']}` |",
        "",
        "## System Metrics",
        "",
        f"- M16 completion rate: `{summary['system_metrics']['M16_end_to_end_completion_rate']['mean']}`",
        f"- M17 successful recourse resolution rate: `{summary['system_metrics']['M17_successful_recourse_resolution_rate']['mean']}`",
        f"- M18 clarification rate: `{summary['system_metrics']['M18_clarification_rate']['mean']}`",
        f"- M19 average clarification rounds: `{summary['system_metrics']['M19_average_clarification_rounds']['mean']}`",
        f"- M20 merge success rate: `{summary['system_metrics']['M20_merge_success_rate']['mean']}`",
        f"- M21 conflict rate: `{summary['system_metrics']['M21_conflict_rate']['mean']}`",
        f"- M22 unsupported request rate: `{summary['system_metrics']['M22_unsupported_request_rate']['mean']}`",
        f"- M23 clarification exhaustion rate: `{summary['system_metrics']['M23_clarification_exhaustion_rate']['mean']}`",
        f"- M24 average turns to terminal state: `{summary['system_metrics']['M24_average_turns_to_terminal_state']['mean']}`",
        f"- M25 end-to-end latency ms: `{summary['system_metrics']['M25_end_to_end_latency_ms']}`",
        "",
        "## Aggregate Validation",
        "",
        f"- ok: `{summary['aggregate_validation']['ok']}`",
        f"- difference_count: `{summary['aggregate_validation']['difference_count']}`",
        "",
    ]
    script_mismatch_summary = summary.get("script_mismatch_summary") or {}
    lines.extend(
        [
            "## Script Mismatches",
            "",
            f"- count: `{script_mismatch_summary.get('count')}`",
            f"- reason_counts: `{script_mismatch_summary.get('reason_counts')}`",
            f"- case_identifiers: `{script_mismatch_summary.get('case_identifiers')}`",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
