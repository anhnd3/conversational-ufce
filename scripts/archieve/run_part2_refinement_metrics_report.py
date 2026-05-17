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
from llm.src.part2_eval.annotation_scoring import score_refinement_delta_cases
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
from llm.src.product.service import RefinementNotAllowedError
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.utils.hashing import sha256_file
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "part2_refinement_metrics"
RUNNER_SCOPE = "part2_level2_refinement_eval"
SCORER_VERSION = "part2_refinement_metrics_report_v1"
SCOPE_NOTE = (
    "This runner measures the existing bounded refinement subsystem as implemented Part II functionality. "
    "It does not treat Level 2 refinement as a greenfield feature build."
)


def parse_args() -> argparse.Namespace:
    product_config = ProductConfig.load()
    parser = argparse.ArgumentParser(description="Generate the Part II refinement metrics report.")
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
    summary = run_refinement_metrics_report(args=args, command=command)
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


def run_refinement_metrics_report(*, args: argparse.Namespace, command: str) -> dict[str, Any]:
    baseline_catalog = load_catalog(args.baseline_catalog)
    model_registry = ModelRegistry()
    policy_registry = PolicyRegistry(model_registry)
    bank_bundle = model_registry.get_bundle("bank")
    bank_policy = policy_registry.get_policy("bank")

    tier_a_corpus = load_tier_a_annotation_corpus(args.tier_a_corpus)
    refinement_cases = [case for case in tier_a_corpus["cases"] if case["annotation_type"] == "refinement_delta"]
    parser_adapter = LiveLmStudioParserAdapter(
        model_alias=args.model_alias,
        api_base=args.lm_studio_api_base,
        benchmark_path=args.benchmark,
    )
    benchmark = parser_adapter.load_benchmark()
    tier_a_summary = score_refinement_delta_cases(
        parser_adapter=parser_adapter,
        benchmark=benchmark,
        cases=refinement_cases,
        progress_enabled=not args.no_progress,
    )

    del bank_bundle
    del bank_policy
    full_tier_b_corpus = load_tier_b_bank_corpus(args.tier_b_corpus)
    live_refinement_cases = [case for case in full_tier_b_corpus["cases"] if case["group"] == "REFINEMENT"]
    refinement_corpus = {
        "scope": "tier_b_group:REFINEMENT",
        "case_count": len(live_refinement_cases),
    }
    refinement_corpus["corpus_sha256"] = sha256_json_payload(refinement_corpus)

    run_id = "part2_refinement_metrics_" + local_now_compact()
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
    version_payload, dataset_entry, live_results = collect_refinement_sessions(
        handle=handle,
        cases=live_refinement_cases,
        dataset_key="bank",
        progress_enabled=not args.no_progress,
    )
    summary = build_refinement_summary(
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
        tier_b_corpus=full_tier_b_corpus,
        tier_b_corpus_path=args.tier_b_corpus,
        refinement_corpus=refinement_corpus,
        live_results=live_results,
        benchmark_path=args.benchmark,
    )
    write_json(layout["run_root"] / "refinement_metrics_report.json", summary)
    (layout["run_root"] / "refinement_metrics_report.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def collect_refinement_sessions(
    *,
    handle,
    cases: list[dict[str, Any]],
    dataset_key: str,
    progress_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    version_payload = call_with_legacy_stdout_redirect(handle.service.version)
    dataset_catalog = call_with_legacy_stdout_redirect(handle.service.list_dataset_catalog)
    dataset_entry = get_dataset_entry(dataset_catalog, dataset_key=dataset_key)

    rows: list[dict[str, Any]] = []
    for case in progress_iter(cases, enabled=progress_enabled, desc="Refinement sessions", unit="session"):
        replay = replay_scripted_session_case(handle=handle, case=case)
        session_id = replay["session_id"]
        initial_turns = list(replay["turn_payloads"])
        prior_turn_payload = initial_turns[-1] if initial_turns else {
            "current_public_state": replay["session_detail"].get("current_public_state"),
            "case_completion_reason": replay["session_detail"].get("case_completion_reason"),
            "refinement_rounds_used": replay["session_detail"].get("refinement_rounds_used"),
            "refinement_round_limit": replay["session_detail"].get("refinement_round_limit"),
            "active_constraint_spec": replay["session_detail"].get("active_constraint_spec") or {},
            "restart_required": replay["session_detail"].get("restart_required"),
            "explanation_payload": {},
            "debug_summary": {},
        }

        refinement_rounds: list[dict[str, Any]] = []
        previous_payload = prior_turn_payload
        if replay["script_execution_status"] == "completed":
            for feedback_text in case.get("refinement_feedback", []):
                try:
                    stored_turn = call_with_legacy_stdout_redirect(handle.service.submit_refinement, session_id, feedback_text)
                    payload = handle.service.build_turn_response(stored_turn)
                    response_status_code = 200
                except RefinementNotAllowedError as exc:
                    payload = build_refinement_not_allowed_payload(previous_payload=previous_payload, detail=str(exc))
                    response_status_code = 409
                refinement_rounds.append(
                    build_refinement_round_result(
                        case=case,
                        session_id=session_id,
                        previous_payload=previous_payload,
                        feedback_text=feedback_text,
                        response_status_code=response_status_code,
                        response_payload=payload,
                    )
                )
                previous_payload = payload
                if response_status_code != 200:
                    break
        session_detail = build_session_detail_payload(handle.repository.get_session(session_id))
        rows.append(
            {
                "case_id": case["case_id"],
                "session_id": session_id,
                "initial_turn": prior_turn_payload,
                "initial_turns": initial_turns,
                "refinement_rounds": refinement_rounds,
                "session_detail": session_detail,
                "scripted_turn_count": replay["scripted_turn_count"],
                "executed_turn_count": replay["executed_turn_count"],
                "script_execution_status": replay["script_execution_status"],
                "failed_turn_index": replay["failed_turn_index"],
                "script_mismatch_reason": replay["script_mismatch_reason"],
                "premature_terminal_state": replay["premature_terminal_state"],
                "premature_case_completion_reason": replay["premature_case_completion_reason"],
            }
        )
    return version_payload, dataset_entry, rows


def build_refinement_not_allowed_payload(*, previous_payload: dict[str, Any], detail: str) -> dict[str, Any]:
    public_state = previous_payload.get("public_state") or previous_payload.get("current_public_state")
    explanation_payload = dict(previous_payload.get("explanation_payload") or {})
    debug_summary = dict(previous_payload.get("debug_summary") or {})
    timing_metrics = dict(debug_summary.get("timing_metrics") or {})
    return {
        "error_code": "refinement_not_allowed",
        "detail": detail,
        "current_public_state": public_state,
        "case_completion_reason": previous_payload.get("case_completion_reason"),
        "refinement_status": "not_allowed",
        "refinement_rounds_used": previous_payload.get("refinement_rounds_used"),
        "refinement_round_limit": previous_payload.get("refinement_round_limit"),
        "active_constraint_spec": previous_payload.get("active_constraint_spec") or {},
        "restart_required": previous_payload.get("restart_required"),
        "explanation_payload": explanation_payload,
        "debug_summary": {
            **debug_summary,
            "timing_metrics": timing_metrics,
        },
    }


def build_refinement_round_result(
    *,
    case: dict[str, Any],
    session_id: str,
    previous_payload: dict[str, Any],
    feedback_text: str,
    response_status_code: int,
    response_payload: dict[str, Any],
) -> dict[str, Any]:
    public_state = response_payload.get("public_state") or response_payload.get("current_public_state")
    summary_type = (response_payload.get("explanation_payload") or {}).get("summary_type")
    refinement_status = response_payload.get("refinement_status")
    reject_class = classify_reject_class(response_payload=response_payload)
    solution_changed = 0 if response_status_code != 200 else compare_solution_payloads(previous_payload, response_payload)
    debug_summary = response_payload.get("debug_summary") or {}
    timing_metrics = debug_summary.get("timing_metrics") or {}
    return {
        "case_id": case["case_id"],
        "session_id": session_id,
        "feedback_text": feedback_text,
        "http_status_code": response_status_code,
        "public_state": public_state,
        "summary_type": summary_type,
        "refinement_status": refinement_status,
        "reject_class": reject_class,
        "solution_changed": solution_changed,
        "refinement_rounds_used": response_payload.get("refinement_rounds_used"),
        "refinement_round_limit": response_payload.get("refinement_round_limit"),
        "active_constraint_spec": response_payload.get("active_constraint_spec") or {},
        "restart_required": response_payload.get("restart_required"),
        "refinement_latency_ms": timing_metrics.get("end_to_end_latency_ms"),
        "runtime_latency_ms": timing_metrics.get("runtime_latency_ms"),
    }


def compare_solution_payloads(previous_payload: dict[str, Any], current_payload: dict[str, Any]) -> int:
    previous_state = previous_payload.get("public_state") or previous_payload.get("current_public_state")
    current_state = current_payload.get("public_state") or current_payload.get("current_public_state")
    if previous_state != current_state:
        return 1
    previous_explanation = previous_payload.get("explanation_payload") or {}
    current_explanation = current_payload.get("explanation_payload") or {}
    if previous_explanation.get("summary_type") != current_explanation.get("summary_type"):
        return 1
    previous_cf = previous_explanation.get("counterfactual_summary") or {}
    current_cf = current_explanation.get("counterfactual_summary") or {}
    if previous_cf.get("profile") != current_cf.get("profile"):
        return 1
    return 0


def classify_reject_class(*, response_payload: dict[str, Any]) -> str | None:
    public_state = response_payload.get("public_state") or response_payload.get("current_public_state")
    case_completion_reason = response_payload.get("case_completion_reason")
    explanation_payload = response_payload.get("explanation_payload") or {}
    debug_summary = response_payload.get("debug_summary") or {}
    runtime_summary = debug_summary.get("runtime_summary") or {}
    reason_codes = list(explanation_payload.get("reason_codes") or runtime_summary.get("reason_codes") or [])
    if public_state == "RUNTIME_REJECT":
        if "REQUEST_CONSTRAINTS_BLOCKED" in reason_codes:
            return "request_constraints_blocked"
        if "INVALID_COUNTERFACTUAL_BLOCKED" in reason_codes:
            return "invariant_blocked"
        if "NO_FEASIBLE_CF_FOUND" in reason_codes:
            return "no_feasible_cf"
        return "system_error"
    if case_completion_reason == "clarification_limit_reached":
        return "clarification_limit_reached"
    return None


def flatten_refinement_rounds(live_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_row in live_results:
        for round_result in session_row["refinement_rounds"]:
            rows.append(dict(round_result))
    return rows


def compute_refinement_aggregate_blocks(
    live_results: list[dict[str, Any]],
    tier_a_summary: dict[str, Any],
) -> dict[str, Any]:
    all_rounds = flatten_refinement_rounds(live_results)
    total_rounds = len(all_rounds)
    applied_rounds = sum(
        1 for item in all_rounds if item["refinement_status"] == "applied" and item["http_status_code"] == 200
    )
    changed_rounds = sum(int(item["solution_changed"]) for item in all_rounds)
    blocked_rounds = sum(1 for item in all_rounds if item["reject_class"] == "request_constraints_blocked")
    status_counts = counter_dict(item["refinement_status"] for item in all_rounds if item["refinement_status"] is not None)
    final_rounds_used = [
        row["refinement_rounds"][-1]["refinement_rounds_used"]
        for row in live_results
        if row["refinement_rounds"] and isinstance(row["refinement_rounds"][-1]["refinement_rounds_used"], int)
    ]
    return {
        "refinement_metrics": {
            "round_count": total_rounds,
            "status_counts": status_counts,
            "reject_class_counts": counter_dict(item["reject_class"] for item in all_rounds if item["reject_class"] is not None),
            "M32_refinement_success_rate": {
                "numerator": applied_rounds,
                "denominator": total_rounds,
                "mean": safe_mean(applied_rounds, total_rounds),
            },
            "M33_solution_change_rate_after_feedback": {
                "numerator": changed_rounds,
                "denominator": total_rounds,
                "mean": safe_mean(changed_rounds, total_rounds),
            },
            "M34_constraint_induced_blocking_rate": {
                "numerator": blocked_rounds,
                "denominator": total_rounds,
                "mean": safe_mean(blocked_rounds, total_rounds),
            },
            "M35_average_refinement_rounds_to_stable_outcome": {
                "numerator": sum(final_rounds_used),
                "denominator": len(final_rounds_used),
                "mean": safe_mean(sum(final_rounds_used), len(final_rounds_used)),
            },
            "M36_constraint_delta_fidelity": tier_a_summary["M36_constraint_delta_fidelity"],
            "refinement_rounds_used_distribution": counter_dict(final_rounds_used),
            "initial_turn_latency_ms": summarize_latency_ms(
                [
                    ((row["initial_turn"].get("debug_summary") or {}).get("timing_metrics") or {}).get("end_to_end_latency_ms")
                    for row in live_results
                ]
            ),
            "refinement_round_latency_ms": summarize_latency_ms([item["refinement_latency_ms"] for item in all_rounds]),
        }
    }


def build_refinement_summary(
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
    tier_b_corpus: dict[str, Any],
    tier_b_corpus_path: Path,
    refinement_corpus: dict[str, Any],
    live_results: list[dict[str, Any]],
    benchmark_path: Path,
) -> dict[str, Any]:
    baseline_catalog_sha256 = sha256_file(baseline_catalog.source_path)
    aggregate_blocks = compute_refinement_aggregate_blocks(live_results, tier_a_summary)
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
        "corpus_version": tier_b_corpus["corpus_version"],
        "corpus_sha256": tier_b_corpus["corpus_sha256"],
        "report_json_path": str((run_root / "refinement_metrics_report.json").resolve()),
        "report_markdown_path": str((run_root / "refinement_metrics_report.md").resolve()),
        "loaded_corpora": {
            "tier_a": {
                "corpus_path": str(Path(tier_a_corpus_path).resolve()),
                "corpus_version": tier_a_corpus["corpus_version"],
                "corpus_sha256": tier_a_corpus["corpus_sha256"],
            },
            "tier_b": {
                "corpus_path": str(Path(tier_b_corpus_path).resolve()),
                "corpus_version": tier_b_corpus["corpus_version"],
                "corpus_sha256": tier_b_corpus["corpus_sha256"],
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
            "corpus_path": str(Path(tier_b_corpus_path).resolve()),
            "corpus_version": tier_b_corpus["corpus_version"],
            "corpus_sha256": tier_b_corpus["corpus_sha256"],
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
            "tier_b_corpus_path": str(Path(tier_b_corpus_path).resolve()),
            "refinement_scope": refinement_corpus,
        },
        "tier_a_delta_fidelity": tier_a_summary,
        "refinement_metrics": aggregate_blocks["refinement_metrics"],
        "per_round_results": flatten_refinement_rounds(live_results),
        "per_session_results": live_results,
    }
    summary["aggregate_validation"] = recompute_and_validate_aggregates(
        expected_blocks={"refinement_metrics": summary["refinement_metrics"]},
        recomputed_blocks=aggregate_blocks,
    )
    summary["script_mismatch_summary"] = build_script_mismatch_summary(live_results)
    summary["aggregate_validation"] = apply_script_mismatch_validation(
        summary["aggregate_validation"],
        script_mismatch_summary=summary["script_mismatch_summary"],
    )
    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Part II Refinement Metrics Report",
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
        "",
        "## Scope",
        "",
        SCOPE_NOTE,
        "",
        "## Tier A",
        "",
        f"- M36 exact_match_accuracy: `{summary['tier_a_delta_fidelity']['M36_constraint_delta_fidelity']['mean']}`",
        f"- M36 component_mean: `{summary['tier_a_delta_fidelity']['M36_constraint_delta_fidelity']['component_mean']}`",
        f"- Tier A annotation schema sha256: `{summary['frozen_inputs']['tier_a_annotation_schema_sha256']}`",
        f"- Tier A scorer output schema sha256: `{summary['frozen_inputs']['tier_a_scorer_output_schema_sha256']}`",
        "",
        "## Refinement Metrics",
        "",
        f"- status_counts: `{summary['refinement_metrics']['status_counts']}`",
        f"- reject_class_counts: `{summary['refinement_metrics']['reject_class_counts']}`",
        f"- M32 refinement success rate: `{summary['refinement_metrics']['M32_refinement_success_rate']['mean']}`",
        f"- M33 solution change rate after feedback: `{summary['refinement_metrics']['M33_solution_change_rate_after_feedback']['mean']}`",
        f"- M34 constraint-induced blocking rate: `{summary['refinement_metrics']['M34_constraint_induced_blocking_rate']['mean']}`",
        f"- M35 average refinement rounds to stable outcome: `{summary['refinement_metrics']['M35_average_refinement_rounds_to_stable_outcome']['mean']}`",
        f"- rounds_used_distribution: `{summary['refinement_metrics']['refinement_rounds_used_distribution']}`",
        f"- initial_turn_latency_ms: `{summary['refinement_metrics']['initial_turn_latency_ms']}`",
        f"- refinement_round_latency_ms: `{summary['refinement_metrics']['refinement_round_latency_ms']}`",
        "",
        "## Aggregate Validation",
        "",
        f"- ok: `{summary['aggregate_validation']['ok']}`",
        f"- difference_count: `{summary['aggregate_validation']['difference_count']}`",
        "",
        "## Script Mismatches",
        "",
        f"- count: `{(summary.get('script_mismatch_summary') or {}).get('count')}`",
        f"- reason_counts: `{(summary.get('script_mismatch_summary') or {}).get('reason_counts')}`",
        f"- case_identifiers: `{(summary.get('script_mismatch_summary') or {}).get('case_identifiers')}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
