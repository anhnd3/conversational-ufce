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

from llm.src.part2_eval.common import (
    add_summary_output_args,
    apply_script_mismatch_validation,
    build_script_mismatch_summary,
    build_in_process_service,
    build_runner_command,
    build_session_detail_payload,
    call_with_legacy_stdout_redirect,
    get_dataset_entry,
    prepare_run_layout,
    progress_iter,
    replay_scripted_session_case,
    recompute_and_validate_aggregates,
    safe_mean,
    summarize_latency_ms,
    write_optional_summary_outputs,
)
from llm.src.part2_eval.corpora import G5_AGENT_PORTABILITY_CORPUS_PATH, load_g5_agent_portability_corpus
from llm.src.phase3.phase3_2_catalog import DEFAULT_CATALOG_PATH, load_catalog
from llm.src.product.config import ProductConfig
from llm.src.runtime.constraint_spec import effective_blocked_fields
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.utils.hashing import sha256_file
from llm.src.utils.io import write_json
from llm.src.utils.time import local_now_compact, local_now_iso


DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "part2_agent_portability"
RUNNER_SCOPE = "part2_g5_agent_portability"
SCORER_VERSION = "part2_agent_portability_report_v1"
DEFAULT_BACKENDS = ("ufce", "dice", "ar")


def parse_args() -> argparse.Namespace:
    product_config = ProductConfig.load()
    parser = argparse.ArgumentParser(description="Generate the Part II G5 agent portability report.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--baseline-catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "llm_eval" / "benchmarks" / "ufce_bank_cf_parser_benchmark_v1.yaml",
    )
    parser.add_argument("--lm-studio-api-base", default=product_config.lm_studio_api_base)
    parser.add_argument("--model-alias", default=product_config.model_alias)
    parser.add_argument("--product-mode", default=product_config.product_mode)
    parser.add_argument("--api-version", default=product_config.api_version)
    parser.add_argument("--app-version", default=product_config.app_version)
    parser.add_argument("--g5-corpus", type=Path, default=G5_AGENT_PORTABILITY_CORPUS_PATH)
    parser.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    parser.add_argument("--attempts-per-case", type=int, default=3)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    add_summary_output_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = build_runner_command(Path(__file__).resolve())
    summary = run_agent_portability_report(args=args, command=command)
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


def run_agent_portability_report(*, args: argparse.Namespace, command: str) -> dict[str, Any]:
    if args.attempts_per_case <= 0:
        raise ValueError("--attempts-per-case must be positive")
    baseline_catalog = load_catalog(args.baseline_catalog)
    model_registry = ModelRegistry()
    policy_registry = PolicyRegistry(model_registry)
    bank_bundle = model_registry.get_bundle("bank")
    bank_policy = policy_registry.get_policy("bank")
    del bank_policy
    corpus = load_g5_agent_portability_corpus(args.g5_corpus)
    cases = list(corpus["cases"])
    if args.case_limit is not None:
        if args.case_limit <= 0:
            raise ValueError("--case-limit must be positive when provided")
        cases = cases[: args.case_limit]
    feature_ranges = build_feature_ranges(bank_bundle.dataset_df, list(bank_bundle.feature_order))
    backends = parse_backend_names(args.backends)

    run_id = "part2_agent_portability_" + local_now_compact()
    run_root = args.out_dir / run_id
    run_root.mkdir(parents=True, exist_ok=False)

    fairness_contract = {
        "dataset": "bank",
        "scope": "G5_agent_portability",
        "rules": [
            "same parser contracts and schemas",
            "same canonical validation",
            "same clarification and merge logic",
            "same constraint_spec semantics",
            "same invariant validator",
            "same explanation rendering",
            "same persistence and artifacts",
            "same public conversation states",
            "only backend generator changes",
        ],
    }

    backend_results: dict[str, Any] = {}
    comparison_rows: dict[str, Any] = {}
    validated_blocks: dict[str, Any] = {}
    version_payloads: dict[str, Any] = {}

    for backend_name in progress_iter(backends, enabled=not args.no_progress, desc="G5 backends", unit="backend"):
        layout = prepare_run_layout(out_dir=run_root / "backend_runs", run_id=backend_name)
        handle = build_in_process_service(
            layout=layout,
            lm_studio_api_base=args.lm_studio_api_base,
            model_alias=args.model_alias,
            product_mode=args.product_mode,
            api_version=args.api_version,
            app_version=args.app_version,
            benchmark_path=args.benchmark,
            counterfactual_backend_name=backend_name,
        )
        version_payload = call_with_legacy_stdout_redirect(handle.service.version)
        dataset_entry = get_dataset_entry(
            call_with_legacy_stdout_redirect(handle.service.list_dataset_catalog),
            dataset_key="bank",
        )
        case_attempts = collect_backend_attempts(
            handle=handle,
            cases=cases,
            feature_ranges=feature_ranges,
            dataset_entry=dataset_entry,
            attempts_per_case=args.attempts_per_case,
            progress_enabled=not args.no_progress,
            backend_name=backend_name,
        )
        aggregate = build_backend_aggregate(
            backend_name=backend_name,
            case_attempts=case_attempts,
        )
        backend_results[backend_name] = {
            "backend_name": backend_name,
            "execution_mode": handle.execution_mode,
            "artifact_root": str(handle.artifact_root),
            "sqlite_path": str(handle.sqlite_path),
            "version_payload": version_payload,
            "aggregate": aggregate,
            "case_attempts": case_attempts,
        }
        comparison_rows[f"agent + {backend_name.upper()}"] = aggregate["portability_row"]
        validated_blocks[backend_name] = aggregate["validated_aggregate"]
        version_payloads[backend_name] = version_payload

    summary = build_agent_portability_summary(
        run_id=run_id,
        command=command,
        run_root=run_root,
        baseline_catalog=baseline_catalog,
        corpus=corpus,
        g5_corpus_path=args.g5_corpus,
        cases=cases,
        backends=backends,
        attempts_per_case=args.attempts_per_case,
        fairness_contract=fairness_contract,
        backend_results=backend_results,
        comparison_rows=comparison_rows,
        validated_blocks=validated_blocks,
        benchmark_path=args.benchmark,
        version_payloads=version_payloads,
    )
    write_json(run_root / "agent_portability_report.json", summary)
    (run_root / "agent_portability_report.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def parse_backend_names(raw_value: str) -> list[str]:
    items = [item.strip().lower() for item in str(raw_value).split(",") if item.strip()]
    if not items:
        raise ValueError("At least one backend must be provided.")
    unsupported = sorted({item for item in items if item not in DEFAULT_BACKENDS})
    if unsupported:
        raise ValueError(f"Unsupported backend names: {', '.join(unsupported)}")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def collect_backend_attempts(
    *,
    handle,
    cases: list[dict[str, Any]],
    feature_ranges: dict[str, tuple[float, float]],
    dataset_entry: dict[str, Any],
    attempts_per_case: int,
    progress_enabled: bool,
    backend_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in progress_iter(cases, enabled=progress_enabled, desc=f"G5 {backend_name} cases", unit="case"):
        attempts: list[dict[str, Any]] = []
        for attempt_index in range(1, attempts_per_case + 1):
            replay = replay_scripted_session_case(handle=handle, case=case)
            attempt_row = build_attempt_result(
                backend_name=backend_name,
                case=case,
                attempt_index=attempt_index,
                session_id=replay["session_id"],
                turn_payloads=replay["turn_payloads"],
                session_detail=replay["session_detail"],
                dataset_entry=dataset_entry,
                feature_ranges=feature_ranges,
            )
            attempt_row.update(
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
            attempts.append(attempt_row)
        rows.append(
            {
                "case_id": case["case_id"],
                "group": case["group"],
                "session_shape": case["session_shape"],
                "backend_name": backend_name,
                "attempts": attempts,
                "stable": compute_attempt_stability(attempts),
                "reproducibility_signature_count": count_unique_signatures(attempts),
            }
        )
    return rows


def build_attempt_result(
    *,
    backend_name: str,
    case: dict[str, Any],
    attempt_index: int,
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
    reject_class = classify_reject_class(
        public_state=public_state,
        case_completion_reason=case_completion_reason,
        explanation_payload=explanation_payload,
        debug_summary=debug_summary,
    )
    counterfactual_summary = explanation_payload.get("counterfactual_summary") or {}
    changed_fields = [str(item) for item in counterfactual_summary.get("changed_fields", []) if isinstance(item, str)]
    applicable_counterfactual = public_state == "RUNTIME_SUCCESS" and summary_type == "counterfactual_found"
    proximity = None
    sparsity = None
    if applicable_counterfactual and counterfactual_summary.get("profile"):
        proximity = compute_normalized_proximity(
            factual_profile=case["seed_profile"],
            counterfactual_profile=counterfactual_summary["profile"],
            feature_ranges=feature_ranges,
            feature_order=[str(item) for item in dataset_entry.get("full_feature_list", []) if isinstance(item, str)],
        )
        sparsity = len(changed_fields)
    active_constraint_spec = session_detail.get("active_constraint_spec") or {}
    feature_order = [str(item) for item in dataset_entry.get("full_feature_list", []) if isinstance(item, str)]
    policy_f2change = [str(item) for item in dataset_entry.get("f2change", []) if isinstance(item, str)]
    return {
        "backend_name": backend_name,
        "case_id": case["case_id"],
        "group": case["group"],
        "session_shape": case["session_shape"],
        "attempt_index": attempt_index,
        "session_id": session_id,
        "turn_count": len(turn_payloads),
        "final_public_state": public_state,
        "is_case_complete": bool(final_turn["is_case_complete"]),
        "case_completion_reason": case_completion_reason,
        "summary_type": summary_type,
        "reject_class": reject_class,
        "active_constraint_spec": active_constraint_spec,
        "had_clarification": any(turn["public_state"] == "NEEDS_CLARIFICATION" for turn in turn_payloads),
        "clarification_rounds": sum(1 for turn in turn_payloads if turn["public_state"] == "NEEDS_CLARIFICATION"),
        "successful_resolution": summary_type in {"no_recourse_needed", "counterfactual_found"},
        "final_latency_ms": ((debug_summary.get("timing_metrics") or {}).get("end_to_end_latency_ms")),
        "runtime_latency_ms": ((debug_summary.get("timing_metrics") or {}).get("runtime_latency_ms")),
        "runtime_executed": bool(runtime_summary.get("executed")),
        "runtime_controller_state": runtime_summary.get("controller_state"),
        "counterfactual_profile": counterfactual_summary.get("profile"),
        "final_cf_validity": int(applicable_counterfactual),
        "applicable_counterfactual": applicable_counterfactual,
        "proximity": proximity,
        "sparsity": sparsity,
        "actionability": None
        if not applicable_counterfactual
        else compute_actionability(
            counterfactual_summary=counterfactual_summary,
            active_constraint_spec=active_constraint_spec,
            policy_f2change=policy_f2change,
            feature_order=feature_order,
        ),
        "plausibility": None if not applicable_counterfactual else compute_plausibility(debug_summary=debug_summary),
        "feasibility": None
        if not applicable_counterfactual
        else compute_feasibility(
            public_state=public_state,
            summary_type=summary_type,
            debug_summary=debug_summary,
        ),
        "constraint_blocked": int(reject_class == "request_constraints_blocked"),
    }


def build_backend_aggregate(*, backend_name: str, case_attempts: list[dict[str, Any]]) -> dict[str, Any]:
    representative_attempts = [row["attempts"][0] for row in case_attempts if row.get("attempts")]
    total_cases = len(representative_attempts)
    completed_count = sum(1 for row in representative_attempts if row["is_case_complete"])
    successful_resolution_count = sum(1 for row in representative_attempts if row["successful_resolution"])
    clarification_count = sum(1 for row in representative_attempts if row["had_clarification"])
    clarification_exhausted_count = sum(
        1 for row in representative_attempts if row["case_completion_reason"] == "clarification_limit_reached"
    )
    conflict_count = sum(1 for row in representative_attempts if row["reject_class"] == "conflict")
    unsupported_count = sum(1 for row in representative_attempts if row["reject_class"] == "unsupported_request")
    stable_count = sum(1 for row in case_attempts if row["stable"])
    applicable_counterfactuals = [row for row in representative_attempts if row["applicable_counterfactual"]]
    constrained_attempts = [row for row in representative_attempts if isinstance(row["active_constraint_spec"], dict) and row["active_constraint_spec"]]
    portability_row = {
        "completion_rate": build_binary_metric(completed_count, total_cases),
        "successful_recourse_resolution": build_binary_metric(successful_resolution_count, total_cases),
        "clarification_rate": build_binary_metric(clarification_count, total_cases),
        "average_turns": summarize_latency_ms([row["turn_count"] for row in representative_attempts]),
        "end_to_end_latency_ms": summarize_latency_ms([row["final_latency_ms"] for row in representative_attempts]),
        "reproducibility_stability": build_binary_metric(stable_count, len(case_attempts)),
        "final_cf_validity": build_binary_metric(
            sum(int(row["final_cf_validity"]) for row in representative_attempts),
            total_cases,
        ),
        "actionability": build_optional_binary_metric("actionability", applicable_counterfactuals),
        "plausibility": build_optional_binary_metric("plausibility", applicable_counterfactuals),
        "feasibility": build_optional_binary_metric("feasibility", applicable_counterfactuals),
        "constraint_blocked_rate": build_binary_metric(
            sum(int(row["constraint_blocked"]) for row in constrained_attempts),
            len(constrained_attempts),
        ),
    }
    validated_aggregate = {
        "system_metrics": {
            "completion_rate": portability_row["completion_rate"],
            "successful_recourse_resolution": portability_row["successful_recourse_resolution"],
            "clarification_rate": portability_row["clarification_rate"],
            "clarification_exhaustion_rate": build_binary_metric(clarification_exhausted_count, total_cases),
            "conflict_rate": build_binary_metric(conflict_count, total_cases),
            "unsupported_request_rate": build_binary_metric(unsupported_count, total_cases),
            "average_turns_to_terminal_state": portability_row["average_turns"],
            "end_to_end_latency_ms": portability_row["end_to_end_latency_ms"],
            "reproducibility_stability": portability_row["reproducibility_stability"],
        },
        "recommendation_metrics": {
            "final_cf_validity": portability_row["final_cf_validity"],
            "actionability": portability_row["actionability"],
            "plausibility": portability_row["plausibility"],
            "feasibility": portability_row["feasibility"],
            "proximity": {
                **summarize_latency_ms([row["proximity"] for row in applicable_counterfactuals]),
                "denominator": len(applicable_counterfactuals),
            },
            "sparsity": {
                **summarize_latency_ms([row["sparsity"] for row in applicable_counterfactuals]),
                "denominator": len(applicable_counterfactuals),
            },
            "constraint_blocked_rate": portability_row["constraint_blocked_rate"],
        },
    }
    return {
        "backend_name": backend_name,
        "portability_row": portability_row,
        "validated_aggregate": validated_aggregate,
    }


def build_agent_portability_summary(
    *,
    run_id: str,
    command: str,
    run_root: Path,
    baseline_catalog,
    corpus: dict[str, Any],
    g5_corpus_path: Path,
    cases: list[dict[str, Any]],
    backends: list[str],
    attempts_per_case: int,
    fairness_contract: dict[str, Any],
    backend_results: dict[str, Any],
    comparison_rows: dict[str, Any],
    validated_blocks: dict[str, Any],
    benchmark_path: Path,
    version_payloads: dict[str, Any],
) -> dict[str, Any]:
    baseline_catalog_sha256 = sha256_file(baseline_catalog.source_path)
    aggregate_blocks = {
        backend_name: payload["aggregate"]["validated_aggregate"]
        for backend_name, payload in backend_results.items()
    }
    recomputed_blocks = recompute_backend_aggregates(backend_results)
    summary = {
        "run_id": run_id,
        "runner_scope": RUNNER_SCOPE,
        "scorer_version": SCORER_VERSION,
        "timestamp_local": local_now_iso(),
        "timezone": "UTC+07:00",
        "command": command,
        "catalog_version": baseline_catalog.catalog_version,
        "catalog_path": str(baseline_catalog.source_path),
        "catalog_sha256": baseline_catalog_sha256,
        "git_commit": next((payload.get("git_commit") for payload in version_payloads.values() if payload.get("git_commit")), None),
        "benchmark_path": str(Path(benchmark_path).resolve()),
        "corpus_path": str(Path(g5_corpus_path).resolve()),
        "corpus_version": corpus["corpus_version"],
        "corpus_sha256": corpus["corpus_sha256"],
        "report_json_path": str((run_root / "agent_portability_report.json").resolve()),
        "report_markdown_path": str((run_root / "agent_portability_report.md").resolve()),
        "backends": backends,
        "attempts_per_case": attempts_per_case,
        "evaluated_case_count": len(cases),
        "fairness_contract": fairness_contract,
        "loaded_corpora": {
            "g5_corpus": {
                "corpus_path": str(Path(g5_corpus_path).resolve()),
                "corpus_version": corpus["corpus_version"],
                "corpus_sha256": corpus["corpus_sha256"],
            }
        },
        "agent_portability_table": comparison_rows,
        "backend_results": backend_results,
        "aggregate_validation_inputs": aggregate_blocks,
    }
    summary["aggregate_validation"] = recompute_and_validate_aggregates(
        expected_blocks=aggregate_blocks,
        recomputed_blocks=recomputed_blocks,
    )
    mismatch_rows = [
        dict(attempt)
        for backend_payload in backend_results.values()
        for case_payload in backend_payload.get("case_attempts", [])
        for attempt in case_payload.get("attempts", [])
    ]
    summary["script_mismatch_summary"] = build_script_mismatch_summary(
        mismatch_rows,
        identifier_builder=lambda row: (
            f"{row.get('backend_name')}:{row.get('case_id')}:attempt_{row.get('attempt_index')}"
        ),
    )
    summary["aggregate_validation"] = apply_script_mismatch_validation(
        summary["aggregate_validation"],
        script_mismatch_summary=summary["script_mismatch_summary"],
    )
    return summary


def recompute_backend_aggregates(backend_results: dict[str, Any]) -> dict[str, Any]:
    recomputed: dict[str, Any] = {}
    for backend_name, payload in backend_results.items():
        recomputed[backend_name] = build_backend_aggregate(
            backend_name=backend_name,
            case_attempts=list(payload["case_attempts"]),
        )["validated_aggregate"]
    return recomputed


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


def compute_feasibility(*, public_state: str, summary_type: str | None, debug_summary: dict[str, Any]) -> int:
    runtime_summary = debug_summary.get("runtime_summary") or {}
    return int(
        public_state == "RUNTIME_SUCCESS"
        and summary_type == "counterfactual_found"
        and runtime_summary.get("executed") is True
        and runtime_summary.get("controller_state") == "TERMINAL_SUCCESS"
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


def build_binary_metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "mean": safe_mean(numerator, denominator),
    }


def build_optional_binary_metric(field_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [row[field_name] for row in rows if row.get(field_name) is not None]
    numerator = sum(int(value) for value in values)
    return {
        "numerator": numerator,
        "denominator": len(values),
        "mean": safe_mean(numerator, len(values)),
    }


def compute_attempt_stability(attempts: list[dict[str, Any]]) -> bool:
    if not attempts:
        return False
    signatures = [build_reproducibility_signature(item) for item in attempts]
    return all(signature == signatures[0] for signature in signatures[1:])


def count_unique_signatures(attempts: list[dict[str, Any]]) -> int:
    return len({build_reproducibility_signature(item) for item in attempts})


def build_reproducibility_signature(payload: dict[str, Any]) -> str:
    signature = {
        "final_public_state": payload["final_public_state"],
        "summary_type": payload["summary_type"],
        "reject_class": payload["reject_class"],
        "active_constraint_spec": payload["active_constraint_spec"],
        "counterfactual_profile": payload["counterfactual_profile"],
    }
    return json.dumps(signature, ensure_ascii=True, sort_keys=True)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Part II Agent Portability Report",
        "",
        "## Provenance",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- runner_scope: `{summary['runner_scope']}`",
        f"- scorer_version: `{summary['scorer_version']}`",
        f"- timestamp_local: `{summary['timestamp_local']}`",
        f"- timezone: `{summary['timezone']}`",
        f"- corpus_version: `{summary['corpus_version']}`",
        f"- corpus_sha256: `{summary['corpus_sha256']}`",
        f"- attempts_per_case: `{summary['attempts_per_case']}`",
        f"- evaluated_case_count: `{summary['evaluated_case_count']}`",
        "",
        "## Fairness Contract",
        "",
        f"- rules: `{summary['fairness_contract']['rules']}`",
        "",
        "## G5_agent_portability",
        "",
        "| Row | Completion | Successful Resolution | Clarification | Avg Turns | End-to-End Latency | Reproducibility | Final CF Validity | Actionability | Plausibility | Feasibility |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row_name, row in summary["agent_portability_table"].items():
        lines.append(
            "| `{row_name}` | `{completion}` | `{success}` | `{clarification}` | `{avg_turns}` | `{latency}` | `{repro}` | `{validity}` | `{actionability}` | `{plausibility}` | `{feasibility}` |".format(
                row_name=row_name,
                completion=row["completion_rate"]["mean"],
                success=row["successful_recourse_resolution"]["mean"],
                clarification=row["clarification_rate"]["mean"],
                avg_turns=row["average_turns"],
                latency=row["end_to_end_latency_ms"],
                repro=row["reproducibility_stability"]["mean"],
                validity=row["final_cf_validity"]["mean"],
                actionability=row["actionability"]["mean"],
                plausibility=row["plausibility"]["mean"],
                feasibility=row["feasibility"]["mean"],
            )
        )
    lines.extend(
        [
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
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
