from __future__ import annotations

from typing import Any

from llm.src.conversation.canonical_validator import BankCanonicalValidator
from llm.src.parser.output_repair import collect_repair_errors, should_attempt_repair
from llm.src.parser.parser_quality import finalize_parser_quality_metadata, run_parser_quality
from llm.src.part2_eval.common import progress_iter, safe_mean, summarize_latency_ms
from llm.src.parser.response_normalizer import normalize_and_parse
from llm.src.refinement.validation import validate_refinement_prediction


def score_initial_constraint_cases(
    *,
    parser_adapter,
    benchmark,
    cases: list[dict[str, Any]],
    progress_enabled: bool = True,
) -> dict[str, Any]:
    canonical_validator = BankCanonicalValidator()
    rows: list[dict[str, Any]] = []
    for case in progress_iter(cases, enabled=progress_enabled, desc="Tier A initial cases", unit="case"):
        parser_result = parser_adapter.parse(user_text=case["input_text"], benchmark=benchmark)
        quality_result = run_parser_quality(
            message_text=parser_result.message_text,
            benchmark_spec=benchmark,
            user_text=case["input_text"],
            api_error=parser_result.api_error,
        )
        normalized = quality_result.normalized
        schema_validation = quality_result.schema_validation
        canonical_validation = canonical_validator.validate(
            candidate=normalized.parsed_json,
            schema_validation=schema_validation,
        )
        repair_invoked = False
        errors = collect_repair_errors(
            parser_result=parser_result,
            normalized=normalized,
            schema_validation=schema_validation,
            canonical_validation=canonical_validation,
        )
        if should_attempt_repair(
            raw_output=parser_result.message_text,
            api_error=parser_result.api_error,
            errors=errors,
        ):
            repair_invoked = True
            repair_result = parser_adapter.repair(
                invalid_output=parser_result.message_text,
                errors=errors,
                benchmark=benchmark,
            )
            parser_result = repair_result
            quality_result = run_parser_quality(
                message_text=repair_result.message_text,
                benchmark_spec=benchmark,
                user_text=case["input_text"],
                api_error=repair_result.api_error,
            )
            normalized = quality_result.normalized
            schema_validation = quality_result.schema_validation
            canonical_validation = canonical_validator.validate(
                candidate=normalized.parsed_json,
                schema_validation=schema_validation,
            )

        predicted_spec = None
        if isinstance(normalized.parsed_json, dict):
            predicted_spec = normalized.parsed_json.get("constraint_spec")
        expected_spec = case["expected_constraint_spec"]
        rows.append(
            {
                "case_id": case["case_id"],
                "annotation_type": case["annotation_type"],
                "valid_json": isinstance(normalized.parsed_json, dict),
                "schema_valid": bool(schema_validation.is_valid),
                "canonical_pass": bool(canonical_validation.ready_for_runtime),
                "repair_invoked": repair_invoked,
                "final_parser_failure": not bool(schema_validation.is_valid),
                "exact_match": predicted_spec == expected_spec,
                "component_accuracy": score_component_accuracy(
                    expected=expected_spec,
                    predicted=predicted_spec,
                    component_names=("disallowed_changes", "numeric_bounds", "max_changed_features", "prefer_fewer_changes"),
                ),
                "predicted_constraint_spec": predicted_spec,
                "expected_constraint_spec": expected_spec,
                "parser_quality": finalize_parser_quality_metadata(
                    quality_result.metadata,
                    canonical_pass_after_quality=bool(canonical_validation.ready_for_runtime),
                    repair_invoked=repair_invoked,
                ),
                "request_latency_ms": parser_result.derived_metrics.get("request_latency_ms"),
            }
        )
    return build_annotation_score_summary(
        rows=rows,
        exact_metric_key="M6_constraint_extraction_fidelity",
    )


def score_refinement_delta_cases(
    *,
    parser_adapter,
    benchmark,
    cases: list[dict[str, Any]],
    progress_enabled: bool = True,
) -> dict[str, Any]:
    del benchmark
    rows: list[dict[str, Any]] = []
    for case in progress_iter(cases, enabled=progress_enabled, desc="Tier A refinement cases", unit="case"):
        parser_result = parser_adapter.parse_refinement(
            user_text=case["input_text"],
            active_constraint_spec=case["active_constraint_spec"],
            pending_refinement_clarification=case["pending_refinement_clarification"],
        )
        normalized = normalize_and_parse(parser_result.message_text)
        validation = validate_refinement_prediction(
            normalized.parsed_json,
            feature_order=list(BankCanonicalValidator().required_fields),
        )
        repair_invoked = False
        errors = collect_refinement_errors(parser_result=parser_result, normalized=normalized, validation=validation)
        if should_attempt_repair(
            raw_output=parser_result.message_text,
            api_error=parser_result.api_error,
            errors=errors,
        ):
            repair_invoked = True
            repair_result = parser_adapter.repair_refinement(
                invalid_output=parser_result.message_text,
                errors=errors,
                active_constraint_spec=case["active_constraint_spec"],
                pending_refinement_clarification=case["pending_refinement_clarification"],
            )
            parser_result = repair_result
            normalized = normalize_and_parse(repair_result.message_text)
            validation = validate_refinement_prediction(
                normalized.parsed_json,
                feature_order=list(BankCanonicalValidator().required_fields),
            )
        predicted_delta = None if validation.normalized_delta is None else dict(validation.normalized_delta)
        expected_delta = case["expected_delta"]
        rows.append(
            {
                "case_id": case["case_id"],
                "annotation_type": case["annotation_type"],
                "valid_json": isinstance(normalized.parsed_json, dict),
                "schema_valid": bool(validation.is_valid),
                "canonical_pass": bool(validation.is_valid),
                "repair_invoked": repair_invoked,
                "final_parser_failure": not bool(validation.is_valid),
                "exact_match": predicted_delta == expected_delta,
                "component_accuracy": score_component_accuracy(
                    expected=expected_delta,
                    predicted=predicted_delta,
                    component_names=(
                        "add_blocked_fields",
                        "remove_blocked_fields",
                        "set_numeric_bounds",
                        "clear_numeric_bounds",
                        "set_max_changed_features",
                        "clear_max_changed_features",
                        "set_prefer_fewer_changes",
                        "clear_prefer_fewer_changes",
                    ),
                ),
                "predicted_delta": predicted_delta,
                "expected_delta": expected_delta,
                "request_latency_ms": parser_result.derived_metrics.get("request_latency_ms"),
            }
        )
    return build_annotation_score_summary(
        rows=rows,
        exact_metric_key="M36_constraint_delta_fidelity",
    )


def collect_refinement_errors(*, parser_result, normalized, validation) -> list[str]:
    errors: list[str] = []
    if parser_result.api_error:
        errors.append(str(parser_result.api_error))
    if normalized.parse_error:
        errors.append(str(normalized.parse_error))
    errors.extend(str(item) for item in validation.errors)
    ordered: list[str] = []
    seen: set[str] = set()
    for item in errors:
        clean = " ".join(item.split()).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def score_component_accuracy(
    *,
    expected: dict[str, Any] | None,
    predicted: dict[str, Any] | None,
    component_names: tuple[str, ...],
) -> dict[str, float]:
    expected_dict = expected or {}
    predicted_dict = predicted or {}
    scores: dict[str, float] = {}
    for component_name in component_names:
        scores[component_name] = 1.0 if expected_dict.get(component_name) == predicted_dict.get(component_name) else 0.0
    scores["mean"] = round(sum(scores.values()) / len(component_names), 6) if component_names else 0.0
    return scores


def build_annotation_score_summary(*, rows: list[dict[str, Any]], exact_metric_key: str) -> dict[str, Any]:
    total_cases = len(rows)
    valid_json = sum(1 for row in rows if row["valid_json"])
    schema_valid = sum(1 for row in rows if row["schema_valid"])
    canonical_pass = sum(1 for row in rows if row["canonical_pass"])
    repair_invocations = sum(1 for row in rows if row["repair_invoked"])
    final_failures = sum(1 for row in rows if row["final_parser_failure"])
    exact_matches = sum(1 for row in rows if row["exact_match"])
    component_mean = round(sum(row["component_accuracy"]["mean"] for row in rows) / total_cases, 6) if total_cases else None
    return {
        "total_cases": total_cases,
        "M1_json_validity_rate": {
            "numerator": valid_json,
            "denominator": total_cases,
            "mean": safe_mean(valid_json, total_cases),
        },
        "M2_schema_compliance_rate": {
            "numerator": schema_valid,
            "denominator": total_cases,
            "mean": safe_mean(schema_valid, total_cases),
        },
        "M3_canonical_validation_pass_rate": {
            "numerator": canonical_pass,
            "denominator": total_cases,
            "mean": safe_mean(canonical_pass, total_cases),
        },
        "M4_repair_rate": {
            "numerator": repair_invocations,
            "denominator": total_cases,
            "mean": safe_mean(repair_invocations, total_cases),
        },
        "M5_final_parser_failure_rate": {
            "numerator": final_failures,
            "denominator": total_cases,
            "mean": safe_mean(final_failures, total_cases),
        },
        exact_metric_key: {
            "numerator": exact_matches,
            "denominator": total_cases,
            "mean": safe_mean(exact_matches, total_cases),
            "component_mean": component_mean,
        },
        "M7_parser_latency_ms": summarize_latency_ms([row["request_latency_ms"] for row in rows]),
        "per_case_results": rows,
    }
