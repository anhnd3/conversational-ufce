from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from llm.src.validation.schema_validator import ValidationResult

from llm_eval.models import BenchmarkCase, BenchmarkDefinition


def score_prediction(
    benchmark: BenchmarkDefinition,
    case: BenchmarkCase,
    parsed_json: dict[str, Any] | None,
    validation_result: ValidationResult,
) -> dict[str, Any]:
    expected = case.expected_output
    predicted_cf = parsed_json.get("cf_request") if isinstance(parsed_json, dict) else {}
    if not isinstance(predicted_cf, dict):
        predicted_cf = {}
    expected_cf = expected.get("cf_request", {})
    exact_match = parsed_json == expected
    field_accuracy = compute_field_accuracy(
        benchmark.allowed_field_names,
        expected_cf,
        predicted_cf,
    )
    return {
        "valid_json": parsed_json is not None,
        "schema_valid": validation_result.is_valid,
        "exact_match": exact_match,
        "field_accuracy": field_accuracy,
        "status_correct": parsed_json.get("status") == expected.get("status")
        if isinstance(parsed_json, dict)
        else False,
        "missing_fields_correct": parsed_json.get("missing_fields") == expected.get("missing_fields")
        if isinstance(parsed_json, dict)
        else False,
        "conflicts_correct": parsed_json.get("conflicts") == expected.get("conflicts")
        if isinstance(parsed_json, dict)
        else False,
        "hallucination_count": compute_hallucination_count(
            expected_cf,
            predicted_cf,
            validation_result,
        ),
    }


def compute_field_accuracy(
    allowed_fields: tuple[str, ...],
    expected_cf: dict[str, Any],
    predicted_cf: dict[str, Any],
) -> float:
    correct = 0
    for field_name in allowed_fields:
        expected_has = field_name in expected_cf
        predicted_has = field_name in predicted_cf
        if expected_has != predicted_has:
            continue
        if not expected_has:
            correct += 1
            continue
        if predicted_cf[field_name] == expected_cf[field_name]:
            correct += 1
    return correct / len(allowed_fields)


def compute_hallucination_count(
    expected_cf: dict[str, Any],
    predicted_cf: dict[str, Any],
    validation_result: ValidationResult,
) -> int:
    unexpected_cf_fields = set(validation_result.unexpected_cf_fields)
    invented_fields = sum(
        1
        for field_name in predicted_cf
        if field_name not in expected_cf and field_name not in unexpected_cf_fields
    )
    return invented_fields + len(validation_result.unexpected_cf_fields) + len(
        validation_result.unexpected_top_level_keys
    )


def attach_stability_scores(rows: list[dict[str, Any]]) -> None:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)

    for case_rows in by_case.values():
        stability = compute_case_stability(case_rows)
        for row in case_rows:
            row["stability_case"] = stability


def compute_case_stability(case_rows: list[dict[str, Any]]) -> float:
    serialized_predictions: list[str] = []
    for row in case_rows:
        parsed_json = row.get("parsed_json")
        if not isinstance(parsed_json, dict):
            return 0.0
        serialized_predictions.append(json.dumps(parsed_json, sort_keys=True))
    if not serialized_predictions:
        return 0.0
    return 1.0 if len(set(serialized_predictions)) == 1 else 0.0
