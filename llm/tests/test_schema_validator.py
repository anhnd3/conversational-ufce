from __future__ import annotations

from copy import deepcopy

import pytest

from llm_eval.config import benchmark_from_dict
from llm.src.validation.schema_validator import validate_prediction


def test_validate_prediction_accepts_valid_payload(sample_benchmark):
    case = sample_benchmark.case_map["A01"]

    result = validate_prediction(case.expected_output, sample_benchmark)

    assert result.is_valid is True
    assert result.errors == ()


def test_validate_prediction_rejects_extra_fields_and_wrong_types(sample_benchmark):
    candidate = {
        "task": "extract_cf_request",
        "status": "partial",
        "cf_request": {
            "Income": "40",
            "Online": True,
            "BadField": 1,
        },
        "missing_fields": ["CCAvg"],
        "conflicts": [],
        "notes": [],
        "extra_top_level": 1,
    }

    result = validate_prediction(candidate, sample_benchmark)

    assert result.is_valid is False
    assert "Unexpected top-level keys: extra_top_level" in result.errors
    assert "Unexpected cf_request fields: BadField" in result.errors
    assert "Income must be numeric." in result.errors
    assert "Online must be binary 0 or 1." in result.errors


def test_validate_prediction_validates_constraint_spec_shape(sample_benchmark):
    candidate = {
        "task": "extract_cf_request",
        "status": "complete",
        "cf_request": {
            "Income": 72,
            "Family": 1,
            "CCAvg": 4.8,
            "Education": 2,
            "Mortgage": 200,
            "SecuritiesAccount": 1,
            "CDAccount": 1,
            "Online": 0,
            "CreditCard": 0,
        },
        "missing_fields": [],
        "conflicts": [],
        "notes": [],
        "constraint_spec": {
            "disallowed_changes": ["CreditCard", "Income"],
            "numeric_bounds": {"Online": {"min": 1}},
            "max_changed_features": 4,
        },
    }

    result = validate_prediction(candidate, sample_benchmark)

    assert result.is_valid is False
    assert "constraint_spec.numeric_bounds contains unsupported fields: Online" in result.errors
    assert "constraint_spec.max_changed_features must be one of 1, 2, 3." in result.errors


@pytest.fixture
def sample_benchmark_v3(sample_benchmark_payload):
    payload = deepcopy(sample_benchmark_payload)
    payload["benchmark_name"] = "ufce_bank_cf_parser_v3_en"
    return benchmark_from_dict(payload)


def test_validate_prediction_requires_field_evidence_for_v3(sample_benchmark_v3):
    candidate = {
        "task": "extract_cf_request",
        "status": "partial",
        "cf_request": {"Income": 40, "Online": 1},
        "missing_fields": ["CCAvg", "Family", "Education", "Mortgage", "SecuritiesAccount", "CDAccount", "CreditCard"],
        "conflicts": [],
        "notes": [],
    }

    result = validate_prediction(candidate, sample_benchmark_v3)

    assert result.is_valid is False
    assert "Missing top-level keys: field_evidence" in result.errors


def test_validate_prediction_rejects_invalid_field_evidence_shape(sample_benchmark_v3):
    candidate = {
        "task": "extract_cf_request",
        "status": "partial",
        "cf_request": {"Income": 40, "Online": 1},
        "field_evidence": {
            "Income": {
                "source_text": "Income 40",
                "evidence_kind": "explicit_numeric",
                "normalized_from": "40",
                "language": "en",
            },
            "Online": {
                "source_text": "Online yes",
                "evidence_kind": "bad_kind",
                "normalized_from": "yes",
                "language": "en",
            },
        },
        "missing_fields": ["CCAvg", "Family", "Education", "Mortgage", "SecuritiesAccount", "CDAccount", "CreditCard"],
        "conflicts": [],
        "notes": [],
    }

    result = validate_prediction(candidate, sample_benchmark_v3)

    assert result.is_valid is False
    assert any("field_evidence.Online.evidence_kind" in error for error in result.errors)
