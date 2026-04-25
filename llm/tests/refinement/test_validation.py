from __future__ import annotations

from llm.src.refinement.validation import validate_refinement_prediction


FEATURE_ORDER = [
    "Income",
    "Family",
    "CCAvg",
    "Education",
    "Mortgage",
    "SecuritiesAccount",
    "CDAccount",
    "Online",
    "CreditCard",
]


def test_validate_refinement_prediction_normalizes_supported_output():
    result = validate_refinement_prediction(
        {
            "task": "extract_constraint_feedback",
            "status": "apply",
            "constraint_feedback_delta": {
                "add_blocked_fields": ["CreditCard", "Income", "Income"],
                "set_numeric_bounds": {"Mortgage": {"max": 120}},
                "set_prefer_fewer_changes": True,
            },
            "ambiguities": [],
            "unsupported_feedback": [],
            "notes": ["keep it narrow", "keep it narrow"],
        },
        feature_order=FEATURE_ORDER,
    )

    assert result.is_valid is True
    assert result.parser_status == "apply"
    assert result.normalized_delta == {
        "add_blocked_fields": ["Income", "CreditCard"],
        "set_numeric_bounds": {"Mortgage": {"max": 120.0}},
        "set_prefer_fewer_changes": True,
    }
    assert result.normalized_output == {
        "task": "extract_constraint_feedback",
        "status": "apply",
        "constraint_feedback_delta": {
            "add_blocked_fields": ["Income", "CreditCard"],
            "set_numeric_bounds": {"Mortgage": {"max": 120.0}},
            "set_prefer_fewer_changes": True,
        },
        "ambiguities": [],
        "unsupported_feedback": [],
        "notes": ["keep it narrow"],
    }


def test_validate_refinement_prediction_converts_empty_apply_to_unsupported_feedback():
    result = validate_refinement_prediction(
        {
            "task": "extract_constraint_feedback",
            "status": "apply",
            "constraint_feedback_delta": {},
            "ambiguities": [],
            "unsupported_feedback": [],
            "notes": [],
        },
        feature_order=FEATURE_ORDER,
    )

    assert result.is_valid is True
    assert result.parser_status == "apply"
    assert result.normalized_output["status"] == "unsupported_feedback"
    assert result.unsupported_reasons == (
        "No supported constraint update was extracted from the refinement feedback.",
    )


def test_validate_refinement_prediction_rejects_wrong_task_and_schema_shape():
    result = validate_refinement_prediction(
        {
            "task": "extract_cf_request",
            "status": "apply",
            "constraint_feedback_delta": [],
            "ambiguities": [],
            "unsupported_feedback": [],
            "notes": [],
        },
        feature_order=FEATURE_ORDER,
    )

    assert result.is_valid is False
    assert "task must equal 'extract_constraint_feedback'" in result.errors
    assert "constraint_feedback_delta must be an object." in result.errors
