from __future__ import annotations

import pytest

from llm.src.runtime.constraint_spec import (
    apply_constraint_spec_to_candidates,
    validate_and_normalize_constraint_spec,
)
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.reason_codes import REQUEST_CONSTRAINTS_BLOCKED
from llm.src.runtime.reproducibility import sort_counterfactual_candidates
from llm.src.runtime.types import CounterfactualCandidate, CounterfactualResult


pytestmark = pytest.mark.filterwarnings("ignore:tostring\\(\\) is deprecated:DeprecationWarning")


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


def test_validate_and_normalize_constraint_spec_orders_and_deduplicates_lists():
    normalized, errors = validate_and_normalize_constraint_spec(
        {
            "immutable": ["CreditCard", "Income", "CreditCard"],
            "disallowed_changes": ["Mortgage", "Income", "Mortgage"],
            "numeric_bounds": {
                "Mortgage": {"max": 120},
                "Income": {"min": 90},
            },
            "max_changed_features": 2,
            "prefer_fewer_changes": True,
        },
        feature_order=FEATURE_ORDER,
    )

    assert errors == []
    assert normalized == {
        "immutable": ["Income", "CreditCard"],
        "disallowed_changes": ["Income", "Mortgage"],
        "numeric_bounds": {
            "Income": {"min": 90.0},
            "Mortgage": {"max": 120.0},
        },
        "max_changed_features": 2,
        "prefer_fewer_changes": True,
    }


def test_validate_and_normalize_constraint_spec_rejects_unsupported_fields():
    _, errors = validate_and_normalize_constraint_spec(
        {
            "immutable": ["BadField"],
            "numeric_bounds": {"Online": {"min": 1}},
        },
        feature_order=FEATURE_ORDER,
    )

    assert "constraint_spec.immutable contains unsupported fields: BadField" in errors
    assert "constraint_spec.numeric_bounds contains unsupported fields: Online" in errors


def test_runtime_orchestrator_returns_distinct_request_constraints_blocked_reason():
    orchestrator = RuntimeOrchestrator()
    result = orchestrator.handle(
        {
            "dataset": "bank",
            "profile": {
                "Income": 100,
                "Family": 1,
                "CCAvg": 2.7,
                "Education": 2,
                "Mortgage": 0,
                "SecuritiesAccount": 0,
                "CDAccount": 0,
                "Online": 0,
                "CreditCard": 0,
            },
            "constraint_spec": {
                "disallowed_changes": ["CDAccount"],
            },
        },
        include_debug_trace=True,
    )

    assert result.controller_state == "TERMINAL_REJECT"
    assert result.reason_codes == [REQUEST_CONSTRAINTS_BLOCKED]
    assert result.counterfactual is not None
    assert result.counterfactual.feasible is False
    assert result.debug_trace is not None
    assert result.debug_trace.constraint_filter is not None
    assert result.debug_trace.constraint_filter["blocked_reason_counts"] == {"blocked_change_field": 1}


def test_runtime_orchestrator_enforces_max_changed_features_and_numeric_bounds():
    orchestrator = RuntimeOrchestrator()
    base_request = {
        "dataset": "bank",
        "profile": {
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
    }

    max_changed_result = orchestrator.handle(
        {
            **base_request,
            "constraint_spec": {"max_changed_features": 1},
        }
    )
    numeric_bounds_result = orchestrator.handle(
        {
            **base_request,
            "constraint_spec": {"numeric_bounds": {"Income": {"min": 90}}},
        }
    )

    assert max_changed_result.controller_state == "TERMINAL_SUCCESS"
    assert max_changed_result.counterfactual is not None
    assert [candidate.changed_features for candidate in max_changed_result.counterfactual.candidates] == [["Income"]]

    assert numeric_bounds_result.controller_state == "TERMINAL_SUCCESS"
    assert numeric_bounds_result.counterfactual is not None
    assert [candidate.profile["Income"] for candidate in numeric_bounds_result.counterfactual.candidates] == [91.0]


def test_apply_constraint_spec_prefer_fewer_changes_reorders_after_filtering():
    candidates = [
        CounterfactualCandidate(
            method="sfexp",
            rank=1,
            profile={"Income": 88.0, "CCAvg": 5.1},
            changed_features=["Income", "CCAvg"],
        ),
        CounterfactualCandidate(
            method="tfexp",
            rank=1,
            profile={"Income": 90.0, "CCAvg": 4.8},
            changed_features=["Income"],
        ),
    ]

    filtered, debug_summary = apply_constraint_spec_to_candidates(
        result=CounterfactualResult(feasible=True, candidates=candidates, reason_codes=[]),
        constraint_spec={"prefer_fewer_changes": True},
        feature_order=["Income", "CCAvg"],
        sort_candidates=sort_counterfactual_candidates,
        request_constraints_blocked_code=REQUEST_CONSTRAINTS_BLOCKED,
    )

    assert filtered.feasible is True
    assert [candidate.method for candidate in filtered.candidates] == ["tfexp", "sfexp"]
    assert debug_summary is not None
    assert debug_summary["prefer_fewer_changes"] is True
