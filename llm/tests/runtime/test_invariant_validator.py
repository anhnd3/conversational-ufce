from __future__ import annotations

import pandas as pd

from llm.src.runtime.invariant_validator import RuntimeInvariantValidator
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.runtime.types import CounterfactualCandidate, CounterfactualResult, PredictionResult, RuntimeResult


def _context():
    return PolicyRegistry(ModelRegistry()).get_runtime_context("bank")


def test_invariant_validator_passes_known_feasible_counterfactual():
    context = _context()
    validator = RuntimeInvariantValidator()
    current_profile = {
        "Income": 100.0,
        "Family": 1,
        "CCAvg": 2.7,
        "Education": 2,
        "Mortgage": 0.0,
        "SecuritiesAccount": 0,
        "CDAccount": 0,
        "Online": 0,
        "CreditCard": 0,
    }
    candidate_profile = dict(current_profile)
    candidate_profile["CDAccount"] = 1
    prediction = PredictionResult(
        dataset="bank",
        predicted_label=0,
        predicted_proba=0.12,
        feature_order_used=list(context.bundle.feature_order),
    )
    result = RuntimeResult(
        dataset="bank",
        controller_state="TERMINAL_SUCCESS",
        prediction=prediction,
        counterfactual=CounterfactualResult(
            feasible=True,
            candidates=[
                CounterfactualCandidate(
                    method="sfexp",
                    rank=1,
                    profile=candidate_profile,
                    changed_features=["CDAccount"],
                )
            ],
            reason_codes=[],
        ),
        reason_codes=[],
        runtime_mode="stable_demo",
    )

    validation = validator.validate(result=result, current_profile=current_profile, context=context)

    assert validation.status == "passed"
    assert validation.validated_summary_type == "counterfactual_found"
    assert validation.validated_changed_fields == ["CDAccount"]


def test_invariant_validator_blocks_disallowed_field_change():
    context = _context()
    validator = RuntimeInvariantValidator()
    current_profile = {
        "Income": 100.0,
        "Family": 1,
        "CCAvg": 2.7,
        "Education": 2,
        "Mortgage": 0.0,
        "SecuritiesAccount": 0,
        "CDAccount": 0,
        "Online": 0,
        "CreditCard": 0,
    }
    candidate_profile = dict(current_profile)
    candidate_profile["Family"] = 4
    prediction = PredictionResult(
        dataset="bank",
        predicted_label=0,
        predicted_proba=0.12,
        feature_order_used=list(context.bundle.feature_order),
    )
    result = RuntimeResult(
        dataset="bank",
        controller_state="TERMINAL_SUCCESS",
        prediction=prediction,
        counterfactual=CounterfactualResult(
            feasible=True,
            candidates=[
                CounterfactualCandidate(
                    method="sfexp",
                    rank=1,
                    profile=candidate_profile,
                    changed_features=["Family"],
                )
            ],
            reason_codes=[],
        ),
        reason_codes=[],
        runtime_mode="stable_demo",
    )

    validation = validator.validate(result=result, current_profile=current_profile, context=context)

    assert validation.status == "failed"
    assert validation.reason_codes == ["INVALID_COUNTERFACTUAL_BLOCKED"]
    assert "changed_outside_allowed_fields" in validation.details["violations"]
