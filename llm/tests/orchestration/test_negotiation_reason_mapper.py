from __future__ import annotations

from llm.src.orchestration.negotiation_reason_mapper import build_negotiation_explanation
from llm.src.runtime.reason_codes import (
    INVALID_COUNTERFACTUAL_BLOCKED,
    NO_FEASIBLE_CF_FOUND,
    REQUEST_CONSTRAINTS_BLOCKED,
)


class StubPolicy:
    dataset_name = "bank"
    f2change = ["Income", "CCAvg", "Mortgage"]


def test_no_feasible_reason_maps_to_explanation_and_next_actions():
    blocked, effects, actions = build_negotiation_explanation(
        transition_reason="runtime_reject_no_feasible_cf",
        reason_codes=[NO_FEASIBLE_CF_FOUND],
        active_constraint_spec=None,
        policy=StubPolicy(),
        included_suggestion_types=["revise_target_profile"],
    )

    assert blocked
    assert blocked[0].code == NO_FEASIBLE_CF_FOUND
    assert effects == []
    assert any(action.action_type == "revise_profile" for action in actions)


def test_constraints_blocked_maps_to_constraint_explanation():
    blocked, effects, actions = build_negotiation_explanation(
        transition_reason="runtime_reject_no_feasible_cf",
        reason_codes=[REQUEST_CONSTRAINTS_BLOCKED],
        active_constraint_spec={
            "disallowed_changes": ["Income"],
            "numeric_bounds": {"Mortgage": {"max": 0}},
            "max_changed_features": 1,
        },
        policy=StubPolicy(),
    )

    assert blocked
    assert blocked[0].code == REQUEST_CONSTRAINTS_BLOCKED
    assert any(effect.constraint_key.startswith("disallowed_changes:") for effect in effects)
    assert any(effect.constraint_key == "max_changed_features" for effect in effects)
    assert any(action.action_type == "relax_constraints" for action in actions)


def test_invalid_counterfactual_maps_to_validation_blocked_explanation():
    blocked, effects, actions = build_negotiation_explanation(
        transition_reason="runtime_reject_system_error",
        reason_codes=[INVALID_COUNTERFACTUAL_BLOCKED],
        active_constraint_spec=None,
        policy=StubPolicy(),
    )

    assert blocked
    assert blocked[0].code == INVALID_COUNTERFACTUAL_BLOCKED
    assert effects == []
    assert any(action.action_type == "start_new_case" for action in actions)


def test_disallowed_changes_numeric_bounds_and_change_limit_are_extracted():
    _, effects, _ = build_negotiation_explanation(
        transition_reason="runtime_reject_no_feasible_cf",
        reason_codes=[REQUEST_CONSTRAINTS_BLOCKED],
        active_constraint_spec={
            "disallowed_changes": ["Income"],
            "numeric_bounds": {"Mortgage": {"min": 0, "max": 120}},
            "max_changed_features": 1,
        },
        policy=StubPolicy(),
    )

    keys = [effect.constraint_key for effect in effects]
    assert "disallowed_changes:Income" in keys
    assert "numeric_bounds:Mortgage" in keys
    assert "max_changed_features" in keys
