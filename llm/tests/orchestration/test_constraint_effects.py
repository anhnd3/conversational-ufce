from __future__ import annotations

from llm.src.orchestration.constraint_effects import (
    describe_active_constraint_effects,
    describe_allowed_change_policy,
)


class StubPolicy:
    dataset_name = "bank"
    f2change = ["Income", "CCAvg", "Mortgage"]


def test_disallowed_changes_are_converted_to_constraint_effects():
    effects = describe_active_constraint_effects(
        active_constraint_spec={"disallowed_changes": ["Income", "Mortgage"]},
        policy=None,
    )

    keys = [effect.constraint_key for effect in effects]
    assert "disallowed_changes:Income" in keys
    assert "disallowed_changes:Mortgage" in keys


def test_numeric_bounds_are_converted_to_constraint_effects():
    effects = describe_active_constraint_effects(
        active_constraint_spec={"numeric_bounds": {"Mortgage": {"min": 0, "max": 120}}},
        policy=None,
    )

    assert effects
    assert effects[0].constraint_key == "numeric_bounds:Mortgage"
    assert "between 0 and 120" in effects[0].detail


def test_max_changed_features_is_converted_to_constraint_effect():
    effects = describe_active_constraint_effects(
        active_constraint_spec={"max_changed_features": 1},
        policy=None,
    )

    assert effects
    assert effects[0].constraint_key == "max_changed_features"


def test_policy_f2change_is_converted_to_constraint_effect():
    effect = describe_allowed_change_policy(StubPolicy())

    assert effect is not None
    assert effect.constraint_key == "policy.f2change"
    assert "Income" in effect.detail
