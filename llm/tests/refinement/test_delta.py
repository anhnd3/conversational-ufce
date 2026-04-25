from __future__ import annotations

from llm.src.refinement.delta import (
    apply_refinement_delta_to_active_constraint_spec,
    build_active_constraint_spec,
    validate_and_normalize_refinement_delta,
)


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


def test_build_active_constraint_spec_collapses_to_effective_blocked_fields():
    active = build_active_constraint_spec(
        {
            "immutable": ["CreditCard", "Income"],
            "disallowed_changes": ["Mortgage", "Income"],
            "numeric_bounds": {"Mortgage": {"max": 120}},
        },
        feature_order=FEATURE_ORDER,
    )

    assert active == {
        "disallowed_changes": ["Income", "Mortgage", "CreditCard"],
        "numeric_bounds": {"Mortgage": {"max": 120.0}},
    }


def test_validate_and_normalize_refinement_delta_detects_same_turn_conflicts():
    normalized, errors, clarification_reasons = validate_and_normalize_refinement_delta(
        {
            "add_blocked_fields": ["Income"],
            "remove_blocked_fields": ["Income"],
            "set_numeric_bounds": {"Mortgage": {"max": 120}},
            "clear_numeric_bounds": {"Mortgage": ["max"]},
            "set_max_changed_features": 1,
            "clear_max_changed_features": True,
        },
        feature_order=FEATURE_ORDER,
    )

    assert errors == []
    assert normalized["add_blocked_fields"] == ["Income"]
    assert normalized["remove_blocked_fields"] == ["Income"]
    assert normalized["set_numeric_bounds"] == {"Mortgage": {"max": 120.0}}
    assert normalized["clear_numeric_bounds"] == {"Mortgage": ["max"]}
    assert clarification_reasons == [
        "The same blocked fields were both added and removed in one refinement turn: Income",
        "The same numeric bound was both set and cleared for Mortgage: max",
        "max_changed_features was both set and cleared in one refinement turn.",
    ]


def test_apply_refinement_delta_canonicalizes_and_overrides_active_constraints():
    updated = apply_refinement_delta_to_active_constraint_spec(
        {
            "immutable": ["CreditCard"],
            "disallowed_changes": ["Mortgage"],
            "numeric_bounds": {"Income": {"min": 80}},
            "prefer_fewer_changes": False,
        },
        {
            "add_blocked_fields": ["Income", "Mortgage"],
            "remove_blocked_fields": ["Mortgage"],
            "set_numeric_bounds": {"Mortgage": {"max": 120}},
            "set_max_changed_features": 1,
            "set_prefer_fewer_changes": True,
        },
        feature_order=FEATURE_ORDER,
    )

    assert updated == {
        "disallowed_changes": ["Income", "CreditCard"],
        "numeric_bounds": {
            "Income": {"min": 80.0},
            "Mortgage": {"max": 120.0},
        },
        "max_changed_features": 1,
        "prefer_fewer_changes": True,
    }
