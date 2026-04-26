from __future__ import annotations

from llm.src.orchestration.explanation_flow import (
    build_explanation_payload,
    build_user_response_payload_from_explanation,
)
from llm.src.runtime.reason_codes import (
    INVALID_COUNTERFACTUAL_BLOCKED,
    NO_FEASIBLE_CF_FOUND,
    NO_RECOURSE_NEEDED,
    REQUEST_CONSTRAINTS_BLOCKED,
)


class StubPolicy:
    dataset_name = "bank"
    f2change = ["Income", "CCAvg", "Mortgage"]


def test_no_recourse_payload_has_success_tone_and_no_changed_items():
    explanation_payload = build_explanation_payload(
        runtime_result={
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.95},
            "counterfactual": None,
            "reason_codes": [NO_RECOURSE_NEEDED],
        },
        current_profile={"Income": 100},
    )

    payload = build_user_response_payload_from_explanation(
        explanation_payload=explanation_payload,
        runtime_result={"reason_codes": [NO_RECOURSE_NEEDED]},
        current_profile={"Income": 100},
        policy=StubPolicy(),
        dataset_label="bank profile",
    )

    assert payload.response_kind == "no_recourse_needed"
    assert payload.tone == "success"
    assert payload.changed_items == []


def test_counterfactual_found_payload_converts_profile_diff_to_changed_items():
    explanation_payload = build_explanation_payload(
        runtime_result={
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 0, "predicted_proba": 0.21},
            "counterfactual": {
                "candidates": [
                    {
                        "rank": 1,
                        "method": "sfexp",
                        "profile": {"Income": 80, "CCAvg": 2.0},
                        "changed_features": ["Income", "CCAvg"],
                    }
                ]
            },
            "reason_codes": [],
        },
        current_profile={"Income": 45, "CCAvg": 1.2},
    )

    payload = build_user_response_payload_from_explanation(
        explanation_payload=explanation_payload,
        runtime_result={"reason_codes": []},
        current_profile={"Income": 45, "CCAvg": 1.2},
        policy=StubPolicy(),
        dataset_label="bank profile",
    )

    assert payload.response_kind == "counterfactual_found"
    assert [item.field_name for item in payload.changed_items] == ["Income", "CCAvg"]
    assert payload.changed_items[0].before == 45
    assert payload.changed_items[0].after == 80


def test_runtime_reject_constraints_blocked_has_blocked_reason_and_relax_action():
    explanation_payload = build_explanation_payload(
        runtime_result={
            "controller_state": "TERMINAL_REJECT",
            "prediction": {"predicted_label": 0, "predicted_proba": 0.11},
            "counterfactual": {"candidates": []},
            "reason_codes": [REQUEST_CONSTRAINTS_BLOCKED],
        },
        current_profile={"Income": 45},
        included_suggestion_types=["revise_target_profile"],
        policy=StubPolicy(),
    )

    payload = build_user_response_payload_from_explanation(
        explanation_payload=explanation_payload,
        runtime_result={"reason_codes": [REQUEST_CONSTRAINTS_BLOCKED]},
        current_profile={"Income": 45},
        policy=StubPolicy(),
        dataset_label="bank profile",
        active_constraint_spec={"disallowed_changes": ["Income"]},
    )

    assert payload.response_kind == "runtime_reject_constraints_blocked"
    assert payload.blocked_reasons
    assert payload.blocked_reasons[0].code == REQUEST_CONSTRAINTS_BLOCKED
    assert any(action.action_type == "relax_constraints" for action in payload.next_actions)


def test_runtime_reject_invalid_counterfactual_does_not_expose_changed_items():
    explanation_payload = build_explanation_payload(
        runtime_result={
            "controller_state": "TERMINAL_REJECT",
            "prediction": {"predicted_label": 0, "predicted_proba": 0.11},
            "counterfactual": {"candidates": [{"profile": {"Income": 999}}]},
            "reason_codes": [INVALID_COUNTERFACTUAL_BLOCKED],
        },
        current_profile={"Income": 45},
    )

    payload = build_user_response_payload_from_explanation(
        explanation_payload=explanation_payload,
        runtime_result={"reason_codes": [INVALID_COUNTERFACTUAL_BLOCKED]},
        current_profile={"Income": 45},
        policy=StubPolicy(),
        dataset_label="bank profile",
    )

    assert payload.response_kind == "runtime_reject_invalid_counterfactual_blocked"
    assert payload.changed_items == []


def test_runtime_reject_no_feasible_suggests_revise_or_broaden_changes():
    explanation_payload = build_explanation_payload(
        runtime_result={
            "controller_state": "TERMINAL_REJECT",
            "prediction": {"predicted_label": 0, "predicted_proba": 0.12},
            "counterfactual": {"candidates": []},
            "reason_codes": [NO_FEASIBLE_CF_FOUND],
        },
        current_profile={"Income": 45},
        included_suggestion_types=["broaden_allowed_financial_changes", "revise_target_profile"],
        policy=StubPolicy(),
    )

    payload = build_user_response_payload_from_explanation(
        explanation_payload=explanation_payload,
        runtime_result={"reason_codes": [NO_FEASIBLE_CF_FOUND]},
        current_profile={"Income": 45},
        policy=StubPolicy(),
        dataset_label="bank profile",
    )

    assert payload.response_kind == "runtime_reject_no_feasible_cf"
    action_types = [action.action_type for action in payload.next_actions]
    assert "relax_constraints" in action_types or "revise_profile" in action_types
