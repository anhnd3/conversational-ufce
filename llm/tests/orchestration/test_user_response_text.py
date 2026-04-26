from __future__ import annotations

from llm.src.orchestration.user_response_payload import (
    BlockedReason,
    ChangeItem,
    ConstraintEffect,
    NextAction,
    UserResponsePayload,
)
from llm.src.orchestration.user_response_text import render_deterministic_user_response_text


def test_success_text_avoids_counterfactual_term_in_first_line():
    payload = UserResponsePayload(
        response_kind="no_recourse_needed",
        tone="success",
        headline="Current profile already qualifies",
        short_summary="No changes needed.",
    )

    text = render_deterministic_user_response_text(payload, dataset_label="bank profile")
    first_line = text.splitlines()[0].lower()
    assert "counterfactual" not in first_line


def test_counterfactual_found_text_contains_bullets_with_exact_values():
    payload = UserResponsePayload(
        response_kind="counterfactual_found",
        tone="success",
        headline="Found recommendation",
        short_summary="",
        changed_items=[
            ChangeItem(field_name="Income", display_name="Income", before=45, after=60),
            ChangeItem(field_name="CCAvg", display_name="CCAvg", before=1.2, after=1.8),
        ],
    )

    text = render_deterministic_user_response_text(payload, dataset_label="bank profile")

    assert "* Income: 45 -> 60" in text
    assert "* CCAvg: 1.2 -> 1.8" in text


def test_invalid_blocked_text_does_not_expose_invalid_candidate_values():
    payload = UserResponsePayload(
        response_kind="runtime_reject_invalid_counterfactual_blocked",
        tone="danger",
        headline="Validation blocked",
        short_summary="",
        technical_facts={"invalid_candidate": {"Income": 999}},
    )

    text = render_deterministic_user_response_text(payload, dataset_label="bank profile")

    assert "999" not in text


def test_constraints_blocked_text_includes_blocked_fields_when_available():
    payload = UserResponsePayload(
        response_kind="runtime_reject_constraints_blocked",
        tone="warning",
        headline="Blocked",
        short_summary="",
        blocked_reasons=[
            BlockedReason(
                code="REQUEST_CONSTRAINTS_BLOCKED",
                title="Blocked",
                detail="constraints",
                fields=["Income"],
                source="constraint",
            )
        ],
        constraint_effects=[
            ConstraintEffect(
                constraint_key="numeric_bounds:Mortgage",
                title="Mortgage bound",
                detail="Mortgage <= 0",
                affected_fields=["Mortgage"],
            )
        ],
        next_actions=[
            NextAction(
                action_type="relax_constraints",
                label="Relax",
                detail="relax one constraint",
                primary=True,
            )
        ],
    )

    text = render_deterministic_user_response_text(payload, dataset_label="bank profile")

    assert "Blocked fields:" in text
    assert "* Income" in text
    assert "* Mortgage" in text


def test_all_rendered_texts_are_non_empty():
    payloads = [
        UserResponsePayload(
            response_kind="clarification_required",
            tone="info",
            headline="Missing",
            short_summary="",
            next_actions=[
                NextAction(
                    action_type="provide_missing_fields",
                    label="Provide",
                    detail="Provide missing",
                    fields=["Income"],
                    primary=True,
                )
            ],
            technical_facts={"missing_fields": ["Income"]},
        ),
        UserResponsePayload(
            response_kind="clarification_limit_reached",
            tone="warning",
            headline="Limit",
            short_summary="",
        ),
        UserResponsePayload(
            response_kind="parser_failure",
            tone="danger",
            headline="Parse failed",
            short_summary="",
        ),
    ]

    for payload in payloads:
        text = render_deterministic_user_response_text(payload, dataset_label="bank profile")
        assert text.strip()
