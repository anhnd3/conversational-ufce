from __future__ import annotations

from llm.src.orchestration.llm_response_guard import accept_llm_rewrite
from llm.src.orchestration.user_response_payload import ChangeItem, UserResponsePayload


def _payload() -> UserResponsePayload:
    return UserResponsePayload(
        response_kind="counterfactual_found",
        tone="success",
        headline="Found path",
        short_summary="",
        changed_items=[
            ChangeItem(field_name="Income", display_name="Income", before=45, after=60),
        ],
    )


def test_empty_llm_text_returns_fallback():
    fallback = "Fallback text"
    result = accept_llm_rewrite(llm_text="", fallback_text=fallback, payload=_payload())
    assert result == fallback


def test_missing_required_changed_value_returns_fallback():
    fallback = "Fallback text"
    result = accept_llm_rewrite(
        llm_text="Increase Income to improve your case.",
        fallback_text=fallback,
        payload=_payload(),
    )
    assert result == fallback


def test_rewrite_over_max_length_returns_fallback():
    fallback = "Fallback text"
    long_text = "word " * 250
    result = accept_llm_rewrite(
        llm_text=long_text,
        fallback_text=fallback,
        payload=_payload(),
    )
    assert result == fallback


def test_forbidden_outcome_flip_returns_fallback():
    fallback = "Fallback text"
    payload = UserResponsePayload(
        response_kind="runtime_reject_no_feasible_cf",
        tone="warning",
        headline="No recommendation",
        short_summary="",
    )
    result = accept_llm_rewrite(
        llm_text="Good news, your profile is approved and recommendation found.",
        fallback_text=fallback,
        payload=payload,
    )
    assert result == fallback


def test_good_rewrite_is_accepted():
    fallback = "Fallback text"
    result = accept_llm_rewrite(
        llm_text="Increase Income from 45 to 60 to reach the desired outcome.",
        fallback_text=fallback,
        payload=_payload(),
    )
    assert result == "Increase Income from 45 to 60 to reach the desired outcome."
