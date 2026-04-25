from __future__ import annotations

from llm.src.orchestration.clarification_flow import (
    build_clarification_payload,
    render_clarification_text,
)
from llm.src.orchestration.explanation_flow import (
    build_explanation_payload,
    render_explanation_text,
)


class PromptCapturingAdapter:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, object]] = []

    def generate_conversational_response(self, *, system_prompt: str, user_prompt: str, max_tokens: int | None = None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
            }
        )
        return self.response_text


def test_clarification_builder_orders_missing_fields_and_requests_full_profile():
    payload = build_clarification_payload(
        required_fields=[
            "Income",
            "Family",
            "CCAvg",
            "Education",
            "Mortgage",
            "SecuritiesAccount",
            "CDAccount",
            "Online",
            "CreditCard",
        ],
        missing_fields=["Online", "Income", "CreditCard"],
        conflicts=[],
        carried_forward_fields=["Family", "CCAvg", "Education"],
    )

    assert payload.clarification_type == "missing_information"
    assert payload.reply_strategy == "missing_fields_only"
    assert payload.missing_fields == ["Income", "Online", "CreditCard"]
    assert payload.carried_forward_fields == ["Family", "CCAvg", "Education"]
    assert "Reply with only the missing fields: Income, Online, and CreditCard." in payload.next_required_input
    assert "I'll keep the values already provided for Family, CCAvg, and Education." in payload.next_required_input
    assert "Reply with only the missing fields" in render_clarification_text(payload)


def test_clarification_and_explanation_rendering_use_conversational_adapter_when_available():
    clarification_adapter = PromptCapturingAdapter("Friendly clarification from LLM.")
    explanation_adapter = PromptCapturingAdapter("Friendly explanation from LLM.")

    clarification_payload = build_clarification_payload(
        required_fields=["Income", "Family", "CCAvg"],
        missing_fields=["Family", "CCAvg"],
        conflicts=[],
        carried_forward_fields=["Income"],
    )
    clarification_text = render_clarification_text(
        clarification_payload,
        parser_adapter=clarification_adapter,
    )

    explanation_payload = build_explanation_payload(
        runtime_result={
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 0, "predicted_proba": 0.42},
            "counterfactual": {
                "candidates": [
                    {
                        "method": "sfexp",
                        "rank": 1,
                        "profile": {"Income": 120.0},
                        "changed_features": ["Income"],
                    }
                ]
            },
            "reason_codes": [],
        },
        current_profile={"Income": 100.0},
    )
    explanation_text = render_explanation_text(
        explanation_payload,
        parser_adapter=explanation_adapter,
    )

    assert clarification_text == "Friendly clarification from LLM."
    assert explanation_text == "Friendly explanation from LLM."
    assert clarification_adapter.calls[0]["max_tokens"] == 112
    assert "reply with only the missing fields" in clarification_adapter.calls[0]["system_prompt"].lower()
    assert "fallback_meaning" in clarification_adapter.calls[0]["user_prompt"]
    assert explanation_adapter.calls[0]["max_tokens"] == 112
    assert "feasible counterfactual" in explanation_adapter.calls[0]["system_prompt"]
    assert "fallback_meaning" in explanation_adapter.calls[0]["user_prompt"]


def test_explanation_builder_uses_first_candidate_only():
    runtime_result = {
        "controller_state": "TERMINAL_SUCCESS",
        "prediction": {"predicted_label": 0, "predicted_proba": 0.42},
        "counterfactual": {
            "candidates": [
                {
                    "method": "sfexp",
                    "rank": 1,
                    "profile": {"Income": 120.0, "Online": 1},
                    "changed_features": ["Income", "Online"],
                },
                {
                    "method": "dfexp",
                    "rank": 1,
                    "profile": {"Income": 130.0, "Online": 1},
                    "changed_features": ["Income", "Online"],
                },
            ]
        },
        "reason_codes": [],
    }

    payload = build_explanation_payload(
        runtime_result=runtime_result,
        current_profile={"Income": 100.0, "Online": 0},
    )

    assert payload.summary_type == "counterfactual_found"
    assert payload.counterfactual_summary["method"] == "sfexp"
    assert payload.changed_fields == ["Income", "Online"]
    assert "Using the first runtime candidate only" in render_explanation_text(payload)


def test_runtime_reject_explanation_can_include_bounded_suggestions():
    runtime_result = {
        "controller_state": "TERMINAL_REJECT",
        "prediction": {"predicted_label": 1, "predicted_proba": 0.08},
        "counterfactual": {"candidates": []},
        "reason_codes": ["NO_FEASIBLE_CF_FOUND"],
    }

    payload = build_explanation_payload(
        runtime_result=runtime_result,
        current_profile={"Income": 49.0, "Online": 0},
        included_suggestion_types=["revise_target_profile", "broaden_allowed_financial_changes"],
    )

    assert payload.summary_type == "runtime_reject"
    assert payload.included_suggestion_types == [
        "revise_target_profile",
        "broaden_allowed_financial_changes",
    ]
    assert len(payload.next_step_suggestions) == 2
    assert "Optional next steps" in render_explanation_text(payload)
