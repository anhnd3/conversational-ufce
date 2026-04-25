from __future__ import annotations

from typing import Any


NO_RECOURSE_NEEDED = "no_recourse_needed"
COUNTERFACTUAL_FOUND = "counterfactual_found"
RUNTIME_REJECT = "runtime_reject"
CLARIFICATION = "clarification"
SUPPLEMENTAL_FOLLOWUP = "supplemental_followup"
CONFLICT = "conflict"
PARSER_FAILURE = "parser_failure"
UNKNOWN = "unknown"

PRIMARY_LABELS = (
    NO_RECOURSE_NEEDED,
    COUNTERFACTUAL_FOUND,
    RUNTIME_REJECT,
    CLARIFICATION,
)

PRIMARY_ACCEPTANCE_TARGET = {
    NO_RECOURSE_NEEDED: 2,
    COUNTERFACTUAL_FOUND: 2,
    RUNTIME_REJECT: 2,
    CLARIFICATION: 2,
}

EXPLANATION_SUMMARY_TYPES = (
    NO_RECOURSE_NEEDED,
    COUNTERFACTUAL_FOUND,
    RUNTIME_REJECT,
)

MANDATORY_WORKED_EXAMPLE_BUCKETS = (
    NO_RECOURSE_NEEDED,
    COUNTERFACTUAL_FOUND,
    "supplemental_followup_merge_to_success",
)


def classify_turn_result(*, stage: str | None, explanation_payload: Any) -> str:
    summary_type = extract_summary_type(explanation_payload)
    if summary_type in EXPLANATION_SUMMARY_TYPES:
        return summary_type
    if stage == "NEEDS_CLARIFICATION":
        return CLARIFICATION
    if stage == "CONFLICT":
        return CONFLICT
    if stage == "PARSER_FAILURE":
        return PARSER_FAILURE
    if isinstance(stage, str) and stage:
        return stage.lower()
    return UNKNOWN


def extract_summary_type(explanation_payload: Any) -> str | None:
    if explanation_payload is None:
        return None
    if hasattr(explanation_payload, "summary_type"):
        value = getattr(explanation_payload, "summary_type")
        return str(value) if isinstance(value, str) and value else None
    if isinstance(explanation_payload, dict):
        value = explanation_payload.get("summary_type")
        return str(value) if isinstance(value, str) and value else None
    return None

