from __future__ import annotations

from typing import Any

from llm.src.orchestration.constraint_effects import describe_active_constraint_effects
from llm.src.orchestration.user_response_payload import BlockedReason, ConstraintEffect, NextAction
from llm.src.runtime.reason_codes import (
    INVALID_COUNTERFACTUAL_BLOCKED,
    NO_FEASIBLE_CF_FOUND,
    REQUEST_CONSTRAINTS_BLOCKED,
)


def build_negotiation_explanation(
    *,
    transition_reason: str | None,
    reason_codes: list[str],
    active_constraint_spec: dict[str, Any] | None,
    policy=None,
    included_suggestion_types: list[str] | None = None,
) -> tuple[list[BlockedReason], list[ConstraintEffect], list[NextAction]]:
    del transition_reason
    normalized_codes = [str(code) for code in list(reason_codes or []) if isinstance(code, str)]
    suggestions = [str(item) for item in list(included_suggestion_types or []) if isinstance(item, str)]
    blocked_reasons: list[BlockedReason] = []
    next_actions: list[NextAction] = []
    constraint_effects: list[ConstraintEffect] = []

    if REQUEST_CONSTRAINTS_BLOCKED in normalized_codes:
        blocked_fields = _extract_blocked_fields(active_constraint_spec)
        blocked_reasons.append(
            BlockedReason(
                code=REQUEST_CONSTRAINTS_BLOCKED,
                title="Active constraints blocked recommendations",
                detail=(
                    "A candidate existed, but active request constraints blocked showing it under the current settings."
                ),
                fields=blocked_fields,
                source="constraint",
            )
        )
        constraint_effects = describe_active_constraint_effects(
            active_constraint_spec=active_constraint_spec,
            policy=policy,
        )
        next_actions.append(
            NextAction(
                action_type="relax_constraints",
                label="Relax constraints",
                detail="Relax one active constraint or revise bounds, then rerun the same case.",
                fields=blocked_fields,
                primary=True,
            )
        )
        next_actions.append(
            NextAction(
                action_type="revise_profile",
                label="Revise profile",
                detail="Revise the target profile values and rerun.",
                fields=[],
                primary=False,
            )
        )
        return blocked_reasons, constraint_effects, next_actions

    if INVALID_COUNTERFACTUAL_BLOCKED in normalized_codes:
        blocked_reasons.append(
            BlockedReason(
                code=INVALID_COUNTERFACTUAL_BLOCKED,
                title="Validation blocked recommendation",
                detail="A generated candidate failed post-generation invariant validation checks.",
                fields=[],
                source="invariant",
            )
        )
        next_actions.append(
            NextAction(
                action_type="start_new_case",
                label="Start new case",
                detail="Start a new case with corrected values.",
                primary=True,
            )
        )
        next_actions.append(
            NextAction(
                action_type="check_technical_details",
                label="Check technical details",
                detail="Review the technical drawer for the validation reason.",
                primary=False,
            )
        )
        return blocked_reasons, constraint_effects, next_actions

    if NO_FEASIBLE_CF_FOUND in normalized_codes:
        blocked_reasons.append(
            BlockedReason(
                code=NO_FEASIBLE_CF_FOUND,
                title="No feasible recommendation found",
                detail=(
                    "No runtime candidate reached the desired outcome while satisfying the current policy and checks."
                ),
                fields=[],
                source="runtime",
            )
        )
        next_actions.extend(_next_actions_from_suggestion_types(suggestions))
        if not next_actions:
            next_actions.append(
                NextAction(
                    action_type="revise_profile",
                    label="Revise profile",
                    detail="Revise the target profile or broaden allowed financial changes.",
                    primary=True,
                )
            )
        return blocked_reasons, constraint_effects, next_actions

    if "MISSING_REQUIRED_FIELDS" in normalized_codes:
        blocked_reasons.append(
            BlockedReason(
                code="MISSING_REQUIRED_FIELDS",
                title="Missing required profile values",
                detail="Required profile values are still missing for this case.",
                fields=[],
                source="clarification",
            )
        )
        next_actions.append(
            NextAction(
                action_type="provide_missing_fields",
                label="Provide missing fields",
                detail="Reply with only the listed missing fields.",
                primary=True,
            )
        )
        return blocked_reasons, constraint_effects, next_actions

    if "PARSER_FAILURE" in normalized_codes:
        blocked_reasons.append(
            BlockedReason(
                code="PARSER_FAILURE",
                title="Request parsing failed",
                detail="The parser output remained invalid after repair.",
                fields=[],
                source="parser",
            )
        )
        next_actions.append(
            NextAction(
                action_type="start_new_case",
                label="Start new case",
                detail="Start a new case with one complete profile.",
                primary=True,
            )
        )
        return blocked_reasons, constraint_effects, next_actions

    return blocked_reasons, constraint_effects, next_actions


def _extract_blocked_fields(active_constraint_spec: dict[str, Any] | None) -> list[str]:
    if not isinstance(active_constraint_spec, dict):
        return []
    fields: list[str] = []
    disallowed = active_constraint_spec.get("disallowed_changes")
    if isinstance(disallowed, list):
        fields.extend(str(item) for item in disallowed if isinstance(item, str))
    numeric_bounds = active_constraint_spec.get("numeric_bounds")
    if isinstance(numeric_bounds, dict):
        fields.extend(str(item) for item in numeric_bounds if isinstance(item, str))
    deduped: list[str] = []
    seen: set[str] = set()
    for field_name in fields:
        if field_name in seen:
            continue
        seen.add(field_name)
        deduped.append(field_name)
    return deduped


def _next_actions_from_suggestion_types(suggestions: list[str]) -> list[NextAction]:
    actions: list[NextAction] = []
    if "broaden_allowed_financial_changes" in suggestions:
        actions.append(
            NextAction(
                action_type="relax_constraints",
                label="Broaden allowed changes",
                detail="Broaden allowed financial changes under the current policy.",
                primary=True,
            )
        )
    if "revise_target_profile" in suggestions:
        actions.append(
            NextAction(
                action_type="revise_profile",
                label="Revise profile",
                detail="Revise target profile values and rerun the case.",
                primary=not actions,
            )
        )
    return actions
