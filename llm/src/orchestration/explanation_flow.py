from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.src.conversation.types import ExplanationPayload
from llm.src.orchestration.llm_response_guard import accept_llm_rewrite
from llm.src.orchestration.negotiation_reason_mapper import build_negotiation_explanation
from llm.src.orchestration.user_response_payload import ChangeItem, NextAction, UserResponsePayload
from llm.src.orchestration.user_response_text import render_deterministic_user_response_text
from llm.src.runtime.reason_codes import (
    INVALID_COUNTERFACTUAL_BLOCKED,
    NO_FEASIBLE_CF_FOUND,
    NO_RECOURSE_NEEDED,
    REQUEST_CONSTRAINTS_BLOCKED,
)


ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = ROOT / "llm" / "prompts"
EXPLANATION_NO_RECOURSE_PROMPT_PATH = PROMPT_DIR / "explanation_no_recourse_system_prompt_v1.txt"
EXPLANATION_COUNTERFACTUAL_PROMPT_PATH = PROMPT_DIR / "explanation_counterfactual_system_prompt_v1.txt"
EXPLANATION_REJECT_PROMPT_PATH = PROMPT_DIR / "explanation_reject_system_prompt_v1.txt"


def build_explanation_payload(
    *,
    runtime_result: dict[str, Any],
    current_profile: dict[str, Any],
    included_suggestion_types: list[str] | None = None,
    policy=None,
    dataset_label: str = "bank profile",
) -> ExplanationPayload:
    del dataset_label
    prediction = runtime_result.get("prediction") or {}
    counterfactual = runtime_result.get("counterfactual") or {}
    reason_codes = list(runtime_result.get("reason_codes") or [])
    prediction_snapshot = {
        "predicted_label": prediction.get("predicted_label"),
        "predicted_proba": prediction.get("predicted_proba"),
    }
    controller_state = runtime_result.get("controller_state")
    candidates = counterfactual.get("candidates") if isinstance(counterfactual, dict) else []

    if reason_codes == [NO_RECOURSE_NEEDED]:
        return ExplanationPayload(
            summary_type="no_recourse_needed",
            prediction_snapshot=prediction_snapshot,
            counterfactual_summary=None,
            reason_codes=reason_codes,
            changed_fields=[],
            included_suggestion_types=[],
            next_step_suggestions=[],
        )

    first_candidate = candidates[0] if candidates else None
    if controller_state == "TERMINAL_SUCCESS" and isinstance(first_candidate, dict):
        counterfactual_summary = {
            "method": first_candidate.get("method"),
            "rank": first_candidate.get("rank"),
            "profile": dict(first_candidate.get("profile") or {}),
            "changed_fields": list(first_candidate.get("changed_features") or []),
            "profile_diff": build_profile_diff(
                current_profile=current_profile,
                candidate_profile=first_candidate.get("profile") or {},
            ),
        }
        return ExplanationPayload(
            summary_type="counterfactual_found",
            prediction_snapshot=prediction_snapshot,
            counterfactual_summary=counterfactual_summary,
            reason_codes=reason_codes,
            changed_fields=list(first_candidate.get("changed_features") or []),
            included_suggestion_types=[],
            next_step_suggestions=[],
        )

    suggestion_types = list(included_suggestion_types or [])
    return ExplanationPayload(
        summary_type="runtime_reject",
        prediction_snapshot=prediction_snapshot,
        counterfactual_summary=None,
        reason_codes=reason_codes,
        changed_fields=[],
        included_suggestion_types=suggestion_types,
        next_step_suggestions=build_next_step_suggestions(suggestion_types=suggestion_types, policy=policy),
    )


def build_user_response_payload_from_explanation(
    *,
    explanation_payload: ExplanationPayload,
    runtime_result: dict[str, Any],
    current_profile: dict[str, Any],
    policy=None,
    dataset_label: str,
    active_constraint_spec: dict[str, Any] | None = None,
    transition_reason: str | None = None,
) -> UserResponsePayload:
    del current_profile
    reason_codes = list(explanation_payload.reason_codes)

    if explanation_payload.summary_type == "no_recourse_needed":
        return UserResponsePayload(
            response_kind="no_recourse_needed",
            tone="success",
            headline=f"Your current {dataset_label} already reaches the desired outcome",
            short_summary="No changes are needed right now.",
            changed_items=[],
            blocked_reasons=[],
            constraint_effects=[],
            next_actions=[
                NextAction(
                    action_type="none",
                    label="No action required",
                    detail="This case is already complete.",
                    primary=True,
                )
            ],
            technical_facts={"reason_codes": reason_codes},
        )

    if explanation_payload.summary_type == "counterfactual_found":
        changed_items = _build_changed_items(explanation_payload)
        return UserResponsePayload(
            response_kind="counterfactual_found",
            tone="success",
            headline="A valid improvement path was found",
            short_summary="The recommendation below was validated before being shown.",
            changed_items=changed_items,
            blocked_reasons=[],
            constraint_effects=[],
            next_actions=[
                NextAction(
                    action_type="refine_recommendation",
                    label="Refine recommendation",
                    detail="Refine this recommendation in the same case if needed.",
                    primary=True,
                )
            ],
            technical_facts={
                "reason_codes": reason_codes,
                "profile_diff": dict((explanation_payload.counterfactual_summary or {}).get("profile_diff") or {}),
                "prediction_snapshot": dict(explanation_payload.prediction_snapshot),
            },
        )

    blocked_reasons, constraint_effects, next_actions = build_negotiation_explanation(
        transition_reason=transition_reason,
        reason_codes=reason_codes,
        active_constraint_spec=active_constraint_spec,
        policy=policy,
        included_suggestion_types=list(explanation_payload.included_suggestion_types),
    )

    response_kind = "runtime_reject_no_feasible_cf"
    tone = "warning"
    headline = "No recommendation is available under the current request"
    short_summary = "No checked candidate reached the desired outcome under the current runtime rules."
    if INVALID_COUNTERFACTUAL_BLOCKED in reason_codes:
        response_kind = "runtime_reject_invalid_counterfactual_blocked"
        tone = "danger"
        headline = "A generated recommendation was blocked by validation"
        short_summary = "A candidate was generated but failed post-generation validation checks."
    elif REQUEST_CONSTRAINTS_BLOCKED in reason_codes:
        response_kind = "runtime_reject_constraints_blocked"
        tone = "warning"
        headline = "Active constraints blocked recommendation exposure"
        short_summary = "Candidates were generated, but active constraints prevented showing them."
    elif NO_FEASIBLE_CF_FOUND in reason_codes:
        response_kind = "runtime_reject_no_feasible_cf"
        tone = "warning"
        headline = "No feasible recommendation was found"
        short_summary = "No checked candidate reached the desired outcome under the current policy."

    return UserResponsePayload(
        response_kind=response_kind,
        tone=tone,
        headline=headline,
        short_summary=short_summary,
        changed_items=[],
        blocked_reasons=blocked_reasons,
        constraint_effects=constraint_effects,
        next_actions=next_actions,
        technical_facts={
            "reason_codes": reason_codes,
            "prediction_snapshot": dict(explanation_payload.prediction_snapshot),
            "runtime_result": dict(runtime_result or {}),
        },
    )


def render_explanation_text_with_dataset(
    payload: ExplanationPayload,
    *,
    dataset_label: str = "bank profile",
    parser_adapter=None,
    runtime_result: dict[str, Any] | None = None,
    current_profile: dict[str, Any] | None = None,
    policy=None,
    active_constraint_spec: dict[str, Any] | None = None,
    transition_reason: str | None = None,
) -> str:
    user_payload = build_user_response_payload_from_explanation(
        explanation_payload=payload,
        runtime_result=dict(runtime_result or {}),
        current_profile=dict(current_profile or {}),
        policy=policy,
        dataset_label=dataset_label,
        active_constraint_spec=active_constraint_spec,
        transition_reason=transition_reason,
    )
    fallback_text = render_deterministic_user_response_text(
        user_payload,
        dataset_label=dataset_label,
    )

    if parser_adapter is not None and hasattr(parser_adapter, "generate_conversational_response"):
        system_prompt = _load_system_prompt(_prompt_path_for_payload(payload))
        user_prompt = _build_explanation_user_prompt(
            payload,
            user_response_payload=user_payload,
            dataset_label=dataset_label,
            fallback_text=fallback_text,
        )

        try:
            llm_text = parser_adapter.generate_conversational_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=112,
            )
            return accept_llm_rewrite(
                llm_text=str(llm_text or ""),
                fallback_text=fallback_text,
                payload=user_payload,
            )
        except Exception:
            pass

    return fallback_text


def render_explanation_text(
    payload: ExplanationPayload,
    *,
    dataset_label: str = "bank profile",
    parser_adapter=None,
    runtime_result: dict[str, Any] | None = None,
    current_profile: dict[str, Any] | None = None,
    policy=None,
    active_constraint_spec: dict[str, Any] | None = None,
    transition_reason: str | None = None,
) -> str:
    return render_explanation_text_with_dataset(
        payload,
        dataset_label=dataset_label,
        parser_adapter=parser_adapter,
        runtime_result=runtime_result,
        current_profile=current_profile,
        policy=policy,
        active_constraint_spec=active_constraint_spec,
        transition_reason=transition_reason,
    )


def build_explanation_response(
    *,
    runtime_result: dict[str, Any],
    current_profile: dict[str, Any],
    included_suggestion_types: list[str] | None = None,
    policy=None,
    parser_adapter=None,
    dataset_label: str = "bank profile",
    active_constraint_spec: dict[str, Any] | None = None,
    transition_reason: str | None = None,
) -> dict[str, object]:
    payload = build_explanation_payload(
        runtime_result=runtime_result,
        current_profile=current_profile,
        included_suggestion_types=included_suggestion_types,
        policy=policy,
        dataset_label=dataset_label,
    )
    user_payload = build_user_response_payload_from_explanation(
        explanation_payload=payload,
        runtime_result=runtime_result,
        current_profile=current_profile,
        policy=policy,
        dataset_label=dataset_label,
        active_constraint_spec=active_constraint_spec,
        transition_reason=transition_reason,
    )
    return {
        "explanation_payload": payload.to_dict(),
        "user_response_payload": user_payload.to_dict(),
        "response_text": render_explanation_text(
            payload,
            dataset_label=dataset_label,
            parser_adapter=parser_adapter,
            runtime_result=runtime_result,
            current_profile=current_profile,
            policy=policy,
            active_constraint_spec=active_constraint_spec,
            transition_reason=transition_reason,
        ),
    }


def _build_explanation_user_prompt(
    payload: ExplanationPayload,
    *,
    user_response_payload: UserResponsePayload,
    dataset_label: str,
    fallback_text: str,
) -> str:
    factual_summary = {
        "dataset_label": dataset_label,
        "explanation_payload": payload.to_dict(),
        "response_payload": user_response_payload.to_dict(),
        "fallback_text": fallback_text,
        "rules": [
            "Keep all field names exactly as provided.",
            "Keep all before/after values exactly as provided.",
            "Do not add recommendations that are not listed.",
            "Do not change success/reject meaning.",
            "Return plain text only.",
        ],
    }
    return (
        "Rewrite the explanation as a brief user-facing response.\n"
        "Use only the factual summary below.\n"
        "Do not invent new fields, values, constraints, or outcomes.\n\n"
        f"{json.dumps(factual_summary, ensure_ascii=True, indent=2)}"
    )


def _build_changed_items(payload: ExplanationPayload) -> list[ChangeItem]:
    summary = payload.counterfactual_summary or {}
    profile_diff = summary.get("profile_diff")
    if not isinstance(profile_diff, dict):
        return []
    items: list[ChangeItem] = []
    for field_name, values in profile_diff.items():
        if not isinstance(field_name, str) or not isinstance(values, dict):
            continue
        if "from" not in values or "to" not in values:
            continue
        before = values.get("from")
        after = values.get("to")
        items.append(
            ChangeItem(
                field_name=field_name,
                display_name=field_name,
                before=before,
                after=after,
                direction=_direction_for_change(before=before, after=after),
            )
        )
    return items


def _direction_for_change(*, before: Any, after: Any) -> str:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if after > before:
            return "increase"
        if after < before:
            return "decrease"
        return "change"
    if isinstance(before, bool) and isinstance(after, bool):
        if before is False and after is True:
            return "enable"
        if before is True and after is False:
            return "disable"
    return "unknown"


def build_profile_diff(
    *,
    current_profile: dict[str, Any],
    candidate_profile: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    diff: dict[str, dict[str, Any]] = {}
    for field_name, factual_value in current_profile.items():
        candidate_value = candidate_profile.get(field_name)
        if candidate_value != factual_value:
            diff[field_name] = {
                "from": factual_value,
                "to": candidate_value,
            }
    return diff


def build_next_step_suggestions(*, suggestion_types: list[str], policy=None) -> list[str]:
    suggestions: list[str] = []
    for suggestion_type in suggestion_types:
        if suggestion_type == "revise_target_profile":
            subject_label = "profile"
            if policy is not None and getattr(policy, "dataset_name", None):
                subject_label = f"{policy.dataset_name} profile"
            suggestions.append(f"Revise the requested target values and resubmit a less restrictive {subject_label}.")
        elif suggestion_type == "broaden_allowed_financial_changes":
            allowed_fields = []
            if policy is not None:
                allowed_fields = list(getattr(policy, "f2change", []) or [])
            dataset_name = "current dataset"
            if policy is not None and getattr(policy, "dataset_name", None):
                dataset_name = f"{policy.dataset_name} policy"
            allowed_text = ", ".join(allowed_fields) if allowed_fields else f"the current {dataset_name} changeable fields"
            suggestions.append(
                "If acceptable, broaden allowed changes within the current dataset policy assumptions: "
                f"{allowed_text}."
            )
    return suggestions


def _prompt_path_for_payload(payload: ExplanationPayload) -> Path:
    if payload.summary_type == "no_recourse_needed":
        return EXPLANATION_NO_RECOURSE_PROMPT_PATH
    if payload.summary_type == "counterfactual_found":
        return EXPLANATION_COUNTERFACTUAL_PROMPT_PATH
    return EXPLANATION_REJECT_PROMPT_PATH


def _load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")
