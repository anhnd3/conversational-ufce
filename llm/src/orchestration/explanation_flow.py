from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.src.conversation.types import ExplanationPayload
from llm.src.runtime.reason_codes import REQUEST_CONSTRAINTS_BLOCKED


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
    prediction = runtime_result.get("prediction") or {}
    counterfactual = runtime_result.get("counterfactual") or {}
    reason_codes = list(runtime_result.get("reason_codes") or [])
    prediction_snapshot = {
        "predicted_label": prediction.get("predicted_label"),
        "predicted_proba": prediction.get("predicted_proba"),
    }
    controller_state = runtime_result.get("controller_state")
    candidates = counterfactual.get("candidates") if isinstance(counterfactual, dict) else []

    if reason_codes == ["NO_RECOURSE_NEEDED"]:
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


def _build_programmatic_explanation(payload: ExplanationPayload, *, dataset_label: str = "bank profile") -> str:
    label = payload.prediction_snapshot.get("predicted_label")
    proba = payload.prediction_snapshot.get("predicted_proba")
    prediction_text = (
        "prediction label {0} at probability {1:.4f}".format(
            label,
            float(proba),
        )
        if isinstance(proba, (int, float))
        else "prediction unavailable"
    )

    if payload.summary_type == "no_recourse_needed":
        return (
            f"Your current {dataset_label} already reaches the desired outcome. "
            f"Current {prediction_text}. No recourse changes are needed."
        )

    if payload.summary_type == "counterfactual_found":
        summary = payload.counterfactual_summary or {}
        diff = summary.get("profile_diff") or {}
        if diff:
            ordered_changes = ", ".join(
                "{0}: {1} -> {2}".format(field, values["from"], values["to"])
                for field, values in diff.items()
            )
        else:
            ordered_changes = "no feature changes recorded"
        return (
            "A feasible counterfactual was found for your current profile. "
            f"Current {prediction_text}. "
            "Using the first runtime candidate only, the suggested changes are: "
            f"{ordered_changes}."
        )

    reason_text = ", ".join(payload.reason_codes) if payload.reason_codes else "no reason code provided"
    suggestion_text = ""
    if payload.next_step_suggestions:
        suggestion_text = " Optional next steps: " + " ".join(payload.next_step_suggestions)

    if "INVALID_COUNTERFACTUAL_BLOCKED" in payload.reason_codes:
        return (
            "The system could not safely present a recommendation for this request. "
            f"Current {prediction_text}. "
            "A runtime candidate was generated, but it did not pass the post-generation safety checks."
        )
    if REQUEST_CONSTRAINTS_BLOCKED in payload.reason_codes:
        return (
            "Runtime generated candidate options, but none could be shown under the current request-specific constraints. "
            f"Current {prediction_text}. "
            "All generated options were blocked by the current immutable fields, disallowed changes, final-value bounds, or maximum-change limit."
            f"{suggestion_text}"
        )
    return (
        "Runtime completed without a feasible counterfactual. "
        f"Current {prediction_text}. "
        f"Reason codes: {reason_text}."
        f"{suggestion_text}"
    )


def render_explanation_text_with_dataset(
    payload: ExplanationPayload,
    *,
    dataset_label: str = "bank profile",
    parser_adapter=None,
) -> str:
    fallback_text = _build_programmatic_explanation(payload, dataset_label=dataset_label)

    if parser_adapter is not None and hasattr(parser_adapter, "generate_conversational_response"):
        system_prompt = _load_system_prompt(_prompt_path_for_payload(payload))
        user_prompt = _build_explanation_user_prompt(
            payload,
            dataset_label=dataset_label,
            fallback_text=fallback_text,
        )

        try:
            llm_text = parser_adapter.generate_conversational_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=112,
            )
            if llm_text:
                return llm_text
        except Exception:
            pass

    return fallback_text


def render_explanation_text(
    payload: ExplanationPayload,
    *,
    dataset_label: str = "bank profile",
    parser_adapter=None,
) -> str:
    return render_explanation_text_with_dataset(
        payload,
        dataset_label=dataset_label,
        parser_adapter=parser_adapter,
    )


def build_explanation_response(
    *,
    runtime_result: dict[str, Any],
    current_profile: dict[str, Any],
    included_suggestion_types: list[str] | None = None,
    policy=None,
    parser_adapter=None,
) -> dict[str, object]:
    payload = build_explanation_payload(
        runtime_result=runtime_result,
        current_profile=current_profile,
        included_suggestion_types=included_suggestion_types,
        policy=policy,
    )
    return {
        "explanation_payload": payload.to_dict(),
        "response_text": render_explanation_text(
            payload,
            parser_adapter=parser_adapter,
        ),
    }


def _build_explanation_user_prompt(
    payload: ExplanationPayload,
    *,
    dataset_label: str,
    fallback_text: str,
) -> str:
    factual_summary = {
        "dataset_label": dataset_label,
        "explanation_payload": payload.to_dict(),
        "fallback_meaning": fallback_text,
    }
    return (
        "Rewrite the explanation as a brief, natural response to the user.\n"
        "Keep the meaning exactly aligned with the factual summary.\n"
        "Do not invent new facts or change the outcome classification.\n"
        "Return plain text only.\n\n"
        f"{json.dumps(factual_summary, ensure_ascii=True, indent=2)}"
    )


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
