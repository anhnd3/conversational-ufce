from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.src.conversation.types import ClarificationPayload
from llm.src.orchestration.llm_response_guard import accept_llm_rewrite
from llm.src.orchestration.user_response_payload import NextAction, UserResponsePayload
from llm.src.orchestration.user_response_text import render_deterministic_user_response_text


ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = ROOT / "llm" / "prompts"
MISSING_FIELDS_PROMPT_PATH = PROMPT_DIR / "clarification_missing_fields_system_prompt_v1.txt"
RESTART_PROMPT_PATH = PROMPT_DIR / "clarification_restart_system_prompt_v1.txt"


def build_clarification_payload(
    *,
    required_fields: list[str],
    missing_fields: list[str],
    conflicts: list[str],
    carried_forward_fields: list[str] | None = None,
    remaining_rounds: int | None = None,
    restart_required: bool = False,
    dataset_label: str = "bank profile",
) -> ClarificationPayload:
    ordered_missing = order_fields(missing_fields, required_fields)
    ordered_required = order_fields(required_fields, required_fields)
    clean_conflicts = [item for item in conflicts if item]
    clarification_type = "conflict_resolution" if clean_conflicts else "missing_information"
    reply_strategy = "start_new_case" if clean_conflicts or restart_required else "missing_fields_only"
    if carried_forward_fields is not None:
        ordered_carried_forward = order_fields(carried_forward_fields, required_fields)
    elif reply_strategy == "missing_fields_only":
        ordered_carried_forward = order_fields(
            [field for field in ordered_required if field not in ordered_missing],
            required_fields,
        )
    else:
        ordered_carried_forward = []
    if reply_strategy == "missing_fields_only":
        next_required_input = _build_missing_fields_next_input(
            dataset_label=dataset_label,
            missing_fields=ordered_missing,
            carried_forward_fields=ordered_carried_forward,
        )
    else:
        next_required_input = _build_restart_required_next_input(dataset_label=dataset_label)
    return ClarificationPayload(
        clarification_type=clarification_type,
        missing_fields=ordered_missing,
        conflicts=clean_conflicts,
        next_required_input=next_required_input,
        remaining_rounds=remaining_rounds,
        restart_required=restart_required,
        reply_strategy=reply_strategy,
        carried_forward_fields=ordered_carried_forward,
    )


def build_clarification_limit_reached_payload(*, dataset_label: str = "bank profile") -> ClarificationPayload:
    return ClarificationPayload(
        clarification_type="clarification_limit_reached",
        missing_fields=[],
        conflicts=[],
        next_required_input=_build_restart_required_next_input(dataset_label=dataset_label),
        remaining_rounds=0,
        restart_required=True,
        reply_strategy="start_new_case",
        carried_forward_fields=[],
    )


def build_user_response_payload_from_clarification(
    payload: ClarificationPayload,
    *,
    dataset_label: str,
) -> UserResponsePayload:
    facts = {
        "missing_fields": list(payload.missing_fields),
        "conflicts": list(payload.conflicts),
        "carried_forward_fields": list(payload.carried_forward_fields),
        "remaining_rounds": payload.remaining_rounds,
        "reply_strategy": payload.reply_strategy,
        "restart_required": payload.restart_required,
    }

    if payload.clarification_type == "clarification_limit_reached":
        return UserResponsePayload(
            response_kind="clarification_limit_reached",
            tone="warning",
            headline="Clarification limit reached",
            short_summary=(
                "The case could not be completed within the allowed clarification rounds."
            ),
            next_actions=[
                NextAction(
                    action_type="start_new_case",
                    label="Start new case",
                    detail=f"Start a new case with one complete corrected {dataset_label}.",
                    primary=True,
                )
            ],
            technical_facts=facts,
        )

    if payload.clarification_type == "refinement_clarification":
        return UserResponsePayload(
            response_kind="refinement_clarification",
            tone="info",
            headline="Refinement needs clarification",
            short_summary="The refinement request is still ambiguous and needs one clear follow-up.",
            next_actions=[
                NextAction(
                    action_type="provide_missing_fields",
                    label="Clarify refinement",
                    detail=payload.next_required_input,
                    fields=[],
                    primary=True,
                )
            ],
            technical_facts=facts,
        )

    if payload.reply_strategy == "start_new_case":
        return UserResponsePayload(
            response_kind="conflict",
            tone="danger",
            headline="Conflicting values were found",
            short_summary="A restart is required to avoid merging inconsistent values.",
            next_actions=[
                NextAction(
                    action_type="start_new_case",
                    label="Start new case",
                    detail=_build_restart_required_next_input(dataset_label=dataset_label),
                    primary=True,
                )
            ],
            technical_facts=facts,
        )

    return UserResponsePayload(
        response_kind="clarification_required",
        tone="info",
        headline="More profile details are required",
        short_summary="Reply only with the missing fields. Existing values will be carried forward.",
        next_actions=[
            NextAction(
                action_type="provide_missing_fields",
                label="Provide missing fields",
                detail="Reply with only the listed missing values.",
                fields=list(payload.missing_fields),
                primary=True,
            )
        ],
        technical_facts=facts,
    )


def render_clarification_text(
    payload: ClarificationPayload,
    *,
    dataset_label: str = "bank profile",
    parser_adapter=None,
) -> str:
    return render_clarification_text_with_dataset(
        payload,
        dataset_label=dataset_label,
        parser_adapter=parser_adapter,
    )


def render_clarification_text_with_dataset(
    payload: ClarificationPayload,
    *,
    dataset_label: str = "bank profile",
    parser_adapter=None,
) -> str:
    user_payload = build_user_response_payload_from_clarification(payload, dataset_label=dataset_label)
    fallback_text = render_deterministic_user_response_text(user_payload, dataset_label=dataset_label)

    if parser_adapter is not None and hasattr(parser_adapter, "generate_conversational_response"):
        system_prompt = _load_system_prompt(_prompt_path_for_payload(payload))
        user_prompt = _build_clarification_user_prompt(
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


def build_clarification_response(
    *,
    required_fields: list[str],
    missing_fields: list[str],
    conflicts: list[str],
    carried_forward_fields: list[str] | None = None,
    parser_adapter=None,
    dataset_label: str = "bank profile",
) -> dict[str, object]:
    payload = build_clarification_payload(
        required_fields=required_fields,
        missing_fields=missing_fields,
        conflicts=conflicts,
        carried_forward_fields=carried_forward_fields,
        dataset_label=dataset_label,
    )
    user_payload = build_user_response_payload_from_clarification(payload, dataset_label=dataset_label)
    return {
        "clarification_payload": payload.to_dict(),
        "user_response_payload": user_payload.to_dict(),
        "response_text": render_clarification_text(
            payload,
            parser_adapter=parser_adapter,
            dataset_label=dataset_label,
        ),
    }


def _build_clarification_user_prompt(
    payload: ClarificationPayload,
    *,
    user_response_payload: UserResponsePayload,
    dataset_label: str,
    fallback_text: str,
) -> str:
    factual_summary = {
        "dataset_label": dataset_label,
        "clarification_payload": payload.to_dict(),
        "response_payload": user_response_payload.to_dict(),
        "fallback_text": fallback_text,
        "rules": [
            "Do not ask for fields not listed in missing_fields.",
            "Do not drop any missing fields.",
            "Do not change reply_strategy.",
            "Do not suggest continuing when restart_required is true.",
            "Return plain text only.",
        ],
    }
    return (
        "Rewrite the clarification as a brief user-facing response.\n"
        "Keep the meaning exactly aligned with the factual summary.\n\n"
        f"{json.dumps(factual_summary, ensure_ascii=True, indent=2)}"
    )


def _build_missing_fields_next_input(
    *,
    dataset_label: str,
    missing_fields: list[str],
    carried_forward_fields: list[str],
) -> str:
    missing_text = _format_field_list(missing_fields) if missing_fields else f"the remaining {dataset_label} fields"
    example = _build_missing_field_example(missing_fields)
    if carried_forward_fields:
        return (
            f"Reply with only the missing fields: {missing_text}. "
            f"I will keep {_format_field_list(carried_forward_fields)}. "
            f"Example: {example}."
        )
    return f"Reply with only the missing fields: {missing_text}. Example: {example}."


def _build_restart_required_next_input(*, dataset_label: str) -> str:
    return f"Start a new case and submit one corrected {dataset_label}."


def _build_missing_field_example(missing_fields: list[str]) -> str:
    if not missing_fields:
        return "Field = value"
    sample_values = ["42", "80", "3", "2"]
    parts: list[str] = []
    for index, field_name in enumerate(missing_fields[:3]):
        sample = sample_values[index] if index < len(sample_values) else "value"
        parts.append(f"{field_name} = {sample}")
    return ", ".join(parts)


def _format_field_list(fields: list[str]) -> str:
    ordered = [field for field in fields if field]
    if not ordered:
        return ""
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) == 2:
        return f"{ordered[0]} and {ordered[1]}"
    return ", ".join(ordered[:-1]) + f", and {ordered[-1]}"


def _prompt_path_for_payload(payload: ClarificationPayload) -> Path:
    if payload.reply_strategy == "missing_fields_only":
        return MISSING_FIELDS_PROMPT_PATH
    return RESTART_PROMPT_PATH


def _load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def order_fields(fields: list[str], canonical_order: list[str]) -> list[str]:
    seen = set()
    ordered = []
    field_set = set(fields)
    for field in canonical_order:
        if field in field_set and field not in seen:
            seen.add(field)
            ordered.append(field)
    for field in fields:
        if field not in seen:
            seen.add(field)
            ordered.append(field)
    return ordered
