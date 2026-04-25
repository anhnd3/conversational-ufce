from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm.src.conversation.types import ClarificationPayload


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
    fallback_text = _build_programmatic_clarification_text(payload, dataset_label=dataset_label)

    if parser_adapter is not None and hasattr(parser_adapter, "generate_conversational_response"):
        system_prompt = _load_system_prompt(_prompt_path_for_payload(payload))
        user_prompt = _build_clarification_user_prompt(
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


def build_clarification_response(
    *,
    required_fields: list[str],
    missing_fields: list[str],
    conflicts: list[str],
    carried_forward_fields: list[str] | None = None,
    parser_adapter=None,
) -> dict[str, object]:
    payload = build_clarification_payload(
        required_fields=required_fields,
        missing_fields=missing_fields,
        conflicts=conflicts,
        carried_forward_fields=carried_forward_fields,
    )
    return {
        "clarification_payload": payload.to_dict(),
        "response_text": render_clarification_text(
            payload,
            parser_adapter=parser_adapter,
        ),
    }


def _build_clarification_user_prompt(
    payload: ClarificationPayload,
    *,
    dataset_label: str,
    fallback_text: str,
) -> str:
    factual_summary = {
        "dataset_label": dataset_label,
        "clarification_payload": payload.to_dict(),
        "fallback_meaning": fallback_text,
    }
    return (
        "Rewrite the clarification for the end user as a brief, natural response.\n"
        "Keep the meaning exactly aligned with the factual summary.\n"
        "Do not invent any missing values or change the reply strategy.\n"
        "Return plain text only.\n\n"
        f"{json.dumps(factual_summary, ensure_ascii=True, indent=2)}"
    )


def _build_programmatic_clarification_text(payload: ClarificationPayload, *, dataset_label: str = "bank profile") -> str:
    if payload.clarification_type == "clarification_limit_reached":
        return (
            "I couldn't resolve this case within the clarification limit. "
            f"Please start a new case and submit one complete {dataset_label}."
        )

    if payload.reply_strategy == "start_new_case":
        conflict_text = "; ".join(payload.conflicts) if payload.conflicts else None
        if conflict_text:
            return (
                "Your request contains conflicting instructions: "
                f"{conflict_text}. "
                f"{_build_restart_required_next_input(dataset_label=dataset_label)}"
            )
        return (
            f"I can't continue this case as-is. "
            f"{_build_restart_required_next_input(dataset_label=dataset_label)}"
        )

    missing_text = _format_field_list(payload.missing_fields) if payload.missing_fields else f"the remaining {dataset_label} fields"
    carried_text = _format_field_list(payload.carried_forward_fields)
    if carried_text:
        return (
            f"Reply with only the missing fields: {missing_text}. "
            f"I'll keep the values already provided for {carried_text}."
        )
    return (
        f"Reply with only the missing fields: {missing_text}."
    )


def _build_missing_fields_next_input(
    *,
    dataset_label: str,
    missing_fields: list[str],
    carried_forward_fields: list[str],
) -> str:
    missing_text = _format_field_list(missing_fields) if missing_fields else f"the remaining {dataset_label} fields"
    if carried_forward_fields:
        return (
            f"Reply with only the missing fields: {missing_text}. "
            f"I'll keep the values already provided for {_format_field_list(carried_forward_fields)}."
        )
    return f"Reply with only the missing fields: {missing_text}."


def _build_restart_required_next_input(*, dataset_label: str) -> str:
    return f"Start a new case and submit one corrected {dataset_label}."


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
