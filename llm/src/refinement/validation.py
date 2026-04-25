from __future__ import annotations

from typing import Any

from llm.src.refinement.delta import (
    is_empty_refinement_delta,
    validate_and_normalize_refinement_delta,
)
from llm.src.refinement.types import (
    REFINEMENT_PARSER_TASK,
    RefinementValidationResult,
)


REQUIRED_TOP_LEVEL_KEYS = (
    "task",
    "status",
    "constraint_feedback_delta",
    "ambiguities",
    "unsupported_feedback",
    "notes",
)
ALLOWED_TOP_LEVEL_KEYS = REQUIRED_TOP_LEVEL_KEYS
ALLOWED_STATUSES = {
    "apply",
    "clarification_required",
    "unsupported_feedback",
}


def validate_refinement_prediction(
    candidate: dict[str, Any] | None,
    *,
    feature_order: list[str],
    numeric_bound_fields: list[str] | tuple[str, ...] | None = None,
) -> RefinementValidationResult:
    if candidate is None:
        return RefinementValidationResult(
            is_valid=False,
            parser_status=None,
            normalized_output=None,
            normalized_delta=None,
            errors=("No parsed JSON object available.",),
            clarification_reasons=(),
            unsupported_reasons=(),
        )

    errors: list[str] = []
    missing_top_level = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in candidate]
    unexpected_top_level = sorted(key for key in candidate if key not in ALLOWED_TOP_LEVEL_KEYS)
    if missing_top_level:
        errors.append("Missing top-level keys: " + ", ".join(missing_top_level))
    if unexpected_top_level:
        errors.append("Unexpected top-level keys: " + ", ".join(unexpected_top_level))

    task = candidate.get("task")
    if task != REFINEMENT_PARSER_TASK:
        errors.append(f"task must equal {REFINEMENT_PARSER_TASK!r}")

    status = candidate.get("status")
    if status not in ALLOWED_STATUSES:
        errors.append("status must be one of " + ", ".join(repr(item) for item in sorted(ALLOWED_STATUSES)))

    ambiguities = candidate.get("ambiguities")
    if not _is_string_list(ambiguities):
        errors.append("ambiguities must be an array of strings.")
    unsupported_feedback = candidate.get("unsupported_feedback")
    if not _is_string_list(unsupported_feedback):
        errors.append("unsupported_feedback must be an array of strings.")
    notes = candidate.get("notes")
    if not _is_string_list(notes):
        errors.append("notes must be an array of strings.")

    normalized_delta, delta_errors, clarification_reasons = validate_and_normalize_refinement_delta(
        candidate.get("constraint_feedback_delta"),
        feature_order=feature_order,
        numeric_bound_fields=numeric_bound_fields,
    )
    errors.extend(delta_errors)

    if errors:
        return RefinementValidationResult(
            is_valid=False,
            parser_status=status if isinstance(status, str) else None,
            normalized_output=None,
            normalized_delta=None,
            errors=tuple(errors),
            clarification_reasons=tuple(clarification_reasons),
            unsupported_reasons=tuple(_normalize_string_list(unsupported_feedback)),
        )

    normalized_output = {
        "task": REFINEMENT_PARSER_TASK,
        "status": str(status),
        "constraint_feedback_delta": dict(normalized_delta),
        "ambiguities": _normalize_string_list(ambiguities),
        "unsupported_feedback": _normalize_string_list(unsupported_feedback),
        "notes": _normalize_string_list(notes),
    }
    if is_empty_refinement_delta(normalized_delta):
        if normalized_output["status"] == "apply":
            normalized_output["status"] = "unsupported_feedback"
            normalized_output["unsupported_feedback"] = list(normalized_output["unsupported_feedback"]) or [
                "No supported constraint update was extracted from the refinement feedback."
            ]

    return RefinementValidationResult(
        is_valid=True,
        parser_status=str(status),
        normalized_output=normalized_output,
        normalized_delta=normalized_delta,
        errors=(),
        clarification_reasons=tuple(clarification_reasons),
        unsupported_reasons=tuple(normalized_output["unsupported_feedback"]),
    )


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = " ".join(item.split()).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized
