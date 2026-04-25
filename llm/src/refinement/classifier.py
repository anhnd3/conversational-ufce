from __future__ import annotations

import re
from typing import Any

from llm.src.refinement.types import (
    REFINEMENT_STATUS_APPLIED,
    REFINEMENT_STATUS_CLARIFICATION_REQUIRED,
    REFINEMENT_STATUS_LIMIT_REACHED,
    REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK,
)


VAGUE_REFINEMENT_CLARIFICATION_REASON = (
    "The feedback asks for improvement without specifying allowed fields, blocked fields, bounds, or a change limit."
)

_EXPLICIT_SOFT_PREFERENCE_PATTERNS = (
    re.compile(r"\bprefer\s+fewer\s+changes\b", re.IGNORECASE),
    re.compile(r"\bprefer\s+smaller\s+edits?\b", re.IGNORECASE),
)
_VAGUE_GOAL_PATTERNS = (
    re.compile(r"\bmake\s+it\s+better\b", re.IGNORECASE),
    re.compile(r"\bmake\s+the\s+[a-z\s]+?\s+result\s+better\b", re.IGNORECASE),
    re.compile(r"\bimprove\s+it\b", re.IGNORECASE),
    re.compile(r"\bimprove\s+the\s+[a-z\s]+?\s+result\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+changing\s+too\s+much\b", re.IGNORECASE),
)
_CONCRETE_DELTA_KEYS = frozenset(
    {
        "add_blocked_fields",
        "remove_blocked_fields",
        "set_numeric_bounds",
        "clear_numeric_bounds",
        "set_max_changed_features",
        "clear_max_changed_features",
    }
)


def classify_refinement_outcome(
    *,
    user_feedback: str,
    parser_status: str | None,
    normalized_delta: dict[str, Any] | None,
    clarification_reasons: list[str] | tuple[str, ...] | None,
    unsupported_reasons: list[str] | tuple[str, ...] | None,
    limit_reached: bool = False,
) -> str:
    if limit_reached:
        return REFINEMENT_STATUS_LIMIT_REACHED
    if parser_status == "unsupported_feedback" or bool(unsupported_reasons):
        return REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK
    if parser_status == "clarification_required" or bool(clarification_reasons):
        return REFINEMENT_STATUS_CLARIFICATION_REQUIRED
    if parser_status == "apply" and feedback_requires_refinement_clarification(
        user_feedback=user_feedback,
        normalized_delta=normalized_delta,
    ):
        return REFINEMENT_STATUS_CLARIFICATION_REQUIRED
    return REFINEMENT_STATUS_APPLIED


def build_refinement_clarification_reasons(
    *,
    user_feedback: str,
    parser_status: str | None,
    normalized_delta: dict[str, Any] | None,
    clarification_reasons: list[str] | tuple[str, ...] | None,
    parser_ambiguities: list[str] | tuple[str, ...] | None,
) -> list[str]:
    reasons = _dedupe_strings([*(clarification_reasons or ()), *(parser_ambiguities or ())])
    if reasons:
        return reasons
    if parser_status == "apply" and feedback_requires_refinement_clarification(
        user_feedback=user_feedback,
        normalized_delta=normalized_delta,
    ):
        return [VAGUE_REFINEMENT_CLARIFICATION_REASON]
    return []


def feedback_requires_refinement_clarification(
    *,
    user_feedback: str,
    normalized_delta: dict[str, Any] | None,
) -> bool:
    if not _contains_vague_goal_language(user_feedback):
        return False
    return not has_concrete_refinement_delta(
        user_feedback=user_feedback,
        normalized_delta=normalized_delta,
    )


def has_concrete_refinement_delta(
    *,
    user_feedback: str,
    normalized_delta: dict[str, Any] | None,
) -> bool:
    delta = dict(normalized_delta or {})
    for key in _CONCRETE_DELTA_KEYS:
        value = delta.get(key)
        if value not in (None, False, [], {}):
            return True
    if any(key in delta for key in ("set_prefer_fewer_changes", "clear_prefer_fewer_changes")):
        return _contains_explicit_soft_preference_language(user_feedback)
    return False


def _contains_explicit_soft_preference_language(user_feedback: str) -> bool:
    text = _normalize_feedback_text(user_feedback)
    return any(pattern.search(text) for pattern in _EXPLICIT_SOFT_PREFERENCE_PATTERNS)


def _contains_vague_goal_language(user_feedback: str) -> bool:
    text = _normalize_feedback_text(user_feedback)
    return any(pattern.search(text) for pattern in _VAGUE_GOAL_PATTERNS)


def _normalize_feedback_text(user_feedback: str) -> str:
    return " ".join(str(user_feedback or "").split())


def _dedupe_strings(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output
