from __future__ import annotations

from llm.src.orchestration.user_response_payload import UserResponsePayload


MAX_LLM_REWRITE_WORDS = 180


def accept_llm_rewrite(
    *,
    llm_text: str,
    fallback_text: str,
    payload: UserResponsePayload,
) -> str:
    candidate = str(llm_text or "").strip()
    if not candidate:
        return fallback_text
    if _too_long(candidate):
        return fallback_text
    if _missing_required_values(candidate, payload):
        return fallback_text
    if _contains_forbidden_outcome_flip(candidate, payload):
        return fallback_text
    return candidate


def _too_long(text: str) -> bool:
    return len(text.split()) > MAX_LLM_REWRITE_WORDS


def _missing_required_values(text: str, payload: UserResponsePayload) -> bool:
    lowered = text.lower()
    for item in payload.changed_items:
        if item.display_name.lower() not in lowered and item.field_name.lower() not in lowered:
            return True
        before = _normalize_value(item.before)
        after = _normalize_value(item.after)
        if before and before not in lowered:
            return True
        if after and after not in lowered:
            return True
    return False


def _contains_forbidden_outcome_flip(text: str, payload: UserResponsePayload) -> bool:
    lowered = text.lower()
    reject_kinds = {
        "runtime_reject_no_feasible_cf",
        "runtime_reject_constraints_blocked",
        "runtime_reject_invalid_counterfactual_blocked",
    }
    success_kinds = {
        "no_recourse_needed",
        "counterfactual_found",
    }
    if payload.response_kind in reject_kinds:
        if "recommendation found" in lowered or "approved" in lowered:
            return True
    if payload.response_kind in success_kinds:
        if "no recommendation" in lowered or "blocked" in lowered:
            return True
    return False

def _normalize_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return ("{0:.6f}".format(value)).rstrip("0").rstrip(".")
    return str(value).strip().lower()
