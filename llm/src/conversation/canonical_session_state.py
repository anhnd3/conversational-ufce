from __future__ import annotations

from typing import Any

from llm.src.conversation.types import ConversationStage
from llm.src.refinement.delta import build_active_constraint_spec
from llm.src.runtime.contracts import SessionNegotiationState


SOFT_PREFERENCE_KEYS = ("prefer_fewer_changes",)
USABLE_CANONICAL_STAGES = frozenset(
    {
        ConversationStage.NEEDS_CLARIFICATION,
        ConversationStage.READY_FOR_RUNTIME,
    }
)
RESET_DECISION_FRESH_REQUEST = "fresh_request"
RESET_DECISION_RESET_NO_MERGE = "reset_no_merge"


def split_constraint_buckets(
    constraint_spec: dict[str, Any] | None,
    *,
    feature_order: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(constraint_spec, dict):
        return {}, {}
    try:
        active_constraint_spec = build_active_constraint_spec(constraint_spec, feature_order=feature_order)
    except Exception:
        active_constraint_spec = {
            str(key): value
            for key, value in constraint_spec.items()
            if isinstance(key, str)
        }
    hard_constraints = dict(active_constraint_spec)
    soft_preferences: dict[str, Any] = {}
    for key in SOFT_PREFERENCE_KEYS:
        if key in hard_constraints:
            soft_preferences[key] = hard_constraints.pop(key)
    return hard_constraints, soft_preferences


def combine_constraint_buckets(
    *,
    hard_constraints: dict[str, Any] | None,
    soft_preferences: dict[str, Any] | None,
) -> dict[str, Any]:
    combined = {
        str(key): value
        for key, value in dict(hard_constraints or {}).items()
        if isinstance(key, str)
    }
    for key, value in dict(soft_preferences or {}).items():
        if isinstance(key, str):
            combined[key] = value
    return combined


def extract_canonical_buckets_from_turn_result(
    *,
    result,
    feature_order: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    builder_result = getattr(result, "builder_result", None)
    runtime_request = None if builder_result is None else builder_result.runtime_request

    profile_facts: dict[str, Any] = {}
    if isinstance(runtime_request, dict) and isinstance(runtime_request.get("profile"), dict):
        profile_facts = {
            field_name: runtime_request["profile"][field_name]
            for field_name in feature_order
            if field_name in runtime_request["profile"]
        }
        hard_constraints, soft_preferences = split_constraint_buckets(
            runtime_request.get("constraint_spec"),
            feature_order=feature_order,
        )
        return profile_facts, hard_constraints, soft_preferences

    normalized_parse = getattr(result, "normalized_parse", None)
    if not isinstance(normalized_parse, dict):
        return {}, {}, {}
    cf_request = normalized_parse.get("cf_request")
    if isinstance(cf_request, dict):
        profile_facts = {
            field_name: cf_request[field_name]
            for field_name in feature_order
            if field_name in cf_request
        }
    hard_constraints, soft_preferences = split_constraint_buckets(
        normalized_parse.get("constraint_spec"),
        feature_order=feature_order,
    )
    return profile_facts, hard_constraints, soft_preferences


def build_canonical_session_state_for_turn_result(
    *,
    session_id: str,
    prior_state: dict[str, Any] | None,
    result,
    backend_id: str | None,
    feature_order: list[str],
    dataset_id: str = "bank",
) -> dict[str, Any]:
    payload = dict(prior_state or {})
    state = SessionNegotiationState(
        session_id=session_id,
        dataset_id=str(payload.get("dataset_id", dataset_id)),
        backend_id=payload.get("backend_id", backend_id),
        profile_facts=dict(payload.get("profile_facts") or {}),
        hard_constraints=dict(payload.get("hard_constraints") or {}),
        soft_preferences=dict(payload.get("soft_preferences") or {}),
        rejected_candidate_ids=list(payload.get("rejected_candidate_ids") or []),
        rejected_features=list(payload.get("rejected_features") or []),
        accepted_tradeoffs=list(payload.get("accepted_tradeoffs") or []),
        last_reason_codes=list(payload.get("last_reason_codes") or []),
        terminal_status=payload.get("terminal_status"),
        turn_count=int(payload.get("turn_count") or 0),
        canonical_mirror_ok=bool(payload.get("canonical_mirror_ok", True)),
    )
    builder_result = getattr(result, "builder_result", None)
    builder_status = None if builder_result is None else builder_result.builder_status
    builder_provenance = {} if builder_result is None else dict(builder_result.provenance or {})
    reset_decision = str(builder_provenance.get("reset_decision") or "none")
    should_mutate = builder_status in USABLE_CANONICAL_STAGES

    if reset_decision == RESET_DECISION_FRESH_REQUEST:
        state.profile_facts = {}
        state.hard_constraints = {}
        state.soft_preferences = {}
        state.rejected_candidate_ids = []
        state.rejected_features = []
        state.accepted_tradeoffs = []

    extracted_profile_facts: dict[str, Any] = {}
    extracted_hard_constraints: dict[str, Any] = {}
    extracted_soft_preferences: dict[str, Any] = {}
    if should_mutate:
        (
            extracted_profile_facts,
            extracted_hard_constraints,
            extracted_soft_preferences,
        ) = extract_canonical_buckets_from_turn_result(
            result=result,
            feature_order=feature_order,
        )
        state.profile_facts = dict(extracted_profile_facts)
        state.hard_constraints = dict(extracted_hard_constraints)
        state.soft_preferences = dict(extracted_soft_preferences)
    state.last_reason_codes = list(
        ((getattr(result, "runtime_result", None) or {}).get("reason_codes") or [])
    )
    state.terminal_status = getattr(result, "stage", None)
    state.backend_id = backend_id
    state.turn_count += 1

    if should_mutate:
        state.canonical_mirror_ok = (
            state.profile_facts == extracted_profile_facts
            and state.hard_constraints == extracted_hard_constraints
            and state.soft_preferences == extracted_soft_preferences
        )
    elif reset_decision == RESET_DECISION_FRESH_REQUEST:
        state.canonical_mirror_ok = True

    return {
        "state": state.to_dict(),
        "source": "canonical_authoritative",
        "mirror_ok": bool(state.canonical_mirror_ok),
    }
