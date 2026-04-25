from __future__ import annotations

from typing import Any

from llm.src.runtime.contracts import SessionNegotiationState


def merge_session_state(
    state: SessionNegotiationState,
    *,
    profile_updates: dict[str, Any] | None = None,
    hard_constraints: dict[str, Any] | None = None,
    soft_preferences: dict[str, Any] | None = None,
    rejected_features: list[str] | None = None,
    rejected_candidate_ids: list[str] | None = None,
    accepted_tradeoffs: list[dict[str, Any]] | None = None,
    terminal_status: str | None = None,
    last_reason_codes: list[str] | None = None,
    backend_id: str | None = None,
) -> SessionNegotiationState:
    if profile_updates:
        state.profile_facts.update(profile_updates)
    if hard_constraints:
        state.hard_constraints.update(hard_constraints)
    if soft_preferences:
        state.soft_preferences.update(soft_preferences)
    if rejected_features:
        for feature in rejected_features:
            if feature not in state.rejected_features:
                state.rejected_features.append(feature)
    if rejected_candidate_ids:
        for candidate_id in rejected_candidate_ids:
            if candidate_id not in state.rejected_candidate_ids:
                state.rejected_candidate_ids.append(candidate_id)
    if accepted_tradeoffs:
        for tradeoff in accepted_tradeoffs:
            state.accepted_tradeoffs.append(dict(tradeoff))
    if terminal_status is not None:
        state.terminal_status = terminal_status
    if last_reason_codes is not None:
        state.last_reason_codes = list(last_reason_codes)
    if backend_id is not None:
        state.backend_id = backend_id
    state.turn_count += 1
    return state
