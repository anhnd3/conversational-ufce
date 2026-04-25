from __future__ import annotations

from typing import Any

from llm.src.runtime.contracts import SessionNegotiationState


def build_session_state_from_turn(
    *,
    session_id: str,
    dataset_id: str,
    backend_id: str | None,
    runtime_request: dict[str, Any] | None,
    active_constraint_spec: dict[str, Any] | None,
    reason_codes: list[str] | None,
    terminal_status: str | None,
    prior_state: dict[str, Any] | None = None,
) -> SessionNegotiationState:
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
    if isinstance(runtime_request, dict):
        profile = runtime_request.get("profile")
        if isinstance(profile, dict):
            state.profile_facts.update(profile)
    if isinstance(active_constraint_spec, dict):
        state.hard_constraints.update(active_constraint_spec)
    state.last_reason_codes = list(reason_codes or [])
    state.terminal_status = terminal_status
    state.backend_id = backend_id
    state.turn_count += 1
    return state
