from __future__ import annotations

import time
from typing import Any

from llm.src.conversation.canonical_session_state import (
    build_canonical_session_state_for_turn_result,
    combine_constraint_buckets,
)
from llm.src.conversation.types import ConversationStage, PendingClarification
from llm.src.utils.hashing import make_run_id
from dataclasses import dataclass


CLARIFICATION_TURN_LIMIT = 3


class SessionCaseCompleteError(RuntimeError):
    pass


@dataclass
class InteractiveSessionState:
    session_id: str
    dataset_id: str = "bank"
    turn_index: int = 0
    pending_clarification: PendingClarification | None = None
    canonical_session_state: dict[str, Any] | None = None
    clarification_turns_used: int = 0
    clarification_turn_limit: int = CLARIFICATION_TURN_LIMIT
    is_case_complete: bool = False
    case_completion_reason: str | None = None
    restart_required: bool = False


def create_interactive_session_state(session_id: str | None = None, dataset_id: str = "bank") -> InteractiveSessionState:
    active_session_id = session_id or make_run_id().replace("run_", "session_")
    return InteractiveSessionState(session_id=active_session_id, dataset_id=dataset_id)


def handle_session_turn(
    orchestrator,
    state: InteractiveSessionState,
    *,
    user_input: str,
    save_artifacts: bool,
    scenario_slug: str | None,
    debug_trace_enabled: bool,
    command: str | None,
    dataset_id: str | None = None,
):
    started = time.perf_counter()
    if state.is_case_complete:
        raise SessionCaseCompleteError(
            "This case is complete. Start a new case before sending another message."
        )
    active_dataset_id = str(dataset_id or state.dataset_id or "bank").strip().lower() or "bank"
    state.dataset_id = active_dataset_id
    state.turn_index += 1
    result = orchestrator.finalize_turn(
        **orchestrator.prepare_turn(user_input=user_input, dataset_id=active_dataset_id),
        user_input=user_input,
        save_artifacts=save_artifacts,
        scenario_slug=scenario_slug,
        debug_trace_enabled=debug_trace_enabled,
        command=command,
        session_trace=build_session_trace(state),
        pending_clarification=state.pending_clarification,
        canonical_session_state=state.canonical_session_state,
        clarification_turns_used=state.clarification_turns_used,
        clarification_turn_limit=state.clarification_turn_limit,
        dataset_id=state.dataset_id,
    )
    state.clarification_turns_used = result.clarification_turns_used
    state.is_case_complete = bool(result.is_case_complete)
    state.case_completion_reason = result.case_completion_reason
    state.restart_required = bool(result.restart_required)
    state.canonical_session_state = build_canonical_session_state_for_turn_result(
        session_id=state.session_id,
        prior_state=state.canonical_session_state,
        result=result,
        backend_id=getattr(orchestrator.runtime_orchestrator, "counterfactual_backend_name", "ufce"),
        feature_order=list(orchestrator.canonical_validator.required_fields),
        dataset_id=active_dataset_id,
    )["state"]
    state.pending_clarification = build_pending_clarification(
        orchestrator,
        result,
        canonical_session_state=state.canonical_session_state,
    )
    timing_metrics = dict(result.timing_metrics or {})
    timing_metrics["end_to_end_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    result.timing_metrics = timing_metrics
    return result


def build_session_trace(
    state: InteractiveSessionState,
) -> dict[str, Any]:
    return {
        "session_id": state.session_id,
        "dataset_id": state.dataset_id,
        "turn_index": state.turn_index,
        "parent_turn_id": None,
        "merge_applied": False,
        "carried_fields": [],
        "carried_constraint_keys": [],
        "clarification_turns_used": state.clarification_turns_used,
    }


def build_pending_clarification(
    orchestrator,
    result,
    *,
    canonical_session_state: dict[str, Any] | None,
) -> PendingClarification | None:
    payload = result.clarification_payload
    state_payload = dict(canonical_session_state or {})
    current_request = state_payload.get("profile_facts")
    current_constraint_spec = combine_constraint_buckets(
        hard_constraints=state_payload.get("hard_constraints"),
        soft_preferences=state_payload.get("soft_preferences"),
    )
    builder_provenance = result.builder_result.provenance if result.builder_result is not None else {}
    current_field_provenance = (
        builder_provenance.get("field_provenance")
        if isinstance(builder_provenance, dict)
        else {}
    )
    if result.is_case_complete:
        return None
    if result.stage != ConversationStage.NEEDS_CLARIFICATION:
        return None
    if payload is None or payload.clarification_type != "missing_information":
        return None
    if not isinstance(current_request, dict) or not current_request:
        return None
    return PendingClarification(
        prior_cf_request=dict(current_request),
        prior_constraint_spec=dict(current_constraint_spec),
        missing_fields=list(payload.missing_fields),
        required_field_order=list(orchestrator.canonical_validator.required_fields),
        originating_turn_id=result.turn_id,
        prior_field_provenance=(
            {
                str(field_name): str(value)
                for field_name, value in current_field_provenance.items()
                if isinstance(field_name, str) and isinstance(value, str)
            }
            if isinstance(current_field_provenance, dict)
            else {}
        ),
    )
