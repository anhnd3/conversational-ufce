from __future__ import annotations

from llm.src.conversation.types import ConversationStage, NegotiationTransition


MISSING_REQUIRED_FIELDS = "missing_required_fields"
CONFLICTING_VALUES = "conflicting_values"
UNSUPPORTED_INTENT = "unsupported_intent"
FOLLOWUP_RESET_NEW_REQUEST = "followup_reset_new_request"
RUNTIME_READY = "runtime_ready"
RUNTIME_SUCCESS_NO_RECOURSE = "runtime_success_no_recourse"
RUNTIME_SUCCESS_COUNTERFACTUAL_FOUND = "runtime_success_counterfactual_found"
RUNTIME_REJECT_NO_FEASIBLE_CF = "runtime_reject_no_feasible_cf"
RUNTIME_REJECT_SYSTEM_ERROR = "runtime_reject_system_error"

ALLOWED_TRANSITION_REASONS = frozenset(
    {
        MISSING_REQUIRED_FIELDS,
        CONFLICTING_VALUES,
        UNSUPPORTED_INTENT,
        FOLLOWUP_RESET_NEW_REQUEST,
        RUNTIME_READY,
        RUNTIME_SUCCESS_NO_RECOURSE,
        RUNTIME_SUCCESS_COUNTERFACTUAL_FOUND,
        RUNTIME_REJECT_NO_FEASIBLE_CF,
        RUNTIME_REJECT_SYSTEM_ERROR,
    }
)

ALLOWED_TRANSITIONS = {
    None: {
        ConversationStage.NEEDS_CLARIFICATION,
        ConversationStage.CONFLICT,
        ConversationStage.UNSUPPORTED_REQUEST,
        ConversationStage.READY_FOR_RUNTIME,
    },
    ConversationStage.READY_FOR_RUNTIME: {
        ConversationStage.RUNTIME_SUCCESS,
        ConversationStage.RUNTIME_REJECT,
    },
    ConversationStage.NEEDS_CLARIFICATION: set(),
    ConversationStage.CONFLICT: set(),
    ConversationStage.UNSUPPORTED_REQUEST: set(),
    ConversationStage.RUNTIME_SUCCESS: set(),
    ConversationStage.RUNTIME_REJECT: set(),
}


class ConversationNegotiationController:
    def __init__(self) -> None:
        self.current_state: str | None = None
        self.state_trace: list[str] = []

    def transition(
        self,
        *,
        next_state: str,
        transition_reason: str,
        merge_applied: bool,
        bounded_suggestion_available: bool,
    ) -> NegotiationTransition:
        if transition_reason not in ALLOWED_TRANSITION_REASONS:
            raise ValueError(f"Unsupported transition_reason: {transition_reason!r}")
        allowed = ALLOWED_TRANSITIONS.get(self.current_state, set())
        if next_state not in allowed:
            raise ValueError(
                "Invalid conversation transition from {0!r} to {1!r}.".format(
                    self.current_state,
                    next_state,
                )
            )
        source_state = self.current_state
        self.current_state = next_state
        self.state_trace.append(next_state)
        return NegotiationTransition(
            source_state=source_state,
            target_state=next_state,
            transition_reason=transition_reason,
            state_trace=list(self.state_trace),
            merge_applied=bool(merge_applied),
            bounded_suggestion_available=bool(bounded_suggestion_available),
        )
