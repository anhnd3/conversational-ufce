from __future__ import annotations

import pytest

from llm.src.conversation.negotiation_controller import (
    ALLOWED_TRANSITION_REASONS,
    CONFLICTING_VALUES,
    FOLLOWUP_RESET_NEW_REQUEST,
    MISSING_REQUIRED_FIELDS,
    RUNTIME_READY,
    RUNTIME_REJECT_NO_FEASIBLE_CF,
    RUNTIME_REJECT_SYSTEM_ERROR,
    RUNTIME_SUCCESS_COUNTERFACTUAL_FOUND,
    RUNTIME_SUCCESS_NO_RECOURSE,
    UNSUPPORTED_INTENT,
    ConversationNegotiationController,
)
from llm.src.conversation.types import ConversationStage


def test_conversation_controller_rejects_invalid_transition_reason():
    controller = ConversationNegotiationController()

    with pytest.raises(ValueError, match="Unsupported transition_reason"):
        controller.transition(
            next_state=ConversationStage.NEEDS_CLARIFICATION,
            transition_reason="free_text_reason",
            merge_applied=False,
            bounded_suggestion_available=False,
        )


def test_conversation_controller_tracks_internal_ready_then_final_public_state():
    controller = ConversationNegotiationController()

    controller.transition(
        next_state=ConversationStage.READY_FOR_RUNTIME,
        transition_reason=RUNTIME_READY,
        merge_applied=False,
        bounded_suggestion_available=False,
    )
    final_transition = controller.transition(
        next_state=ConversationStage.RUNTIME_SUCCESS,
        transition_reason=RUNTIME_SUCCESS_NO_RECOURSE,
        merge_applied=False,
        bounded_suggestion_available=False,
    )

    assert final_transition.source_state == ConversationStage.READY_FOR_RUNTIME
    assert final_transition.target_state == ConversationStage.RUNTIME_SUCCESS
    assert final_transition.state_trace == ["READY_FOR_RUNTIME", "RUNTIME_SUCCESS"]


def test_conversation_controller_blocks_terminal_transitions():
    controller = ConversationNegotiationController()
    controller.transition(
        next_state=ConversationStage.NEEDS_CLARIFICATION,
        transition_reason=MISSING_REQUIRED_FIELDS,
        merge_applied=False,
        bounded_suggestion_available=False,
    )

    with pytest.raises(ValueError, match="Invalid conversation transition"):
        controller.transition(
            next_state=ConversationStage.RUNTIME_SUCCESS,
            transition_reason=RUNTIME_SUCCESS_NO_RECOURSE,
            merge_applied=False,
            bounded_suggestion_available=False,
        )


def test_allowed_transition_reason_taxonomy_is_locked():
    assert ALLOWED_TRANSITION_REASONS == frozenset(
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
