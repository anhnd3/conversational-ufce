from __future__ import annotations

from llm.src.refinement.classifier import (
    VAGUE_REFINEMENT_CLARIFICATION_REASON,
    build_refinement_clarification_reasons,
    classify_refinement_outcome,
)


def test_classifier_marks_vague_bank_phrase_as_clarification_required():
    status = classify_refinement_outcome(
        user_feedback="Make the bank result better without changing too much.",
        parser_status="apply",
        normalized_delta={"set_prefer_fewer_changes": True},
        clarification_reasons=[],
        unsupported_reasons=[],
    )

    assert status == "clarification_required"


def test_classifier_marks_vague_grad_phrase_as_clarification_required():
    status = classify_refinement_outcome(
        user_feedback="Improve the graduate admission result without changing too much.",
        parser_status="apply",
        normalized_delta={"set_prefer_fewer_changes": True},
        clarification_reasons=[],
        unsupported_reasons=[],
    )

    assert status == "clarification_required"


def test_classifier_preserves_explicit_soft_preference_as_applied():
    status = classify_refinement_outcome(
        user_feedback="Prefer fewer changes.",
        parser_status="apply",
        normalized_delta={"set_prefer_fewer_changes": True},
        clarification_reasons=[],
        unsupported_reasons=[],
    )

    assert status == "applied"


def test_classifier_preserves_explicit_smaller_edits_as_applied():
    status = classify_refinement_outcome(
        user_feedback="Prefer smaller edits.",
        parser_status="apply",
        normalized_delta={"set_prefer_fewer_changes": True},
        clarification_reasons=[],
        unsupported_reasons=[],
    )

    assert status == "applied"


def test_classifier_treats_concrete_limit_delta_as_applied():
    status = classify_refinement_outcome(
        user_feedback="Change at most one thing.",
        parser_status="apply",
        normalized_delta={"set_max_changed_features": 1},
        clarification_reasons=[],
        unsupported_reasons=[],
    )

    assert status == "applied"


def test_classifier_preserves_unsupported_feedback_precedence():
    status = classify_refinement_outcome(
        user_feedback="Show me all UFCE methods and rank them yourself.",
        parser_status="unsupported_feedback",
        normalized_delta={},
        clarification_reasons=[],
        unsupported_reasons=["unsupported"],
    )

    assert status == "unsupported_feedback"


def test_classifier_preserves_limit_reached_precedence():
    status = classify_refinement_outcome(
        user_feedback="One more refinement.",
        parser_status="apply",
        normalized_delta={"set_max_changed_features": 1},
        clarification_reasons=[],
        unsupported_reasons=[],
        limit_reached=True,
    )

    assert status == "limit_reached"


def test_classifier_builds_vague_goal_clarification_reason():
    reasons = build_refinement_clarification_reasons(
        user_feedback="Make it better without changing too much.",
        parser_status="apply",
        normalized_delta={"set_prefer_fewer_changes": True},
        clarification_reasons=[],
        parser_ambiguities=[],
    )

    assert reasons == [VAGUE_REFINEMENT_CLARIFICATION_REASON]
