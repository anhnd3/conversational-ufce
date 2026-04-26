from __future__ import annotations

from llm.src.product.app import serialize_turn_response


def _base_payload() -> dict:
    return {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "turn_index": 1,
        "user_input": "input",
        "assistant_text": "assistant",
        "public_state": "RUNTIME_SUCCESS",
        "clarification_payload": None,
        "explanation_payload": None,
        "artifact_refs": {
            "turn_id": "turn-1",
            "artifact_dir": None,
            "files": [],
            "download_urls": {},
            "preview_urls": {},
        },
        "debug_summary": {
            "builder_status": "READY_FOR_RUNTIME",
            "builder_reason_codes": [],
            "transition_reason": "runtime_success_counterfactual_found",
            "merge_applied": False,
            "runtime_summary": {
                "executed": True,
                "controller_state": "TERMINAL_SUCCESS",
                "reason_codes": [],
                "prediction_score": 0.8,
            },
            "invariant_validation_status": "passed",
            "artifact_dir": None,
            "timing_metrics": None,
        },
        "clarification_turns_used": 0,
        "is_case_complete": True,
        "case_completion_reason": "runtime_success",
        "restart_required": True,
        "turn_kind": "message",
        "refinement_status": None,
        "refinement_revision_index": None,
        "parent_terminal_turn_id": None,
        "parent_refinement_revision_index": None,
        "active_constraint_spec": None,
        "constraint_feedback_delta": None,
        "refinement_rounds_used": None,
        "refinement_round_limit": None,
        "ui_response_summary": None,
    }


def test_runtime_success_counterfactual_maps_to_success_tone_and_result_anchor():
    payload = _base_payload()
    payload["public_state"] = "RUNTIME_SUCCESS"
    payload["explanation_payload"] = {
        "summary_type": "counterfactual_found",
        "counterfactual_summary": {"profile_diff": {"Income": {"from": 45, "to": 60}}},
        "changed_fields": ["Income"],
        "reason_codes": [],
    }
    payload["ui_response_summary"] = {
        "response_kind": "counterfactual_found",
        "tone": "success",
        "headline": "A valid improvement path was found",
        "short_summary": "Validated and safe.",
        "changed_items": [],
        "blocked_reasons": [],
        "next_actions": [],
    }

    response = serialize_turn_response(payload)

    assert response.ui_response_summary is not None
    assert response.ui_response_summary.tone == "success"
    assert response.render_hints is not None
    assert response.render_hints.right_rail_anchor == "result-card"


def test_no_recourse_maps_to_no_action_required():
    payload = _base_payload()
    payload["public_state"] = "RUNTIME_SUCCESS"
    payload["explanation_payload"] = {
        "summary_type": "no_recourse_needed",
        "counterfactual_summary": None,
        "changed_fields": [],
        "reason_codes": ["NO_RECOURSE_NEEDED"],
    }
    payload["ui_response_summary"] = {
        "response_kind": "no_recourse_needed",
        "tone": "success",
        "headline": "Already qualifies",
        "short_summary": "No changes needed.",
        "changed_items": [],
        "blocked_reasons": [],
        "next_actions": [{"action_type": "none", "label": "No action", "detail": "Complete", "fields": [], "primary": True}],
    }

    response = serialize_turn_response(payload)

    assert response.render_hints is not None
    assert response.render_hints.primary_action_type == "no_action_required"


def test_runtime_reject_constraints_blocked_maps_to_warning_tone_and_relax_action():
    payload = _base_payload()
    payload["public_state"] = "RUNTIME_REJECT"
    payload["case_completion_reason"] = "runtime_reject"
    payload["explanation_payload"] = {
        "summary_type": "runtime_reject",
        "counterfactual_summary": None,
        "changed_fields": [],
        "reason_codes": ["REQUEST_CONSTRAINTS_BLOCKED"],
    }
    payload["ui_response_summary"] = {
        "response_kind": "runtime_reject_constraints_blocked",
        "tone": "warning",
        "headline": "Constraints blocked",
        "short_summary": "Relax constraints.",
        "changed_items": [],
        "blocked_reasons": [{"code": "REQUEST_CONSTRAINTS_BLOCKED", "title": "Blocked", "detail": "detail", "fields": ["Income"]}],
        "next_actions": [{"action_type": "relax_constraints", "label": "Relax constraints", "detail": "Relax one.", "fields": ["Income"], "primary": True}],
    }

    response = serialize_turn_response(payload)

    assert response.ui_response_summary is not None
    assert response.ui_response_summary.tone == "warning"
    assert response.ui_response_summary.next_actions[0].action_type == "relax_constraints"
    assert response.render_hints is not None
    assert response.render_hints.primary_action_type == "relax_constraints_or_restart"


def test_needs_clarification_maps_to_info_tone_and_missing_fields_action():
    payload = _base_payload()
    payload["public_state"] = "NEEDS_CLARIFICATION"
    payload["is_case_complete"] = False
    payload["restart_required"] = False
    payload["case_completion_reason"] = None
    payload["clarification_payload"] = {
        "clarification_type": "missing_information",
        "missing_fields": ["Income", "Family"],
        "conflicts": [],
        "next_required_input": "Reply with missing fields.",
        "remaining_rounds": 2,
        "restart_required": False,
        "reply_strategy": "missing_fields_only",
        "carried_forward_fields": ["CCAvg"],
    }
    payload["explanation_payload"] = None
    payload["ui_response_summary"] = {
        "response_kind": "clarification_required",
        "tone": "info",
        "headline": "More info needed",
        "short_summary": "Provide missing fields.",
        "changed_items": [],
        "blocked_reasons": [],
        "next_actions": [{"action_type": "provide_missing_fields", "label": "Provide missing fields", "detail": "Reply with missing fields.", "fields": ["Income", "Family"], "primary": True}],
    }

    response = serialize_turn_response(payload)

    assert response.ui_response_summary is not None
    assert response.ui_response_summary.tone == "info"
    assert response.render_hints is not None
    assert response.render_hints.primary_action_type == "provide_missing_fields"


def test_invalid_counterfactual_blocked_keeps_technical_without_exposing_candidate_values():
    payload = _base_payload()
    payload["public_state"] = "RUNTIME_REJECT"
    payload["case_completion_reason"] = "runtime_reject"
    payload["explanation_payload"] = {
        "summary_type": "runtime_reject",
        "counterfactual_summary": None,
        "changed_fields": [],
        "reason_codes": ["INVALID_COUNTERFACTUAL_BLOCKED"],
    }
    payload["ui_response_summary"] = {
        "response_kind": "runtime_reject_invalid_counterfactual_blocked",
        "tone": "danger",
        "headline": "Validation blocked",
        "short_summary": "Candidate blocked.",
        "changed_items": [],
        "blocked_reasons": [{"code": "INVALID_COUNTERFACTUAL_BLOCKED", "title": "Validation", "detail": "failed", "fields": []}],
        "next_actions": [],
    }

    response = serialize_turn_response(payload)

    assert response.ui_response_summary is not None
    assert response.ui_response_summary.changed_items == []
    assert response.render_hints is not None
    assert response.render_hints.right_rail_anchor == "result-card"
