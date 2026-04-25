from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ConversationStage:
    READY_FOR_RUNTIME = "READY_FOR_RUNTIME"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    CONFLICT = "CONFLICT"
    UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"
    RUNTIME_SUCCESS = "RUNTIME_SUCCESS"
    RUNTIME_REJECT = "RUNTIME_REJECT"
    PARSER_FAILURE = "PARSER_FAILURE"


@dataclass(frozen=True)
class ParserAdapterResult:
    message_text: str
    api_error: str | None
    http_status_code: int | None
    raw_response_text: str | None
    response_json: dict[str, Any] | None
    reasoning_text: str
    usage: dict[str, Any]
    stats: dict[str, Any]
    derived_metrics: dict[str, Any]
    request_payload: dict[str, Any]
    failure_cause: str | None = None
    task_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_text": self.message_text,
            "api_error": self.api_error,
            "http_status_code": self.http_status_code,
            "raw_response_text": self.raw_response_text,
            "response_json": self.response_json,
            "reasoning_text": self.reasoning_text,
            "usage": dict(self.usage),
            "stats": dict(self.stats),
            "derived_metrics": dict(self.derived_metrics),
            "request_payload": dict(self.request_payload),
            "failure_cause": self.failure_cause,
            "task_type": self.task_type,
        }


@dataclass(frozen=True)
class CanonicalValidationResult:
    parser_status: str | None
    final_stage: str
    is_usable: bool
    ready_for_runtime: bool
    missing_runtime_fields: list[str]
    confirmed_conflicts: list[str]
    provided_fields: list[str]
    errors: list[str]
    runtime_request: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "parser_status": self.parser_status,
            "final_stage": self.final_stage,
            "is_usable": bool(self.is_usable),
            "ready_for_runtime": bool(self.ready_for_runtime),
            "missing_runtime_fields": list(self.missing_runtime_fields),
            "confirmed_conflicts": list(self.confirmed_conflicts),
            "provided_fields": list(self.provided_fields),
            "errors": list(self.errors),
            "runtime_request": None if self.runtime_request is None else dict(self.runtime_request),
        }


@dataclass(frozen=True)
class PendingClarification:
    prior_cf_request: dict[str, Any]
    prior_constraint_spec: dict[str, Any]
    missing_fields: list[str]
    required_field_order: list[str]
    originating_turn_id: str
    prior_field_provenance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior_cf_request": dict(self.prior_cf_request),
            "prior_constraint_spec": dict(self.prior_constraint_spec),
            "missing_fields": list(self.missing_fields),
            "required_field_order": list(self.required_field_order),
            "originating_turn_id": self.originating_turn_id,
            "prior_field_provenance": dict(self.prior_field_provenance),
        }


@dataclass(frozen=True)
class ClarificationPayload:
    clarification_type: str
    missing_fields: list[str]
    conflicts: list[str]
    next_required_input: str
    remaining_rounds: int | None = None
    restart_required: bool = False
    reply_strategy: str = "start_new_case"
    carried_forward_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clarification_type": self.clarification_type,
            "missing_fields": list(self.missing_fields),
            "conflicts": list(self.conflicts),
            "next_required_input": self.next_required_input,
            "remaining_rounds": self.remaining_rounds,
            "restart_required": bool(self.restart_required),
            "reply_strategy": self.reply_strategy,
            "carried_forward_fields": list(self.carried_forward_fields),
        }


@dataclass(frozen=True)
class ExplanationPayload:
    summary_type: str
    prediction_snapshot: dict[str, Any]
    counterfactual_summary: dict[str, Any] | None
    reason_codes: list[str]
    changed_fields: list[str]
    included_suggestion_types: list[str] = field(default_factory=list)
    next_step_suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_type": self.summary_type,
            "prediction_snapshot": dict(self.prediction_snapshot),
            "counterfactual_summary": None
            if self.counterfactual_summary is None
            else dict(self.counterfactual_summary),
            "reason_codes": list(self.reason_codes),
            "changed_fields": list(self.changed_fields),
            "included_suggestion_types": list(self.included_suggestion_types),
            "next_step_suggestions": list(self.next_step_suggestions),
        }


@dataclass(frozen=True)
class RequestBuildResult:
    builder_status: str
    builder_reason_codes: list[str]
    partial_profile_snapshot: dict[str, Any] | None
    runtime_request: dict[str, Any] | None
    missing_fields: list[str]
    conflicts: list[str]
    policy_version: str
    canonical_field_order: list[str]
    provenance: dict[str, Any]
    merge_applied: bool = False
    carried_fields: list[str] = field(default_factory=list)
    carried_constraint_keys: list[str] = field(default_factory=list)
    carried_preference_keys: list[str] = field(default_factory=list)
    pending_reset: bool = False
    _normalized_candidate: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    _schema_validation: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _canonical_validation: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "builder_status": self.builder_status,
            "builder_reason_codes": list(self.builder_reason_codes),
            "partial_profile_snapshot": None
            if self.partial_profile_snapshot is None
            else dict(self.partial_profile_snapshot),
            "runtime_request": None if self.runtime_request is None else dict(self.runtime_request),
            "missing_fields": list(self.missing_fields),
            "conflicts": list(self.conflicts),
            "policy_version": self.policy_version,
            "canonical_field_order": list(self.canonical_field_order),
            "provenance": dict(self.provenance),
            "merge_applied": bool(self.merge_applied),
            "carried_fields": list(self.carried_fields),
            "carried_constraint_keys": list(self.carried_constraint_keys),
            "carried_preference_keys": list(self.carried_preference_keys),
            "pending_reset": bool(self.pending_reset),
        }


@dataclass(frozen=True)
class NegotiationTransition:
    source_state: str | None
    target_state: str
    transition_reason: str
    state_trace: list[str]
    merge_applied: bool
    bounded_suggestion_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_state": self.source_state,
            "target_state": self.target_state,
            "transition_reason": self.transition_reason,
            "state_trace": list(self.state_trace),
            "merge_applied": bool(self.merge_applied),
            "bounded_suggestion_available": bool(self.bounded_suggestion_available),
        }


@dataclass(frozen=True)
class ResponseDecision:
    final_public_state: str
    template_type: str
    included_suggestion_types: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_public_state": self.final_public_state,
            "template_type": self.template_type,
            "included_suggestion_types": list(self.included_suggestion_types),
        }


@dataclass(frozen=True)
class ArtifactRecord:
    turn_id: str
    stage: str
    output_dir: str
    saved_files: list[str]
    model_alias: str
    command: str
    timestamp_utc: str
    debug_trace_enabled: bool
    repair_used: bool
    session_id: str | None = None
    turn_index: int | None = None
    parent_turn_id: str | None = None
    merge_applied: bool = False
    carried_fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "stage": self.stage,
            "output_dir": self.output_dir,
            "saved_files": list(self.saved_files),
            "model_alias": self.model_alias,
            "command": self.command,
            "timestamp_utc": self.timestamp_utc,
            "debug_trace_enabled": bool(self.debug_trace_enabled),
            "repair_used": bool(self.repair_used),
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "parent_turn_id": self.parent_turn_id,
            "merge_applied": bool(self.merge_applied),
            "carried_fields": list(self.carried_fields),
        }


@dataclass
class ConversationTurnResult:
    turn_id: str
    timestamp_utc: str
    model_alias: str
    user_input: str
    stage: str
    stage_trace: list[str]
    parser_result: ParserAdapterResult
    repair_result: ParserAdapterResult | None
    normalized_parse: dict[str, Any] | None
    schema_validation: dict[str, Any]
    canonical_validation: dict[str, Any]
    builder_result: RequestBuildResult | None
    negotiation_transition: NegotiationTransition | None
    response_decision: ResponseDecision | None
    runtime_result: dict[str, Any] | None
    runtime_debug_trace: dict[str, Any] | None
    invariant_validation: dict[str, Any] | None
    field_provenance: dict[str, str] | None
    clarification_payload: ClarificationPayload | None
    explanation_payload: ExplanationPayload | None
    response_text: str
    parser_quality_metadata: dict[str, Any] | None = None
    parser_failure_cause: str | None = None
    artifact_record: ArtifactRecord | None = None
    is_case_complete: bool = False
    case_completion_reason: str | None = None
    restart_required: bool = False
    clarification_turns_used: int = 0
    timing_metrics: dict[str, Any] | None = None
    turn_kind: str = "message"
    refinement_status: str | None = None
    refinement_revision_index: int | None = None
    parent_terminal_turn_id: str | None = None
    parent_refinement_revision_index: int | None = None
    active_constraint_spec: dict[str, Any] | None = None
    active_constraint_spec_before: dict[str, Any] | None = None
    constraint_feedback_delta: dict[str, Any] | None = None
    refinement_rounds_used: int | None = None
    refinement_round_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "timestamp_utc": self.timestamp_utc,
            "model_alias": self.model_alias,
            "user_input": self.user_input,
            "stage": self.stage,
            "stage_trace": list(self.stage_trace),
            "raw_parser_output": self.parser_result.message_text,
            "repaired_output": None if self.repair_result is None else self.repair_result.message_text,
            "parser_result": self.parser_result.to_dict(),
            "repair_result": None if self.repair_result is None else self.repair_result.to_dict(),
            "normalized_parse": serialize_normalized_parse_payload(
                self.normalized_parse,
                self.field_provenance,
                self.parser_quality_metadata,
            ),
            "schema_validation": dict(self.schema_validation),
            "canonical_validation": dict(self.canonical_validation),
            "builder_result": None if self.builder_result is None else self.builder_result.to_dict(),
            "negotiation_transition": None
            if self.negotiation_transition is None
            else self.negotiation_transition.to_dict(),
            "response_decision": None if self.response_decision is None else self.response_decision.to_dict(),
            "runtime_result": None if self.runtime_result is None else dict(self.runtime_result),
            "runtime_debug_trace": None if self.runtime_debug_trace is None else dict(self.runtime_debug_trace),
            "invariant_validation": None if self.invariant_validation is None else dict(self.invariant_validation),
            "clarification_payload": None
            if self.clarification_payload is None
            else self.clarification_payload.to_dict(),
            "explanation_payload": None
            if self.explanation_payload is None
            else self.explanation_payload.to_dict(),
            "response_text": self.response_text,
            "parser_failure_cause": self.parser_failure_cause,
            "artifact_record": None
            if self.artifact_record is None
            else self.artifact_record.to_dict(),
            "is_case_complete": bool(self.is_case_complete),
            "case_completion_reason": self.case_completion_reason,
            "restart_required": bool(self.restart_required),
            "clarification_turns_used": int(self.clarification_turns_used),
            "timing_metrics": None if self.timing_metrics is None else dict(self.timing_metrics),
            "turn_kind": self.turn_kind,
            "refinement_status": self.refinement_status,
            "refinement_revision_index": self.refinement_revision_index,
            "parent_terminal_turn_id": self.parent_terminal_turn_id,
            "parent_refinement_revision_index": self.parent_refinement_revision_index,
            "active_constraint_spec": None if self.active_constraint_spec is None else dict(self.active_constraint_spec),
            "active_constraint_spec_before": None
            if self.active_constraint_spec_before is None
            else dict(self.active_constraint_spec_before),
            "constraint_feedback_delta": None
            if self.constraint_feedback_delta is None
            else dict(self.constraint_feedback_delta),
            "refinement_rounds_used": self.refinement_rounds_used,
            "refinement_round_limit": self.refinement_round_limit,
        }


def serialize_normalized_parse_payload(
    normalized_parse: dict[str, Any] | None,
    field_provenance: dict[str, str] | None,
    parser_quality_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if normalized_parse is None:
        return None
    payload = dict(normalized_parse)
    if field_provenance:
        payload["_field_provenance"] = {
            str(field_name): str(value)
            for field_name, value in field_provenance.items()
            if isinstance(field_name, str) and isinstance(value, str)
        }
    payload["_parser_quality"] = normalize_parser_quality_payload(parser_quality_metadata)
    return payload


def normalize_parser_quality_payload(parser_quality_metadata: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(parser_quality_metadata or {})
    return {
        "reason_codes": [str(code) for code in list(payload.get("reason_codes", [])) if isinstance(code, str)],
        "flags": {
            "deterministic_recovery_applied": bool(dict(payload.get("flags") or {}).get("deterministic_recovery_applied")),
            "post_quality_schema_valid": bool(dict(payload.get("flags") or {}).get("post_quality_schema_valid")),
            "canonical_pass_after_quality": bool(dict(payload.get("flags") or {}).get("canonical_pass_after_quality")),
            "repair_invoked": bool(dict(payload.get("flags") or {}).get("repair_invoked")),
            "still_failed_after_quality": bool(dict(payload.get("flags") or {}).get("still_failed_after_quality")),
            "constraint_extraction_absent": bool(dict(payload.get("flags") or {}).get("constraint_extraction_absent")),
        },
        "semantic_buckets": {
            "profile_facts": dict(dict(payload.get("semantic_buckets") or {}).get("profile_facts") or {}),
            "hard_constraints": dict(dict(payload.get("semantic_buckets") or {}).get("hard_constraints") or {}),
            "soft_preferences": dict(dict(payload.get("semantic_buckets") or {}).get("soft_preferences") or {}),
        },
    }
