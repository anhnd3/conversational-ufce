from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any

from llm.src.conversation.artifacts import DEFAULT_OUTPUT_ROOT, save_conversation_artifacts
from llm.src.conversation.types import (
    ClarificationPayload,
    ConversationStage,
    ConversationTurnResult,
    ParserAdapterResult,
    ResponseDecision,
)
from llm.src.orchestration.explanation_flow import build_explanation_payload, render_explanation_text
from llm.src.parser.output_repair import should_attempt_repair
from llm.src.refinement.classifier import (
    build_refinement_clarification_reasons,
    classify_refinement_outcome,
)
from llm.src.refinement.delta import (
    apply_refinement_delta_to_active_constraint_spec,
    build_active_constraint_spec,
)
from llm.src.refinement.types import (
    PendingRefinementClarification,
    REFINEMENT_STATUS_APPLIED,
    REFINEMENT_STATUS_CLARIFICATION_REQUIRED,
    REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK,
)
from llm.src.refinement.validation import validate_refinement_prediction
from llm.src.runtime.reason_codes import INVALID_COUNTERFACTUAL_BLOCKED
from llm.src.utils.hashing import make_run_id, sha256_file, sha256_text, utc_now_iso


DEFAULT_REFINEMENT_NEXT_INPUT = (
    "State the allowed or blocked fields, bounds, or change-limit update in one clear sentence."
)


class ConstraintRefinementOrchestrator:
    def __init__(
        self,
        *,
        parser_adapter,
        runtime_orchestrator,
        benchmark,
        output_root: Path | None = None,
        model_alias: str,
    ) -> None:
        self.parser_adapter = parser_adapter
        self.runtime_orchestrator = runtime_orchestrator
        self.benchmark = benchmark
        self.dataset_registry = getattr(runtime_orchestrator, "dataset_registry", None)
        self.output_root = Path(output_root or DEFAULT_OUTPUT_ROOT)
        self.model_alias = model_alias

    def run_turn(
        self,
        *,
        user_feedback: str,
        dataset_id: str,
        active_constraint_spec: dict[str, Any] | None,
        last_runtime_request: dict[str, Any],
        parent_terminal_turn_id: str,
        parent_refinement_revision_index: int | None,
        refinement_revision_index: int,
        refinement_rounds_used: int,
        refinement_round_limit: int,
        parent_public_state: str,
        parent_case_completion_reason: str,
        pending_refinement_clarification: PendingRefinementClarification | None,
        session_trace: dict[str, Any] | None = None,
        save_artifacts: bool = True,
        scenario_slug: str | None = None,
        debug_trace_enabled: bool = False,
        command: str | None = None,
    ) -> tuple[ConversationTurnResult, dict[str, Any]]:
        started = time.perf_counter()
        turn_id = make_run_id()
        timestamp_utc = utc_now_iso()
        dataset_package = self._resolve_dataset_package(dataset_id)
        feature_order = list(dataset_package.profile_schema()["field_order"])
        numeric_bound_fields = dataset_package.numeric_bound_fields()
        benchmark = dataset_package.live_primary_benchmark()
        parse_payload, repair_result, validation = self._prepare_refinement(
            user_feedback=user_feedback,
            active_constraint_spec=active_constraint_spec,
            pending_refinement_clarification=pending_refinement_clarification,
            dataset_package=dataset_package,
            benchmark=benchmark,
            feature_order=feature_order,
            numeric_bound_fields=numeric_bound_fields,
        )

        normalized_output = validation.normalized_output
        normalized_delta = validation.normalized_delta or {}
        stage_trace: list[str] = []
        runtime_result = None
        runtime_debug_trace = None
        invariant_validation = None
        clarification_payload = None
        explanation_payload = None
        response_decision = None
        assistant_text = ""
        stage = parent_public_state
        public_state = parent_public_state
        is_case_complete = True
        case_completion_reason = parent_case_completion_reason
        restart_required = True
        active_before = build_active_constraint_spec(
            active_constraint_spec,
            feature_order=feature_order,
            numeric_bound_fields=numeric_bound_fields,
        )
        active_after = dict(active_before)
        refinement_status = REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK
        clarification_reasons: list[str] = []
        next_pending_refinement = None

        if not validation.is_valid:
            refinement_status = REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK
            assistant_text = render_refinement_unsupported_text(list(validation.errors))
            stage = ConversationStage.UNSUPPORTED_REQUEST
            public_state = parent_public_state
            restart_required = False
            response_decision = ResponseDecision(
                final_public_state=public_state,
                template_type="refinement_unsupported_feedback",
                included_suggestion_types=[],
            )
            stage_trace = [stage]
        else:
            refinement_status = classify_refinement_outcome(
                user_feedback=user_feedback,
                parser_status=validation.parser_status,
                normalized_delta=normalized_delta,
                clarification_reasons=validation.clarification_reasons,
                unsupported_reasons=validation.unsupported_reasons,
            )
            clarification_reasons = build_refinement_clarification_reasons(
                user_feedback=user_feedback,
                parser_status=validation.parser_status,
                normalized_delta=normalized_delta,
                clarification_reasons=validation.clarification_reasons,
                parser_ambiguities=(validation.normalized_output or {}).get("ambiguities") or [],
            )
        if not validation.is_valid:
            pass
        elif refinement_status == REFINEMENT_STATUS_CLARIFICATION_REQUIRED:
            refinement_status = REFINEMENT_STATUS_CLARIFICATION_REQUIRED
            clarification_payload = ClarificationPayload(
                clarification_type="refinement_clarification",
                missing_fields=[],
                conflicts=list(clarification_reasons),
                next_required_input=build_refinement_next_input(dataset_package.primary_subject_label()),
                remaining_rounds=max(refinement_round_limit - refinement_rounds_used, 0),
                restart_required=False,
                reply_strategy="start_new_case",
                carried_forward_fields=[],
            )
            next_pending_refinement = PendingRefinementClarification(
                originating_turn_id=turn_id,
                ambiguities=list(clarification_payload.conflicts),
                next_required_input=clarification_payload.next_required_input,
                parent_terminal_turn_id=parent_terminal_turn_id,
                parent_refinement_revision_index=parent_refinement_revision_index,
            )
            assistant_text = render_refinement_clarification_text(clarification_payload)
            stage = ConversationStage.NEEDS_CLARIFICATION
            public_state = parent_public_state
            restart_required = False
            response_decision = ResponseDecision(
                final_public_state=public_state,
                template_type="refinement_clarification",
                included_suggestion_types=[],
            )
            stage_trace = [stage]
        elif refinement_status == REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK:
            refinement_status = REFINEMENT_STATUS_UNSUPPORTED_FEEDBACK
            assistant_text = render_refinement_unsupported_text(list(validation.unsupported_reasons))
            stage = ConversationStage.UNSUPPORTED_REQUEST
            public_state = parent_public_state
            restart_required = False
            response_decision = ResponseDecision(
                final_public_state=public_state,
                template_type="refinement_unsupported_feedback",
                included_suggestion_types=[],
            )
            stage_trace = [stage]
        else:
            refinement_status = REFINEMENT_STATUS_APPLIED
            active_after = apply_refinement_delta_to_active_constraint_spec(
                active_before,
                normalized_delta,
                feature_order=feature_order,
                numeric_bound_fields=numeric_bound_fields,
            )
            runtime_payload = dict(last_runtime_request)
            runtime_payload["profile"] = dict(last_runtime_request["profile"])
            if active_after:
                runtime_payload["constraint_spec"] = dict(active_after)
            else:
                runtime_payload.pop("constraint_spec", None)

            runtime_started = time.perf_counter()
            runtime_obj = self.runtime_orchestrator.handle(runtime_payload, include_debug_trace=True)
            runtime_latency_ms = (time.perf_counter() - runtime_started) * 1000.0
            runtime_result = runtime_obj.to_dict(include_debug_trace=False)
            runtime_debug_trace = None if runtime_obj.debug_trace is None else runtime_obj.debug_trace.to_dict()
            if runtime_debug_trace is not None:
                runtime_debug_trace["request_latency_ms"] = round(runtime_latency_ms, 3)
            invariant_validation = (
                None if runtime_obj.invariant_validation is None else runtime_obj.invariant_validation.to_dict()
            )
            invariant_failed = bool(
                runtime_obj.invariant_validation is not None and runtime_obj.invariant_validation.status == "failed"
            )
            if runtime_obj.controller_state == "TERMINAL_SUCCESS" and not invariant_failed:
                stage = ConversationStage.RUNTIME_SUCCESS
                public_state = ConversationStage.RUNTIME_SUCCESS
                case_completion_reason = "runtime_success"
                restart_required = True
                template_type = (
                    "refinement_explanation_no_recourse"
                    if runtime_obj.reason_codes == ["NO_RECOURSE_NEEDED"]
                    else "refinement_explanation_counterfactual"
                )
            else:
                stage = ConversationStage.RUNTIME_REJECT
                public_state = ConversationStage.RUNTIME_REJECT
                case_completion_reason = "runtime_reject"
                restart_required = True
                template_type = "refinement_explanation_reject"
            explanation_runtime_result = runtime_result
            if invariant_failed:
                explanation_runtime_result = dict(runtime_result or {})
                explanation_runtime_result["controller_state"] = "TERMINAL_REJECT"
                explanation_runtime_result["counterfactual"] = None
                explanation_runtime_result["reason_codes"] = [INVALID_COUNTERFACTUAL_BLOCKED]
            explanation_payload = build_explanation_payload(
                runtime_result=explanation_runtime_result,
                current_profile=runtime_payload["profile"],
                included_suggestion_types=[],
                policy=dataset_package.runtime_context().policy,
                dataset_label=dataset_package.primary_subject_label(),
            )
            assistant_text = render_explanation_text(
                explanation_payload,
                dataset_label=dataset_package.primary_subject_label(),
                parser_adapter=self.parser_adapter,
            )
            response_decision = ResponseDecision(
                final_public_state=public_state,
                template_type=template_type,
                included_suggestion_types=list(explanation_payload.included_suggestion_types),
            )
            stage_trace = [stage]

        end_to_end_latency_ms = (time.perf_counter() - started) * 1000.0
        timing_metrics = {
            "end_to_end_latency_ms": round(end_to_end_latency_ms, 3),
        }
        result = ConversationTurnResult(
            turn_id=turn_id,
            timestamp_utc=timestamp_utc,
            model_alias=self.model_alias,
            user_input=user_feedback,
            stage=stage,
            stage_trace=stage_trace,
            parser_result=parse_payload,
            repair_result=repair_result,
            normalized_parse=normalized_output,
            schema_validation=validation.to_dict(),
            canonical_validation={},
            builder_result=None,
            negotiation_transition=None,
            response_decision=response_decision,
            runtime_result=runtime_result,
            runtime_debug_trace=runtime_debug_trace,
            invariant_validation=invariant_validation,
            field_provenance=None,
            clarification_payload=clarification_payload,
            explanation_payload=explanation_payload,
            response_text=assistant_text,
            parser_failure_cause=None,
            is_case_complete=is_case_complete,
            case_completion_reason=case_completion_reason,
            restart_required=restart_required,
            clarification_turns_used=0,
            timing_metrics=timing_metrics,
            turn_kind="refinement",
            refinement_status=refinement_status,
            refinement_revision_index=refinement_revision_index,
            parent_terminal_turn_id=parent_terminal_turn_id,
            parent_refinement_revision_index=parent_refinement_revision_index,
            active_constraint_spec=active_after,
            active_constraint_spec_before=active_before,
            constraint_feedback_delta=normalized_delta,
            refinement_rounds_used=refinement_rounds_used,
            refinement_round_limit=refinement_round_limit,
        )
        if save_artifacts:
            config_snapshot = self._build_config_snapshot(
                parser_result=parse_payload,
                repair_result=repair_result,
                timing_metrics=timing_metrics,
                dataset_package=dataset_package,
            )
            result.artifact_record = save_conversation_artifacts(
                result,
                output_root=self.output_root,
                scenario_slug=scenario_slug,
                command=command or "refinement",
                config_snapshot=config_snapshot,
                debug_trace_enabled=debug_trace_enabled,
                session_trace=session_trace,
            )
        response_payload = {
            "public_state": public_state,
            "pending_refinement_clarification": None
            if next_pending_refinement is None
            else next_pending_refinement.to_dict(),
            "active_constraint_spec": active_after,
            "last_runtime_request": None,
            "latest_runtime_backed_turn_id": parent_terminal_turn_id,
        }
        if refinement_status == REFINEMENT_STATUS_APPLIED:
            response_payload["last_runtime_request"] = dict(last_runtime_request)
            if active_after:
                response_payload["last_runtime_request"]["constraint_spec"] = dict(active_after)
            else:
                response_payload["last_runtime_request"].pop("constraint_spec", None)
            response_payload["latest_runtime_backed_turn_id"] = turn_id
        return result, response_payload

    def _prepare_refinement(
        self,
        *,
        user_feedback: str,
        active_constraint_spec: dict[str, Any] | None,
        pending_refinement_clarification: PendingRefinementClarification | None,
        dataset_package,
        benchmark,
        feature_order: list[str],
        numeric_bound_fields: list[str],
    ) -> tuple[ParserAdapterResult, ParserAdapterResult | None, Any]:
        parser_result = invoke_parser_adapter_method(
            self.parser_adapter.parse_refinement,
            user_text=user_feedback,
            active_constraint_spec=active_constraint_spec,
            pending_refinement_clarification=None
            if pending_refinement_clarification is None
            else pending_refinement_clarification.to_dict(),
            benchmark=benchmark,
            dataset_package=dataset_package,
        )
        repair_result = None
        validation = validate_refinement_prediction(
            _safe_parse_json(parser_result.message_text),
            feature_order=feature_order,
            numeric_bound_fields=numeric_bound_fields,
        )
        if should_attempt_repair(
            raw_output=parser_result.message_text,
            api_error=parser_result.api_error,
            errors=list(validation.errors),
        ):
            repair_result = invoke_parser_adapter_method(
                self.parser_adapter.repair_refinement,
                invalid_output=parser_result.message_text,
                errors=list(validation.errors),
                active_constraint_spec=active_constraint_spec,
                pending_refinement_clarification=None
                if pending_refinement_clarification is None
                else pending_refinement_clarification.to_dict(),
                benchmark=benchmark,
                dataset_package=dataset_package,
            )
            validation = validate_refinement_prediction(
                _safe_parse_json(repair_result.message_text),
                feature_order=feature_order,
                numeric_bound_fields=numeric_bound_fields,
            )
        return parser_result, repair_result, validation

    def _build_config_snapshot(
        self,
        *,
        parser_result: ParserAdapterResult,
        repair_result: ParserAdapterResult | None,
        timing_metrics: dict[str, Any],
        dataset_package,
    ) -> dict[str, Any]:
        parse_profile = (
            invoke_parser_adapter_method(
                self.parser_adapter.describe_request_profile,
                "parse_refinement",
                dataset_package=dataset_package,
            )
            if hasattr(self.parser_adapter, "describe_request_profile")
            else {}
        )
        repair_profile = (
            invoke_parser_adapter_method(
                self.parser_adapter.describe_request_profile,
                "repair_refinement",
                dataset_package=dataset_package,
            )
            if repair_result is not None and hasattr(self.parser_adapter, "describe_request_profile")
            else None
        )
        prompt_sha256 = None
        system_prompt = getattr(self.parser_adapter, "system_prompt", None)
        if isinstance(system_prompt, str):
            prompt_sha256 = sha256_text(system_prompt)
        system_prompt_path = getattr(self.parser_adapter, "system_prompt_path", None)
        system_prompt_file_sha256 = None
        if isinstance(system_prompt_path, Path) and system_prompt_path.exists():
            system_prompt_file_sha256 = sha256_file(system_prompt_path)
        return {
            "mode": "refinement",
            "dataset": dataset_package.dataset_id,
            "model_alias": self.model_alias,
            "parser_api_base": getattr(self.parser_adapter, "api_base", None),
            "structured_output_mode": getattr(self.parser_adapter, "structured_output_mode", None),
            "system_prompt_sha256": prompt_sha256,
            "system_prompt_file": None if system_prompt_path is None else str(system_prompt_path),
            "system_prompt_file_sha256": system_prompt_file_sha256,
            "request_profiles": {
                "parse_refinement": parse_profile,
                "repair_refinement": repair_profile,
            },
            "timing_metrics": dict(timing_metrics),
            "parser_task_type": parser_result.task_type,
            "repair_used": repair_result is not None,
        }

    def _resolve_dataset_package(self, dataset_id: str):
        normalized = str(dataset_id or "bank").strip().lower() or "bank"
        if self.dataset_registry is not None and self.dataset_registry.has(normalized):
            return self.dataset_registry.get(normalized)
        if self.dataset_registry is not None:
            return self.dataset_registry.get("bank")
        raise KeyError(f"Unsupported refinement dataset: {dataset_id}")


def render_refinement_clarification_text(payload: ClarificationPayload) -> str:
    reasons = "; ".join(payload.conflicts) if payload.conflicts else "the requested update was ambiguous"
    return (
        "The refinement feedback could not be applied yet. "
        f"Clarification is required because {reasons}. "
        f"{payload.next_required_input}"
    )


def render_refinement_unsupported_text(reasons: list[str]) -> str:
    detail = " ".join(reasons) if reasons else "The feedback was outside the supported refinement language."
    return (
        "The refinement feedback could not be applied. "
        "Active constraints were left unchanged. "
        f"{detail}"
    )


def build_refinement_next_input(dataset_label: str) -> str:
    return (
        "State the allowed or blocked fields, bounds, or change-limit update for the active "
        f"{dataset_label} in one clear sentence."
    )


def _safe_parse_json(message_text: str) -> dict[str, Any] | None:
    try:
        parsed = __import__("json").loads(message_text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def invoke_parser_adapter_method(method, *args, **kwargs):
    if "dataset_package" in kwargs and not _supports_keyword_argument(method, "dataset_package"):
        kwargs = {key: value for key, value in kwargs.items() if key != "dataset_package"}
    return method(*args, **kwargs)


def _supports_keyword_argument(method, keyword: str) -> bool:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return keyword in signature.parameters
