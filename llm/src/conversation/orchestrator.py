from __future__ import annotations

import inspect
import sys
from pathlib import Path
import time
from typing import Any

from llm.src.conversation.artifacts import DEFAULT_OUTPUT_ROOT, save_conversation_artifacts
from llm.src.conversation.canonical_validator import BankCanonicalValidator, DatasetRoutedCanonicalValidator
from llm.src.conversation.negotiation_controller import (
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
from llm.src.conversation.parser_adapter import (
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_MODEL_ALIAS,
    DEFAULT_SYSTEM_PROMPT_PATH,
    LiveLmStudioParserAdapter,
)
from llm.src.conversation.request_builder import ConversationRequestBuilder
from llm.src.conversation.types import (
    ConversationStage,
    ConversationTurnResult,
    PendingClarification,
    ResponseDecision,
)
from llm.src.orchestration.clarification_flow import (
    build_clarification_payload,
    build_clarification_limit_reached_payload,
    build_user_response_payload_from_clarification,
    render_clarification_text,
)
from llm.src.orchestration.explanation_flow import (
    build_explanation_payload,
    build_user_response_payload_from_explanation,
    render_explanation_text,
)
from llm.src.orchestration.unsupported_flow import render_unsupported_request_text
from llm.src.parser.output_repair import collect_repair_errors, should_attempt_repair
from llm.src.parser.parser_quality import finalize_parser_quality_metadata, run_parser_quality
from llm.src.runtime.datasets import BankDatasetPackage, GradDatasetPackage
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.registries.dataset_registry import DatasetRegistry
from llm.src.runtime.reason_codes import INVALID_COUNTERFACTUAL_BLOCKED, REQUEST_CONSTRAINTS_BLOCKED
from llm.src.utils.hashing import make_run_id, sha256_file, sha256_text, utc_now_iso
from llm_eval.config import load_benchmark


class BankConversationOrchestrator:
    def __init__(
        self,
        *,
        parser_adapter=None,
        runtime_orchestrator: RuntimeOrchestrator | None = None,
        canonical_validator: DatasetRoutedCanonicalValidator | BankCanonicalValidator | None = None,
        benchmark=None,
        benchmark_path: Path | None = None,
        system_prompt_path: Path | None = None,
        output_root: Path | None = None,
        model_alias: str = DEFAULT_MODEL_ALIAS,
    ) -> None:
        self.benchmark_path = Path(benchmark_path or DEFAULT_BENCHMARK_PATH)
        self.system_prompt_path = Path(system_prompt_path or DEFAULT_SYSTEM_PROMPT_PATH)
        self.benchmark = benchmark or load_benchmark(self.benchmark_path)
        self.parser_adapter = parser_adapter or LiveLmStudioParserAdapter(
            model_alias=model_alias,
            benchmark_path=self.benchmark_path,
            system_prompt_path=self.system_prompt_path,
        )
        self.runtime_orchestrator = runtime_orchestrator or RuntimeOrchestrator()
        self.canonical_validator = canonical_validator or DatasetRoutedCanonicalValidator()
        active_dataset_registry = getattr(self.runtime_orchestrator, "dataset_registry", None)
        if active_dataset_registry is None:
            active_dataset_registry = getattr(self.canonical_validator, "dataset_registry", None)
        if active_dataset_registry is None:
            active_model_registry = getattr(self.runtime_orchestrator, "model_registry", None) or ModelRegistry()
            active_dataset_registry = DatasetRegistry(
                {
                    "bank": BankDatasetPackage(active_model_registry),
                    "grad": GradDatasetPackage(active_model_registry),
                }
            )
        self.dataset_registry = active_dataset_registry
        self.request_builder = ConversationRequestBuilder(
            canonical_validator=self.canonical_validator,
            benchmark=self.benchmark,
            policy=self.canonical_validator.context.policy,
        )
        self.output_root = Path(output_root or DEFAULT_OUTPUT_ROOT)
        self.model_alias = model_alias

    def run_turn(
        self,
        *,
        user_input: str,
        save_artifacts: bool = True,
        scenario_slug: str | None = None,
        debug_trace_enabled: bool = False,
        command: str | None = None,
        session_trace: dict[str, Any] | None = None,
        pending_clarification: PendingClarification | None = None,
        canonical_session_state: dict[str, Any] | None = None,
        clarification_turns_used: int = 0,
        clarification_turn_limit: int = 3,
        dataset_id: str = "bank",
    ) -> ConversationTurnResult:
        started = time.perf_counter()
        prepared = self.prepare_turn(user_input=user_input, dataset_id=dataset_id)
        result = self.finalize_turn(
            user_input=user_input,
            parser_result=prepared["parser_result"],
            repair_result=prepared["repair_result"],
            normalized_parse=prepared["normalized_parse"],
            parser_quality=prepared["parser_quality"],
            field_provenance=prepared["field_provenance"],
            schema_validation=prepared["schema_validation"],
            canonical_validation=prepared["canonical_validation"],
            dataset_package=prepared["dataset_package"],
            benchmark=prepared["benchmark"],
            save_artifacts=save_artifacts,
            scenario_slug=scenario_slug,
            debug_trace_enabled=debug_trace_enabled,
            command=command,
            session_trace=session_trace,
            pending_clarification=pending_clarification,
            canonical_session_state=canonical_session_state,
            clarification_turns_used=clarification_turns_used,
            clarification_turn_limit=clarification_turn_limit,
            dataset_id=dataset_id,
        )
        timing_metrics = dict(result.timing_metrics or {})
        timing_metrics["end_to_end_latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        result.timing_metrics = timing_metrics
        return result

    def prepare_turn(self, *, user_input: str, dataset_id: str = "bank") -> dict[str, Any]:
        dataset_package = self._resolve_shell_dataset_package(dataset_id)
        benchmark = dataset_package.live_primary_benchmark()
        parser_result = invoke_parser_adapter_method(
            self.parser_adapter.parse,
            user_text=user_input,
            benchmark=benchmark,
            dataset_package=dataset_package,
        )
        repair_result = None

        normalized, parser_quality, field_provenance, schema_validation, canonical_validation = self.evaluate_parser_output(
            user_input=user_input,
            message_text=parser_result.message_text,
            api_error=parser_result.api_error,
            dataset_package=dataset_package,
            benchmark=benchmark,
        )
        repair_errors = collect_repair_errors(
            parser_result=parser_result,
            normalized=normalized,
            schema_validation=schema_validation,
            canonical_validation=canonical_validation,
        )

        if should_attempt_repair(
            raw_output=parser_result.message_text,
            api_error=parser_result.api_error,
            errors=repair_errors,
        ):
            repair_result = invoke_parser_adapter_method(
                self.parser_adapter.repair,
                invalid_output=parser_result.message_text,
                errors=repair_errors,
                benchmark=benchmark,
                dataset_package=dataset_package,
            )
            normalized, parser_quality, field_provenance, schema_validation, canonical_validation = self.evaluate_parser_output(
                user_input=user_input,
                message_text=repair_result.message_text,
                api_error=repair_result.api_error,
                dataset_package=dataset_package,
                benchmark=benchmark,
            )

        return {
            "parser_result": parser_result,
            "repair_result": repair_result,
            "normalized_parse": normalized.parsed_json,
            "parser_quality": finalize_parser_quality_metadata(
                parser_quality,
                canonical_pass_after_quality=bool(canonical_validation.ready_for_runtime),
                repair_invoked=repair_result is not None,
            ),
            "field_provenance": field_provenance,
            "schema_validation": schema_validation,
            "canonical_validation": canonical_validation,
            "dataset_package": dataset_package,
            "benchmark": benchmark,
        }

    def evaluate_parser_output(
        self,
        *,
        user_input: str,
        message_text: str,
        api_error: str | None,
        dataset_package,
        benchmark,
    ):
        quality_result = run_parser_quality(
            message_text=message_text,
            benchmark_spec=benchmark,
            user_text=user_input,
            api_error=api_error,
            dataset_id=dataset_package.dataset_id,
            numeric_bound_fields=dataset_package.numeric_bound_fields(),
        )
        canonical_validation = self.canonical_validator.validate(
            candidate=quality_result.normalized.parsed_json,
            schema_validation=quality_result.schema_validation,
            dataset_id=dataset_package.dataset_id,
        )
        return (
            quality_result.normalized,
            quality_result.metadata,
            quality_result.field_provenance,
            quality_result.schema_validation,
            canonical_validation,
        )

    def finalize_turn(
        self,
        *,
        user_input: str,
        parser_result,
        repair_result,
        normalized_parse: dict[str, Any] | None,
        parser_quality: dict[str, Any] | None,
        field_provenance: dict[str, str] | None,
        schema_validation,
        canonical_validation,
        dataset_package,
        benchmark,
        save_artifacts: bool = True,
        scenario_slug: str | None = None,
        debug_trace_enabled: bool = False,
        command: str | None = None,
        session_trace: dict[str, Any] | None = None,
        pending_clarification: PendingClarification | None = None,
        canonical_session_state: dict[str, Any] | None = None,
        clarification_turns_used: int = 0,
        clarification_turn_limit: int = 3,
        dataset_id: str = "bank",
    ) -> ConversationTurnResult:
        turn_id = make_run_id()
        timestamp_utc = utc_now_iso()
        stage_trace: list[str] = []
        runtime_result = None
        runtime_debug_trace = None
        invariant_validation = None
        clarification_payload = None
        explanation_payload = None
        user_response_payload = None
        parser_failure_cause = None
        is_case_complete = False
        case_completion_reason = None
        restart_required = False
        clarification_turns_used_result = clarification_turns_used
        self.request_builder.benchmark = benchmark
        builder_result = self.request_builder.build(
            user_input=user_input,
            normalized_candidate=normalized_parse,
            schema_validation=schema_validation,
            canonical_validation=canonical_validation,
            parser_quality=parser_quality,
            pending_clarification=pending_clarification,
            canonical_session_state=canonical_session_state,
            field_provenance=field_provenance,
            policy=dataset_package.runtime_context().policy,
            required_fields=list(self.canonical_validator.required_fields),
            dataset_id=dataset_package.dataset_id,
            supported_dataset_ids=self.dataset_registry.keys(),
        )
        negotiation_transition = None
        response_decision = None
        effective_normalized_parse = normalized_parse
        effective_parser_quality = dict(parser_quality or {})
        effective_field_provenance = dict(field_provenance or {})
        effective_schema_validation = schema_validation.to_dict()
        effective_canonical_validation = canonical_validation.to_dict()
        timing_metrics: dict[str, Any] = {}

        if builder_result is not None:
            effective_normalized_parse = builder_result._normalized_candidate
            effective_parser_quality = dict(builder_result.provenance.get("parser_quality", {}))
            effective_field_provenance = (
                dict(builder_result.provenance.get("field_provenance", {}))
                if isinstance(builder_result.provenance, dict)
                else {}
            )
            effective_schema_validation = dict(builder_result._schema_validation)
            effective_canonical_validation = builder_result._canonical_validation.to_dict()

        if builder_result is not None and builder_result.builder_status == ConversationStage.READY_FOR_RUNTIME:
            controller = ConversationNegotiationController()
            controller.transition(
                next_state=ConversationStage.READY_FOR_RUNTIME,
                transition_reason=RUNTIME_READY,
                merge_applied=builder_result.merge_applied,
                bounded_suggestion_available=False,
            )
            runtime_payload = builder_result.runtime_request
            runtime_started = time.perf_counter()
            runtime_obj = self.runtime_orchestrator.handle(
                runtime_payload,
                include_debug_trace=True,
            )
            runtime_latency_ms = round((time.perf_counter() - runtime_started) * 1000.0, 3)
            runtime_result = runtime_obj.to_dict(include_debug_trace=False)
            runtime_debug_trace = None if runtime_obj.debug_trace is None else runtime_obj.debug_trace.to_dict()
            if runtime_debug_trace is not None:
                runtime_debug_trace["request_latency_ms"] = runtime_latency_ms
            timing_metrics["runtime_latency_ms"] = runtime_latency_ms
            invariant_validation = (
                None if runtime_obj.invariant_validation is None else runtime_obj.invariant_validation.to_dict()
            )
            suggestion_types = determine_bounded_suggestion_types(runtime_obj.reason_codes)
            invariant_failed = bool(
                runtime_obj.invariant_validation is not None and runtime_obj.invariant_validation.status == "failed"
            )
            if runtime_obj.controller_state == "TERMINAL_SUCCESS" and not invariant_failed:
                final_stage = ConversationStage.RUNTIME_SUCCESS
                is_case_complete = True
                case_completion_reason = "runtime_success"
                restart_required = True
                if runtime_obj.reason_codes == ["NO_RECOURSE_NEEDED"]:
                    transition_reason = RUNTIME_SUCCESS_NO_RECOURSE
                    template_type = "explanation_no_recourse"
                else:
                    transition_reason = RUNTIME_SUCCESS_COUNTERFACTUAL_FOUND
                    template_type = "explanation_counterfactual"
            else:
                final_stage = ConversationStage.RUNTIME_REJECT
                is_case_complete = True
                case_completion_reason = "runtime_reject"
                restart_required = True
                transition_reason = (
                    RUNTIME_REJECT_NO_FEASIBLE_CF
                    if "NO_FEASIBLE_CF_FOUND" in runtime_obj.reason_codes and not invariant_failed
                    else RUNTIME_REJECT_SYSTEM_ERROR
                )
                template_type = "explanation_reject"
            negotiation_transition = controller.transition(
                next_state=final_stage,
                transition_reason=transition_reason,
                merge_applied=builder_result.merge_applied,
                bounded_suggestion_available=bool(suggestion_types) and not invariant_failed,
            )
            stage_trace = list(negotiation_transition.state_trace)
            explanation_runtime_result = runtime_result
            if invariant_failed:
                explanation_runtime_result = dict(runtime_result or {})
                explanation_runtime_result["controller_state"] = "TERMINAL_REJECT"
                explanation_runtime_result["counterfactual"] = None
                explanation_runtime_result["reason_codes"] = [INVALID_COUNTERFACTUAL_BLOCKED]
            explanation_payload = build_explanation_payload(
                runtime_result=explanation_runtime_result,
                current_profile=runtime_payload["profile"],
                included_suggestion_types=[] if invariant_failed else suggestion_types,
                policy=self.canonical_validator.context.policy,
                dataset_label=dataset_package.primary_subject_label(),
            )
            response_payload = build_user_response_payload_from_explanation(
                explanation_payload=explanation_payload,
                runtime_result=explanation_runtime_result,
                current_profile=runtime_payload["profile"],
                policy=self.canonical_validator.context.policy,
                dataset_label=dataset_package.primary_subject_label(),
                active_constraint_spec=runtime_payload.get("constraint_spec"),
                transition_reason=transition_reason,
            )
            user_response_payload = response_payload.to_dict()
            response_text = render_explanation_text(
                explanation_payload,
                dataset_label=dataset_package.primary_subject_label(),
                parser_adapter=self.parser_adapter,
                runtime_result=explanation_runtime_result,
                current_profile=runtime_payload["profile"],
                policy=self.canonical_validator.context.policy,
                active_constraint_spec=runtime_payload.get("constraint_spec"),
                transition_reason=transition_reason,
            )
            response_decision = ResponseDecision(
                final_public_state=final_stage,
                template_type=template_type,
                included_suggestion_types=list(explanation_payload.included_suggestion_types),
            )
        elif builder_result is not None and builder_result.builder_status == ConversationStage.CONFLICT:
            controller = ConversationNegotiationController()
            final_stage = ConversationStage.CONFLICT
            is_case_complete = True
            case_completion_reason = "conflict"
            restart_required = True
            transition_reason = (
                FOLLOWUP_RESET_NEW_REQUEST
                if builder_result.pending_reset and pending_clarification is not None
                else CONFLICTING_VALUES
            )
            negotiation_transition = controller.transition(
                next_state=final_stage,
                transition_reason=transition_reason,
                merge_applied=builder_result.merge_applied,
                bounded_suggestion_available=False,
            )
            stage_trace = list(negotiation_transition.state_trace)
            clarification_payload = build_clarification_payload(
                required_fields=list(self.canonical_validator.required_fields),
                missing_fields=builder_result.missing_fields,
                conflicts=builder_result.conflicts,
                carried_forward_fields=list((builder_result.partial_profile_snapshot or {}).keys()),
                dataset_label=dataset_package.primary_subject_label(),
            )
            user_response_payload = build_user_response_payload_from_clarification(
                clarification_payload,
                dataset_label=dataset_package.primary_subject_label(),
            ).to_dict()
            response_text = render_clarification_text(
                clarification_payload,
                dataset_label=dataset_package.primary_subject_label(),
                parser_adapter=self.parser_adapter,
            )
            response_decision = ResponseDecision(
                final_public_state=final_stage,
                template_type="clarification_conflict",
                included_suggestion_types=[],
            )
        elif builder_result is not None and builder_result.builder_status == ConversationStage.NEEDS_CLARIFICATION:
            controller = ConversationNegotiationController()
            final_stage = ConversationStage.NEEDS_CLARIFICATION
            next_clarification_turns_used = clarification_turns_used + 1
            clarification_turns_used_result = next_clarification_turns_used
            transition_reason = (
                FOLLOWUP_RESET_NEW_REQUEST
                if builder_result.pending_reset and pending_clarification is not None
                else MISSING_REQUIRED_FIELDS
            )
            negotiation_transition = controller.transition(
                next_state=final_stage,
                transition_reason=transition_reason,
                merge_applied=builder_result.merge_applied,
                bounded_suggestion_available=False,
            )
            stage_trace = list(negotiation_transition.state_trace)
            remaining_rounds = max(clarification_turn_limit - next_clarification_turns_used, 0)
            if next_clarification_turns_used >= clarification_turn_limit:
                clarification_payload = build_clarification_limit_reached_payload(
                    dataset_label=dataset_package.primary_subject_label(),
                )
                is_case_complete = True
                case_completion_reason = "clarification_limit_reached"
                restart_required = True
            else:
                clarification_payload = build_clarification_payload(
                    required_fields=list(self.canonical_validator.required_fields),
                    missing_fields=builder_result.missing_fields,
                    conflicts=builder_result.conflicts,
                    carried_forward_fields=list((builder_result.partial_profile_snapshot or {}).keys()),
                    remaining_rounds=remaining_rounds,
                    restart_required=False,
                    dataset_label=dataset_package.primary_subject_label(),
                )
            user_response_payload = build_user_response_payload_from_clarification(
                clarification_payload,
                dataset_label=dataset_package.primary_subject_label(),
            ).to_dict()
            response_text = render_clarification_text(
                clarification_payload,
                dataset_label=dataset_package.primary_subject_label(),
                parser_adapter=self.parser_adapter,
            )
            response_decision = ResponseDecision(
                final_public_state=final_stage,
                template_type=(
                    "clarification_limit_reached"
                    if is_case_complete
                    else "clarification_missing_information"
                ),
                included_suggestion_types=[],
            )
        elif builder_result is not None and builder_result.builder_status == ConversationStage.UNSUPPORTED_REQUEST:
            controller = ConversationNegotiationController()
            final_stage = ConversationStage.UNSUPPORTED_REQUEST
            is_case_complete = True
            case_completion_reason = "unsupported_request"
            restart_required = True
            negotiation_transition = controller.transition(
                next_state=final_stage,
                transition_reason=UNSUPPORTED_INTENT,
                merge_applied=builder_result.merge_applied,
                bounded_suggestion_available=False,
            )
            stage_trace = list(negotiation_transition.state_trace)
            builder_provenance = builder_result.provenance if isinstance(builder_result.provenance, dict) else {}
            response_text = render_unsupported_request_text(
                required_fields=list(self.canonical_validator.required_fields),
                dataset_label=dataset_package.primary_subject_label(),
                dataset_id=dataset_package.dataset_id,
                unsupported_intent_type=builder_provenance.get("unsupported_intent_type"),
                requested_dataset_label=builder_provenance.get("requested_dataset_label"),
            )
            response_decision = ResponseDecision(
                final_public_state=final_stage,
                template_type="unsupported_request",
                included_suggestion_types=[],
            )
        else:
            final_stage = ConversationStage.PARSER_FAILURE
            is_case_complete = True
            case_completion_reason = "parser_failure"
            restart_required = True
            stage_trace.append(final_stage)
            active_parser_result = repair_result or parser_result
            parser_failure_cause = determine_parser_failure_cause_from_parse(
                parser_result=active_parser_result,
                normalized_parse=normalized_parse,
            )
            parser_failure_errors = collect_repair_errors(
                parser_result=active_parser_result,
                normalized=build_normalized_proxy(normalized_parse),
                schema_validation=schema_validation,
                canonical_validation=canonical_validation,
            )
            response_text = build_parser_failure_text(
                required_fields=list(self.canonical_validator.required_fields),
                errors=parser_failure_errors,
                failure_cause=parser_failure_cause,
                repair_attempted=repair_result is not None,
                dataset_label=dataset_package.primary_subject_label(),
            )
            response_decision = ResponseDecision(
                final_public_state=final_stage,
                template_type="parser_failure",
                included_suggestion_types=[],
            )

        turn_result = ConversationTurnResult(
            turn_id=turn_id,
            timestamp_utc=timestamp_utc,
            model_alias=self.model_alias,
            user_input=user_input,
            stage=final_stage,
            stage_trace=stage_trace,
            parser_result=parser_result,
            repair_result=repair_result,
            normalized_parse=effective_normalized_parse,
            schema_validation=effective_schema_validation,
            canonical_validation=effective_canonical_validation,
            builder_result=builder_result,
            negotiation_transition=negotiation_transition,
            response_decision=response_decision,
            runtime_result=runtime_result,
            runtime_debug_trace=runtime_debug_trace,
            invariant_validation=invariant_validation,
            field_provenance=effective_field_provenance,
            parser_quality_metadata=effective_parser_quality,
            clarification_payload=clarification_payload,
            explanation_payload=explanation_payload,
            response_text=response_text,
            user_response_payload=user_response_payload,
            parser_failure_cause=parser_failure_cause,
            is_case_complete=is_case_complete,
            case_completion_reason=case_completion_reason,
            restart_required=restart_required,
            clarification_turns_used=clarification_turns_used_result,
            timing_metrics=timing_metrics,
        )

        if save_artifacts:
            snapshot = self.build_config_snapshot(
                command=command or shell_command_from_argv(),
                debug_trace_enabled=debug_trace_enabled,
                dataset_package=dataset_package,
                benchmark=benchmark,
            )
            resolved_session_trace = merge_session_trace(
                base_session_trace=session_trace,
                builder_result=builder_result,
                pending_clarification=pending_clarification,
            )
            save_conversation_artifacts(
                turn_result,
                output_root=self.output_root,
                scenario_slug=scenario_slug,
                command=command or shell_command_from_argv(),
                config_snapshot=snapshot,
                debug_trace_enabled=debug_trace_enabled,
                session_trace=resolved_session_trace,
            )
        return turn_result

    def build_config_snapshot(
        self,
        *,
        command: str,
        debug_trace_enabled: bool,
        dataset_package,
        benchmark,
    ) -> dict[str, Any]:
        return {
            "timestamp_utc": utc_now_iso(),
            "dataset": dataset_package.dataset_id,
            "benchmark_path": (
                str(self.benchmark_path)
                if dataset_package.dataset_id == "bank"
                else f"generated::{dataset_package.dataset_id}::primary_benchmark"
            ),
            "benchmark_sha256": sha256_file(self.benchmark_path) if dataset_package.dataset_id == "bank" else None,
            "system_prompt_path": str(self.system_prompt_path),
            "system_prompt_sha256": sha256_file(self.system_prompt_path),
            "response_schema_path": describe_schema_path(self.parser_adapter, dataset_package=dataset_package),
            "response_schema_sha256": describe_schema_sha256(self.parser_adapter, dataset_package=dataset_package),
            "model_alias": self.model_alias,
            "lm_studio_model": self.model_alias,
            "runtime_mode": getattr(self.runtime_orchestrator, "runtime_mode", None),
            "counterfactual_backend_name": getattr(self.runtime_orchestrator, "counterfactual_backend_name", "ufce"),
            "api_base": getattr(self.parser_adapter, "api_base", None),
            "structured_output_mode": getattr(self.parser_adapter, "structured_output_mode", None),
            "parser_contract": {
                "task": benchmark.output_contract.task,
                "status_enum": list(benchmark.output_contract.status_enum),
                "allowed_fields": list(benchmark.allowed_field_names),
            },
            "parser_request_profiles": describe_request_profiles(self.parser_adapter, dataset_package=dataset_package),
            "llm_task_token_policy": describe_token_policy(self.parser_adapter),
            "deterministic_template_output": True,
            "repair_enabled": True,
            "debug_trace_enabled": bool(debug_trace_enabled),
            "command": command,
            "system_prompt_preview_sha256": sha256_text(getattr(self.parser_adapter, "system_prompt", "")),
        }

    def _resolve_shell_dataset_package(self, dataset_id: str):
        normalized = str(dataset_id or "bank").strip().lower() or "bank"
        if self.dataset_registry.has(normalized):
            return self.dataset_registry.get(normalized)
        return self.dataset_registry.get("bank")


def build_parser_failure_text(
    *,
    required_fields: list[str],
    errors: list[str],
    failure_cause: str | None,
    repair_attempted: bool,
    dataset_label: str = "bank profile",
) -> str:
    ordered_fields = ", ".join(required_fields)
    reason_text = "; ".join(errors) if errors else "parser output remained invalid"
    if failure_cause == "timeout_no_body":
        failure_text = "LM Studio timed out before returning any parser response body."
    elif failure_cause == "unsupported_structured_output":
        failure_text = "The current LM Studio server or model rejected strict structured JSON output."
    elif failure_cause == "invalid_json_body":
        failure_text = "The parser returned a body, but it did not contain one valid JSON object."
    else:
        failure_text = f"Reason: {reason_text}."
    attempt_text = "after one repair attempt" if repair_attempted else "from the initial parser call"
    return (
        f"I could not validate the parser output {attempt_text}. "
        f"{failure_text} "
        f"Please start a new case and submit one complete corrected {dataset_label} using all required fields: "
        f"{ordered_fields}."
    )


def shell_command_from_argv() -> str:
    return " ".join(sys.argv)


def determine_parser_failure_cause(*, parser_result, normalized) -> str | None:
    if getattr(parser_result, "failure_cause", None):
        return parser_result.failure_cause
    if normalized.parse_error and parser_result.message_text.strip():
        return "invalid_json_body"
    return None


def determine_parser_failure_cause_from_parse(*, parser_result, normalized_parse: dict[str, Any] | None) -> str | None:
    if getattr(parser_result, "failure_cause", None):
        return parser_result.failure_cause
    if parser_result.message_text.strip() and normalized_parse is None:
        return "invalid_json_body"
    return None


def build_normalized_proxy(normalized_parse: dict[str, Any] | None):
    class _NormalizedProxy:
        def __init__(self, parsed_json: dict[str, Any] | None) -> None:
            self.parsed_json = parsed_json
            self.parse_error = None if parsed_json is not None else "No parsed JSON object available."

    return _NormalizedProxy(normalized_parse)


def prune_field_provenance(
    *,
    candidate: dict[str, Any] | None,
    field_provenance: dict[str, str] | None,
) -> dict[str, str]:
    if not isinstance(candidate, dict) or not isinstance(field_provenance, dict):
        return {}
    cf_request = candidate.get("cf_request")
    if not isinstance(cf_request, dict):
        return {}
    retained_fields = set(cf_request)
    retained_fields.update(
        field_name
        for field_name, provenance in field_provenance.items()
        if provenance == "conflict"
    )
    return {
        str(field_name): str(value)
        for field_name, value in field_provenance.items()
        if field_name in retained_fields and isinstance(field_name, str) and isinstance(value, str)
    }


def describe_schema_path(parser_adapter, *, dataset_package=None) -> str | None:
    if dataset_package is not None and getattr(dataset_package, "dataset_id", "bank") != "bank":
        schema_name = getattr(dataset_package, "primary_response_schema_name", None)
        if callable(schema_name):
            return f"generated::{schema_name()}"
        return f"generated::{getattr(dataset_package, 'dataset_id', 'dataset')}::primary_schema"
    schema_path = getattr(parser_adapter, "schema_path", None)
    if schema_path is None:
        return None
    return str(schema_path)


def describe_schema_sha256(parser_adapter, *, dataset_package=None) -> str | None:
    if dataset_package is not None and getattr(dataset_package, "dataset_id", "bank") != "bank":
        return None
    schema_path = getattr(parser_adapter, "schema_path", None)
    if not isinstance(schema_path, Path) or not schema_path.exists():
        return None
    return sha256_file(schema_path)


def describe_request_profiles(parser_adapter, *, dataset_package=None) -> dict[str, Any]:
    describe = getattr(parser_adapter, "describe_request_profile", None)
    if callable(describe):
        return {
            "parse": invoke_parser_adapter_method(describe, "parse", dataset_package=dataset_package),
            "repair": invoke_parser_adapter_method(describe, "repair", dataset_package=dataset_package),
        }
    return {}


def describe_token_policy(parser_adapter) -> dict[str, Any]:
    describe = getattr(parser_adapter, "describe_token_policy", None)
    if callable(describe):
        return describe()
    return {}


def determine_bounded_suggestion_types(reason_codes: list[str]) -> list[str]:
    if REQUEST_CONSTRAINTS_BLOCKED in reason_codes:
        return [
            "revise_target_profile",
        ]
    if "NO_FEASIBLE_CF_FOUND" not in reason_codes:
        return []
    return [
        "revise_target_profile",
        "broaden_allowed_financial_changes",
    ]


def merge_session_trace(
    *,
    base_session_trace: dict[str, Any] | None,
    builder_result,
    pending_clarification: PendingClarification | None,
) -> dict[str, Any] | None:
    if not isinstance(base_session_trace, dict):
        return None
    merged = dict(base_session_trace)
    if builder_result is None:
        return merged
    if builder_result.pending_reset:
        merged["parent_turn_id"] = None
        merged["merge_applied"] = False
        merged["carried_fields"] = []
        merged["carried_constraint_keys"] = []
        return merged
    if builder_result.merge_applied and pending_clarification is not None:
        merged["parent_turn_id"] = pending_clarification.originating_turn_id
        merged["merge_applied"] = True
        merged["carried_fields"] = list(builder_result.carried_fields)
        merged["carried_constraint_keys"] = list(builder_result.carried_constraint_keys)
    return merged


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
