from __future__ import annotations

from dataclasses import dataclass

from llm.src.conversation.orchestrator import BankConversationOrchestrator


@dataclass(frozen=True)
class StubResult:
    message_text: str
    api_error: str | None = None
    http_status_code: int | None = 200
    raw_response_text: str | None = None
    response_json: dict | None = None
    reasoning_text: str = ""
    usage: dict | None = None
    stats: dict | None = None
    derived_metrics: dict | None = None
    request_payload: dict | None = None
    failure_cause: str | None = None
    task_type: str | None = "parse"

    def to_dict(self) -> dict:
        return {
            "message_text": self.message_text,
            "api_error": self.api_error,
            "http_status_code": self.http_status_code,
            "raw_response_text": self.raw_response_text,
            "response_json": self.response_json,
            "reasoning_text": self.reasoning_text,
            "usage": dict(self.usage or {}),
            "stats": dict(self.stats or {}),
            "derived_metrics": dict(self.derived_metrics or {}),
            "request_payload": dict(self.request_payload or {}),
            "failure_cause": self.failure_cause,
            "task_type": self.task_type,
        }


class StubParserAdapter:
    def __init__(
        self,
        parse_result: StubResult,
        repair_result: StubResult | None = None,
        refinement_parse_result: StubResult | None = None,
        refinement_repair_result: StubResult | None = None,
    ) -> None:
        self.parse_result = parse_result
        self.repair_result = repair_result
        self.refinement_parse_result = refinement_parse_result
        self.refinement_repair_result = refinement_repair_result
        self.repair_calls = 0
        self.refinement_repair_calls = 0
        self.api_base = "stub://adapter"
        self.system_prompt = "stub prompt"
        self.structured_output_mode = "json_schema_strict"

    def parse(self, *, user_text: str, benchmark=None):
        del user_text
        del benchmark
        return self.parse_result

    def repair(self, *, invalid_output: str, errors: list[str], benchmark=None):
        del invalid_output
        del errors
        del benchmark
        self.repair_calls += 1
        if self.repair_result is None:
            raise AssertionError("repair should not be called without a repair result")
        return self.repair_result

    def parse_refinement(
        self,
        *,
        user_text: str,
        active_constraint_spec=None,
        pending_refinement_clarification=None,
        benchmark=None,
    ):
        del user_text
        del active_constraint_spec
        del pending_refinement_clarification
        del benchmark
        if self.refinement_parse_result is None:
            raise AssertionError("unexpected refinement parse call")
        return self.refinement_parse_result

    def repair_refinement(
        self,
        *,
        invalid_output: str,
        errors: list[str],
        active_constraint_spec=None,
        pending_refinement_clarification=None,
        benchmark=None,
    ):
        del invalid_output
        del errors
        del active_constraint_spec
        del pending_refinement_clarification
        del benchmark
        self.refinement_repair_calls += 1
        if self.refinement_repair_result is None:
            raise AssertionError("unexpected refinement repair call")
        return self.refinement_repair_result

    def describe_request_profile(self, task_type: str) -> dict:
        max_tokens = 768 if task_type in {"repair", "repair_refinement"} else 512
        return {
            "task_type": task_type,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
            "structured_output_mode": "json_schema_strict",
            "response_schema_name": (
                "ufce_bank_refinement_feedback_output_v1"
                if task_type in {"parse_refinement", "repair_refinement"}
                else "ufce_bank_cf_parser_output_v1"
            ),
        }

    def describe_token_policy(self) -> dict:
        return {
            "parse": 512,
            "repair": 768,
            "future_explanation": {"min": 1536, "max": 2048},
            "future_negotiation": {"min": 2048, "max": 3072},
        }


@dataclass(frozen=True)
class StubRuntimeResponse:
    payload: dict

    @property
    def controller_state(self) -> str:
        return str(self.payload["controller_state"])

    @property
    def reason_codes(self) -> list[str]:
        return list(self.payload.get("reason_codes", []))

    @property
    def debug_trace(self):
        value = self.payload.get("debug_trace")
        if value is None:
            return None
        return _RuntimeTraceProxy(value)

    @property
    def invariant_validation(self):
        value = self.payload.get("invariant_validation")
        if value is None:
            return None
        return _InvariantValidationProxy(value)

    def to_dict(self, include_debug_trace: bool = False) -> dict:
        payload = dict(self.payload)
        if not include_debug_trace:
            payload.pop("debug_trace", None)
        return payload


@dataclass(frozen=True)
class _RuntimeTraceProxy:
    payload: dict

    def to_dict(self) -> dict:
        return dict(self.payload)


@dataclass(frozen=True)
class _InvariantValidationProxy:
    payload: dict

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    def to_dict(self) -> dict:
        return dict(self.payload)


class StubRuntimeOrchestrator:
    def __init__(self, payload: dict) -> None:
        self.payload = dict(payload)

    def handle(self, request, include_debug_trace: bool = False):
        del request
        del include_debug_trace
        return StubRuntimeResponse(self.payload)


def test_conversation_runtime_success_without_recourse(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input=(
            "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
            "SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0."
        ),
        save_artifacts=False,
    )

    assert result.stage == "RUNTIME_SUCCESS"
    assert result.stage_trace == ["READY_FOR_RUNTIME", "RUNTIME_SUCCESS"]
    assert result.explanation_payload is not None
    assert result.explanation_payload.summary_type == "no_recourse_needed"
    assert result.runtime_result is not None
    assert result.runtime_result["reason_codes"] == ["NO_RECOURSE_NEEDED"]
    assert result.negotiation_transition is not None
    assert result.negotiation_transition.transition_reason == "runtime_success_no_recourse"
    assert result.response_decision is not None
    assert result.response_decision.final_public_state == "RUNTIME_SUCCESS"


def test_conversation_runtime_success_with_counterfactual(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":100,"Family":1,"CCAvg":2.7,"Education":2,"Mortgage":0,'
                '"SecuritiesAccount":0,"CDAccount":0,"Online":0,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input=(
            "Income 100, Family 1, CCAvg 2.7, Education 2, Mortgage 0, "
            "SecuritiesAccount 0, CDAccount 0, Online 0, CreditCard 0."
        ),
        save_artifacts=False,
    )

    assert result.stage == "RUNTIME_SUCCESS"
    assert result.explanation_payload is not None
    assert result.explanation_payload.summary_type == "counterfactual_found"
    assert result.explanation_payload.counterfactual_summary is not None
    assert result.explanation_payload.counterfactual_summary["rank"] == 1
    assert result.negotiation_transition is not None
    assert result.negotiation_transition.transition_reason == "runtime_success_counterfactual_found"
    assert result.response_decision is not None
    assert result.response_decision.final_public_state == "RUNTIME_SUCCESS"


def test_conversation_runtime_reject_with_infeasibility(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":49,"Family":4,"CCAvg":1.6,"Education":1,"Mortgage":0,'
                '"SecuritiesAccount":1,"CDAccount":0,"Online":0,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input=(
            "Income 49, Family 4, CCAvg 1.6, Education 1, Mortgage 0, "
            "SecuritiesAccount 1, CDAccount 0, Online 0, CreditCard 0."
        ),
        save_artifacts=False,
    )

    assert result.stage == "RUNTIME_REJECT"
    assert result.explanation_payload is not None
    assert result.explanation_payload.summary_type == "runtime_reject"
    assert result.explanation_payload.included_suggestion_types == [
        "revise_target_profile",
        "broaden_allowed_financial_changes",
    ]
    assert result.runtime_result is not None
    assert result.runtime_result["reason_codes"] == ["NO_FEASIBLE_CF_FOUND"]
    assert result.negotiation_transition is not None
    assert result.negotiation_transition.transition_reason == "runtime_reject_no_feasible_cf"
    assert result.response_decision is not None
    assert result.response_decision.included_suggestion_types == [
        "revise_target_profile",
        "broaden_allowed_financial_changes",
    ]


def test_conversation_complete_but_missing_fields_attempts_repair_then_requests_clarification(
    sample_benchmark, tmp_path
):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":{"Income":40,"Online":1},'
                '"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","SecuritiesAccount","CreditCard"],'
                '"conflicts":[],"notes":[]}'
            )
        ),
        repair_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"Online":1},'
                '"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","SecuritiesAccount","CreditCard"],'
                '"conflicts":[],"notes":[]}'
            ),
            task_type="repair",
        ),
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(user_input="Income 40 and Online yes.", save_artifacts=False)

    assert adapter.repair_calls == 1
    assert result.repair_result is not None
    assert result.stage == "NEEDS_CLARIFICATION"
    assert result.clarification_payload is not None
    assert result.clarification_payload.clarification_type == "missing_information"
    assert result.negotiation_transition is not None
    assert result.negotiation_transition.transition_reason == "missing_required_fields"
    assert result.response_decision is not None
    assert result.response_decision.final_public_state == "NEEDS_CLARIFICATION"
    assert "Reply with only the missing fields" in result.response_text


def test_conversation_clarification_uses_conversational_adapter_when_available(sample_benchmark, tmp_path):
    class ConversationalParserAdapter(StubParserAdapter):
        def __init__(self, *, parse_result: StubResult, conversational_response: str) -> None:
            super().__init__(parse_result=parse_result)
            self.conversational_response = conversational_response
            self.conversation_calls: list[dict[str, object]] = []

        def generate_conversational_response(self, *, system_prompt: str, user_prompt: str, max_tokens: int | None = None):
            self.conversation_calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "max_tokens": max_tokens,
                }
            )
            return self.conversational_response

    adapter = ConversationalParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40},'
                '"missing_fields":["Family","Education","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                '"conflicts":[],"notes":[]}'
            )
        ),
        conversational_response="Please send just the missing fields.",
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(user_input="Income 40.", save_artifacts=False)

    assert result.stage == "NEEDS_CLARIFICATION"
    assert result.response_text == "Please send just the missing fields."
    assert len(adapter.conversation_calls) == 1
    assert "reply with only the missing fields" in adapter.conversation_calls[0]["system_prompt"].lower()
    assert "fallback_meaning" in adapter.conversation_calls[0]["user_prompt"]


def test_conversation_runtime_explanation_uses_conversational_adapter_when_available(sample_benchmark, tmp_path):
    class ConversationalParserAdapter(StubParserAdapter):
        def __init__(self, *, parse_result: StubResult, conversational_response: str) -> None:
            super().__init__(parse_result=parse_result)
            self.conversational_response = conversational_response
            self.conversation_calls: list[dict[str, object]] = []

        def generate_conversational_response(self, *, system_prompt: str, user_prompt: str, max_tokens: int | None = None):
            self.conversation_calls.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "max_tokens": max_tokens,
                }
            )
            return self.conversational_response

    adapter = ConversationalParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        ),
        conversational_response="Friendly runtime explanation from LLM.",
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input=(
            "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
            "SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0."
        ),
        save_artifacts=False,
    )

    assert result.stage == "RUNTIME_SUCCESS"
    assert result.response_text == "Friendly runtime explanation from LLM."
    assert len(adapter.conversation_calls) == 1
    assert "no recourse changes are needed" in adapter.conversation_calls[0]["system_prompt"].lower()
    assert "fallback_meaning" in adapter.conversation_calls[0]["user_prompt"]


def test_conversation_conflict_requires_confirmed_conflict_payload(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"conflict","cf_request":{"Income":40},'
                '"missing_fields":[],"conflicts":["Income cannot be both 40 and 60."],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(user_input="bank conflict", save_artifacts=False)

    assert result.stage == "CONFLICT"
    assert result.clarification_payload is not None
    assert result.clarification_payload.clarification_type == "conflict_resolution"
    assert result.canonical_validation["confirmed_conflicts"] == ["Income cannot be both 40 and 60."]
    assert result.negotiation_transition is not None
    assert result.negotiation_transition.transition_reason == "conflicting_values"
    assert result.response_decision is not None
    assert result.response_decision.final_public_state == "CONFLICT"


def test_conversation_unsupported_request_stays_outside_parser_failure(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"needs_clarification","cf_request":{},'
                '"missing_fields":["Income","Family","CCAvg","Education","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                '"conflicts":[],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input="Give me general financial advice about how to optimize my finances.",
        save_artifacts=False,
    )

    assert result.stage == "UNSUPPORTED_REQUEST"
    assert result.builder_result is not None
    assert result.builder_result.builder_reason_codes == ["unsupported_intent"]
    assert result.negotiation_transition is not None
    assert result.negotiation_transition.transition_reason == "unsupported_intent"
    assert result.response_decision is not None
    assert result.response_decision.template_type == "unsupported_request"


def test_conversation_runtime_reject_system_error_uses_system_error_transition(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":49,"Family":4,"CCAvg":1.6,"Education":1,"Mortgage":0,'
                '"SecuritiesAccount":1,"CDAccount":0,"Online":0,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_REJECT",
            "prediction": {"predicted_label": 0, "predicted_proba": 0.12},
            "counterfactual": None,
            "reason_codes": ["UFCE_EXECUTION_ERROR"],
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input=(
            "Income 49, Family 4, CCAvg 1.6, Education 1, Mortgage 0, "
            "SecuritiesAccount 1, CDAccount 0, Online 0, CreditCard 0."
        ),
        save_artifacts=False,
    )

    assert result.stage == "RUNTIME_REJECT"
    assert result.explanation_payload is not None
    assert result.explanation_payload.summary_type == "runtime_reject"
    assert result.explanation_payload.included_suggestion_types == []
    assert result.negotiation_transition is not None
    assert result.negotiation_transition.transition_reason == "runtime_reject_system_error"
    assert result.response_decision is not None
    assert result.response_decision.final_public_state == "RUNTIME_REJECT"
    assert result.response_decision.included_suggestion_types == []


def test_conversation_repair_success(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(message_text='{"task": "extract_cf_request", }'),
        repair_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"Online":1},'
                '"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","SecuritiesAccount","CreditCard"],'
                '"conflicts":[],"notes":[]}'
            )
        ),
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(user_input="repair success", save_artifacts=False)

    assert adapter.repair_calls == 1
    assert result.repair_result is not None
    assert result.stage == "NEEDS_CLARIFICATION"


def test_conversation_complete_with_implicit_boolean_defaults_downgrades_to_clarification(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":0,"CDAccount":0,"Online":0,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input="I want Income 140, Family 2, CCAvg 7.7376709303, Education 2, and Mortgage 32.",
        save_artifacts=False,
    )

    assert result.stage == "NEEDS_CLARIFICATION"
    assert result.normalized_parse["status"] == "partial"
    assert result.normalized_parse["cf_request"] == {
        "Income": 140,
        "Family": 2,
        "CCAvg": 7.7376709303,
        "Education": 2,
        "Mortgage": 32,
    }
    assert result.normalized_parse["missing_fields"] == [
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]


def test_conversation_repair_complete_with_implicit_boolean_defaults_downgrades_to_clarification(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(message_text='{"task": "extract_cf_request", }'),
        repair_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":0,"CDAccount":0,"Online":0,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            ),
            task_type="repair",
        ),
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input="I want Income 140, Family 2, CCAvg 7.7376709303, Education 2, and Mortgage 32.",
        save_artifacts=False,
    )

    assert adapter.repair_calls == 1
    assert result.stage == "NEEDS_CLARIFICATION"
    assert result.normalized_parse["status"] == "partial"
    assert result.normalized_parse["missing_fields"] == [
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]


def test_conversation_dense_profile_recovery_restores_missing_fields_ref001(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"partial","cf_request":'
                '{"Income":68,"Family":1,"Education":2,"Mortgage":0,"SecuritiesAccount":0,"CreditCard":0},'
                '"missing_fields":["CCAvg","Online","CDAccount"],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.91},
            "counterfactual": None,
            "reason_codes": ["NO_RECOURSE_NEEDED"],
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input=(
            "Income 68, Family 1, CCAvg 1.5, Education 2, Mortgage 0, "
            "SecuritiesAccount no, CDAccount no, Online no, CreditCard no."
        ),
        save_artifacts=False,
    )

    assert result.stage == "RUNTIME_SUCCESS"
    assert result.builder_result is not None
    assert result.builder_result.builder_status == "READY_FOR_RUNTIME"
    assert result.normalized_parse["cf_request"]["CCAvg"] == 1.5
    assert result.normalized_parse["cf_request"]["CDAccount"] == 0
    assert result.normalized_parse["cf_request"]["Online"] == 0
    assert result.field_provenance["CCAvg"] == "deterministic_extractor"
    assert result.field_provenance["CDAccount"] == "deterministic_extractor"
    assert result.field_provenance["Online"] == "deterministic_extractor"


def test_conversation_dense_profile_recovery_restores_missing_fields_ref002(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":84,"Family":1,"Education":3,"Mortgage":0,"SecuritiesAccount":1,"CreditCard":0},'
                '"missing_fields":["CCAvg","Online","CDAccount"],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.93},
            "counterfactual": None,
            "reason_codes": ["NO_RECOURSE_NEEDED"],
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input=(
            "Income 84, Family 1, CCAvg 1.3, Education 3, Mortgage 0, "
            "SecuritiesAccount yes, CDAccount no, Online yes, CreditCard no."
        ),
        save_artifacts=False,
    )

    assert result.stage == "RUNTIME_SUCCESS"
    assert result.builder_result is not None
    assert result.builder_result.builder_status == "READY_FOR_RUNTIME"
    assert result.normalized_parse["cf_request"]["CCAvg"] == 1.3
    assert result.normalized_parse["cf_request"]["CDAccount"] == 0
    assert result.normalized_parse["cf_request"]["Online"] == 1
    assert result.field_provenance["SecuritiesAccount"] == "parser_and_extractor_agree"
    assert result.field_provenance["CreditCard"] == "parser_and_extractor_agree"


def test_conversation_runtime_path_recovers_constraint_spec_and_emits_parser_quality(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.95},
            "counterfactual": None,
            "reason_codes": ["NO_RECOURSE_NEEDED"],
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input=(
            "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
            "SecuritiesAccount yes, CDAccount yes, Online yes, CreditCard no. Do not change Income."
        ),
        save_artifacts=False,
    )

    assert result.stage == "RUNTIME_SUCCESS"
    assert result.normalized_parse["constraint_spec"] == {"disallowed_changes": ["Income"]}
    assert result.builder_result is not None
    assert result.builder_result.runtime_request["constraint_spec"] == {"disallowed_changes": ["Income"]}
    assert result.parser_quality_metadata["flags"]["canonical_pass_after_quality"] is True
    assert "constraint_spec_recovered" in result.parser_quality_metadata["reason_codes"]
    assert result.builder_result.provenance["parser_quality"] == result.parser_quality_metadata


def test_conversation_bank_session_keeps_complete_profile_with_graduate_wording(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":65000,"Family":2,"CCAvg":1.5,"Education":2,"Mortgage":0,'
                '"SecuritiesAccount":0,"CDAccount":0,"Online":1,"CreditCard":1},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    runtime_orchestrator = StubRuntimeOrchestrator(
        {
            "dataset": "bank",
            "controller_state": "TERMINAL_SUCCESS",
            "prediction": {"predicted_label": 1, "predicted_proba": 0.91},
            "counterfactual": None,
            "reason_codes": ["NO_RECOURSE_NEEDED"],
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input=(
            "I have an annual income of $65,000, a family of 2, and spend about $1.5k monthly on my credit cards. "
            "My education is level 2 (graduate). I have no mortgage, I do not have a securities account or a CD account, "
            "but I do use online banking and own a bank credit card."
        ),
        save_artifacts=False,
    )

    assert result.stage == "RUNTIME_SUCCESS"
    assert result.builder_result is not None
    assert result.builder_result.builder_status == "READY_FOR_RUNTIME"
    assert result.normalized_parse["status"] == "complete"
    assert result.normalized_parse["missing_fields"] == []
    assert result.normalized_parse["cf_request"]["CDAccount"] == 0
    assert result.normalized_parse["cf_request"]["CreditCard"] == 1
    assert result.response_decision is not None
    assert result.response_decision.template_type != "unsupported_request"
    assert result.clarification_payload is None


def test_conversation_explicit_dataset_switch_requires_new_session(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"complete","cf_request":'
                '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                '"missing_fields":[],"conflicts":[],"notes":[]}'
            )
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(
        user_input="Switch this case to the graduate admission dataset and use GRE 320 instead.",
        save_artifacts=False,
    )

    assert result.stage == "UNSUPPORTED_REQUEST"
    assert result.builder_result is not None
    assert result.builder_result.provenance["unsupported_intent_type"] == "dataset_switch"
    assert "start a new session" in result.response_text.lower()
    assert "graduate admission profile" in result.response_text.lower()


def test_conversation_repair_failure(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(message_text='{"task": "extract_cf_request", }'),
        repair_result=StubResult(message_text='{"task": "extract_cf_request", }'),
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(user_input="repair failure", save_artifacts=False)

    assert adapter.repair_calls == 1
    assert result.stage == "PARSER_FAILURE"
    assert "after one repair attempt" in result.response_text


def test_conversation_timeout_without_body_skips_repair_and_reports_timeout(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text="",
            api_error="ReadTimeout: HTTPConnectionPool(host='localhost', port=1234): Read timed out.",
            http_status_code=None,
            failure_cause="timeout_no_body",
        )
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    result = orchestrator.run_turn(user_input="parser timeout", save_artifacts=False)

    assert adapter.repair_calls == 0
    assert result.stage == "PARSER_FAILURE"
    assert result.parser_failure_cause == "timeout_no_body"
    assert "after one repair attempt" not in result.response_text
    assert "timed out before returning any parser response body" in result.response_text
