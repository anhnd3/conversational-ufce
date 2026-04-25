from __future__ import annotations

from dataclasses import dataclass
import json

from llm.src.conversation.orchestrator import BankConversationOrchestrator
from llm.src.conversation.session import create_interactive_session_state, handle_session_turn


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


class QueueParserAdapter:
    def __init__(
        self,
        parse_results: list[StubResult],
        repair_results: list[StubResult] | None = None,
        refinement_results: list[StubResult] | None = None,
        refinement_repair_results: list[StubResult] | None = None,
    ) -> None:
        self.parse_results = list(parse_results)
        self.repair_results = list(repair_results or [])
        self.refinement_results = list(refinement_results or [])
        self.refinement_repair_results = list(refinement_repair_results or [])
        self.api_base = "stub://adapter"
        self.system_prompt = "stub prompt"
        self.structured_output_mode = "json_schema_strict"

    def parse(self, *, user_text: str, benchmark=None):
        del user_text
        del benchmark
        if not self.parse_results:
            raise AssertionError("unexpected parse call")
        return self.parse_results.pop(0)

    def repair(self, *, invalid_output: str, errors: list[str], benchmark=None):
        del invalid_output
        del errors
        del benchmark
        if not self.repair_results:
            raise AssertionError("unexpected repair call")
        return self.repair_results.pop(0)

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
        if not self.refinement_results:
            raise AssertionError("unexpected refinement parse call")
        return self.refinement_results.pop(0)

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
        if not self.refinement_repair_results:
            raise AssertionError("unexpected refinement repair call")
        return self.refinement_repair_results.pop(0)

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


def test_session_missing_information_followup_merges_to_runtime(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32},'
                    '"missing_fields":["SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                    '"missing_fields":["Income","Family","CCAvg","Education","Mortgage"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    first = handle_session_turn(
        orchestrator,
        state,
        user_input="first turn",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second = handle_session_turn(
        orchestrator,
        state,
        user_input="followup",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert first.stage == "NEEDS_CLARIFICATION"
    assert second.stage == "RUNTIME_SUCCESS"
    assert second.explanation_payload is not None
    assert second.normalized_parse["cf_request"]["Income"] == 140
    assert second.normalized_parse["cf_request"]["Online"] == 1
    assert state.pending_clarification is None


def test_session_subthreshold_initial_ccavg_recovery_unblocks_clarification_followup(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"complete","cf_request":'
                    '{"Income":140,"Family":2,"Education":2,"Mortgage":32,'
                    '"SecuritiesAccount":0,"CDAccount":0,"Online":0,"CreditCard":0},'
                    '"missing_fields":["CCAvg"],"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                    '"missing_fields":["Income","Family","CCAvg","Education","Mortgage"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    first = handle_session_turn(
        orchestrator,
        state,
        user_input="Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32.",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert first.stage == "NEEDS_CLARIFICATION"
    assert first.normalized_parse["cf_request"] == {
        "Income": 140,
        "Family": 2,
        "CCAvg": 7.7376709303,
        "Education": 2,
        "Mortgage": 32,
    }
    assert first.normalized_parse["missing_fields"] == [
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]
    assert state.canonical_session_state is not None
    assert state.canonical_session_state["profile_facts"]["CCAvg"] == 7.7376709303
    assert state.pending_clarification is not None
    assert state.pending_clarification.prior_cf_request["CCAvg"] == 7.7376709303

    second = handle_session_turn(
        orchestrator,
        state,
        user_input="SecuritiesAccount yes, CDAccount yes, Online yes, CreditCard no.",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert second.stage == "RUNTIME_SUCCESS"
    assert second.canonical_validation["missing_runtime_fields"] == []
    assert second.normalized_parse["cf_request"]["CCAvg"] == 7.7376709303
    assert state.canonical_session_state is not None
    assert state.canonical_session_state["profile_facts"]["CCAvg"] == 7.7376709303
    assert state.pending_clarification is None


def test_session_followup_merge_can_still_need_clarification(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80},'
                    '"missing_fields":["CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"CDAccount":1,"Online":1},'
                    '"missing_fields":["Income","CCAvg","Family","Education","Mortgage","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    handle_session_turn(
        orchestrator,
        state,
        user_input="first turn",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second = handle_session_turn(
        orchestrator,
        state,
        user_input="followup",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert second.stage == "NEEDS_CLARIFICATION"
    assert second.normalized_parse["missing_fields"] == ["SecuritiesAccount", "CreditCard"]
    assert state.pending_clarification is not None
    assert state.pending_clarification.originating_turn_id == second.turn_id


def test_session_empty_short_boolean_followup_merges_from_explicit_text(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32},'
                    '"missing_fields":["SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"needs_clarification","cf_request":{},'
                    '"missing_fields":["Income","Family","CCAvg","Education","Mortgage","SecuritiesAccount","CDAccount","Online","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    first = handle_session_turn(
        orchestrator,
        state,
        user_input="Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32.",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second = handle_session_turn(
        orchestrator,
        state,
        user_input="CD account no, online no, securities account no, and credit card no.",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert first.stage == "NEEDS_CLARIFICATION"
    assert second.stage == "RUNTIME_SUCCESS"
    assert second.normalized_parse["cf_request"] == {
        "Income": 140,
        "Family": 2,
        "CCAvg": 7.7376709303,
        "Education": 2,
        "Mortgage": 32,
        "SecuritiesAccount": 0,
        "CDAccount": 0,
        "Online": 0,
        "CreditCard": 0,
    }
    assert second.artifact_record is None
    assert state.pending_clarification is None


def test_session_conflict_followup_does_not_merge(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40},'
                    '"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"conflict","cf_request":{"Income":60},'
                    '"missing_fields":[],"conflicts":["Income cannot be both 40 and 60."],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    handle_session_turn(
        orchestrator,
        state,
        user_input="first turn",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second = handle_session_turn(
        orchestrator,
        state,
        user_input="followup",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert second.stage == "CONFLICT"
    assert second.normalized_parse["cf_request"] == {"Income": 40}
    assert second.builder_result is not None
    assert second.builder_result.merge_applied is False
    assert second.builder_result.provenance["followup_classification"] == "ambiguous_followup"
    assert state.pending_clarification is None
    assert state.canonical_session_state is not None
    assert state.canonical_session_state["profile_facts"] == {"Income": 40}


def test_session_parser_failure_followup_does_not_merge(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40},'
                    '"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(message_text='{"task": "extract_cf_request", }'),
        ],
        repair_results=[StubResult(message_text='{"task": "extract_cf_request", }', task_type="repair")],
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    handle_session_turn(
        orchestrator,
        state,
        user_input="first turn",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second = handle_session_turn(
        orchestrator,
        state,
        user_input="followup",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert second.stage == "PARSER_FAILURE"
    assert state.pending_clarification is None


def test_session_unrelated_next_message_resets_pending(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40},'
                    '"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"needs_clarification","cf_request":{},'
                    '"missing_fields":["Income","CCAvg","Family","Education","Mortgage","CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    first = handle_session_turn(
        orchestrator,
        state,
        user_input="first turn",
        save_artifacts=True,
        scenario_slug="unrelated_reset_turn1",
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second = handle_session_turn(
        orchestrator,
        state,
        user_input="unrelated",
        save_artifacts=True,
        scenario_slug="unrelated_reset_turn2",
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert first.stage == "NEEDS_CLARIFICATION"
    assert second.stage == "NEEDS_CLARIFICATION"
    assert second.normalized_parse["cf_request"] == {}
    assert second.builder_result is not None
    assert second.builder_result.partial_profile_snapshot == {}
    assert second.builder_result.provenance["pending_reset"] is True
    assert second.negotiation_transition is not None
    assert second.negotiation_transition.transition_reason == "followup_reset_new_request"
    assert second.artifact_record is not None
    assert second.artifact_record.merge_applied is False
    assert second.artifact_record.parent_turn_id is None
    assert second.artifact_record.carried_fields == []
    assert "Income" not in second.normalized_parse["cf_request"]
    assert "Income" not in second.builder_result.partial_profile_snapshot
    assert state.pending_clarification is None


def test_session_followup_can_explicitly_override_prior_field(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80},'
                    '"missing_fields":["CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"Income":60,"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":1},'
                    '"missing_fields":["CCAvg","Family","Education","Mortgage"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    handle_session_turn(
        orchestrator,
        state,
        user_input="first turn",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second = handle_session_turn(
        orchestrator,
        state,
        user_input="Actually, change Income to 60. CD account yes, online yes, securities account yes, and credit card yes.",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert second.normalized_parse["cf_request"]["Income"] == 60
    assert second.builder_result is not None
    assert second.builder_result.provenance["followup_classification"] == "correction"


def test_session_followup_merge_persists_merge_metadata(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"Income":140,"CCAvg":7.7376709303,"Family":2,"Education":2,"Mortgage":32},'
                    '"missing_fields":["CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":0},'
                    '"missing_fields":["Income","CCAvg","Family","Education","Mortgage"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    first = handle_session_turn(
        orchestrator,
        state,
        user_input="first turn",
        save_artifacts=True,
        scenario_slug="merge_turn1",
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second = handle_session_turn(
        orchestrator,
        state,
        user_input="CD account yes, online yes, securities account yes, and credit card no.",
        save_artifacts=True,
        scenario_slug="merge_turn2",
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert second.artifact_record is not None
    manifest_path = tmp_path / next(path.name for path in tmp_path.iterdir() if path.name.endswith("merge_turn2")) / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert second.stage == "RUNTIME_SUCCESS"
    assert manifest["merge_applied"] is True
    assert manifest["parent_turn_id"] == first.turn_id
    assert manifest["carried_fields"] == ["Income", "Family", "CCAvg", "Education", "Mortgage"]
    assert second.builder_result is not None
    assert second.builder_result.provenance["followup_classification"] == "profile_completion"


def test_session_reset_full_profile_remains_unmerged_in_artifacts(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80},'
                    '"missing_fields":["CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"complete","cf_request":'
                    '{"Income":72,"CCAvg":4.8,"Family":1,"Education":2,"Mortgage":200,'
                    '"CDAccount":1,"Online":0,"SecuritiesAccount":1,"CreditCard":0},'
                    '"missing_fields":[],"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    first = handle_session_turn(
        orchestrator,
        state,
        user_input="first turn",
        save_artifacts=True,
        scenario_slug="reset_turn1",
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second = handle_session_turn(
        orchestrator,
        state,
        user_input=(
            "Income 72, CCAvg 4.8, Family 1, Education 2, Mortgage 200, "
            "CDAccount 1, Online 0, SecuritiesAccount 1, CreditCard 0."
        ),
        save_artifacts=True,
        scenario_slug="reset_turn2",
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert second.artifact_record is not None
    assert first.stage == "NEEDS_CLARIFICATION"
    assert second.stage == "RUNTIME_SUCCESS"
    assert second.builder_result is not None
    assert second.builder_result.provenance["pending_reset"] is True
    expected_turn2_profile = {
        "Income": 72,
        "Family": 1,
        "CCAvg": 4.8,
        "Education": 2,
        "Mortgage": 200,
        "SecuritiesAccount": 1,
        "CDAccount": 1,
        "Online": 0,
        "CreditCard": 0,
    }
    assert second.normalized_parse["cf_request"] == expected_turn2_profile
    assert second.builder_result.partial_profile_snapshot == expected_turn2_profile
    assert second.negotiation_transition is not None
    assert second.negotiation_transition.transition_reason == "runtime_success_counterfactual_found"
    manifest_path = tmp_path / next(path.name for path in tmp_path.iterdir() if path.name.endswith("reset_turn2")) / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["merge_applied"] is False
    assert manifest["parent_turn_id"] is None
    assert manifest["carried_fields"] == []


def test_session_isolation_keeps_pending_state_per_session(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80},'
                    '"missing_fields":["CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"complete","cf_request":'
                    '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
                    '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
                    '"missing_fields":[],"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"CDAccount":1,"Online":1,"SecuritiesAccount":1,"CreditCard":0},'
                    '"missing_fields":["Income","CCAvg","Family","Education","Mortgage"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state_a = create_interactive_session_state("session_A")
    state_b = create_interactive_session_state("session_B")

    first_a = handle_session_turn(
        orchestrator,
        state_a,
        user_input="session A turn 1",
        save_artifacts=True,
        scenario_slug="session_a_turn1",
        debug_trace_enabled=False,
        command="python -m test session",
    )
    first_b = handle_session_turn(
        orchestrator,
        state_b,
        user_input=(
            "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
            "SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0."
        ),
        save_artifacts=True,
        scenario_slug="session_b_turn1",
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second_a = handle_session_turn(
        orchestrator,
        state_a,
        user_input="CD account yes, online yes, securities account yes, and credit card no.",
        save_artifacts=True,
        scenario_slug="session_a_turn2",
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert first_a.stage == "NEEDS_CLARIFICATION"
    assert first_b.stage == "RUNTIME_SUCCESS"
    assert first_b.artifact_record is not None
    assert first_b.artifact_record.session_id == "session_B"
    assert first_b.artifact_record.parent_turn_id is None
    assert first_b.artifact_record.merge_applied is False
    assert first_b.artifact_record.carried_fields == []
    assert second_a.stage == "RUNTIME_SUCCESS"
    assert second_a.artifact_record is not None
    assert second_a.artifact_record.session_id == "session_A"
    assert second_a.artifact_record.parent_turn_id == first_a.turn_id
    assert second_a.artifact_record.merge_applied is True
    assert second_a.artifact_record.carried_fields == ["Income", "Family", "CCAvg", "Education", "Mortgage"]
    assert state_a.pending_clarification is None
    assert state_b.pending_clarification is None


def test_session_explicit_restart_clears_canonical_followup_state(sample_benchmark, tmp_path):
    adapter = QueueParserAdapter(
        parse_results=[
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"partial","cf_request":'
                    '{"Income":40,"CCAvg":1.5,"Family":3,"Education":2,"Mortgage":80},'
                    '"missing_fields":["CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
            StubResult(
                message_text=(
                    '{"task":"extract_cf_request","status":"needs_clarification","cf_request":{},'
                    '"missing_fields":["Income","Family","CCAvg","Education","Mortgage","CDAccount","Online","SecuritiesAccount","CreditCard"],'
                    '"conflicts":[],"notes":[]}'
                )
            ),
        ]
    )
    orchestrator = BankConversationOrchestrator(parser_adapter=adapter, benchmark=sample_benchmark, output_root=tmp_path)
    state = create_interactive_session_state("session_test")

    handle_session_turn(
        orchestrator,
        state,
        user_input="first turn",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )
    second = handle_session_turn(
        orchestrator,
        state,
        user_input="Start over.",
        save_artifacts=False,
        scenario_slug=None,
        debug_trace_enabled=False,
        command="python -m test session",
    )

    assert second.stage == "NEEDS_CLARIFICATION"
    assert second.builder_result is not None
    assert second.builder_result.provenance["followup_classification"] == "fresh_request"
    assert second.builder_result.provenance["reset_decision"] == "fresh_request"
    assert second.normalized_parse["cf_request"] == {}
    assert state.pending_clarification is None
    assert state.canonical_session_state is not None
    assert state.canonical_session_state["profile_facts"] == {}
    assert state.canonical_session_state["hard_constraints"] == {}
    assert state.canonical_session_state["soft_preferences"] == {}
