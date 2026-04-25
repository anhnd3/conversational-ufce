from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from llm.src.conversation.orchestrator import BankConversationOrchestrator
from llm.src.product.config import ProductConfig
from llm.src.product.persistence import SessionRepository
from llm.src.product.service import ProductSessionService, build_debug_summary
from llm.tests.conversation.test_conversation_orchestrator import StubRuntimeOrchestrator
from llm.tests.conversation.test_session import QueueParserAdapter, StubResult


def build_config(tmp_path: Path) -> ProductConfig:
    return ProductConfig(
        lm_studio_api_base="http://localhost:1234",
        model_alias="stub-model",
        product_mode="stable_demo",
        artifact_root=(tmp_path / "artifacts").resolve(),
        sqlite_path=(tmp_path / "sessions.sqlite3").resolve(),
        api_version="v1",
        app_version="phase3_2_test",
        parser_schema_version="parser_schema_v1",
        bank_policy_version="bank_policy_v1",
        host="127.0.0.1",
        port=8000,
    )


def test_build_debug_summary_exposes_carried_constraint_keys():
    payload = build_debug_summary(
        SimpleNamespace(
            runtime_result=None,
            invariant_validation=None,
            builder_result=SimpleNamespace(
                builder_status="NEEDS_CLARIFICATION",
                builder_reason_codes=["missing_required_fields"],
                carried_fields=["Income"],
                carried_constraint_keys=["disallowed_changes", "numeric_bounds.Mortgage"],
            ),
            negotiation_transition=SimpleNamespace(transition_reason="missing_required_fields"),
            artifact_record=SimpleNamespace(output_dir="/tmp/demo", merge_applied=True),
            timing_metrics={"end_to_end_latency_ms": 12.5},
        )
    )

    assert payload["merge_applied"] is True
    assert payload["carried_fields"] == ["Income"]
    assert payload["carried_constraint_keys"] == ["disallowed_changes", "numeric_bounds.Mortgage"]
    assert payload["carried_preference_keys"] == []
    assert payload["followup_classification"] is None
    assert payload["runtime_summary"]["executed"] is False


def test_service_create_session_persists_explicit_dataset_key(sample_benchmark, tmp_path):
    orchestrator = BankConversationOrchestrator(
        parser_adapter=QueueParserAdapter(parse_results=[]),
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    repository = SessionRepository((tmp_path / "sessions.sqlite3").resolve(), app_version="phase3_2_test")
    service = ProductSessionService(orchestrator=orchestrator, repository=repository, config=build_config(tmp_path))

    session = service.create_session(dataset_key="grad")
    persisted = repository.get_session(session.session_id)

    assert session.dataset_key == "grad"
    assert persisted.dataset_key == "grad"


def test_service_create_session_rejects_unsupported_dataset(sample_benchmark, tmp_path):
    orchestrator = BankConversationOrchestrator(
        parser_adapter=QueueParserAdapter(parse_results=[]),
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    repository = SessionRepository((tmp_path / "sessions.sqlite3").resolve(), app_version="phase3_2_test")
    service = ProductSessionService(orchestrator=orchestrator, repository=repository, config=build_config(tmp_path))

    try:
        service.create_session(dataset_key="movie")
    except ValueError as exc:
        assert "Unsupported session dataset" in str(exc)
    else:
        raise AssertionError("Expected create_session to reject unsupported dataset keys")


def test_service_persists_canonical_state_across_clarification_followup(sample_benchmark, tmp_path):
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
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=StubRuntimeOrchestrator(
            {
                "dataset": "bank",
                "backend_id": "ufce",
                "controller_state": "TERMINAL_SUCCESS",
                "prediction": {"predicted_label": 0, "predicted_proba": 0.85},
                "counterfactual": None,
                "reason_codes": ["NO_RECOURSE_NEEDED"],
                "runtime_mode": "stable_demo",
                "invariant_validation": {
                    "status": "skipped_no_counterfactual",
                    "public_safe": True,
                    "reason_codes": [],
                    "validated_summary_type": "no_recourse_needed",
                    "validated_changed_fields": [],
                    "details": {"reason": "no_counterfactual_required"},
                },
            }
        ),
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    repository = SessionRepository((tmp_path / "sessions.sqlite3").resolve(), app_version="phase3_2_test")
    service = ProductSessionService(orchestrator=orchestrator, repository=repository, config=build_config(tmp_path))
    session = service.create_session()

    service.submit_message(session.session_id, "first turn")
    persisted_after_first = repository.get_session(session.session_id)

    assert persisted_after_first.pending_clarification_json is not None
    assert persisted_after_first.canonical_session_state_json is not None
    assert persisted_after_first.canonical_session_state_json["profile_facts"] == {
        "Income": 140,
        "Family": 2,
        "CCAvg": 7.7376709303,
        "Education": 2,
        "Mortgage": 32,
    }

    second_turn = service.submit_message(
        session.session_id,
        "CD account yes, online yes, securities account yes, and credit card no.",
    )
    persisted_after_second = repository.get_session(session.session_id)

    assert second_turn.debug_summary_json is not None
    assert second_turn.debug_summary_json["merge_applied"] is True
    assert second_turn.debug_summary_json["followup_classification"] == "profile_completion"
    assert second_turn.debug_summary_json["carried_fields"] == [
        "Income",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
    ]
    assert persisted_after_second.pending_clarification_json is None
    assert persisted_after_second.canonical_session_state_json is not None
    assert persisted_after_second.canonical_session_state_json["profile_facts"] == {
        "Income": 140,
        "Family": 2,
        "CCAvg": 7.7376709303,
        "Education": 2,
        "Mortgage": 32,
        "SecuritiesAccount": 1,
        "CDAccount": 1,
        "Online": 1,
        "CreditCard": 0,
    }


def test_service_restart_clears_persisted_canonical_state(sample_benchmark, tmp_path):
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
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=StubRuntimeOrchestrator(
            {
                "dataset": "bank",
                "backend_id": "ufce",
                "controller_state": "TERMINAL_SUCCESS",
                "prediction": {"predicted_label": 0, "predicted_proba": 0.85},
                "counterfactual": None,
                "reason_codes": ["NO_RECOURSE_NEEDED"],
                "runtime_mode": "stable_demo",
                "invariant_validation": {
                    "status": "skipped_no_counterfactual",
                    "public_safe": True,
                    "reason_codes": [],
                    "validated_summary_type": "no_recourse_needed",
                    "validated_changed_fields": [],
                    "details": {"reason": "no_counterfactual_required"},
                },
            }
        ),
        benchmark=sample_benchmark,
        output_root=tmp_path / "artifacts",
        model_alias="stub-model",
    )
    repository = SessionRepository((tmp_path / "sessions.sqlite3").resolve(), app_version="phase3_2_test")
    service = ProductSessionService(orchestrator=orchestrator, repository=repository, config=build_config(tmp_path))
    session = service.create_session()

    service.submit_message(session.session_id, "first turn")
    restart_turn = service.submit_message(session.session_id, "Start over.")
    persisted = repository.get_session(session.session_id)

    assert restart_turn.debug_summary_json is not None
    assert restart_turn.debug_summary_json["followup_classification"] == "fresh_request"
    assert restart_turn.debug_summary_json["reset_decision"] == "fresh_request"
    assert persisted.pending_clarification_json is None
    assert persisted.active_constraint_spec_json == {}
    assert persisted.last_runtime_request_json is None
    assert persisted.canonical_session_state_json is not None
    assert persisted.canonical_session_state_json["profile_facts"] == {}
    assert persisted.canonical_session_state_json["hard_constraints"] == {}
    assert persisted.canonical_session_state_json["soft_preferences"] == {}
