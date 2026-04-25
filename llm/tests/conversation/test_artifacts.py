from __future__ import annotations

import json

from llm.src.conversation.orchestrator import BankConversationOrchestrator
from llm.tests.conversation.test_conversation_orchestrator import (
    StubParserAdapter,
    StubResult,
    StubRuntimeOrchestrator,
)


def _load_output_bundle(tmp_path):
    output_dir = tmp_path / next(tmp_path.iterdir()).name
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    turn_result = json.loads((output_dir / "turn_result.json").read_text(encoding="utf-8"))
    runtime_result = json.loads((output_dir / "runtime_result.json").read_text(encoding="utf-8"))
    clarification_payload = json.loads((output_dir / "clarification_payload.json").read_text(encoding="utf-8"))
    explanation_payload = json.loads((output_dir / "explanation_payload.json").read_text(encoding="utf-8"))
    return output_dir, manifest, turn_result, runtime_result, clarification_payload, explanation_payload


def test_artifact_writer_emits_manifest_and_turn_files(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"Online":1},'
                '"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","SecuritiesAccount","CreditCard"],'
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
        user_input="artifact test",
        save_artifacts=True,
        scenario_slug="artifact_test",
        command="python -m test artifact",
    )

    assert result.artifact_record is not None
    output_dir = tmp_path / next(tmp_path.iterdir()).name
    manifest_path = output_dir / "artifact_manifest.json"
    turn_result_path = output_dir / "turn_result.json"
    response_text_path = output_dir / "response_text.txt"
    config_snapshot_path = output_dir / "config_snapshot.json"

    assert manifest_path.exists()
    assert turn_result_path.exists()
    assert response_text_path.exists()
    assert config_snapshot_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_snapshot = json.loads(config_snapshot_path.read_text(encoding="utf-8"))
    runtime_result = json.loads((output_dir / "runtime_result.json").read_text(encoding="utf-8"))
    clarification_payload = json.loads((output_dir / "clarification_payload.json").read_text(encoding="utf-8"))
    explanation_payload = json.loads((output_dir / "explanation_payload.json").read_text(encoding="utf-8"))
    assert manifest["turn_id"] == result.turn_id
    assert manifest["stage"] == "NEEDS_CLARIFICATION"
    assert "turn_result.json" in manifest["saved_files"]
    assert "response_text.txt" in manifest["saved_files"]
    assert "builder_result.json" in manifest["saved_files"]
    assert "negotiation_transition.json" in manifest["saved_files"]
    assert config_snapshot["parser_request_profiles"]["parse"]["max_tokens"] == 512
    assert config_snapshot["parser_request_profiles"]["repair"]["max_tokens"] == 768
    assert config_snapshot["llm_task_token_policy"]["future_explanation"]["max"] == 2048

    builder_result = json.loads((output_dir / "builder_result.json").read_text(encoding="utf-8"))
    normalized_parse = json.loads((output_dir / "normalized_parse.json").read_text(encoding="utf-8"))
    negotiation_transition = json.loads((output_dir / "negotiation_transition.json").read_text(encoding="utf-8"))
    assert builder_result["builder_status"] == "NEEDS_CLARIFICATION"
    assert builder_result["runtime_request"] is None
    assert builder_result["provenance"]["field_provenance"] == {
        "Income": "parser",
        "Online": "parser",
    }
    assert builder_result["provenance"]["parser_quality"] == {
        "reason_codes": ["constraint_spec_absent"],
        "flags": {
            "deterministic_recovery_applied": False,
            "post_quality_schema_valid": True,
            "canonical_pass_after_quality": False,
            "repair_invoked": False,
            "still_failed_after_quality": True,
            "constraint_extraction_absent": True,
        },
        "semantic_buckets": {
            "profile_facts": {"Income": 40, "Online": 1},
            "hard_constraints": {},
            "soft_preferences": {},
        },
    }
    assert normalized_parse["_field_provenance"] == {
        "Income": "parser",
        "Online": "parser",
    }
    assert normalized_parse["_parser_quality"] == builder_result["provenance"]["parser_quality"]
    assert negotiation_transition["transition_reason"] == "missing_required_fields"
    turn_result = json.loads(turn_result_path.read_text(encoding="utf-8"))
    assert turn_result["builder_result"] is not None
    assert turn_result["normalized_parse"]["_field_provenance"] == {
        "Income": "parser",
        "Online": "parser",
    }
    assert turn_result["normalized_parse"]["_parser_quality"] == builder_result["provenance"]["parser_quality"]
    assert turn_result["negotiation_transition"] is not None
    assert turn_result["response_decision"] is not None
    assert runtime_result is None
    assert clarification_payload["clarification_type"] == "missing_information"
    assert explanation_payload is None


def test_artifact_writer_persists_session_trace_fields(sample_benchmark, tmp_path):
    adapter = StubParserAdapter(
        parse_result=StubResult(
            message_text=(
                '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":40,"Online":1},'
                '"missing_fields":["CCAvg","Family","Education","Mortgage","CDAccount","SecuritiesAccount","CreditCard"],'
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
        user_input="session trace artifact test",
        save_artifacts=True,
        scenario_slug="artifact_trace_test",
        command="python -m test artifact-trace",
        session_trace={
            "session_id": "session_demo",
            "turn_index": 2,
            "parent_turn_id": "run_parent",
            "merge_applied": True,
            "carried_fields": ["Income", "Online"],
        },
    )

    assert result.artifact_record is not None
    output_dir = tmp_path / next(tmp_path.iterdir()).name
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    turn_result = json.loads((output_dir / "turn_result.json").read_text(encoding="utf-8"))

    assert manifest["session_id"] == "session_demo"
    assert manifest["turn_index"] == 2
    assert manifest["parent_turn_id"] == "run_parent"
    assert manifest["merge_applied"] is True
    assert manifest["carried_fields"] == ["Income", "Online"]
    assert turn_result["artifact_record"]["session_id"] == "session_demo"
    assert turn_result["artifact_record"]["merge_applied"] is True
    assert "response_decision" in turn_result


def test_artifact_writer_runtime_success_bundle_has_runtime_and_explanation(sample_benchmark, tmp_path):
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

    orchestrator.run_turn(
        user_input=(
            "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
            "SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0."
        ),
        save_artifacts=True,
        scenario_slug="artifact_runtime_success",
        command="python -m test artifact-runtime-success",
    )

    output_dir, manifest, turn_result, runtime_result, clarification_payload, explanation_payload = _load_output_bundle(tmp_path)
    assert manifest["stage"] == "RUNTIME_SUCCESS"
    assert (output_dir / "builder_result.json").exists()
    assert (output_dir / "negotiation_transition.json").exists()
    assert (output_dir / "turn_result.json").exists()
    assert (output_dir / "response_text.txt").exists()
    assert (output_dir / "artifact_manifest.json").exists()
    assert turn_result["builder_result"] is not None
    assert turn_result["negotiation_transition"] is not None
    assert turn_result["response_decision"] is not None
    assert turn_result["response_decision"]["final_public_state"] == "RUNTIME_SUCCESS"
    assert turn_result["stage"] == "RUNTIME_SUCCESS"
    assert "READY_FOR_RUNTIME" in turn_result["stage_trace"]
    assert runtime_result["controller_state"] == "TERMINAL_SUCCESS"
    assert clarification_payload is None
    assert explanation_payload["summary_type"] == "no_recourse_needed"


def test_artifact_writer_emits_parallel_canonical_runtime_files(sample_benchmark, tmp_path):
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
            "counterfactual": {
                "feasible": True,
                "candidates": [
                    {
                        "method": "sfexp",
                        "rank": 1,
                        "profile": {
                            "Income": 100.0,
                            "Family": 1,
                            "CCAvg": 2.7,
                            "Education": 2,
                            "Mortgage": 0.0,
                            "SecuritiesAccount": 0,
                            "CDAccount": 1,
                            "Online": 0,
                            "CreditCard": 0,
                        },
                        "changed_features": ["CDAccount"],
                    }
                ],
                "reason_codes": [],
            },
            "reason_codes": [],
            "canonical_request": {"dataset_id": "bank"},
            "canonical_candidates": [{"candidate_id": "ufce:sfexp:1"}],
            "verification_results": [{"candidate_id": "ufce:sfexp:1", "is_valid": True, "reason_codes": []}],
            "backend_manifest": {"backend_id": "ufce"},
            "backend_id": "ufce",
            "reason_code_version": "reason_codes_v1",
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    orchestrator.run_turn(
        user_input=(
            "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
            "SecuritiesAccount 1, CDAccount 1, Online 1, CreditCard 0."
        ),
        save_artifacts=True,
        scenario_slug="artifact_parallel_canonical",
        command="python -m test artifact-parallel-canonical",
    )

    output_dir = tmp_path / next(tmp_path.iterdir()).name

    assert (output_dir / "canonical_request.json").exists()
    assert (output_dir / "canonical_candidates.json").exists()
    assert (output_dir / "verification_results.json").exists()
    assert (output_dir / "backend_contract.json").exists()


def test_artifact_writer_conflict_bundle_has_null_runtime_and_conflict_payload(sample_benchmark, tmp_path):
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

    orchestrator.run_turn(
        user_input="artifact conflict",
        save_artifacts=True,
        scenario_slug="artifact_conflict",
        command="python -m test artifact-conflict",
    )

    output_dir, manifest, turn_result, runtime_result, clarification_payload, explanation_payload = _load_output_bundle(tmp_path)
    assert manifest["stage"] == "CONFLICT"
    assert (output_dir / "builder_result.json").exists()
    assert (output_dir / "negotiation_transition.json").exists()
    assert turn_result["builder_result"] is not None
    assert turn_result["negotiation_transition"] is not None
    assert turn_result["response_decision"] is not None
    assert runtime_result is None
    assert clarification_payload["clarification_type"] == "conflict_resolution"
    assert explanation_payload is None


def test_artifact_writer_unsupported_bundle_has_null_runtime_and_unsupported_response(sample_benchmark, tmp_path):
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

    orchestrator.run_turn(
        user_input="Give me general financial advice about how to optimize my finances.",
        save_artifacts=True,
        scenario_slug="artifact_unsupported",
        command="python -m test artifact-unsupported",
    )

    output_dir, manifest, turn_result, runtime_result, clarification_payload, explanation_payload = _load_output_bundle(tmp_path)
    assert manifest["stage"] == "UNSUPPORTED_REQUEST"
    assert (output_dir / "builder_result.json").exists()
    assert (output_dir / "negotiation_transition.json").exists()
    assert turn_result["builder_result"] is not None
    assert turn_result["negotiation_transition"] is not None
    assert turn_result["response_decision"]["template_type"] == "unsupported_request"
    assert runtime_result is None
    assert clarification_payload is None
    assert explanation_payload is None


def test_artifact_writer_runtime_reject_bundle_has_runtime_and_reject_explanation(sample_benchmark, tmp_path):
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
            "counterfactual": {"feasible": False, "candidates": []},
            "reason_codes": ["NO_FEASIBLE_CF_FOUND"],
        }
    )
    orchestrator = BankConversationOrchestrator(
        parser_adapter=adapter,
        runtime_orchestrator=runtime_orchestrator,
        benchmark=sample_benchmark,
        output_root=tmp_path,
    )

    orchestrator.run_turn(
        user_input=(
            "Income 49, Family 4, CCAvg 1.6, Education 1, Mortgage 0, "
            "SecuritiesAccount 1, CDAccount 0, Online 0, CreditCard 0."
        ),
        save_artifacts=True,
        scenario_slug="artifact_runtime_reject",
        command="python -m test artifact-runtime-reject",
    )

    output_dir, manifest, turn_result, runtime_result, clarification_payload, explanation_payload = _load_output_bundle(tmp_path)
    assert manifest["stage"] == "RUNTIME_REJECT"
    assert (output_dir / "builder_result.json").exists()
    assert (output_dir / "negotiation_transition.json").exists()
    assert turn_result["builder_result"] is not None
    assert turn_result["negotiation_transition"] is not None
    assert turn_result["response_decision"]["final_public_state"] == "RUNTIME_REJECT"
    assert runtime_result["controller_state"] == "TERMINAL_REJECT"
    assert clarification_payload is None
    assert explanation_payload["summary_type"] == "runtime_reject"
