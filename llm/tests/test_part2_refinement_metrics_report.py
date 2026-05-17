from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.archieve.run_part2_refinement_metrics_report import (
    build_refinement_not_allowed_payload,
    build_refinement_round_result,
    build_refinement_summary,
    compare_solution_payloads,
)
from llm.src.part2_eval.corpora import build_tier_a_annotation_corpus
from llm.src.phase3.phase3_2_catalog import load_catalog


def test_compare_solution_payloads_detects_counterfactual_change():
    previous = {
        "public_state": "RUNTIME_SUCCESS",
        "explanation_payload": {
            "summary_type": "counterfactual_found",
            "counterfactual_summary": {"profile": {"Income": 80.0}},
        },
    }
    current = {
        "public_state": "RUNTIME_SUCCESS",
        "explanation_payload": {
            "summary_type": "counterfactual_found",
            "counterfactual_summary": {"profile": {"Income": 90.0}},
        },
    }

    assert compare_solution_payloads(previous, current) == 1


def test_build_refinement_round_result_handles_limit_reached_payload():
    result = build_refinement_round_result(
        case={"case_id": "TIERB-REF-001"},
        session_id="session_1",
        previous_payload={"public_state": "RUNTIME_SUCCESS", "explanation_payload": {"summary_type": "counterfactual_found"}},
        feedback_text="one more round",
        response_status_code=409,
        response_payload={
            "error_code": "refinement_limit_reached",
            "detail": "limit",
            "current_public_state": "RUNTIME_SUCCESS",
            "case_completion_reason": "runtime_success",
            "refinement_status": "limit_reached",
            "refinement_rounds_used": 3,
            "refinement_round_limit": 3,
            "active_constraint_spec": {},
            "restart_required": True,
        },
    )

    assert result["http_status_code"] == 409
    assert result["refinement_status"] == "limit_reached"
    assert result["solution_changed"] == 0


def test_build_refinement_round_result_preserves_clarification_required_contract():
    result = build_refinement_round_result(
        case={"case_id": "TIERB-REF-CLARIFY"},
        session_id="session_1",
        previous_payload={
            "public_state": "RUNTIME_SUCCESS",
            "explanation_payload": {"summary_type": "no_recourse_needed"},
        },
        feedback_text="Make the bank result better without changing too much.",
        response_status_code=200,
        response_payload={
            "current_public_state": "RUNTIME_SUCCESS",
            "refinement_status": "clarification_required",
            "clarification_payload": {
                "clarification_type": "refinement_clarification",
                "conflicts": ["The feedback asks for improvement without specifying allowed fields."],
            },
            "explanation_payload": None,
            "refinement_rounds_used": 1,
            "refinement_round_limit": 3,
            "active_constraint_spec": {},
            "restart_required": False,
            "debug_summary": {"runtime_summary": {"executed": False}, "timing_metrics": {}},
        },
    )

    assert result["refinement_status"] == "clarification_required"
    assert result["summary_type"] is None
    assert result["solution_changed"] == 1


def test_build_refinement_not_allowed_payload_preserves_parent_state():
    payload = build_refinement_not_allowed_payload(
        previous_payload={
            "public_state": "NEEDS_CLARIFICATION",
            "case_completion_reason": None,
            "refinement_rounds_used": 0,
            "refinement_round_limit": 3,
            "active_constraint_spec": {},
            "restart_required": False,
            "explanation_payload": {"summary_type": "clarification_needed"},
            "debug_summary": {"timing_metrics": {"end_to_end_latency_ms": 12.5}},
        },
        detail="Refinement is only available after a runtime-backed result.",
    )

    assert payload["error_code"] == "refinement_not_allowed"
    assert payload["refinement_status"] == "not_allowed"
    assert payload["current_public_state"] == "NEEDS_CLARIFICATION"
    assert payload["explanation_payload"]["summary_type"] == "clarification_needed"


def test_build_refinement_summary_includes_locked_metrics(tmp_path):
    baseline_catalog = load_catalog()
    tier_a_corpus = build_tier_a_annotation_corpus()
    handle = SimpleNamespace(
        execution_mode="in_process_conversational",
        service_command=None,
        base_url=None,
        sqlite_path=(tmp_path / "sessions.sqlite3"),
        artifact_root=(tmp_path / "artifacts"),
        config=SimpleNamespace(lm_studio_api_base="http://localhost:1234"),
    )
    summary = build_refinement_summary(
        run_id="demo",
        command="python run_part2_refinement_metrics_report.py",
        run_root=tmp_path,
        handle=handle,
        baseline_catalog=baseline_catalog,
        version_payload={
            "api_version": "v1",
            "app_version": "phase3_2_test",
            "model_alias": "stub-model",
            "runtime_mode": "stable_demo",
            "git_commit": "abc123",
        },
        dataset_entry={"dataset_key": "bank"},
        tier_a_corpus=tier_a_corpus,
        tier_a_corpus_path=tmp_path / "tier_a.json",
        tier_a_summary={
            "M36_constraint_delta_fidelity": {"numerator": 20, "denominator": 25, "mean": 0.8, "component_mean": 0.9},
        },
        tier_b_corpus={"corpus_version": "tier_b_full", "corpus_sha256": "tierbhash"},
        tier_b_corpus_path=tmp_path / "tier_b.json",
        refinement_corpus={"corpus_version": "ref", "corpus_sha256": "refhash", "case_count": 50},
        live_results=[
            {
                "case_id": "TIERB-REF-001",
                "initial_turn": {"debug_summary": {"timing_metrics": {"end_to_end_latency_ms": 10.0}}},
                "refinement_rounds": [
                    {
                        "refinement_status": "applied",
                        "http_status_code": 200,
                        "solution_changed": 1,
                        "reject_class": None,
                        "refinement_rounds_used": 1,
                        "refinement_latency_ms": 8.0,
                    }
                ],
            },
            {
                "case_id": "TIERB-REF-002",
                "initial_turn": {"debug_summary": {"timing_metrics": {"end_to_end_latency_ms": 12.0}}},
                "refinement_rounds": [
                    {
                        "refinement_status": "applied",
                        "http_status_code": 200,
                        "solution_changed": 0,
                        "reject_class": "request_constraints_blocked",
                        "refinement_rounds_used": 1,
                        "refinement_latency_ms": 11.0,
                    }
                ],
            },
        ],
        benchmark_path=tmp_path / "bench.yaml",
    )

    assert summary["runner_scope"] == "part2_level2_refinement_eval"
    assert summary["execution_mode"] == "in_process_conversational"
    assert summary["corpus_path"].endswith("tier_b.json")
    assert summary["frozen_inputs"]["tier_a_annotation_schema_sha256"]
    assert summary["refinement_metrics"]["M32_refinement_success_rate"]["mean"] == 1.0
    assert summary["refinement_metrics"]["M34_constraint_induced_blocking_rate"]["mean"] == 0.5
    assert summary["refinement_metrics"]["refinement_round_latency_ms"]["mean"] == 9.5
    assert summary["aggregate_validation"]["ok"] is True
    assert Path(summary["report_json_path"]).name == "refinement_metrics_report.json"


def test_build_refinement_summary_marks_script_mismatch_as_failed_validation(tmp_path):
    baseline_catalog = load_catalog()
    tier_a_corpus = build_tier_a_annotation_corpus()
    handle = SimpleNamespace(
        execution_mode="in_process_conversational",
        service_command=None,
        base_url=None,
        sqlite_path=(tmp_path / "sessions.sqlite3"),
        artifact_root=(tmp_path / "artifacts"),
        config=SimpleNamespace(lm_studio_api_base="http://localhost:1234"),
    )
    summary = build_refinement_summary(
        run_id="demo",
        command="python run_part2_refinement_metrics_report.py",
        run_root=tmp_path,
        handle=handle,
        baseline_catalog=baseline_catalog,
        version_payload={
            "api_version": "v1",
            "app_version": "phase3_2_test",
            "model_alias": "stub-model",
            "runtime_mode": "stable_demo",
            "git_commit": "abc123",
        },
        dataset_entry={"dataset_key": "bank"},
        tier_a_corpus=tier_a_corpus,
        tier_a_corpus_path=tmp_path / "tier_a.json",
        tier_a_summary={
            "M36_constraint_delta_fidelity": {"numerator": 20, "denominator": 25, "mean": 0.8, "component_mean": 0.9},
        },
        tier_b_corpus={"corpus_version": "tier_b_full", "corpus_sha256": "tierbhash"},
        tier_b_corpus_path=tmp_path / "tier_b.json",
        refinement_corpus={"corpus_version": "ref", "corpus_sha256": "refhash", "case_count": 50},
        live_results=[
            {
                "case_id": "TIERB-REF-001",
                "initial_turn": {"debug_summary": {"timing_metrics": {"end_to_end_latency_ms": 10.0}}},
                "refinement_rounds": [],
                "script_execution_status": "script_mismatch",
                "script_mismatch_reason": "premature_case_completion",
            }
        ],
        benchmark_path=tmp_path / "bench.yaml",
    )

    assert summary["script_mismatch_summary"]["count"] == 1
    assert summary["aggregate_validation"]["ok"] is False
