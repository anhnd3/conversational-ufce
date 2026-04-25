from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from llm.src.phase3.catalog import DEFAULT_CATALOG_PATH, Phase31ValidationScenario
from scripts.run_phase3_1_closeout_suite import (
    AutomatedSuiteResult,
    LiveScenarioOutcome,
    build_closeout_summary,
    build_runner_command,
    evaluate_live_outcome,
    main,
    write_closeout_summary,
    write_standalone_report,
)


def _fake_args(tmp_path, mode: str = "both"):
    return SimpleNamespace(
        mode=mode,
        model_alias="qwen/qwen3-14b",
        api_base="http://localhost:1234",
        timeout_s=120.0,
        benchmark="llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml",
        system_prompt="llm/prompts/parser_system_prompt_v1.txt",
        catalog=str(DEFAULT_CATALOG_PATH),
        out_dir=str(tmp_path),
        debug_trace=True,
    )


def _fake_result(
    *,
    turn_id: str,
    final_public_state: str,
    template_type: str = "explanation",
    summary_type: str | None = None,
    included_suggestion_types: list[str] | None = None,
    stage_trace: list[str] | None = None,
    runtime_result: dict | None = None,
    artifact_parent_turn_id: str | None = None,
    merge_applied: bool = False,
    carried_fields: list[str] | None = None,
    builder_partial_profile_snapshot: dict | None = None,
    normalized_cf_request: dict | None = None,
):
    explanation_payload = None
    if summary_type is not None:
        explanation_payload = SimpleNamespace(
            summary_type=summary_type,
            included_suggestion_types=list(included_suggestion_types or []),
        )
    return SimpleNamespace(
        turn_id=turn_id,
        stage=final_public_state,
        stage_trace=list(stage_trace or [final_public_state]),
        explanation_payload=explanation_payload,
        runtime_result=runtime_result,
        response_decision=SimpleNamespace(
            final_public_state=final_public_state,
            template_type=template_type,
            included_suggestion_types=list(included_suggestion_types or []),
        ),
        builder_result=SimpleNamespace(
            partial_profile_snapshot=builder_partial_profile_snapshot,
        ),
        normalized_parse=None
        if normalized_cf_request is None
        else {
            "cf_request": dict(normalized_cf_request),
        },
        artifact_record=SimpleNamespace(
            output_dir=f"/tmp/{turn_id}",
            parent_turn_id=artifact_parent_turn_id,
            merge_applied=merge_applied,
            carried_fields=list(carried_fields or []),
        ),
    )


def test_evaluate_live_outcome_for_runtime_reject_checks_bounded_suggestions():
    scenario = Phase31ValidationScenario(
        scenario_id="P31-RJ-01",
        slug="runtime_reject",
        description="reject",
        turns=("reject",),
        expected_final_state="RUNTIME_REJECT",
        accept={
            "kind": "runtime_reject",
            "summary_type": "runtime_reject",
            "included_suggestion_types": [
                "revise_target_profile",
                "broaden_allowed_financial_changes",
            ],
        },
    )
    result = _fake_result(
        turn_id="turn1",
        final_public_state="RUNTIME_REJECT",
        template_type="explanation_reject",
        summary_type="runtime_reject",
        included_suggestion_types=[
            "revise_target_profile",
            "broaden_allowed_financial_changes",
        ],
        stage_trace=["READY_FOR_RUNTIME", "RUNTIME_REJECT"],
        runtime_result={"reason_codes": ["NO_FEASIBLE_CF_FOUND"]},
    )

    outcome = evaluate_live_outcome(scenario=scenario, results=[result])

    assert outcome.passed is True
    assert outcome.expected_public_ready_hidden is True
    assert outcome.key_payload_checks["included_suggestion_types_match"] is True
    assert outcome.key_payload_checks["ready_for_runtime_trace_only"] is True


def test_evaluate_live_outcome_for_merge_success_checks_parent_link():
    scenario = Phase31ValidationScenario(
        scenario_id="P31-CL-01",
        slug="merge_success",
        description="merge success",
        turns=("turn1", "turn2"),
        expected_final_state="RUNTIME_SUCCESS",
        accept={
            "kind": "clarification_merge_success",
            "turn1_final_state": "NEEDS_CLARIFICATION",
            "turn2_final_state": "RUNTIME_SUCCESS",
            "turn2_merge_applied": True,
        },
    )
    turn1 = _fake_result(
        turn_id="turn1",
        final_public_state="NEEDS_CLARIFICATION",
        template_type="clarification_missing_information",
    )
    turn2 = _fake_result(
        turn_id="turn2",
        final_public_state="RUNTIME_SUCCESS",
        template_type="explanation_counterfactual",
        summary_type="counterfactual_found",
        stage_trace=["READY_FOR_RUNTIME", "RUNTIME_SUCCESS"],
        runtime_result={"reason_codes": ["VALID_COUNTERFACTUAL_FOUND"]},
        artifact_parent_turn_id="turn1",
        merge_applied=True,
    )

    outcome = evaluate_live_outcome(scenario=scenario, results=[turn1, turn2])

    assert outcome.passed is True
    assert outcome.key_payload_checks["merge_applied_match"] is True
    assert outcome.key_payload_checks["parent_turn_link_match"] is True


def test_build_closeout_summary_marks_failed_gates():
    args = _fake_args(Path("/tmp"), mode="both")
    summary = build_closeout_summary(
        args=args,
        catalog_version="phase3_1_validation_catalog_v2",
        run_version="phase3_1_closeout_demo",
        run_root=Path("/tmp/demo"),
        live_output_root=Path("/tmp/demo/live"),
        mode="both",
        automated_suites=[
            AutomatedSuiteResult(
                suite_name="llm/tests",
                command="pytest -q llm/tests",
                exit_code=1,
                passed=False,
                summary_lines=["1 failed"],
            )
        ],
        live_scenarios=[
            LiveScenarioOutcome(
                scenario_id="P31-RJ-01",
                slug="reject",
                description="reject",
                expected_final_state="RUNTIME_REJECT",
                actual_final_state="RUNTIME_SUCCESS",
                passed=False,
                turn_count=1,
                stages=["RUNTIME_SUCCESS"],
                artifact_folders=["/tmp/reject"],
                key_payload_checks={"summary_type_match": False},
                expected_public_ready_hidden=True,
            )
        ],
    )

    assert summary["ready_to_start_phase3_2"] is False
    assert "automated_suites_failed" in summary["failed_gates"]
    assert "live_scenarios_failed" in summary["failed_gates"]


def test_write_closeout_outputs_include_required_sections(tmp_path):
    summary = {
        "run_version": "phase3_1_closeout_demo",
        "mode": "both",
        "provenance": {
            "timestamp_utc": "2026-03-21T12:00:00Z",
            "model_alias": "qwen/qwen3-14b",
            "api_base": "http://localhost:1234",
            "benchmark_path": "bench.yaml",
            "system_prompt_path": "prompt.txt",
            "scenario_catalog_version": "phase3_1_validation_catalog_v2",
            "scenario_catalog_path": "/tmp/catalog.json",
            "output_root": str(tmp_path),
            "live_output_root": str(tmp_path / "live_outputs"),
        },
        "automated_suites": [
            {
                "suite_name": "llm/tests",
                "command": "pytest -q llm/tests",
                "exit_code": 0,
                "passed": True,
                "summary_lines": ["93 passed"],
            }
        ],
        "live_scenarios": [
            {
                "scenario_id": "P31-NR-01",
                "slug": "nr",
                "description": "nr",
                "expected_final_state": "RUNTIME_SUCCESS",
                "actual_final_state": "RUNTIME_SUCCESS",
                "passed": True,
                "turn_count": 1,
                "stages": ["RUNTIME_SUCCESS"],
                "artifact_folders": ["/tmp/nr"],
                "key_payload_checks": {"summary_type_match": True},
                "expected_public_ready_hidden": True,
            }
        ],
        "ready_to_start_phase3_2": True,
        "failed_gates": [],
        "phase3_2_handoff": ["runtime reproducibility hardening pending"],
    }

    write_closeout_summary(tmp_path, summary)
    write_standalone_report(tmp_path, summary)

    json_summary = json.loads((tmp_path / "phase3_1_closeout_summary.json").read_text(encoding="utf-8"))
    markdown_summary = (tmp_path / "phase3_1_closeout_summary.md").read_text(encoding="utf-8")
    standalone_report = (tmp_path / "phase3_1_standalone_report.md").read_text(encoding="utf-8")

    assert json_summary["ready_to_start_phase3_2"] is True
    assert "# Phase 3.1 Closeout Summary" in markdown_summary
    assert "## Automated Suites" in standalone_report
    assert "## Live Scenario Results" in standalone_report
    assert "## Readiness Verdict" in standalone_report
    assert "## Phase 3.2 Handoff" in standalone_report


def test_build_runner_command_includes_catalog_and_mode(tmp_path):
    command = build_runner_command(_fake_args(tmp_path, mode="live"))

    assert "python scripts/run_phase3_1_closeout_suite.py" in command
    assert "--mode live" in command
    assert f"--catalog {Path(DEFAULT_CATALOG_PATH)}" in command
    assert f"--out-dir {tmp_path}" in command
    assert "--debug-trace" in command


def test_evaluate_live_outcome_for_reset_no_merge_unrelated_checks_no_stale_reuse():
    scenario = Phase31ValidationScenario(
        scenario_id="P31-RS-01",
        slug="reset_unrelated",
        description="reset unrelated",
        turns=("turn1", "turn2"),
        expected_final_state="UNSUPPORTED_REQUEST",
        accept={
            "kind": "reset_no_merge",
            "turn1_final_state": "NEEDS_CLARIFICATION",
            "turn2_final_state": "UNSUPPORTED_REQUEST",
            "expected_turn2_runtime_presence": False,
            "forbidden_turn2_fields": ["Income", "CCAvg", "Family", "Education", "Mortgage"],
        },
    )
    turn1 = _fake_result(
        turn_id="turn1",
        final_public_state="NEEDS_CLARIFICATION",
        template_type="clarification_missing_information",
        builder_partial_profile_snapshot={
            "Income": 40,
            "CCAvg": 1.5,
            "Family": 3,
            "Education": 2,
            "Mortgage": 80,
        },
        normalized_cf_request={
            "Income": 40,
            "CCAvg": 1.5,
            "Family": 3,
            "Education": 2,
            "Mortgage": 80,
        },
    )
    turn2 = _fake_result(
        turn_id="turn2",
        final_public_state="UNSUPPORTED_REQUEST",
        template_type="unsupported_request",
        artifact_parent_turn_id=None,
        merge_applied=False,
        carried_fields=[],
        builder_partial_profile_snapshot={},
        normalized_cf_request={},
    )

    outcome = evaluate_live_outcome(scenario=scenario, results=[turn1, turn2])

    assert outcome.passed is True
    assert outcome.key_payload_checks["merge_applied_match"] is True
    assert outcome.key_payload_checks["parent_turn_link_absent"] is True
    assert outcome.key_payload_checks["carried_fields_empty"] is True
    assert outcome.key_payload_checks["turn2_runtime_presence_match"] is True
    assert outcome.key_payload_checks["no_stale_turn1_reuse"] is True


def test_evaluate_live_outcome_for_reset_no_merge_full_profile_checks_expected_profile():
    scenario = Phase31ValidationScenario(
        scenario_id="P31-RS-02",
        slug="reset_full_profile",
        description="reset full profile",
        turns=("turn1", "turn2"),
        expected_final_state="RUNTIME_SUCCESS",
        accept={
            "kind": "reset_no_merge",
            "turn1_final_state": "NEEDS_CLARIFICATION",
            "turn2_final_state": "RUNTIME_SUCCESS",
            "expected_turn2_runtime_presence": True,
            "expected_turn2_profile": {
                "Income": 72,
                "CCAvg": 4.8,
                "Family": 1,
                "Education": 2,
                "Mortgage": 200,
                "CDAccount": 1,
                "Online": 0,
                "SecuritiesAccount": 1,
                "CreditCard": 0,
            },
        },
    )
    turn1 = _fake_result(
        turn_id="turn1",
        final_public_state="NEEDS_CLARIFICATION",
        template_type="clarification_missing_information",
        builder_partial_profile_snapshot={
            "Income": 40,
            "CCAvg": 1.5,
            "Family": 3,
            "Education": 2,
            "Mortgage": 80,
        },
        normalized_cf_request={
            "Income": 40,
            "CCAvg": 1.5,
            "Family": 3,
            "Education": 2,
            "Mortgage": 80,
        },
    )
    expected_profile = scenario.accept["expected_turn2_profile"]
    turn2 = _fake_result(
        turn_id="turn2",
        final_public_state="RUNTIME_SUCCESS",
        template_type="explanation_counterfactual",
        summary_type="counterfactual_found",
        stage_trace=["READY_FOR_RUNTIME", "RUNTIME_SUCCESS"],
        runtime_result={"reason_codes": ["VALID_COUNTERFACTUAL_FOUND"]},
        artifact_parent_turn_id=None,
        merge_applied=False,
        carried_fields=[],
        builder_partial_profile_snapshot=expected_profile,
        normalized_cf_request=expected_profile,
    )

    outcome = evaluate_live_outcome(scenario=scenario, results=[turn1, turn2])

    assert outcome.passed is True
    assert outcome.key_payload_checks["turn2_runtime_presence_match"] is True
    assert outcome.key_payload_checks["no_stale_turn1_reuse"] is True


def test_main_both_mode_writes_closeout_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.run_phase3_1_closeout_suite.build_run_version", lambda timestamp_utc=None: "phase3_1_closeout_fixed")
    monkeypatch.setattr(
        "scripts.run_phase3_1_closeout_suite.run_automated_suites",
        lambda: [
            AutomatedSuiteResult("llm/tests", "pytest -q llm/tests", 0, True, ["93 passed"]),
            AutomatedSuiteResult("llm_eval/tests", "pytest -q llm_eval/tests", 0, True, ["11 passed"]),
        ],
    )
    monkeypatch.setattr("scripts.run_phase3_1_closeout_suite.build_orchestrator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "scripts.run_phase3_1_closeout_suite.run_live_scenarios",
        lambda **kwargs: [
            LiveScenarioOutcome(
                scenario_id="P31-NR-01",
                slug="nr",
                description="nr",
                expected_final_state="RUNTIME_SUCCESS",
                actual_final_state="RUNTIME_SUCCESS",
                passed=True,
                turn_count=1,
                stages=["RUNTIME_SUCCESS"],
                artifact_folders=["/tmp/nr"],
                key_payload_checks={"summary_type_match": True},
                expected_public_ready_hidden=True,
            )
        ],
    )
    monkeypatch.setattr("scripts.run_phase3_1_closeout_suite.write_case_study_indexes", lambda output_root: None)

    exit_code = main(["--mode", "both", "--out-dir", str(tmp_path)])

    assert exit_code == 0
    run_root = tmp_path / "phase3_1_closeout_fixed"
    assert (run_root / "phase3_1_closeout_summary.json").exists()
    summary = json.loads((run_root / "phase3_1_closeout_summary.json").read_text(encoding="utf-8"))
    assert summary["ready_to_start_phase3_2"] is True


def test_main_tests_mode_stays_not_ready(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.run_phase3_1_closeout_suite.build_run_version", lambda timestamp_utc=None: "phase3_1_closeout_tests_only")
    monkeypatch.setattr(
        "scripts.run_phase3_1_closeout_suite.run_automated_suites",
        lambda: [
            AutomatedSuiteResult("llm/tests", "pytest -q llm/tests", 0, True, ["93 passed"]),
        ],
    )

    exit_code = main(["--mode", "tests", "--out-dir", str(tmp_path)])

    assert exit_code == 1
    summary = json.loads(
        (tmp_path / "phase3_1_closeout_tests_only" / "phase3_1_closeout_summary.json").read_text(encoding="utf-8")
    )
    assert "full_closeout_requires_mode_both" in summary["failed_gates"]
    assert "live_scenarios_not_run" in summary["failed_gates"]


def test_main_live_mode_stays_not_ready(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.run_phase3_1_closeout_suite.build_run_version", lambda timestamp_utc=None: "phase3_1_closeout_live_only")
    monkeypatch.setattr("scripts.run_phase3_1_closeout_suite.build_orchestrator", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "scripts.run_phase3_1_closeout_suite.run_live_scenarios",
        lambda **kwargs: [
            LiveScenarioOutcome(
                scenario_id="P31-RJ-01",
                slug="reject",
                description="reject",
                expected_final_state="RUNTIME_REJECT",
                actual_final_state="RUNTIME_REJECT",
                passed=True,
                turn_count=1,
                stages=["RUNTIME_REJECT"],
                artifact_folders=["/tmp/reject"],
                key_payload_checks={"summary_type_match": True},
                expected_public_ready_hidden=True,
            )
        ],
    )
    monkeypatch.setattr("scripts.run_phase3_1_closeout_suite.write_case_study_indexes", lambda output_root: None)

    exit_code = main(["--mode", "live", "--out-dir", str(tmp_path)])

    assert exit_code == 1
    summary = json.loads(
        (tmp_path / "phase3_1_closeout_live_only" / "phase3_1_closeout_summary.json").read_text(encoding="utf-8")
    )
    assert "full_closeout_requires_mode_both" in summary["failed_gates"]
    assert "automated_suites_not_run" in summary["failed_gates"]
