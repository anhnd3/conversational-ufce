from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_phase3_2_metrics_report import (
    IsolatedServiceHandle,
    build_case_result,
    build_metrics_summary,
    classify_reject_class,
    compute_actionability,
    render_markdown,
    run_metrics_report,
)
from llm.src.phase3.phase3_2_catalog import load_catalog


def write_catalog(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "catalog_version": "phase3_2_validation_catalog_v1",
                "runner_compat_version": "phase3_2_acceptance_runner_v1",
                "created_timestamp_utc": "2026-03-24T00:00:00Z",
                "prompt_template_version": "parser_system_prompt_v1",
                "change_notes": ["demo"],
                "scenarios": [
                    {
                        "scenario_id": "P32-CF-01",
                        "slug": "demo_counterfactual",
                        "description": "demo",
                        "turns": ["Income 72, CCAvg 4.8, Family 1, Education 2, Mortgage 200."],
                        "expected_final_state": "RUNTIME_SUCCESS",
                        "accept": {"kind": "counterfactual_found", "summary_type": "counterfactual_found"},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_compute_actionability_enforces_allowed_fields_bounds_and_max_changes():
    feature_order = [
        "Income",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]
    counterfactual_summary = {
        "changed_fields": ["Income"],
        "profile": {"Income": 95, "Mortgage": 100},
    }
    active_constraint_spec = {
        "disallowed_changes": ["Mortgage"],
        "numeric_bounds": {"Income": {"min": 90}},
        "max_changed_features": 1,
    }

    assert compute_actionability(
        counterfactual_summary=counterfactual_summary,
        active_constraint_spec=active_constraint_spec,
        policy_f2change=["Income", "Mortgage"],
        feature_order=feature_order,
    ) == 1
    assert compute_actionability(
        counterfactual_summary={"changed_fields": ["Online"], "profile": {"Online": 1}},
        active_constraint_spec=active_constraint_spec,
        policy_f2change=["Income", "Mortgage"],
        feature_order=feature_order,
    ) == 0
    assert compute_actionability(
        counterfactual_summary={"changed_fields": ["Mortgage"], "profile": {"Mortgage": 100}},
        active_constraint_spec=active_constraint_spec,
        policy_f2change=["Income", "Mortgage"],
        feature_order=feature_order,
    ) == 0
    assert compute_actionability(
        counterfactual_summary={"changed_fields": ["Income"], "profile": {"Income": 80}},
        active_constraint_spec=active_constraint_spec,
        policy_f2change=["Income", "Mortgage"],
        feature_order=feature_order,
    ) == 0
    assert compute_actionability(
        counterfactual_summary={
            "changed_fields": ["Income", "Mortgage"],
            "profile": {"Income": 95, "Mortgage": 100},
        },
        active_constraint_spec={"max_changed_features": 1},
        policy_f2change=["Income", "Mortgage"],
        feature_order=feature_order,
    ) == 0


def test_classify_reject_class_uses_exposed_public_state_not_internal_runtime_hint():
    assert classify_reject_class(
        public_state="RUNTIME_SUCCESS",
        case_completion_reason="runtime_success",
        explanation_payload={"reason_codes": ["NO_FEASIBLE_CF_FOUND"]},
        debug_summary={"runtime_summary": {"reason_codes": ["NO_FEASIBLE_CF_FOUND"]}},
    ) is None


def test_build_case_result_keeps_refinement_like_runtime_success_out_of_reject_bucket():
    scenario = SimpleNamespace(
        scenario_id="P32-CF-01",
        slug="demo_counterfactual",
        expected_final_state="RUNTIME_SUCCESS",
    )
    final_turn = {
        "turn_id": "turn_1",
        "public_state": "RUNTIME_SUCCESS",
        "is_case_complete": True,
        "case_completion_reason": "runtime_success",
        "explanation_payload": {
            "summary_type": "counterfactual_found",
            "counterfactual_summary": {
                "changed_fields": ["Income"],
                "profile": {"Income": 95},
            },
        },
        "debug_summary": {
            "invariant_validation_status": "passed",
            "runtime_summary": {
                "executed": True,
                "controller_state": "TERMINAL_SUCCESS",
                "reason_codes": ["NO_FEASIBLE_CF_FOUND"],
            },
        },
    }
    session_detail = {"active_constraint_spec": {}, "session_id": "session_demo"}
    dataset_entry = {
        "dataset_key": "bank",
        "full_feature_list": ["Income", "Family", "CCAvg", "Education", "Mortgage"],
        "f2change": ["Income", "Mortgage"],
    }

    case_result = build_case_result(
        scenario=scenario,
        session_id="session_demo",
        final_turn=final_turn,
        session_detail=session_detail,
        dataset_entry=dataset_entry,
    )

    assert case_result["reject_class"] is None
    assert case_result["g2_applicable"] is True
    assert case_result["M11_actionability"] == 1
    assert case_result["M12_plausibility"] == 1
    assert case_result["M13_feasibility"] == 1


def test_build_metrics_summary_and_markdown_include_required_contract(tmp_path):
    catalog_path = write_catalog(tmp_path / "catalog.json")
    catalog = load_catalog(catalog_path)
    run_root = tmp_path / "run_root"
    run_root.mkdir()
    service = IsolatedServiceHandle(
        process=None,
        base_url="http://127.0.0.1:9000",
        service_command="python fake_service.py",
        sqlite_path=(run_root / "isolated_sessions.sqlite3"),
        artifact_root=(run_root / "isolated_product_artifacts"),
        stdout_path=(run_root / "stdout.log"),
        stderr_path=(run_root / "stderr.log"),
        stdout_handle=None,
        stderr_handle=None,
    )
    case_results = [
        {
            "scenario_id": "P32-CF-01",
            "slug": "demo_counterfactual",
            "expected_final_state": "RUNTIME_SUCCESS",
            "expected_final_state_match": True,
            "session_id": "session_1",
            "turn_id": "turn_1",
            "final_public_state": "RUNTIME_SUCCESS",
            "is_case_complete": True,
            "case_completion_reason": "runtime_success",
            "summary_type": "counterfactual_found",
            "reject_class": None,
            "invariant_validation_status": "passed",
            "runtime_executed": True,
            "runtime_controller_state": "TERMINAL_SUCCESS",
            "reproducibility_surface": {
                "final_public_state": "RUNTIME_SUCCESS",
                "summary_type": "counterfactual_found",
                "reject_class": None,
                "invariant_validation_status": "passed",
            },
            "g2_applicable": True,
            "g2_not_applicable_reason": None,
            "M11_actionability": 1,
            "M12_plausibility": 1,
            "M13_feasibility": 1,
        }
    ]

    summary = build_metrics_summary(
        run_id="phase3_2_metrics_demo",
        command="python runner.py",
        run_root=run_root,
        service=service,
        catalog=catalog,
        version_payload={
            "api_version": "v1",
            "app_version": "phase3_2_test",
            "model_alias": "stub-model",
            "runtime_mode": "stable_demo",
            "git_commit": "abc123",
        },
        dataset_entry={
            "dataset_key": "bank",
            "full_feature_list": ["Income", "Family", "CCAvg", "Education", "Mortgage"],
            "f2change": ["Income", "Mortgage"],
        },
        case_results=case_results,
    )

    required_keys = {
        "run_id",
        "runner_scope",
        "timestamp_local",
        "timezone",
        "command",
        "service_command",
        "catalog_version",
        "catalog_path",
        "catalog_sha256",
        "isolated_run",
        "sqlite_path",
        "artifact_root",
        "corpus_counts",
        "g1_metrics",
        "g2_metrics",
        "per_case_results",
        "provenance",
    }
    assert required_keys.issubset(summary.keys())

    markdown = render_markdown(summary)
    assert "## Provenance" in markdown
    assert "## Scope" in markdown
    assert "## Corpus" in markdown
    assert "## G1 Metrics" in markdown
    assert "## G2 Metrics" in markdown
    assert "## Case Breakdown" in markdown
    assert summary["catalog_sha256"] in markdown


def test_run_metrics_report_writes_outputs_and_uses_isolated_paths(monkeypatch, tmp_path):
    catalog_path = write_catalog(tmp_path / "catalog.json")
    out_dir = tmp_path / "metrics_out"

    def fake_launch_isolated_service(*, layout, args):
        return IsolatedServiceHandle(
            process=None,
            base_url="http://127.0.0.1:9000",
            service_command="python fake_service.py",
            sqlite_path=layout["sqlite_path"],
            artifact_root=layout["artifact_root"],
            stdout_path=layout["stdout_path"],
            stderr_path=layout["stderr_path"],
            stdout_handle=None,
            stderr_handle=None,
        )

    def fake_collect_metrics_from_api(*, base_url, api_version, catalog):
        del base_url
        del api_version
        del catalog
        return (
            {
                "api_version": "v1",
                "app_version": "phase3_2_test",
                "model_alias": "stub-model",
                "runtime_mode": "stable_demo",
                "git_commit": "abc123",
            },
            {
                "dataset_key": "bank",
                "full_feature_list": ["Income", "Family", "CCAvg", "Education", "Mortgage"],
                "f2change": ["Income", "Mortgage"],
            },
            [
                {
                    "scenario_id": "P32-CF-01",
                    "slug": "demo_counterfactual",
                    "expected_final_state": "RUNTIME_SUCCESS",
                    "expected_final_state_match": True,
                    "session_id": "session_1",
                    "turn_id": "turn_1",
                    "final_public_state": "RUNTIME_SUCCESS",
                    "is_case_complete": True,
                    "case_completion_reason": "runtime_success",
                    "summary_type": "counterfactual_found",
                    "reject_class": None,
                    "invariant_validation_status": "passed",
                    "runtime_executed": True,
                    "runtime_controller_state": "TERMINAL_SUCCESS",
                    "reproducibility_surface": {
                        "final_public_state": "RUNTIME_SUCCESS",
                        "summary_type": "counterfactual_found",
                        "reject_class": None,
                        "invariant_validation_status": "passed",
                    },
                    "g2_applicable": True,
                    "g2_not_applicable_reason": None,
                    "M11_actionability": 1,
                    "M12_plausibility": 1,
                    "M13_feasibility": 1,
                }
            ],
        )

    monkeypatch.setattr("scripts.run_phase3_2_metrics_report.launch_isolated_service", fake_launch_isolated_service)
    monkeypatch.setattr("scripts.run_phase3_2_metrics_report.collect_metrics_from_api", fake_collect_metrics_from_api)
    monkeypatch.setattr("scripts.run_phase3_2_metrics_report.stop_isolated_service", lambda service: None)

    args = SimpleNamespace(
        catalog=catalog_path,
        out_dir=out_dir,
        lm_studio_api_base="http://localhost:1234",
        model_alias="stub-model",
        product_mode="stable_demo",
        api_version="v1",
        app_version="phase3_2_test",
        startup_timeout_s=5.0,
        service_script=Path("scripts/run_phase3_2_demo.py"),
    )

    summary = run_metrics_report(args=args, command="python runner.py")
    report_json_path = Path(summary["report_json_path"])
    report_markdown_path = Path(summary["report_markdown_path"])

    assert summary["isolated_run"] is True
    assert Path(summary["sqlite_path"]).name == "isolated_sessions.sqlite3"
    assert Path(summary["artifact_root"]).name == "isolated_product_artifacts"
    assert report_json_path.exists()
    assert report_markdown_path.exists()

    payload = json.loads(report_json_path.read_text(encoding="utf-8"))
    assert payload["runner_scope"] == "product_facing_g1_g2"
    assert payload["catalog_path"] == str(catalog_path.resolve())
    assert payload["catalog_sha256"]
    assert payload["command"] == "python runner.py"

    markdown = report_markdown_path.read_text(encoding="utf-8")
    assert "## Provenance" in markdown
    assert "## Scope" in markdown
    assert "## Corpus" in markdown
    assert "## G1 Metrics" in markdown
    assert "## G2 Metrics" in markdown
    assert "## Case Breakdown" in markdown
