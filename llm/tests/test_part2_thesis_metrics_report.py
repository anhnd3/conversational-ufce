from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.run_part2_thesis_metrics_report import (
    build_feature_ranges,
    build_session_result,
    build_thesis_summary,
    compute_final_feasibility,
)
from llm.src.part2_eval.corpora import build_tier_a_annotation_corpus
from llm.src.phase3.phase3_2_catalog import load_catalog


def test_compute_final_feasibility_requires_post_runtime_validation():
    assert compute_final_feasibility(
        public_state="RUNTIME_SUCCESS",
        summary_type="counterfactual_found",
        debug_summary={
            "runtime_summary": {"executed": True, "controller_state": "TERMINAL_SUCCESS"},
            "invariant_validation_status": "passed",
        },
    ) == 1
    assert compute_final_feasibility(
        public_state="RUNTIME_SUCCESS",
        summary_type="counterfactual_found",
        debug_summary={
            "runtime_summary": {"executed": True, "controller_state": "TERMINAL_SUCCESS"},
            "invariant_validation_status": "failed",
        },
    ) == 0


def test_build_session_result_scores_g2_counterfactual_case():
    case = {
        "case_id": "TIERB-G2-001",
        "group": "G2",
        "session_shape": "single_turn",
        "seed_profile": {
            "Income": 72.0,
            "Family": 1,
            "CCAvg": 4.8,
            "Education": 2,
            "Mortgage": 200.0,
            "SecuritiesAccount": 1,
            "CDAccount": 1,
            "Online": 0,
            "CreditCard": 0,
        },
        "active_constraint_spec_expected": {"max_changed_features": 1},
    }
    final_turn = {
        "public_state": "RUNTIME_SUCCESS",
        "is_case_complete": True,
        "case_completion_reason": "runtime_success",
        "explanation_payload": {
            "summary_type": "counterfactual_found",
            "counterfactual_summary": {
                "changed_fields": ["Income"],
                "profile": {
                    "Income": 80.0,
                    "Family": 1,
                    "CCAvg": 4.8,
                    "Education": 2,
                    "Mortgage": 200.0,
                    "SecuritiesAccount": 1,
                    "CDAccount": 1,
                    "Online": 0,
                    "CreditCard": 0,
                },
            },
        },
        "debug_summary": {
            "merge_applied": False,
            "runtime_summary": {
                "executed": True,
                "controller_state": "TERMINAL_SUCCESS",
                "reason_codes": [],
            },
            "invariant_validation_status": "passed",
            "timing_metrics": {"end_to_end_latency_ms": 18.0, "runtime_latency_ms": 6.0},
        },
    }
    dataset_entry = {
        "dataset_key": "bank",
        "full_feature_list": ["Income", "Family", "CCAvg", "Education", "Mortgage", "SecuritiesAccount", "CDAccount", "Online", "CreditCard"],
        "f2change": ["Income", "CCAvg", "Mortgage", "CDAccount", "Online"],
    }
    feature_ranges = build_feature_ranges(
        dataset_df=__import__("pandas").DataFrame(
            [
                case["seed_profile"],
                dict(case["seed_profile"], Income=100.0, Mortgage=300.0, CCAvg=6.0),
            ]
        ),
        feature_order=dataset_entry["full_feature_list"],
    )

    result = build_session_result(
        case=case,
        session_id="session_1",
        turn_payloads=[final_turn],
        session_detail={"active_constraint_spec": {"max_changed_features": 1}},
        dataset_entry=dataset_entry,
        feature_ranges=feature_ranges,
    )

    assert result["g2_applicable"] is True
    assert result["M11_actionability"] == 1
    assert result["M12_plausibility"] == 1
    assert result["M13_feasibility"] == 1
    assert result["M14_constraint_satisfaction"] == 1


def test_build_thesis_summary_includes_locked_metadata(tmp_path):
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
    summary = build_thesis_summary(
        run_id="demo",
        command="python run_part2_thesis_metrics_report.py",
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
        dataset_entry={"dataset_key": "bank", "f2change": ["Income"], "full_feature_list": ["Income"]},
        tier_a_corpus=tier_a_corpus,
        tier_a_corpus_path=tmp_path / "tier_a.json",
        tier_a_summary={
            "M6_constraint_extraction_fidelity": {"numerator": 20, "denominator": 25, "mean": 0.8, "component_mean": 0.9},
            "M1_json_validity_rate": {"numerator": 25, "denominator": 25, "mean": 1.0},
            "M2_schema_compliance_rate": {"numerator": 24, "denominator": 25, "mean": 0.96},
            "M3_canonical_validation_pass_rate": {"numerator": 24, "denominator": 25, "mean": 0.96},
            "M4_repair_rate": {"numerator": 2, "denominator": 25, "mean": 0.08},
            "M5_final_parser_failure_rate": {"numerator": 1, "denominator": 25, "mean": 0.04},
            "M7_parser_latency_ms": {"mean": 10.0, "p50": 9.0, "p95": 14.0, "max": 15.0},
        },
        full_tier_b_corpus={"corpus_version": "full", "corpus_sha256": "fullhash", "group_counts": {"G1": 100, "G2": 100, "REFINEMENT": 50}},
        tier_b_corpus_path=tmp_path / "tier_b.json",
        g1g2_scope={"scope": "tier_b_groups:G1,G2", "evaluated_case_count": 200, "evaluated_group_counts": {"G1": 100, "G2": 100}},
        session_results=[
            {
                "group": "G1",
                "final_public_state": "RUNTIME_SUCCESS",
                "is_case_complete": True,
                "successful_resolution": True,
                "had_clarification": False,
                "clarification_rounds": 0,
                "merge_followup_turns": 0,
                "merge_successes": 0,
                "reject_class": None,
                "case_completion_reason": "runtime_success",
                "turn_count": 1,
                "final_latency_ms": 10.0,
            },
            {
                "group": "G2",
                "final_public_state": "RUNTIME_SUCCESS",
                "is_case_complete": True,
                "successful_resolution": True,
                "had_clarification": False,
                "clarification_rounds": 0,
                "merge_followup_turns": 0,
                "merge_successes": 0,
                "reject_class": None,
                "case_completion_reason": "runtime_success",
                "turn_count": 1,
                "final_latency_ms": 12.0,
                "g2_applicable": True,
                "M8_validity": 1,
                "M9_proximity": 0.2,
                "M10_sparsity": 1,
                "M11_actionability": 1,
                "M12_plausibility": 1,
                "M13_feasibility": 1,
                "M14_constraint_satisfaction": 1,
                "M15_constraint_blocked": 0,
            },
        ],
        benchmark_path=tmp_path / "bench.yaml",
    )

    assert summary["runner_scope"] == "part2_g1_g2_system_eval"
    assert summary["execution_mode"] == "in_process_conversational"
    assert summary["corpus_version"] == "full"
    assert summary["corpus_path"].endswith("tier_b.json")
    assert summary["catalog_version"] == baseline_catalog.catalog_version
    assert summary["frozen_inputs"]["tier_a_annotation_schema_sha256"]
    assert summary["frozen_inputs"]["tier_a_scorer_output_schema_sha256"]
    assert summary["g2_metrics"]["M11_actionability"]["mean"] == 1.0
    assert summary["aggregate_validation"]["ok"] is True
    assert Path(summary["report_json_path"]).name == "thesis_metrics_report.json"


def test_build_thesis_summary_marks_script_mismatch_as_failed_validation(tmp_path):
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
    summary = build_thesis_summary(
        run_id="demo",
        command="python run_part2_thesis_metrics_report.py",
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
        dataset_entry={"dataset_key": "bank", "f2change": ["Income"], "full_feature_list": ["Income"]},
        tier_a_corpus=tier_a_corpus,
        tier_a_corpus_path=tmp_path / "tier_a.json",
        tier_a_summary={
            "M6_constraint_extraction_fidelity": {"numerator": 20, "denominator": 25, "mean": 0.8, "component_mean": 0.9},
            "M1_json_validity_rate": {"numerator": 25, "denominator": 25, "mean": 1.0},
            "M2_schema_compliance_rate": {"numerator": 24, "denominator": 25, "mean": 0.96},
            "M3_canonical_validation_pass_rate": {"numerator": 24, "denominator": 25, "mean": 0.96},
            "M4_repair_rate": {"numerator": 2, "denominator": 25, "mean": 0.08},
            "M5_final_parser_failure_rate": {"numerator": 1, "denominator": 25, "mean": 0.04},
            "M7_parser_latency_ms": {"mean": 10.0, "p50": 9.0, "p95": 14.0, "max": 15.0},
        },
        full_tier_b_corpus={"corpus_version": "full", "corpus_sha256": "fullhash", "group_counts": {"G1": 100, "G2": 100, "REFINEMENT": 50}},
        tier_b_corpus_path=tmp_path / "tier_b.json",
        g1g2_scope={"scope": "tier_b_groups:G1,G2", "evaluated_case_count": 200, "evaluated_group_counts": {"G1": 100, "G2": 100}},
        session_results=[
            {
                "group": "G1",
                "final_public_state": "RUNTIME_SUCCESS",
                "is_case_complete": True,
                "successful_resolution": True,
                "had_clarification": True,
                "clarification_rounds": 1,
                "merge_followup_turns": 1,
                "merge_successes": 1,
                "reject_class": None,
                "case_completion_reason": "runtime_success",
                "turn_count": 1,
                "final_latency_ms": 10.0,
                "script_execution_status": "script_mismatch",
                "script_mismatch_reason": "premature_case_completion",
                "case_id": "TIERB2-G1-001",
            }
        ],
        benchmark_path=tmp_path / "bench.yaml",
    )

    assert summary["script_mismatch_summary"]["count"] == 1
    assert summary["aggregate_validation"]["ok"] is False
    assert summary["aggregate_validation"]["differences"][-1]["path"] == "script_mismatch_summary.count"
