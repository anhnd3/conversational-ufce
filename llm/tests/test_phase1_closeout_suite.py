from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from llm.src.phase2.catalog import DEFAULT_CATALOG_PATH, load_catalog
from llm.src.phase2.taxonomy import classify_turn_result
from scripts.archieve.run_phase1_closeout_suite import (
    SuiteOutcome,
    build_default_scenarios,
    build_runner_command,
    build_suite_summary,
    write_suite_summary,
)


def test_build_default_scenarios_covers_phase1_closeout_shapes():
    catalog = load_catalog(DEFAULT_CATALOG_PATH)
    scenarios = build_default_scenarios(DEFAULT_CATALOG_PATH)

    assert len(scenarios) == 7
    assert scenarios[0].turns == catalog.get_case("P-NR-01").turns
    assert scenarios[1].turns == catalog.get_case("P-CF-01").turns
    assert scenarios[2].turns == catalog.get_case("P-RJ-01").turns
    assert scenarios[3].turns == catalog.get_case("P-CL-01").turns
    assert scenarios[4].turns == catalog.get_case("S-MERGE-01").turns
    assert scenarios[5].turns == catalog.get_case("S-MERGE-02").turns
    assert scenarios[6].turns == catalog.get_case("SM-RESET-01").turns


def test_classify_turn_result_maps_summary_type_and_stage():
    success = SimpleNamespace(summary_type="counterfactual_found")
    assert classify_turn_result(stage="RUNTIME_SUCCESS", explanation_payload=success) == "counterfactual_found"
    assert classify_turn_result(stage="NEEDS_CLARIFICATION", explanation_payload=None) == "clarification"


def test_build_suite_summary_counts_passed_and_failed():
    summary = build_suite_summary(
        [
            SuiteOutcome(
                slug="ok",
                description="ok",
                expected_label="no_recourse_needed",
                actual_label="no_recourse_needed",
                passed=True,
                expected_merge_applied=None,
                actual_merge_applied=False,
                stages=["RUNTIME_SUCCESS"],
                output_dirs=["/tmp/ok"],
                turn_count=1,
                first_turn_stage="RUNTIME_SUCCESS",
                expected_first_turn_stage=None,
                actual_carried_fields=[],
            ),
            SuiteOutcome(
                slug="bad",
                description="bad",
                expected_label="clarification",
                actual_label="parser_failure",
                passed=False,
                expected_merge_applied=True,
                actual_merge_applied=False,
                stages=["PARSER_FAILURE"],
                output_dirs=["/tmp/bad"],
                turn_count=2,
                first_turn_stage="NEEDS_CLARIFICATION",
                expected_first_turn_stage="NEEDS_CLARIFICATION",
                actual_carried_fields=[],
            ),
        ]
    )

    assert summary["scenario_count"] == 2
    assert summary["passed_count"] == 1
    assert summary["failed_count"] == 1
    assert summary["all_passed"] is False
    assert summary["actual_label_counts"] == {
        "no_recourse_needed": 1,
        "parser_failure": 1,
    }


def test_write_suite_summary_writes_json_and_markdown(tmp_path):
    summary = build_suite_summary(
        [
            SuiteOutcome(
                slug="suite_demo",
                description="demo",
                expected_label="clarification",
                actual_label="clarification",
                passed=True,
                expected_merge_applied=True,
                actual_merge_applied=True,
                stages=["NEEDS_CLARIFICATION", "NEEDS_CLARIFICATION"],
                output_dirs=["/tmp/demo1", "/tmp/demo2"],
                turn_count=2,
                first_turn_stage="NEEDS_CLARIFICATION",
                expected_first_turn_stage="NEEDS_CLARIFICATION",
                actual_carried_fields=["Income", "Family"],
            )
        ]
    )

    write_suite_summary(tmp_path, summary)

    json_summary = json.loads((tmp_path / "phase1_closeout_suite_summary.json").read_text(encoding="utf-8"))
    markdown_summary = (tmp_path / "phase1_closeout_suite_summary.md").read_text(encoding="utf-8")

    assert json_summary["all_passed"] is True
    assert "# Phase 1 Closeout Suite Summary" in markdown_summary
    assert "suite_demo" in markdown_summary
    assert "NEEDS_CLARIFICATION -> NEEDS_CLARIFICATION" in markdown_summary
    assert "Income, Family" in markdown_summary


def test_build_runner_command_includes_expected_arguments():
    class Args:
        model_alias = "qwen/qwen3-14b"
        api_base = "http://localhost:1234"
        timeout_s = 120.0
        benchmark = "llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml"
        system_prompt = "llm/prompts/parser_system_prompt_v1.txt"
        catalog = str(DEFAULT_CATALOG_PATH)
        out_dir = "outputs/conversations_smoke"
        debug_trace = True

    command = build_runner_command(Args())

    assert "python scripts/run_phase1_closeout_suite.py" in command
    assert f"--catalog {Path(DEFAULT_CATALOG_PATH)}" in command
    assert "--out-dir outputs/conversations_smoke" in command
    assert "--debug-trace" in command
