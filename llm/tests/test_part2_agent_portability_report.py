from __future__ import annotations

from scripts.run_part2_agent_portability_report import (
    build_backend_aggregate,
    build_agent_portability_summary,
    compute_attempt_stability,
    parse_backend_names,
    render_markdown,
)


def test_parse_backend_names_normalizes_and_deduplicates():
    assert parse_backend_names(" ufce , dice,ar,dice ") == ["ufce", "dice", "ar"]


def test_parse_backend_names_rejects_unknown_backend():
    try:
        parse_backend_names("ufce,other")
    except ValueError as exc:
        assert "Unsupported backend names" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unsupported backend")


def test_compute_attempt_stability_uses_locked_reproducibility_signature():
    stable_attempts = [
        {
            "final_public_state": "RUNTIME_SUCCESS",
            "summary_type": "counterfactual_found",
            "reject_class": None,
            "active_constraint_spec": {},
            "counterfactual_profile": {"Income": 80.0},
        },
        {
            "final_public_state": "RUNTIME_SUCCESS",
            "summary_type": "counterfactual_found",
            "reject_class": None,
            "active_constraint_spec": {},
            "counterfactual_profile": {"Income": 80.0},
        },
    ]
    unstable_attempts = stable_attempts + [
        {
            "final_public_state": "RUNTIME_REJECT",
            "summary_type": None,
            "reject_class": "request_constraints_blocked",
            "active_constraint_spec": {"disallowed_changes": ["Income"]},
            "counterfactual_profile": None,
        }
    ]

    assert compute_attempt_stability(stable_attempts) is True
    assert compute_attempt_stability(unstable_attempts) is False


def test_build_backend_aggregate_summarizes_system_and_recommendation_metrics():
    case_attempts = [
        {
            "case_id": "case-1",
            "stable": True,
            "attempts": [
                {
                    "is_case_complete": True,
                    "successful_resolution": True,
                    "had_clarification": False,
                    "case_completion_reason": None,
                    "reject_class": None,
                    "turn_count": 1,
                    "final_latency_ms": 120.0,
                    "applicable_counterfactual": True,
                    "final_cf_validity": 1,
                    "actionability": 1,
                    "plausibility": 1,
                    "feasibility": 1,
                    "proximity": 0.2,
                    "sparsity": 1,
                    "constraint_blocked": 0,
                    "active_constraint_spec": {},
                }
            ],
        },
        {
            "case_id": "case-2",
            "stable": False,
            "attempts": [
                {
                    "is_case_complete": True,
                    "successful_resolution": False,
                    "had_clarification": True,
                    "case_completion_reason": "clarification_limit_reached",
                    "reject_class": "unsupported_request",
                    "turn_count": 2,
                    "final_latency_ms": 240.0,
                    "applicable_counterfactual": False,
                    "final_cf_validity": 0,
                    "actionability": None,
                    "plausibility": None,
                    "feasibility": None,
                    "proximity": None,
                    "sparsity": None,
                    "constraint_blocked": 1,
                    "active_constraint_spec": {"max_changed_features": 1},
                }
            ],
        },
    ]

    aggregate = build_backend_aggregate(backend_name="dice", case_attempts=case_attempts)

    assert aggregate["portability_row"]["completion_rate"]["mean"] == 1.0
    assert aggregate["portability_row"]["successful_recourse_resolution"]["mean"] == 0.5
    assert aggregate["portability_row"]["clarification_rate"]["mean"] == 0.5
    assert aggregate["portability_row"]["reproducibility_stability"]["mean"] == 0.5
    assert aggregate["validated_aggregate"]["system_metrics"]["clarification_exhaustion_rate"]["mean"] == 0.5
    assert aggregate["validated_aggregate"]["recommendation_metrics"]["proximity"]["denominator"] == 1
    assert aggregate["validated_aggregate"]["recommendation_metrics"]["constraint_blocked_rate"]["mean"] == 1.0


def test_render_markdown_includes_g5_section_and_validation():
    summary = {
        "run_id": "demo",
        "runner_scope": "part2_g5_agent_portability",
        "scorer_version": "part2_agent_portability_report_v1",
        "timestamp_local": "2026-03-26T10:00:00+07:00",
        "timezone": "UTC+07:00",
        "corpus_version": "part2_g5_agent_portability_bank_v1",
        "corpus_sha256": "abc",
        "attempts_per_case": 3,
        "evaluated_case_count": 100,
        "fairness_contract": {"rules": ["same parser", "same shell", "different backend only"]},
        "agent_portability_table": {
            "agent + UFCE": {
                "completion_rate": {"mean": 1.0},
                "successful_recourse_resolution": {"mean": 0.8},
                "clarification_rate": {"mean": 0.4},
                "average_turns": {"mean": 1.5, "p50": 1.0, "p95": 2.0, "max": 2.0},
                "end_to_end_latency_ms": {"mean": 100.0, "p50": 100.0, "p95": 100.0, "max": 100.0},
                "reproducibility_stability": {"mean": 1.0},
                "final_cf_validity": {"mean": 0.6},
                "actionability": {"mean": 0.9},
                "plausibility": {"mean": 0.85},
                "feasibility": {"mean": 0.8},
            }
        },
        "aggregate_validation": {"ok": True, "difference_count": 0},
    }

    markdown = render_markdown(summary)

    assert "G5_agent_portability" in markdown
    assert "agent + UFCE" in markdown
    assert "same parser" in markdown
    assert "difference_count" in markdown


def test_build_agent_portability_summary_reads_nested_backend_aggregate(tmp_path):
    class Catalog:
        catalog_version = "phase3_2_validation_catalog_v1"
        source_path = tmp_path / "catalog.json"

    Catalog.source_path.write_text("{}", encoding="utf-8")
    run_root = tmp_path / "run"
    run_root.mkdir()
    summary = build_agent_portability_summary(
        run_id="demo",
        command="python demo.py",
        run_root=run_root,
        baseline_catalog=Catalog(),
        corpus={
            "corpus_version": "part2_g5_agent_portability_bank_v1",
            "corpus_sha256": "abc",
        },
        g5_corpus_path=tmp_path / "g5.json",
        cases=[{"case_id": "case-1"}],
        backends=["ufce"],
        attempts_per_case=2,
        fairness_contract={"rules": ["same parser"]},
        backend_results={
            "ufce": {
                "aggregate": {
                    "validated_aggregate": {
                        "system_metrics": {"completion_rate": {"mean": 1.0}},
                        "recommendation_metrics": {"final_cf_validity": {"mean": 1.0}},
                    }
                },
                "case_attempts": [],
            }
        },
        comparison_rows={"agent + UFCE": {"completion_rate": {"mean": 1.0}}},
        validated_blocks={},
        benchmark_path=tmp_path / "bench.yaml",
        version_payloads={"ufce": {"git_commit": "deadbeef"}},
    )

    assert "ufce" in summary["aggregate_validation"]["validated_aggregates"]
    assert summary["git_commit"] == "deadbeef"
    assert summary["loaded_corpora"]["g5_corpus"]["corpus_version"] == "part2_g5_agent_portability_bank_v1"


def test_build_agent_portability_summary_marks_script_mismatch_as_failed_validation(tmp_path):
    class Catalog:
        catalog_version = "phase3_2_validation_catalog_v1"
        source_path = tmp_path / "catalog.json"

    Catalog.source_path.write_text("{}", encoding="utf-8")
    run_root = tmp_path / "run"
    run_root.mkdir()
    summary = build_agent_portability_summary(
        run_id="demo",
        command="python demo.py",
        run_root=run_root,
        baseline_catalog=Catalog(),
        corpus={
            "corpus_version": "part2_g5_agent_portability_bank_v1",
            "corpus_sha256": "abc",
        },
        g5_corpus_path=tmp_path / "g5.json",
        cases=[{"case_id": "case-1"}],
        backends=["ufce"],
        attempts_per_case=2,
        fairness_contract={"rules": ["same parser"]},
        backend_results={
            "ufce": {
                "aggregate": {
                    "validated_aggregate": {
                        "system_metrics": {"completion_rate": {"mean": 1.0}},
                        "recommendation_metrics": {"final_cf_validity": {"mean": 1.0}},
                    }
                },
                "case_attempts": [
                    {
                        "case_id": "case-1",
                        "stable": False,
                        "reproducibility_signature_count": 1,
                        "attempts": [
                            {
                                "backend_name": "ufce",
                            "case_id": "case-1",
                            "attempt_index": 1,
                            "is_case_complete": True,
                            "successful_resolution": False,
                            "had_clarification": True,
                            "case_completion_reason": "clarification_limit_reached",
                            "reject_class": "clarification_limit_reached",
                            "turn_count": 1,
                            "final_latency_ms": 15.0,
                            "applicable_counterfactual": False,
                            "final_cf_validity": 0,
                            "actionability": None,
                            "plausibility": None,
                            "feasibility": None,
                            "proximity": None,
                            "sparsity": None,
                            "constraint_blocked": 0,
                            "active_constraint_spec": {},
                            "script_execution_status": "script_mismatch",
                            "script_mismatch_reason": "premature_case_completion",
                        }
                        ],
                    }
                ],
            }
        },
        comparison_rows={"agent + UFCE": {"completion_rate": {"mean": 1.0}}},
        validated_blocks={},
        benchmark_path=tmp_path / "bench.yaml",
        version_payloads={"ufce": {"git_commit": "deadbeef"}},
    )

    assert summary["script_mismatch_summary"]["count"] == 1
    assert summary["aggregate_validation"]["ok"] is False
