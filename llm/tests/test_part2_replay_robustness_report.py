from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from llm.src.phase3.phase3_2_catalog import load_catalog
from scripts.run_part2_replay_robustness_report import (
    build_replay_summary,
    compute_replay_aggregate_blocks,
)


def test_compute_replay_aggregate_blocks_marks_tier_d_as_non_semantic():
    aggregate_blocks = compute_replay_aggregate_blocks(
        replay_results=[
            {
                "final_public_state": "RUNTIME_SUCCESS",
                "case_completion_reason": "runtime_success",
                "restart_required": False,
                "final_latency_ms": 12.0,
                "runtime_latency_ms": 6.0,
                "error_class": None,
            },
            {
                "final_public_state": "EXECUTION_ERROR",
                "case_completion_reason": None,
                "restart_required": True,
                "final_latency_ms": None,
                "runtime_latency_ms": None,
                "error_class": "RuntimeError",
            },
        ],
        reproducibility_checks=[
            {"stable": True},
            {"stable": False},
        ],
        persistence_checks=[
            {"restored": True},
            {"restored": True},
        ],
    )

    assert aggregate_blocks["robustness_metrics"]["request_count"] == 2
    assert aggregate_blocks["robustness_metrics"]["error_rate"]["mean"] == 0.5
    assert aggregate_blocks["robustness_metrics"]["excluded_from_semantic_tables"] is True


def test_build_replay_summary_includes_aggregate_validation(tmp_path):
    baseline_catalog = load_catalog()
    handle = SimpleNamespace(
        execution_mode="in_process_conversational",
        config=SimpleNamespace(lm_studio_api_base="http://localhost:1234"),
    )

    summary = build_replay_summary(
        run_id="demo",
        command="python run_part2_replay_robustness_report.py",
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
        corpus={
            "corpus_version": "part2_tier_d_bank_replay_v1",
            "corpus_sha256": "hash",
            "replay_request_count": 1000,
            "source_request_count": 290,
            "source_case_count": 250,
        },
        tier_d_corpus_path=tmp_path / "tier_d.json",
        replay_results=[
            {
                "final_public_state": "RUNTIME_SUCCESS",
                "case_completion_reason": "runtime_success",
                "restart_required": False,
                "final_latency_ms": 10.0,
                "runtime_latency_ms": 4.0,
                "error_class": None,
            }
        ],
        reproducibility_checks=[{"stable": True}],
        persistence_checks=[{"restored": True}],
        benchmark_path=tmp_path / "bench.yaml",
    )

    assert summary["runner_scope"] == "part2_tier_d_replay_robustness"
    assert summary["corpus_path"].endswith("tier_d.json")
    assert summary["robustness_metrics"]["excluded_from_semantic_tables"] is True
    assert summary["aggregate_validation"]["ok"] is True
    assert Path(summary["report_json_path"]).name == "replay_robustness_report.json"
