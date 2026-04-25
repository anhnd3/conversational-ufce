from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

from scripts import run_part2_golden_parity_report as runner


def test_classify_case_delta_marks_unexpected_regression_for_kind_mismatch():
    classification, reason = runner.classify_case_delta(
        expected_kind="runtime_success_counterfactual",
        actual_kind="runtime_reject_infeasible",
        payload={},
        backend_id="ufce",
    )

    assert classification == runner.UNEXPECTED_REGRESSION
    assert "expected kind" in reason


def test_classify_case_delta_marks_expected_verifier_delta_when_invalid_candidates_present():
    classification, reason = runner.classify_case_delta(
        expected_kind="runtime_success_counterfactual",
        actual_kind="runtime_success_counterfactual",
        payload={
            "verification_results": [
                {"candidate_id": "ufce:sfexp:1", "is_valid": False, "reason_codes": ["NO_FLIP"]},
                {"candidate_id": "ufce:sfexp:2", "is_valid": True, "reason_codes": []},
            ],
            "counterfactual": {
                "candidates": [
                    {"method": "sfexp", "rank": 2},
                ]
            },
            "canonical_candidates": [
                {"candidate_id": "ufce:sfexp:1"},
                {"candidate_id": "ufce:sfexp:2"},
            ],
            "backend_id": "ufce",
        },
        backend_id="ufce",
    )

    assert classification == runner.EXPECTED_VERIFIER_DELTA
    assert "verifier filtered" in reason


def test_run_golden_parity_report_allows_written_waiver_for_unexpected_regression(monkeypatch, tmp_path):
    corpus_path = tmp_path / "golden.json"
    corpus_path.write_text(
        json.dumps(
            {
                "corpus_version": "part2_bank_golden_parity_v1",
                "dataset": "bank",
                "cases": [
                    {
                        "case_id": "BANK-GOLDEN-001",
                        "kind": "runtime_success_no_recourse",
                        "runtime_request": {"dataset": "bank", "profile": {"Income": 1}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    waiver_path = tmp_path / "waiver.md"
    waiver_path.write_text("Approved verifier migration delta.", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "evaluate_case",
        lambda **kwargs: {
            "case_id": "BANK-GOLDEN-001",
            "expected_kind": "runtime_success_no_recourse",
            "actual_kind": "unexpected:TERMINAL_REJECT",
            "delta_classification": runner.UNEXPECTED_REGRESSION,
            "delta_reason": "forced regression",
        },
    )
    monkeypatch.setattr(runner, "local_now_compact", lambda: "20260329_120000")
    monkeypatch.setattr(runner, "local_now_iso", lambda: "2026-03-29T12:00:00+07:00")

    summary = runner.run_golden_parity_report(
        args=Namespace(
            out_dir=tmp_path / "out",
            backend="ufce",
            dataset="bank",
            golden_corpus=corpus_path,
            unexpected_regression_waiver=waiver_path,
            no_progress=True,
            summary_json=None,
            summary_md=None,
        ),
        command="python scripts/run_part2_golden_parity_report.py",
    )

    assert summary["unexpected_regression_count"] == 1
    assert summary["unexpected_regression_waiver"]["applied"] is True
    assert summary["parity_ok"] is True
    assert Path(summary["report_json_path"]).exists()
