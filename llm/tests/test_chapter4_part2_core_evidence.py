from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.archieve.build_chapter4_part2_core_evidence import (
    build_conditional_denominators,
    build_evidence_outputs,
    build_failure_taxonomy,
    normalize_session_outcome,
    validate_evidence,
    write_outputs,
)


def session_row(
    *,
    case_id: str,
    group: str,
    summary_type: str,
    reject_class: str | None = None,
    final_public_state: str = "RUNTIME_SUCCESS",
    case_completion_reason: str | None = "runtime_success",
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "session_id": f"session_{case_id}",
        "group": group,
        "final_public_state": final_public_state,
        "summary_type": summary_type,
        "reject_class": reject_class,
        "case_completion_reason": case_completion_reason,
    }


def metric_block(numerator: int = 28, denominator: int = 28) -> dict[str, object]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "mean": numerator / denominator,
        "formula": "fixture formula",
    }


def minimal_report(per_case_results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "aggregate_validation": {"ok": True},
        "run_id": "fixture_run",
        "runner_scope": "part2_g1_g2_system_eval",
        "corpus_version": "part2_tier_b_bank_sessions_v1",
        "corpus_sha256": "fixture_sha",
        "system_metrics": {},
        "g2_metrics": {
            "M8_validity_success_rate": metric_block(),
            "M11_actionability": metric_block(),
            "M12_plausibility": metric_block(),
            "M13_feasibility": metric_block(),
            "M14_constraint_satisfaction": metric_block(),
        },
        "per_case_results": per_case_results,
    }


def full_200_case_fixture() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(1, 28):
        rows.append(session_row(case_id=f"TIERB-G1-CF-{index:03d}", group="G1", summary_type="counterfactual_found"))
    for index in range(1, 5):
        rows.append(session_row(case_id=f"TIERB-G1-NRN-{index:03d}", group="G1", summary_type="no_recourse_needed"))
    for index in range(1, 70):
        rows.append(
            session_row(
                case_id=f"TIERB-G1-REJ-{index:03d}",
                group="G1",
                summary_type="runtime_reject",
                reject_class="no_feasible_cf",
                final_public_state="RUNTIME_REJECT",
                case_completion_reason="runtime_reject",
            )
        )
    for index in range(1, 29):
        rows.append(session_row(case_id=f"TIERB-G2-CF-{index:03d}", group="G2", summary_type="counterfactual_found"))
    for index in range(1, 72):
        rows.append(
            session_row(
                case_id=f"TIERB-G2-REJ-{index:03d}",
                group="G2",
                summary_type="runtime_reject",
                reject_class="no_feasible_cf",
                final_public_state="RUNTIME_REJECT",
                case_completion_reason="runtime_reject",
            )
        )
    rows.append(
        session_row(
            case_id="TIERB-G2-BLOCK-001",
            group="G2",
            summary_type="runtime_reject",
            reject_class="request_constraints_blocked",
            final_public_state="RUNTIME_REJECT",
            case_completion_reason="runtime_reject",
        )
    )
    return rows


def test_normalize_session_outcome_separates_success_subtypes():
    counterfactual = normalize_session_outcome(
        session_row(case_id="case-cf", group="G1", summary_type="counterfactual_found")
    )
    no_recourse = normalize_session_outcome(
        session_row(case_id="case-nrn", group="G1", summary_type="no_recourse_needed")
    )
    reject = normalize_session_outcome(
        session_row(
            case_id="case-reject",
            group="G2",
            summary_type="runtime_reject",
            reject_class="no_feasible_cf",
            final_public_state="RUNTIME_REJECT",
            case_completion_reason="runtime_reject",
        )
    )

    assert counterfactual["is_counterfactual_found"] is True
    assert counterfactual["is_successful_resolution"] is True
    assert counterfactual["primary_reason"] == "counterfactual_found"
    assert no_recourse["is_counterfactual_found"] is False
    assert no_recourse["is_no_recourse_needed"] is True
    assert no_recourse["is_successful_resolution"] is True
    assert no_recourse["primary_reason"] == "no_recourse_needed"
    assert reject["is_successful_resolution"] is False
    assert reject["primary_reason"] == "no_valid_cf"


def test_failure_taxonomy_maps_runtime_reject_classes_and_keeps_zero_rows():
    rows = [
        normalize_session_outcome(
            session_row(case_id="success", group="G1", summary_type="counterfactual_found")
        ),
        normalize_session_outcome(
            session_row(
                case_id="no-cf",
                group="G1",
                summary_type="runtime_reject",
                reject_class="no_feasible_cf",
                final_public_state="RUNTIME_REJECT",
                case_completion_reason="runtime_reject",
            )
        ),
        normalize_session_outcome(
            session_row(
                case_id="blocked",
                group="G2",
                summary_type="runtime_reject",
                reject_class="request_constraints_blocked",
                final_public_state="RUNTIME_REJECT",
                case_completion_reason="runtime_reject",
            )
        ),
    ]

    taxonomy = {row["primary_reason"]: row for row in build_failure_taxonomy(rows)}

    assert taxonomy["no_valid_cf"]["n_sessions"] == 1
    assert taxonomy["constraint_blocked"]["n_sessions"] == 1
    assert taxonomy["parser_schema_failure"]["n_sessions"] == 0
    assert taxonomy["unsupported"]["n_sessions"] == 0
    assert taxonomy["conflict"]["n_sessions"] == 0
    assert taxonomy["clarification_exhausted"]["n_sessions"] == 0
    assert taxonomy["other_runtime_reject"]["n_sessions"] == 0


def test_conditional_denominators_use_g2_metric_scope():
    rows = build_conditional_denominators(minimal_report([]))

    assert [row["metric"] for row in rows] == [
        "validity",
        "actionability",
        "plausibility",
        "feasibility",
        "constraint_satisfaction",
    ]
    assert {row["scope"] for row in rows} == {"G2_exposed_counterfactuals"}
    assert {row["n_denominator"] for row in rows} == {28}
    assert {row["n_pass"] for row in rows} == {28}


def test_export_outputs_validate_expected_200_session_scope(tmp_path: Path):
    outputs = build_evidence_outputs(minimal_report(full_200_case_fixture()), source_report=tmp_path / "source.json")

    validate_evidence(outputs)
    write_outputs(outputs, out_dir=tmp_path)

    with (tmp_path / "part2_session_outcomes_normalized.csv").open(newline="", encoding="utf-8") as handle:
        session_rows = list(csv.DictReader(handle))
    with (tmp_path / "part2_failure_taxonomy.csv").open(newline="", encoding="utf-8") as handle:
        failure_rows = list(csv.DictReader(handle))
    with (tmp_path / "part2_conditional_quality_denominators.csv").open(newline="", encoding="utf-8") as handle:
        denominator_rows = list(csv.DictReader(handle))

    assert len(session_rows) == 200
    assert sum(row["is_successful_resolution"] == "True" for row in session_rows) == 59
    assert sum(int(row["n_sessions"]) for row in failure_rows) == 141
    assert {row["primary_reason"]: int(row["n_sessions"]) for row in failure_rows}["no_valid_cf"] == 140
    assert {row["primary_reason"]: int(row["n_sessions"]) for row in failure_rows}["constraint_blocked"] == 1
    assert {int(row["n_denominator"]) for row in denominator_rows} == {28}
    assert (tmp_path / "part2_core_evidence_summary.md").exists()


def test_validate_evidence_rejects_refinement_in_main_scope(tmp_path: Path):
    fixture = full_200_case_fixture()
    fixture[0] = session_row(case_id="TIERB-REF-001", group="REFINEMENT", summary_type="counterfactual_found")
    outputs = build_evidence_outputs(minimal_report(fixture), source_report=tmp_path / "source.json")

    with pytest.raises(ValueError, match="Expected group G1 count 100"):
        validate_evidence(outputs)
