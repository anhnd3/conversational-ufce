from __future__ import annotations

import json
from types import SimpleNamespace

from llm.src.phase2.taxonomy import PRIMARY_ACCEPTANCE_TARGET
from scripts.run_part2_bank_evidence_pack import (
    AttemptRunOutcome,
    PrimaryCaseOutcome,
    SupplementalCaseOutcome,
    build_attempt_summary,
    build_pack_manifest,
    build_runner_command,
    evaluate_primary_case_semantics,
    evaluate_supplemental_case_semantics,
    meets_primary_acceptance_target,
    write_case_study_indexes,
    write_pack_status,
)


def make_primary_result(*, stage, summary_type, reason_codes=None, changed_fields=None, missing_fields=None):
    explanation_payload = None
    clarification_payload = None
    if summary_type is not None:
        explanation_payload = SimpleNamespace(
            summary_type=summary_type,
            reason_codes=list(reason_codes or []),
            changed_fields=list(changed_fields or []),
            counterfactual_summary=None if summary_type != "counterfactual_found" else {"rank": 1},
        )
    if missing_fields is not None:
        clarification_payload = SimpleNamespace(
            clarification_type="missing_information",
            missing_fields=list(missing_fields),
            conflicts=[],
        )
    return SimpleNamespace(
        stage=stage,
        explanation_payload=explanation_payload,
        clarification_payload=clarification_payload,
        turn_id="run_001",
        artifact_record=SimpleNamespace(
            merge_applied=True,
            parent_turn_id="run_000",
            carried_fields=["Income", "CCAvg", "Family", "Education", "Mortgage"],
        ),
    )


def test_primary_clarification_acceptance_requires_exact_missing_fields():
    case = SimpleNamespace(
        expected_label="clarification",
        accept={
            "clarification_type": "missing_information",
            "missing_fields": ["SecuritiesAccount", "CDAccount", "Online", "CreditCard"],
        },
    )
    result = make_primary_result(
        stage="NEEDS_CLARIFICATION",
        summary_type=None,
        missing_fields=["SecuritiesAccount", "CDAccount", "Online", "CreditCard"],
    )
    assert evaluate_primary_case_semantics(case, result) == []

    bad_result = make_primary_result(
        stage="NEEDS_CLARIFICATION",
        summary_type=None,
        missing_fields=["CDAccount", "Online"],
    )
    assert "missing_fields mismatch" in evaluate_primary_case_semantics(case, bad_result)


def test_primary_runtime_reject_acceptance_uses_exact_unordered_reason_set():
    case = SimpleNamespace(
        expected_label="runtime_reject",
        accept={"reason_codes": ["NO_FEASIBLE_CF_FOUND"]},
    )
    result = make_primary_result(
        stage="RUNTIME_REJECT",
        summary_type="runtime_reject",
        reason_codes=["NO_FEASIBLE_CF_FOUND"],
    )
    assert evaluate_primary_case_semantics(case, result) == []

    bad_result = make_primary_result(
        stage="RUNTIME_REJECT",
        summary_type="runtime_reject",
        reason_codes=["UFCE_EXECUTION_ERROR"],
    )
    assert "runtime reject reason code set mismatch" in evaluate_primary_case_semantics(case, bad_result)


def test_supplemental_acceptance_requires_merge_parent_and_carried_fields():
    case = SimpleNamespace(
        accept={
            "supplemental_type": "supplemental_followup_merge_to_success",
            "turn1_stage": "NEEDS_CLARIFICATION",
            "carried_fields": ["Income", "CCAvg", "Family", "Education", "Mortgage"],
            "final_label": "counterfactual_found",
        }
    )
    turn1 = make_primary_result(stage="NEEDS_CLARIFICATION", summary_type=None, missing_fields=["CDAccount"])
    turn1.turn_id = "run_111"
    turn2 = make_primary_result(stage="RUNTIME_SUCCESS", summary_type="counterfactual_found", changed_fields=["Income"])
    turn2.artifact_record = SimpleNamespace(
        merge_applied=True,
        parent_turn_id="run_111",
        carried_fields=["Income", "CCAvg", "Family", "Education", "Mortgage"],
    )
    assert evaluate_supplemental_case_semantics(case, [turn1, turn2]) == []

    broken_turn2 = make_primary_result(stage="RUNTIME_SUCCESS", summary_type="counterfactual_found", changed_fields=["Income"])
    broken_turn2.artifact_record = SimpleNamespace(
        merge_applied=False,
        parent_turn_id="wrong_turn",
        carried_fields=["Income"],
    )
    errors = evaluate_supplemental_case_semantics(case, [turn1, broken_turn2])
    assert "turn2 merge_applied must be true" in errors
    assert "turn2 parent_turn_id must match turn1 turn_id" in errors
    assert "turn2 carried_fields mismatch" in errors


def test_build_summaries_and_pack_status(tmp_path):
    primary_outcomes = [
        PrimaryCaseOutcome(
            case_id="P-NR-01",
            slug="no_recourse",
            expected_label="no_recourse_needed",
            accepted=True,
            accepted_case_dir=str((tmp_path / "accepted" / "primary" / "P-NR-01__no_recourse").resolve()),
            runs=[
                AttemptRunOutcome(
                    case_id="P-NR-01",
                    slug="no_recourse",
                    run_name="run1",
                    expected_label="no_recourse_needed",
                    actual_label="no_recourse_needed",
                    stage="RUNTIME_SUCCESS",
                    accepted=True,
                    acceptance_errors=[],
                    artifact_dir=None,
                ),
                AttemptRunOutcome(
                    case_id="P-NR-01",
                    slug="no_recourse",
                    run_name="run2",
                    expected_label="no_recourse_needed",
                    actual_label="no_recourse_needed",
                    stage="RUNTIME_SUCCESS",
                    accepted=True,
                    acceptance_errors=[],
                    artifact_dir=None,
                ),
            ],
            rejection_reasons=[],
        ),
        PrimaryCaseOutcome(
            case_id="P-CL-01",
            slug="clarification",
            expected_label="clarification",
            accepted=False,
            accepted_case_dir=None,
            runs=[
                AttemptRunOutcome(
                    case_id="P-CL-01",
                    slug="clarification",
                    run_name="run1",
                    expected_label="clarification",
                    actual_label="clarification",
                    stage="NEEDS_CLARIFICATION",
                    accepted=False,
                    acceptance_errors=["missing_fields mismatch"],
                    artifact_dir=None,
                ),
                AttemptRunOutcome(
                    case_id="P-CL-01",
                    slug="clarification",
                    run_name="run2",
                    expected_label="clarification",
                    actual_label="clarification",
                    stage="NEEDS_CLARIFICATION",
                    accepted=False,
                    acceptance_errors=["missing_fields mismatch"],
                    artifact_dir=None,
                ),
            ],
            rejection_reasons=["missing_fields mismatch"],
        ),
    ]
    supplemental_outcomes = [
        SupplementalCaseOutcome(
            case_id="S-MERGE-01",
            slug="followup",
            expected_label="supplemental_followup",
            supplemental_type="supplemental_followup_merge_to_success",
            accepted=True,
            accepted_case_dir=str((tmp_path / "accepted" / "supplemental" / "S-MERGE-01__followup").resolve()),
            final_label="counterfactual_found",
            final_stage="RUNTIME_SUCCESS",
            merge_applied=True,
            parent_turn_id="run_1",
            carried_fields=["Income"],
            turn_dirs=[],
            rejection_reasons=[],
        )
    ]
    catalog = SimpleNamespace(catalog_version="phase2_bank_catalog_v1")
    indexes_root = tmp_path / "indexes"
    attempt_summary = build_attempt_summary(
        pack_version="phase2_pack_demo",
        catalog=catalog,
        primary_outcomes=primary_outcomes,
        supplemental_outcomes=supplemental_outcomes,
        indexes_root=indexes_root,
    )
    manifest = build_pack_manifest(
        pack_version="phase2_pack_demo",
        catalog=catalog,
        provenance={"scenario_catalog_version": "phase2_bank_catalog_v1"},
        primary_outcomes=primary_outcomes,
        supplemental_outcomes=supplemental_outcomes,
        indexes_root=indexes_root,
    )

    assert attempt_summary["accepted_counts"]["primary_cases"] == 1
    assert manifest["accepted_counts"]["primary_by_label"] == {"no_recourse_needed": 1}
    assert manifest["accepted_counts"]["supplemental_cases"] == 1

    write_pack_status(tmp_path, pack_version="phase2_pack_demo", status="failed", failure_reason="demo")
    pack_status = json.loads((tmp_path / "pack_status.json").read_text(encoding="utf-8"))
    assert pack_status["pack_version"] == "phase2_pack_demo"
    assert pack_status["status"] == "failed"
    assert pack_status["failure_reason"] == "demo"


def test_write_case_study_indexes_reads_accepted_root_recursively(tmp_path):
    accepted_root = tmp_path / "accepted"
    primary_case = accepted_root / "primary" / "P-NR-01__no_recourse"
    supplemental_turn2 = accepted_root / "supplemental" / "S-MERGE-01__followup" / "turn2"
    primary_case.mkdir(parents=True, exist_ok=True)
    supplemental_turn2.mkdir(parents=True, exist_ok=True)

    (primary_case / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "turn_id": "run_001",
                "stage": "RUNTIME_SUCCESS",
                "model_alias": "qwen/qwen3-14b",
                "timestamp_utc": "2026-03-21T00:00:00Z",
                "command": "python demo.py",
                "session_id": None,
                "turn_index": None,
                "parent_turn_id": None,
                "merge_applied": False,
                "carried_fields": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (primary_case / "turn_result.json").write_text(
        json.dumps({"explanation_payload": {"summary_type": "no_recourse_needed"}}) + "\n",
        encoding="utf-8",
    )
    (supplemental_turn2 / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "turn_id": "run_002",
                "stage": "RUNTIME_SUCCESS",
                "model_alias": "qwen/qwen3-14b",
                "timestamp_utc": "2026-03-21T00:00:05Z",
                "command": "python demo.py",
                "session_id": "S-MERGE-01",
                "turn_index": 2,
                "parent_turn_id": "run_001",
                "merge_applied": True,
                "carried_fields": ["Income", "Family"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (supplemental_turn2 / "turn_result.json").write_text(
        json.dumps({"explanation_payload": {"summary_type": "counterfactual_found"}}) + "\n",
        encoding="utf-8",
    )

    indexes_root = tmp_path / "indexes"
    write_case_study_indexes(accepted_root, indexes_root)

    records = json.loads((indexes_root / "case_studies_index.json").read_text(encoding="utf-8"))
    assert {record["case_label"] for record in records} == {"no_recourse_needed", "supplemental_followup"}


def test_meets_primary_acceptance_target_requires_exact_distribution():
    assert meets_primary_acceptance_target(dict(PRIMARY_ACCEPTANCE_TARGET)) is True
    assert meets_primary_acceptance_target({"no_recourse_needed": 2}) is False


def test_build_runner_command_includes_pack_and_catalog():
    class Args:
        model_alias = "qwen/qwen3-14b"
        api_base = "http://localhost:1234"
        timeout_s = 120.0
        benchmark = "llm_eval/benchmarks/ufce_bank_cf_parser_benchmark_v1.yaml"
        system_prompt = "llm/prompts/parser_system_prompt_v1.txt"
        catalog = "docs/thesis/part2/catalogs/phase2_bank_catalog_v1.json"
        out_dir = "outputs/conversations_phase2_pack"
        debug_trace = True

    command = build_runner_command(Args(), pack_version="phase2_pack_demo")
    assert "--catalog docs/thesis/part2/catalogs/phase2_bank_catalog_v1.json" in command
    assert "--pack-version phase2_pack_demo" in command
    assert "--debug-trace" in command
