from __future__ import annotations

from pathlib import Path

from llm.src.part2_eval.p4_portability import (
    P4A_REQUIRED_MATRIX,
    REFINEMENT_SMOKE_CASES,
    classify_primary_failure_stage,
    classify_refinement_failure_stage,
    compute_enabled_combinations,
    render_portability_markdown,
    summarize_backend_conformance,
    summarize_dataset_conformance,
)


def test_dataset_conformance_reports_bank_and_grad_as_supported():
    summary = summarize_dataset_conformance(["bank", "grad"])

    assert summary["bank"]["pass"] is True
    assert summary["grad"]["pass"] is True
    assert summary["bank"]["checks"]["live_runtime_enabled"] is True
    assert summary["grad"]["checks"]["numeric_bound_fields_present"] is True
    assert "ufce" in summary["bank"]["supported_backends"]
    assert "dice" in summary["grad"]["supported_backends"]


def test_backend_conformance_reports_ufce_dice_and_ar():
    summary = summarize_backend_conformance(["ufce", "dice", "ar"])

    assert summary["ufce"]["pass"] is True
    assert summary["dice"]["pass"] is True
    assert summary["ar"]["pass"] is True
    assert summary["ufce"]["checks"]["request_contract_v1"] is True
    assert summary["ar"]["checks"]["candidate_contract_v1"] is True


def test_compute_enabled_combinations_reflects_manifest_support():
    combinations = compute_enabled_combinations(["bank", "grad"], ["ufce", "dice", "ar"])
    enabled = {row["combination"]: row["enabled"] for row in combinations}

    assert enabled["bank+ufce"] is True
    assert enabled["bank+dice"] is True
    assert enabled["grad+ar"] is True


def test_primary_failure_classification_maps_known_states():
    assert classify_primary_failure_stage(public_state="PARSER_FAILURE") == "parser"
    assert classify_primary_failure_stage(public_state="NEEDS_CLARIFICATION") == "canonical_validation"
    assert (
        classify_primary_failure_stage(
            public_state="RUNTIME_SUCCESS",
            builder_status="READY_FOR_RUNTIME",
            runtime_executed=True,
        )
        == "explanation"
    )


def test_refinement_failure_classification_maps_known_states():
    assert (
        classify_refinement_failure_stage(
            public_state="RUNTIME_SUCCESS",
            refinement_status="clarification_required",
            runtime_executed=False,
        )
        == "refinement_parser"
    )
    assert (
        classify_refinement_failure_stage(
            public_state="RUNTIME_REJECT",
            refinement_status="limit_reached",
            runtime_executed=False,
            limit_reached=True,
        )
        == "refinement_application"
    )


def test_refinement_smoke_cases_use_deterministic_clarification_prompts():
    assert REFINEMENT_SMOKE_CASES["bank"][1]["feedbacks"] == (
        "Do not change Income, actually Income can change.",
    )
    assert REFINEMENT_SMOKE_CASES["grad"][1]["feedbacks"] == (
        "Do not change CGPA, actually CGPA can change.",
    )


def test_render_portability_markdown_includes_required_sections(tmp_path):
    summary = {
        "milestone": "P4a",
        "generated_at": "2026-04-01T00:00:00+00:00",
        "datasets": ["bank", "grad"],
        "backends": ["ufce", "dice"],
        "all_required_primary_passed": True,
        "all_required_refinement_passed": None,
        "dataset_conformance": {
            "bank": {"pass": True, "supported_backends": ["ufce", "dice", "ar"]},
            "grad": {"pass": True, "supported_backends": ["ufce", "dice", "ar"]},
        },
        "backend_conformance": {
            "ufce": {
                "pass": True,
                "manifest": {
                    "request_contract_version": "canonical_request_v1",
                    "candidate_contract_version": "canonical_candidate_v1",
                },
            },
            "dice": {
                "pass": True,
                "manifest": {
                    "request_contract_version": "canonical_request_v1",
                    "candidate_contract_version": "canonical_candidate_v1",
                },
            },
        },
        "enabled_combinations": [
            {"combination": "bank+ufce", "enabled": True},
            {"combination": "grad+ufce", "enabled": True},
        ],
        "primary_results": {
            "bank+ufce": {"pass": True, "cases": []},
            "grad+ufce": {"pass": True, "cases": []},
        },
        "refinement_results": None,
    }

    markdown = render_portability_markdown(summary)

    assert "# P4a Portability Report" in markdown
    assert "## Dataset Conformance" in markdown
    assert "## Backend Conformance" in markdown
    assert "bank+ufce" in markdown
    assert "grad+ufce" in markdown
    assert P4A_REQUIRED_MATRIX[0][0] == "bank"
    assert Path(tmp_path).exists()
