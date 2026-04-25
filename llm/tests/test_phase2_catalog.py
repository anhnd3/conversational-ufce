from __future__ import annotations

import pytest

from llm.src.phase2.catalog import DEFAULT_CATALOG_PATH, load_catalog, validate_catalog_payload


def test_load_catalog_reads_locked_phase2_catalog():
    catalog = load_catalog(DEFAULT_CATALOG_PATH)

    assert catalog.catalog_version == "phase2_bank_catalog_v2"
    assert len(catalog.primary_cases) == 8
    assert len(catalog.supplemental_cases) == 2
    assert len(catalog.smoke_only_cases) == 1
    assert catalog.get_case("P-NR-01").expected_label == "no_recourse_needed"
    assert catalog.get_case("P-RJ-02").turns == (
        "My target profile is Income 55, CCAvg 2.0, Family 2, Education 1, Mortgage 0, CDAccount yes, Online yes, SecuritiesAccount no, and CreditCard yes.",
    )
    assert catalog.get_case("S-MERGE-01").accept["supplemental_type"] == "supplemental_followup_merge_to_success"


def test_validate_catalog_rejects_missing_supplemental_accept_metadata():
    payload = {
        "catalog_version": "phase2_bank_catalog_v1",
        "runner_compat_version": "phase2_pack_runner_v1",
        "created_timestamp_utc": "2026-03-21T00:00:00Z",
        "prompt_template_version": "parser_system_prompt_v1",
        "change_notes": [],
        "primary_cases": [
            {
                "case_id": "P-NR-01",
                "slug": "demo",
                "description": "demo",
                "expected_label": "no_recourse_needed",
                "turns": ["demo"],
                "accept": {"reason_codes": ["NO_RECOURSE_NEEDED"]},
            }
        ],
        "supplemental_cases": [
            {
                "case_id": "S-MERGE-01",
                "slug": "supp",
                "description": "supp",
                "expected_label": "supplemental_followup",
                "turns": ["one", "two"],
                "accept": {"turn1_stage": "NEEDS_CLARIFICATION"},
            }
        ],
        "smoke_only_cases": [
            {
                "case_id": "SM-RESET-01",
                "slug": "smoke",
                "description": "smoke",
                "expected_label": "counterfactual_found",
                "turns": ["one", "two"],
                "accept": {"turn1_stage": "NEEDS_CLARIFICATION", "final_label": "counterfactual_found"},
            }
        ],
    }

    with pytest.raises(ValueError, match="supplemental_type"):
        validate_catalog_payload(payload)
