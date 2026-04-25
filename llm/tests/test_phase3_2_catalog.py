from __future__ import annotations

import pytest

from llm.src.phase3.phase3_2_catalog import DEFAULT_CATALOG_PATH, load_catalog, validate_catalog_payload


def test_load_catalog_reads_phase32_validation_catalog():
    catalog = load_catalog(DEFAULT_CATALOG_PATH)

    assert catalog.catalog_version == "phase3_2_validation_catalog_v1"
    assert len(catalog.scenarios) == 9
    assert catalog.get_scenario("P32-NR-01").expected_final_state == "RUNTIME_SUCCESS"
    assert catalog.get_scenario("P32-UN-01").accept["template_type"] == "unsupported_request"
    assert catalog.get_scenario("P32-RJ-01").accept["included_suggestion_types"] == [
        "revise_target_profile",
        "broaden_allowed_financial_changes",
    ]


def test_validate_catalog_rejects_wrong_runner_version():
    payload = {
        "catalog_version": "phase3_2_validation_catalog_v1",
        "runner_compat_version": "phase3_1_closeout_runner_v1",
        "created_timestamp_utc": "2026-03-22T00:00:00Z",
        "prompt_template_version": "parser_system_prompt_v1",
        "change_notes": [],
        "scenarios": [
            {
                "scenario_id": "P32-NR-01",
                "slug": "no_recourse",
                "description": "no recourse",
                "turns": ["Income 140"],
                "expected_final_state": "RUNTIME_SUCCESS",
                "accept": {
                    "kind": "no_recourse_needed",
                    "summary_type": "no_recourse_needed",
                    "included_suggestion_types": [],
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="phase3_2_acceptance_runner_v1"):
        validate_catalog_payload(payload)
