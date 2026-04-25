from __future__ import annotations

import pytest

from llm.src.phase3.catalog import DEFAULT_CATALOG_PATH, load_catalog, validate_catalog_payload


def test_load_catalog_reads_phase31_validation_catalog():
    catalog = load_catalog(DEFAULT_CATALOG_PATH)

    assert catalog.catalog_version == "phase3_1_validation_catalog_v2"
    assert len(catalog.scenarios) == 9
    assert catalog.get_scenario("P31-NR-01").expected_final_state == "RUNTIME_SUCCESS"
    assert catalog.get_scenario("P31-UN-01").accept["template_type"] == "unsupported_request"
    assert catalog.get_scenario("P31-RJ-01").accept["included_suggestion_types"] == [
        "revise_target_profile",
        "broaden_allowed_financial_changes",
    ]
    assert catalog.get_scenario("P31-RS-01").accept["kind"] == "reset_no_merge"
    assert catalog.get_scenario("P31-RS-02").accept["expected_turn2_runtime_presence"] is True


def test_validate_catalog_rejects_missing_accept_fields():
    payload = {
        "catalog_version": "phase3_1_validation_catalog_v1",
        "runner_compat_version": "phase3_1_closeout_runner_v1",
        "created_timestamp_utc": "2026-03-21T00:00:00Z",
        "prompt_template_version": "parser_system_prompt_v1",
        "change_notes": [],
        "scenarios": [
            {
                "scenario_id": "P31-UN-01",
                "slug": "unsupported",
                "description": "unsupported",
                "turns": ["help me optimize"],
                "expected_final_state": "UNSUPPORTED_REQUEST",
                "accept": {
                    "kind": "unsupported",
                    "runtime_result_absent": True,
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="template_type"):
        validate_catalog_payload(payload)


def test_validate_catalog_rejects_reset_no_merge_without_runtime_presence():
    payload = {
        "catalog_version": "phase3_1_validation_catalog_v2",
        "runner_compat_version": "phase3_1_closeout_runner_v1",
        "created_timestamp_utc": "2026-03-21T00:00:00Z",
        "prompt_template_version": "parser_system_prompt_v1",
        "change_notes": [],
        "scenarios": [
            {
                "scenario_id": "P31-RS-01",
                "slug": "reset",
                "description": "reset",
                "turns": ["one", "two"],
                "expected_final_state": "UNSUPPORTED_REQUEST",
                "accept": {
                    "kind": "reset_no_merge",
                    "turn1_final_state": "NEEDS_CLARIFICATION",
                    "turn2_final_state": "UNSUPPORTED_REQUEST",
                    "forbidden_turn2_fields": ["Income"],
                },
            }
        ],
    }

    with pytest.raises(ValueError, match="expected_turn2_runtime_presence"):
        validate_catalog_payload(payload)
