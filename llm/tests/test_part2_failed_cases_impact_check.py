from __future__ import annotations

import pandas as pd

from scripts.final.part2.failed_cases_impact_check import (
    EXPECTED_FAILED_CASE_COUNT,
    build_final_vs_full250_table,
    build_method_summary_table,
    collapse_primary_failed_field,
    derive_failed_field,
    derive_failure_family,
    derive_final_status,
    filter_result_frame_to_selected_cases,
)


def test_derive_final_status_covers_conflict_reconstruction_and_parser_failure() -> None:
    assert derive_final_status({"final_stage": "CONFLICT", "ready_for_runtime": False, "exact_reconstruction": False}) == "conflict"
    assert derive_final_status({"final_stage": "READY_FOR_RUNTIME", "ready_for_runtime": True, "exact_reconstruction": False}) == "reconstruction_mismatch"
    assert derive_final_status({"final_stage": "PARSER_FAILURE", "ready_for_runtime": False, "exact_reconstruction": False}) == "parser_failure"


def test_derive_failure_family_prefers_semantic_family_then_conflict_then_quote_alignment() -> None:
    semantic_row = {
        "field_semantic_mismatch_json": '[{"failure_family":"field_aware_number_prefix_collision","field_name":"Income"}]',
        "final_stage": "READY_FOR_RUNTIME",
        "retry_errors_json": "[]",
        "schema_errors_json": "[]",
        "verifier_errors_json": "[]",
    }
    conflict_row = {
        "field_semantic_mismatch_json": "[]",
        "final_stage": "CONFLICT",
        "retry_errors_json": "[]",
        "schema_errors_json": "[]",
        "verifier_errors_json": "[]",
    }
    quote_row = {
        "field_semantic_mismatch_json": "[]",
        "final_stage": "PARSER_FAILURE",
        "retry_errors_json": '["fields.Mortgage.evidence_quote is not an exact substring of the input."]',
        "schema_errors_json": "[]",
        "verifier_errors_json": "[]",
    }

    assert derive_failure_family(semantic_row) == (
        "field_aware_number_prefix_collision",
        "field_aware_number_prefix_collision",
    )
    assert derive_failure_family(conflict_row) == ("semantic_conflict", "semantic_conflict")
    assert derive_failure_family(quote_row) == ("quote_alignment_error", "quote_alignment_error")


def test_derive_failed_field_supports_semantic_conflict_text_and_global_fallback() -> None:
    semantic_row = {
        "field_semantic_mismatch_json": '[{"field_name":"Income","failure_family":"field_aware_number_prefix_collision"}]',
        "confirmed_conflicts_json": "[]",
        "reconstruction_mismatch_fields_json": "[]",
    }
    conflict_row = {
        "field_semantic_mismatch_json": "[]",
        "confirmed_conflicts_json": '["Income value interpretation ambiguity and CCAvg ambiguity."]',
        "reconstruction_mismatch_fields_json": '["Income","CCAvg"]',
    }
    fallback_row = {
        "field_semantic_mismatch_json": "[]",
        "confirmed_conflicts_json": "[]",
        "reconstruction_mismatch_fields_json": '["Income","Mortgage"]',
    }

    assert derive_failed_field(semantic_row) == ("Income", "Income")
    assert derive_failed_field(conflict_row) == ("Income;CCAvg", "Income")
    assert derive_failed_field(fallback_row) == ("MULTI_FIELD_OR_GLOBAL", None)
    assert collapse_primary_failed_field(None) == "Khác"


def test_filter_result_frame_to_selected_cases_keeps_only_accepted_rows() -> None:
    result_df = pd.DataFrame(
        [
            {"case_id": "c1", "method": "UFCE1", "status": "released_cf"},
            {"case_id": "c1", "method": "UFCE2", "status": "no_valid_cf"},
            {"case_id": "c1", "method": "UFCE3", "status": "no_valid_cf"},
            {"case_id": "c2", "method": "UFCE1", "status": "released_cf"},
            {"case_id": "c2", "method": "UFCE2", "status": "released_cf"},
            {"case_id": "c2", "method": "UFCE3", "status": "released_cf"},
        ]
    )

    filtered_df = filter_result_frame_to_selected_cases(
        result_df=result_df,
        included_case_ids={"c1"},
        expected_case_count=1,
    )

    assert filtered_df["case_id"].tolist() == ["c1", "c1", "c1"]
    assert filtered_df["method"].tolist() == ["UFCE1", "UFCE2", "UFCE3"]


def test_build_method_summary_table_computes_release_loss_and_failed_case_rates() -> None:
    full250_direct_df = pd.DataFrame(
        [
            {"case_id": "c1", "method": "UFCE1", "status": "released_cf"},
            {"case_id": "c1", "method": "UFCE2", "status": "no_valid_cf"},
            {"case_id": "c1", "method": "UFCE3", "status": "no_valid_cf"},
            {"case_id": "c2", "method": "UFCE1", "status": "released_cf"},
            {"case_id": "c2", "method": "UFCE2", "status": "released_cf"},
            {"case_id": "c2", "method": "UFCE3", "status": "released_cf"},
        ]
    )
    filtered_accepted_nl_first_df = pd.DataFrame(
        [
            {"case_id": "c1", "method": "UFCE1", "status": "released_cf"},
            {"case_id": "c1", "method": "UFCE2", "status": "no_valid_cf"},
            {"case_id": "c1", "method": "UFCE3", "status": "no_valid_cf"},
        ]
    )
    failed_cases_df = pd.DataFrame(
        [
            {"has_valid_cf_ufce1": True, "has_valid_cf_ufce2": False, "has_valid_cf_ufce3": False},
            {"has_valid_cf_ufce1": False, "has_valid_cf_ufce2": False, "has_valid_cf_ufce3": False},
            {"has_valid_cf_ufce1": False, "has_valid_cf_ufce2": False, "has_valid_cf_ufce3": False},
            {"has_valid_cf_ufce1": False, "has_valid_cf_ufce2": False, "has_valid_cf_ufce3": False},
            {"has_valid_cf_ufce1": False, "has_valid_cf_ufce2": False, "has_valid_cf_ufce3": False},
            {"has_valid_cf_ufce1": False, "has_valid_cf_ufce2": False, "has_valid_cf_ufce3": False},
            {"has_valid_cf_ufce1": False, "has_valid_cf_ufce2": False, "has_valid_cf_ufce3": False},
        ]
    )

    summary_df = build_method_summary_table(
        full250_direct_df=full250_direct_df,
        filtered_accepted_nl_first_df=filtered_accepted_nl_first_df,
        failed_cases_df=failed_cases_df,
    )
    rows = {row["method"]: row for row in summary_df.to_dict(orient="records")}

    assert rows["UFCE1"]["full250_direct_released_cf_count"] == 2
    assert rows["UFCE1"]["accepted243_nl_first_released_cf_count"] == 1
    assert rows["UFCE1"]["lost_released_cf_count"] == 1
    assert rows["UFCE1"]["failed_cases_with_valid_cf"] == 1
    assert rows["UFCE1"]["failed_cases_with_valid_cf_rate"] == 1 / EXPECTED_FAILED_CASE_COUNT
    assert rows["UFCE2"]["lost_released_cf_count"] == 1
    assert rows["UFCE3"]["lost_released_cf_count"] == 1


def test_build_final_vs_full250_table_preserves_denominators_and_fold_mean_counts() -> None:
    full250_direct_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "fold_index": 0,
                "method": "UFCE1",
                "status": "released_cf",
                "selected_prox_jac": 0.5,
                "selected_prox_euc": 2.0,
                "selected_sparsity": 1.0,
                "selected_actionability_pass": 1.0,
                "selected_plausibility_pass": 1.0,
                "selected_feasibility_pass": 1.0,
            },
            {
                "case_id": "c2",
                "fold_index": 1,
                "method": "UFCE1",
                "status": "released_cf",
                "selected_prox_jac": 1.0,
                "selected_prox_euc": 4.0,
                "selected_sparsity": 2.0,
                "selected_actionability_pass": 1.0,
                "selected_plausibility_pass": 0.0,
                "selected_feasibility_pass": 1.0,
            },
            {"case_id": "c1", "fold_index": 0, "method": "UFCE2", "status": "no_valid_cf"},
            {"case_id": "c2", "fold_index": 1, "method": "UFCE2", "status": "no_valid_cf"},
            {"case_id": "c1", "fold_index": 0, "method": "UFCE3", "status": "no_valid_cf"},
            {"case_id": "c2", "fold_index": 1, "method": "UFCE3", "status": "no_valid_cf"},
        ]
    )
    accepted_nl_first_df = full250_direct_df.loc[full250_direct_df["case_id"] == "c1"].copy()

    table_df = build_final_vs_full250_table(
        full250_direct_df=full250_direct_df,
        filtered_accepted_nl_first_df=accepted_nl_first_df,
        source_case_count=2,
        accepted_case_count=1,
        dropped_case_count=1,
        fold_count=2,
    )
    rows = {
        (str(row["variant"]), str(row["method"])): row
        for row in table_df.to_dict(orient="records")
    }

    full_ufce1 = rows[("direct_full250", "UFCE1")]
    accepted_ufce1 = rows[("nl_first_accepted243", "UFCE1")]
    assert full_ufce1["released_cf"] == 2
    assert full_ufce1["release_per_eval"] == 1.0
    assert full_ufce1["prox_euc"] == 3.0
    assert full_ufce1["actionability"] == 1.0
    assert full_ufce1["plausibility"] == 0.5
    assert accepted_ufce1["released_cf"] == 1
    assert accepted_ufce1["release_per_eval"] == 1.0
    assert accepted_ufce1["release_per_250"] == 0.5
    assert accepted_ufce1["actionability"] == 0.5
