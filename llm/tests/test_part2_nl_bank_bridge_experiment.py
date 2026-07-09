from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from scripts.final.part2.nl_bank_bridge_experiment import (
    BANK_FEATURE_ORDER,
    GROUP_SPECS,
    TEMPLATE_VERSION,
    apply_bank_profile_v2_evidence_hint_fallback,
    build_bank_profile_v2_evidence_hint_candidates,
    build_bank_profile_v2_field_semantic_mismatch,
    build_eval_case_selection_frame,
    build_query_frame,
    build_mismatch_reason_table,
    build_parse_acceptance_metrics,
    build_parse_fail_fast_reasons,
    build_parse_live_quality_metrics,
    copy_parse_checkpoint_artifacts,
    collect_bank_profile_v2_extraction_retry_errors,
    filter_eval_query_frame,
    format_parse_acceptance_failure,
    format_parse_live_quality_progress,
    main,
    pair_variant_results,
    profiles_equal,
    summarize_bridge_pairs,
    validate_parse_checkpoint_matches_query,
)


def test_build_query_frame_assigns_even_group_quota_per_fold() -> None:
    query_df = build_query_frame(seed=42, cases_per_group=10)

    assert len(query_df) == 250
    assert query_df["case_id"].is_unique
    assert set(query_df["language_group"]) == {spec["group_id"] for spec in GROUP_SPECS}

    counts = query_df.groupby(["fold_name", "language_group"]).size().to_dict()
    assert counts
    assert all(count == 10 for count in counts.values())
    assert set(query_df["template_version"]) == {TEMPLATE_VERSION}


def test_single_250_case_set_contains_word_number_templates_without_expansion() -> None:
    query_df = build_query_frame(seed=42, cases_per_group=10)

    assert len(query_df) == 250
    assert query_df["numeric_style"].value_counts().to_dict() == {"digits": 150, "word_numbers": 100}
    assert len(query_df.loc[query_df["language_group"] == "G2"]) == 50
    assert len(query_df.loc[query_df["language_group"] == "G4"]) == 50

    vi_word_case = str(query_df.loc[query_df["language_group"] == "G2", "user_text"].iloc[0])
    en_word_case = str(query_df.loc[query_df["language_group"] == "G4", "user_text"].iloc[0])
    assert not any(char.isdigit() for char in vi_word_case)
    assert not any(char.isdigit() for char in en_word_case)
    assert any(token in vi_word_case for token in ["mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin"])
    assert any(token in en_word_case for token in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])


def test_parse_checkpoint_validation_rejects_stale_user_text() -> None:
    query_df = build_query_frame(seed=42, cases_per_group=10).head(1).copy()
    parse_df = pd.DataFrame(
        [
            {
                "case_id": query_df.iloc[0]["case_id"],
                "user_text": "stale text from an older template",
            }
        ]
    )

    try:
        validate_parse_checkpoint_matches_query(query_df=query_df, parse_df=parse_df)
    except RuntimeError as exc:
        assert "Parse checkpoint text does not match" in str(exc)
    else:
        raise AssertionError("Expected stale parse checkpoint validation to fail.")


def test_copy_parse_checkpoint_artifacts_seeds_downstream_output_dir(tmp_path) -> None:
    source_dir = tmp_path / "parse_source"
    out_dir = tmp_path / "accepted_eval"
    source_dir.mkdir()
    out_dir.mkdir()
    (source_dir / "bank_250_queries.csv").write_text("case_id\nc1\n", encoding="utf-8")
    (source_dir / "parsed_requests.jsonl").write_text('{"case_id":"c1"}\n', encoding="utf-8")
    (source_dir / "parse_acceptance.json").write_text('{"accepted_count":1}\n', encoding="utf-8")

    copy_parse_checkpoint_artifacts(source_dir=source_dir, out_dir=out_dir)

    assert (out_dir / "bank_250_queries.csv").read_text(encoding="utf-8") == "case_id\nc1\n"
    assert (out_dir / "parsed_requests.jsonl").read_text(encoding="utf-8") == '{"case_id":"c1"}\n'
    snapshot = json.loads((out_dir / "parse_source_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["parse_source_dir"] == str(source_dir.resolve())
    assert snapshot["copied_files"] == [
        "bank_250_queries.csv",
        "parsed_requests.jsonl",
        "parse_acceptance.json",
    ]


def test_eval_case_selection_keeps_only_accepted_parser_cases_in_fold_order() -> None:
    query_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "fold_index": 0,
                "fold_name": "testfold_0_pred_0.csv",
                "query_pos": 0,
                "language_group": "G1",
                "language_label": "vi_clear",
                "benchmark_key": "vi",
            },
            {
                "case_id": "c2",
                "fold_index": 0,
                "fold_name": "testfold_0_pred_0.csv",
                "query_pos": 1,
                "language_group": "G2",
                "language_label": "vi_number_words",
                "benchmark_key": "vi",
            },
            {
                "case_id": "c3",
                "fold_index": 1,
                "fold_name": "testfold_1_pred_0.csv",
                "query_pos": 0,
                "language_group": "G3",
                "language_label": "en_clear",
                "benchmark_key": "en",
            },
        ]
    )
    parse_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "ready_for_runtime": True,
                "exact_reconstruction": True,
                "parser_status": "complete",
                "final_stage": "READY_FOR_RUNTIME",
                "parser_api_error": None,
                "repair_api_error": None,
                "verifier_api_error": None,
                "schema_errors_json": "[]",
                "verifier_errors_json": "[]",
            },
            {
                "case_id": "c2",
                "ready_for_runtime": True,
                "exact_reconstruction": False,
                "parser_status": "complete",
                "final_stage": "READY_FOR_RUNTIME",
                "parser_api_error": None,
                "repair_api_error": None,
                "verifier_api_error": None,
                "schema_errors_json": "[]",
                "verifier_errors_json": "[]",
            },
            {
                "case_id": "c3",
                "ready_for_runtime": False,
                "exact_reconstruction": False,
                "parser_status": "needs_clarification",
                "final_stage": "NEEDS_CLARIFICATION",
                "parser_api_error": None,
                "repair_api_error": None,
                "verifier_api_error": None,
                "schema_errors_json": "[\"Income missing\"]",
                "verifier_errors_json": "[]",
            },
        ]
    )

    selection_df = build_eval_case_selection_frame(query_df=query_df, parse_df=parse_df, scope="accepted")

    included = selection_df.loc[selection_df["eval_included"], "case_id"].tolist()
    reasons = dict(zip(selection_df["case_id"], selection_df["eval_drop_reason"]))
    assert included == ["c1"]
    assert reasons["c2"] == "not_exact_reconstruction"
    assert reasons["c3"] == "not_ready_for_runtime"

    eval_query_df = filter_eval_query_frame(query_df=query_df, selection_df=selection_df)
    assert eval_query_df["case_id"].tolist() == ["c1"]
    assert eval_query_df["fold_name"].tolist() == ["testfold_0_pred_0.csv"]
    assert eval_query_df["query_pos"].tolist() == [0]


def test_profiles_equal_handles_bank_float_and_int_contract() -> None:
    left = {
        "Income": 72.0,
        "Family": 1,
        "CCAvg": 4.8,
        "Education": 2,
        "Mortgage": 200.0,
        "SecuritiesAccount": 1,
        "CDAccount": 0,
        "Online": 1,
        "CreditCard": 0,
    }
    right = dict(left, Income=72, Mortgage=200, CCAvg=4.8000000001)
    wrong = dict(left, Online=0)

    assert profiles_equal(left, right) is True
    assert profiles_equal(left, wrong) is False


def test_field_semantic_mismatch_classifies_vietnamese_income_prefix_collision() -> None:
    records = build_bank_profile_v2_field_semantic_mismatch(
        original_profile={"Income": 13},
        reconstructed_profile={"Income": 53},
        candidate_profile=None,
        payload={
            "fields": {
                "Income": {
                    "value": 53,
                    "evidence_quote": "thu nhap nam muoi ba",
                }
            }
        },
        errors=[],
    )

    assert records == [
        {
            "field_name": "Income",
            "expected_value": 13,
            "observed_value": 53,
            "evidence_quote": "thu nhap nam muoi ba",
            "failure_family": "field_aware_number_prefix_collision",
            "errors": [],
        }
    ]


def test_pair_and_summary_capture_parse_fail_and_selected_profile_mismatch() -> None:
    query_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "fold_index": 0,
                "fold_name": "testfold_0_pred_0.csv",
                "query_pos": 0,
                "language_group": "G1",
                "language_label": "vi_clear",
                "benchmark_key": "vi",
            },
            {
                "case_id": "c2",
                "fold_index": 0,
                "fold_name": "testfold_0_pred_0.csv",
                "query_pos": 1,
                "language_group": "G2",
                "language_label": "vi_paraphrase",
                "benchmark_key": "vi",
            },
            {
                "case_id": "c3",
                "fold_index": 0,
                "fold_name": "testfold_0_pred_0.csv",
                "query_pos": 2,
                "language_group": "G3",
                "language_label": "en_clear",
                "benchmark_key": "en",
            },
        ]
    )
    parse_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "ready_for_runtime": True,
                "exact_reconstruction": True,
                "parser_status": "complete",
                "final_stage": "READY_FOR_RUNTIME",
                "prediction_before_reconstructed": 0,
            },
            {
                "case_id": "c2",
                "ready_for_runtime": False,
                "exact_reconstruction": False,
                "parser_status": "needs_clarification",
                "final_stage": "NEEDS_CLARIFICATION",
                "prediction_before_reconstructed": None,
            },
            {
                "case_id": "c3",
                "ready_for_runtime": True,
                "exact_reconstruction": True,
                "parser_status": "complete",
                "final_stage": "READY_FOR_RUNTIME",
                "prediction_before_reconstructed": 0,
            },
        ]
    )
    direct_df = pd.DataFrame(
        [
            {"case_id": "c1", "method": "UFCE2", "status": "released_cf", "selected_profile_signature": '{"Income":80}'},
            {"case_id": "c2", "method": "UFCE2", "status": "no_valid_cf", "selected_profile_signature": None},
            {"case_id": "c3", "method": "UFCE2", "status": "released_cf", "selected_profile_signature": '{"Income":90}'},
        ]
    )
    nl_df = pd.DataFrame(
        [
            {"case_id": "c1", "method": "UFCE2", "status": "released_cf", "selected_profile_signature": '{"Income":80}'},
            {"case_id": "c2", "method": "UFCE2", "status": "parse_or_handoff_error", "selected_profile_signature": None},
            {"case_id": "c3", "method": "UFCE2", "status": "released_cf", "selected_profile_signature": '{"Income":91}'},
        ]
    )

    pair_df = pair_variant_results(
        query_df=query_df,
        parse_df=parse_df,
        direct_df=direct_df,
        nl_df=nl_df,
    )

    reasons = dict(zip(pair_df["case_id"], pair_df["match_reason"]))
    assert reasons["c1"] == "released_cf_exact_match"
    assert reasons["c2"] == "parse_or_handoff_error"
    assert reasons["c3"] == "selected_profile_mismatch"

    summary_df = summarize_bridge_pairs(pair_df=pair_df, group_cols=["method"])
    ufce2 = summary_df.loc[summary_df["method"] == "UFCE2"].iloc[0]
    assert ufce2["pair_count"] == 3
    assert abs(float(ufce2["strict_outcome_match_rate"]) - (1 / 3)) < 1e-12
    assert abs(float(ufce2["exact_reconstruction_rate"]) - (2 / 3)) < 1e-12

    mismatch_df = pair_df.loc[(~pair_df["strict_outcome_match"]) | (~pair_df["nl_exact_reconstruction"])].copy()
    mismatch_reason_df = build_mismatch_reason_table(mismatch_df)
    assert set(mismatch_reason_df["match_reason"]) == {"parse_or_handoff_error", "selected_profile_mismatch"}
    assert all(column in pair_df.columns for column in ["direct_status", "nl_status", "selected_profile_match"])
    assert BANK_FEATURE_ORDER[0] == "Income"


def test_pair_variant_results_prefers_parse_contract_when_nl_runtime_repeats_columns() -> None:
    query_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "fold_index": 0,
                "fold_name": "testfold_0_pred_0.csv",
                "query_pos": 0,
                "language_group": "G1",
                "language_label": "vi_clear",
                "benchmark_key": "vi",
            }
        ]
    )
    parse_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "ready_for_runtime": True,
                "exact_reconstruction": True,
                "parser_status": "complete",
                "final_stage": "READY_FOR_RUNTIME",
                "prediction_before_reconstructed": 0,
            }
        ]
    )
    direct_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "method": "UFCE2",
                "status": "released_cf",
                "selected_profile_signature": '{"Income":80}',
            }
        ]
    )
    nl_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "method": "UFCE2",
                "status": "released_cf",
                "selected_profile_signature": '{"Income":80}',
                "parse_ready": 1,
                "exact_reconstruction": 1,
                "parser_status": "complete",
            }
        ]
    )

    pair_df = pair_variant_results(
        query_df=query_df,
        parse_df=parse_df,
        direct_df=direct_df,
        nl_df=nl_df,
    )

    row = pair_df.iloc[0]
    assert bool(row["nl_parse_ready"]) is True
    assert bool(row["nl_exact_reconstruction"]) is True
    assert row["nl_parser_status"] == "complete"
    assert bool(row["strict_outcome_match"]) is True
    assert bool(row["end_to_end_exact_match"]) is True


def test_parse_acceptance_metrics_report_error_rate_and_group_breakdown() -> None:
    rows = [
        {
            "case_id": "c1",
            "language_group": "G1",
            "ready_for_runtime": True,
            "exact_reconstruction": True,
            "schema_errors_json": "[]",
            "verifier_errors_json": "[]",
            "parser_api_error": None,
            "repair_api_error": None,
            "verifier_api_error": None,
        },
        {
            "case_id": "c2",
            "language_group": "G2",
            "ready_for_runtime": False,
            "exact_reconstruction": False,
            "schema_errors_json": "[\"Missing runtime fields after semantic parse: Income\"]",
            "verifier_errors_json": "[\"verifier reported: not entailed\"]",
            "parser_api_error": None,
            "repair_api_error": None,
            "verifier_api_error": None,
        },
    ]

    metrics = build_parse_acceptance_metrics(parse_rows=rows, total_cases=2)

    assert metrics["accepted_count"] == 1
    assert metrics["error_count"] == 1
    assert metrics["observed_error_count"] == 1
    assert metrics["missing_unrun_case_count"] == 0
    assert metrics["acceptance_error_count"] == 1
    assert metrics["observed_error_rate"] == 0.5
    assert metrics["error_rate"] == 0.5
    assert metrics["schema_error_count"] == 1
    assert metrics["verifier_reported_error_count"] == 1
    assert metrics["acceptance_passed"] is False
    assert metrics["by_language_group"]["G2"]["error_rate"] == 1.0
    assert "error_rate=0.5000" in format_parse_acceptance_failure(metrics)


def test_parse_live_quality_metrics_use_completed_cases_as_denominator() -> None:
    rows = [
        {
            "case_id": "c1",
            "language_group": "G1",
            "ready_for_runtime": True,
            "exact_reconstruction": True,
            "schema_errors_json": "[]",
            "verifier_errors_json": "[]",
            "parser_api_error": None,
            "repair_api_error": None,
            "verifier_api_error": None,
        },
        {
            "case_id": "c2",
            "language_group": "G2",
            "ready_for_runtime": True,
            "exact_reconstruction": False,
            "schema_errors_json": "[]",
            "verifier_errors_json": "[]",
            "parser_api_error": None,
            "repair_api_error": None,
            "verifier_api_error": None,
        },
    ]

    metrics = build_parse_live_quality_metrics(parse_rows=rows, total_cases=250)

    assert metrics["completed_cases"] == 2
    assert metrics["remaining_cases"] == 248
    assert metrics["error_count"] == 1
    assert metrics["error_rate"] == 0.5
    assert metrics["completion_rate"] == 2 / 250
    assert "fail=1/2 (50.0%)" in format_parse_live_quality_progress(metrics)


def test_extraction_retry_errors_catch_complete_payload_with_silent_omissions(sample_benchmark) -> None:
    parser_result = SimpleNamespace(api_error=None)
    payload = {
        "task": "extract_bank_profile_v2",
        "stage": "evidence_extraction",
        "status": "complete",
        "fields": {
            "Income": {"evidence_quote": "income ninety four", "confidence": 1.0, "reason": "word number"},
            "Family": {"evidence_quote": "family size one", "confidence": 1.0, "reason": "word number"},
            "Education": {"evidence_quote": "education level three", "confidence": 1.0, "reason": "word number"},
            "Mortgage": {"evidence_quote": "mortgage two hundred", "confidence": 1.0, "reason": "word number"},
            "CreditCard": {"evidence_quote": "credit card no", "confidence": 1.0, "reason": "explicit no"},
        },
        "missing_fields": [],
        "conflicts": [],
        "notes": [],
    }
    user_text = "income ninety four; family size one; education level three; mortgage two hundred; credit card no."

    errors = collect_bank_profile_v2_extraction_retry_errors(
        parser_result=parser_result,
        extraction_payload=payload,
        extraction_parse_errors=[],
        benchmark=sample_benchmark,
        user_text=user_text,
    )

    assert "complete extraction omitted field entries: CCAvg, SecuritiesAccount, CDAccount, Online" in errors
    assert "Missing extracted fields after evidence extraction: CCAvg, SecuritiesAccount, CDAccount, Online" in errors


def test_evidence_hints_use_exact_substrings_for_reordered_word_numbers() -> None:
    user_text = (
        "Current profile: family size one, education level three, income ninety four per year, "
        "mortgage two hundred twenty one, and average credit-card spending zero point eight each month. "
        "currently online banking no, currently credit card no, currently CD account no, "
        "currently securities account no."
    )

    hints = build_bank_profile_v2_evidence_hint_candidates(
        user_text=user_text,
        field_names=BANK_FEATURE_ORDER,
    )

    assert hints["Income"] == [
        {"evidence_quote": "income ninety four per year", "source": "alias_nearby_text", "start": 57, "end": 84}
    ]
    assert user_text[57:84] == "income ninety four per year"
    assert hints["CCAvg"][0]["evidence_quote"] == "average credit-card spending zero point eight each month"
    assert hints["SecuritiesAccount"][0]["evidence_quote"] == "currently securities account no"
    assert all(
        candidate["evidence_quote"] in user_text
        for candidates in hints.values()
        for candidate in candidates
    )


def test_evidence_hint_fallback_fills_only_unambiguous_missing_fields(sample_benchmark) -> None:
    user_text = (
        "Ho so hien tai cua toi la: gia dinh bon nguoi; hoc van muc hai; "
        "thu nhap nam hai muoi chin; the chap khong; "
        "chi tieu the trung binh khong phay bon moi thang. "
        "hien tai co online banking, hien tai khong co the tin dung, "
        "hien tai khong co CD account, hien tai khong co tai khoan chung khoan."
    )
    payload = {
        "task": "extract_bank_profile_v2",
        "stage": "evidence_extraction",
        "status": "partial",
        "fields": {
            "Family": {"evidence_quote": "gia dinh bon nguoi", "confidence": 1.0, "reason": "present"},
            "Education": {"evidence_quote": "hoc van muc hai", "confidence": 1.0, "reason": "present"},
            "Mortgage": {"evidence_quote": "the chap khong", "confidence": 1.0, "reason": "present"},
            "Online": {"evidence_quote": "hien tai co online banking", "confidence": 1.0, "reason": "present"},
            "CreditCard": {"evidence_quote": "hien tai khong co the tin dung", "confidence": 1.0, "reason": "present"},
        },
        "missing_fields": ["Income", "CCAvg", "SecuritiesAccount", "CDAccount"],
        "conflicts": [],
        "notes": [],
    }

    repaired, applied_fields, ambiguous_fields = apply_bank_profile_v2_evidence_hint_fallback(
        extraction_payload=payload,
        benchmark=sample_benchmark,
        user_text=user_text,
    )

    assert applied_fields == ["Income", "CCAvg", "SecuritiesAccount", "CDAccount"]
    assert ambiguous_fields == []
    assert repaired["status"] == "complete"
    assert repaired["missing_fields"] == []
    assert repaired["fields"]["Income"]["evidence_quote"] == "thu nhap nam hai muoi chin"
    assert repaired["fields"]["CCAvg"]["evidence_quote"] == "chi tieu the trung binh khong phay bon moi thang"
    assert "value" not in repaired["fields"]["Income"]


def test_evidence_hint_fallback_skips_ambiguous_candidates(sample_benchmark) -> None:
    user_text = "income eighty; income ninety."
    payload = {
        "task": "extract_bank_profile_v2",
        "stage": "evidence_extraction",
        "status": "partial",
        "fields": {},
        "missing_fields": BANK_FEATURE_ORDER,
        "conflicts": [],
        "notes": [],
    }

    repaired, applied_fields, ambiguous_fields = apply_bank_profile_v2_evidence_hint_fallback(
        extraction_payload=payload,
        benchmark=sample_benchmark,
        user_text=user_text,
    )

    assert applied_fields == []
    assert "Income" in ambiguous_fields
    assert "Income" not in repaired["fields"]


def test_parse_acceptance_metrics_apply_baseline_gates() -> None:
    pass_rows = _baseline_rows(error_plan={"G2": 16})
    pass_metrics = build_parse_acceptance_metrics(parse_rows=pass_rows, total_cases=250)

    assert pass_metrics["acceptance_passed"] is True
    assert pass_metrics["strict_acceptance_passed"] is False
    assert pass_metrics["baseline_acceptance_passed"] is True
    assert pass_metrics["baseline_failures"] == []

    overall_fail_metrics = build_parse_acceptance_metrics(
        parse_rows=_baseline_rows(error_plan={"G2": 17}),
        total_cases=250,
    )
    assert overall_fail_metrics["acceptance_passed"] is False
    assert any("overall parse errors 17" in failure for failure in overall_fail_metrics["baseline_failures"])

    group_fail_metrics = build_parse_acceptance_metrics(
        parse_rows=_baseline_rows(error_plan={"G4": 1}),
        total_cases=250,
    )
    assert group_fail_metrics["acceptance_passed"] is False
    assert group_fail_metrics["baseline_failures"] == ["G4 parse errors 1 exceed baseline group target 0"]


def test_parse_fail_fast_reasons_trigger_when_old_baseline_cannot_be_beaten() -> None:
    no_stop_metrics = build_parse_live_quality_metrics(
        parse_rows=_baseline_rows(error_plan={"G2": 16}),
        total_cases=250,
    )
    assert build_parse_fail_fast_reasons(live_quality=no_stop_metrics, total_cases=250) == []

    stop_metrics = build_parse_live_quality_metrics(
        parse_rows=_baseline_rows(error_plan={"G2": 17}),
        total_cases=250,
    )
    stop_reasons = build_parse_fail_fast_reasons(live_quality=stop_metrics, total_cases=250)

    assert any("live parse errors 17 reached the old baseline count 17" in reason for reason in stop_reasons)
    assert any("G2 live parse errors 17 exceed baseline group target 16" in reason for reason in stop_reasons)

    group_stop_metrics = build_parse_live_quality_metrics(
        parse_rows=_baseline_rows(error_plan={"G4": 1}),
        total_cases=250,
    )
    assert build_parse_fail_fast_reasons(live_quality=group_stop_metrics, total_cases=250) == [
        "fail-fast: G4 live parse errors 1 exceed baseline group target 0"
    ]


def test_parse_acceptance_metrics_record_fail_fast_reasons() -> None:
    rows = _baseline_rows(error_plan={"G2": 17})[:100]
    metrics = build_parse_acceptance_metrics(
        parse_rows=rows,
        total_cases=250,
        fail_fast_reasons=["fail-fast: live parse errors 17 reached the old baseline count 17"],
    )

    assert metrics["acceptance_passed"] is False
    assert metrics["baseline_acceptance_passed"] is False
    assert metrics["fail_fast_triggered"] is True
    assert metrics["fail_fast_error_limit"] == 16
    assert metrics["baseline_failure_count"] == 17
    assert metrics["observed_error_count"] == 17
    assert metrics["missing_unrun_case_count"] == 150
    assert metrics["acceptance_error_count"] == 167
    assert metrics["observed_error_rate"] == 0.17
    assert metrics["error_rate"] == 167 / 250
    assert "fail_fast_reasons=fail-fast" in format_parse_acceptance_failure(metrics)
    assert "observed_errors=17/100" in format_parse_acceptance_failure(metrics)


def test_eval_parse_stage_returns_non_zero_when_acceptance_fails(monkeypatch, tmp_path) -> None:
    query_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "fold_index": 0,
                "fold_name": "testfold_0_pred_0.csv",
                "query_pos": 0,
                "language_group": "G1",
                "language_label": "vi_clear",
                "benchmark_key": "vi",
            }
        ]
    )
    parse_df = pd.DataFrame([{"case_id": "c1"}])
    selection_df = pd.DataFrame(
        [
            {
                "case_id": "c1",
                "fold_index": 0,
                "fold_name": "testfold_0_pred_0.csv",
                "query_pos": 0,
                "language_group": "G1",
                "language_label": "vi_clear",
                "benchmark_key": "vi",
                "parse_eval_scope": "accepted",
                "eval_included": False,
                "eval_drop_reason": "not_exact_reconstruction",
                "ready_for_runtime": True,
                "exact_reconstruction": False,
                "parser_status": "complete",
                "final_stage": "READY_FOR_RUNTIME",
            }
        ]
    )
    acceptance = {
        "acceptance_passed": False,
        "accepted_count": 0,
        "ready_for_runtime_count": 1,
        "exact_reconstruction_count": 0,
        "error_rate": 1.0,
        "total_cases": 1,
    }
    args = SimpleNamespace(
        stage="eval_parse",
        out_dir=tmp_path,
        parse_source_dir=None,
        seed=42,
        cases_per_group=10,
        model_alias="qwen3-14b",
        api_base="http://localhost:1234",
        timeout_s=600.0,
        config_profile="final_freeze",
        bundle_mode="table7_author_public",
        mi_k="5",
        no_cf=10,
        contprox_metric="euclidean",
        parse_eval_scope="accepted",
        mi_feature_scope="configured_actionable",
        lmstudio_stream=False,
        no_progress=True,
    )

    class _FakeParser:
        def parse_args(self):
            return args

    monkeypatch.setattr(
        "scripts.final.part2.nl_bank_bridge_experiment.build_arg_parser",
        lambda: _FakeParser(),
    )
    monkeypatch.setattr(
        "scripts.final.part2.nl_bank_bridge_experiment.resolve_out_dir",
        lambda out_dir: tmp_path,
    )
    monkeypatch.setattr(
        "scripts.final.part2.nl_bank_bridge_experiment.build_runner_command",
        lambda _path: "python scripts/final/part2/nl_bank_bridge_experiment.py --stage eval_parse",
    )
    monkeypatch.setattr(
        "scripts.final.part2.nl_bank_bridge_experiment.load_query_frame",
        lambda _out_dir: query_df,
    )
    monkeypatch.setattr(
        "scripts.final.part2.nl_bank_bridge_experiment.load_parse_frame",
        lambda _out_dir: parse_df,
    )
    monkeypatch.setattr(
        "scripts.final.part2.nl_bank_bridge_experiment.validate_parse_checkpoint_matches_query",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.final.part2.nl_bank_bridge_experiment.evaluate_parse_acceptance",
        lambda **_kwargs: acceptance,
    )
    monkeypatch.setattr(
        "scripts.final.part2.nl_bank_bridge_experiment.build_eval_case_selection_frame",
        lambda **_kwargs: selection_df,
    )
    monkeypatch.setattr(
        "scripts.final.part2.nl_bank_bridge_experiment.write_eval_case_selection_artifact",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.final.part2.nl_bank_bridge_experiment.filter_eval_query_frame",
        lambda **_kwargs: query_df.iloc[0:0].copy(),
    )

    assert main() == 1

def _baseline_rows(*, error_plan: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    for group_name in ["G1", "G2", "G3", "G4", "G5"]:
        group_errors = int(error_plan.get(group_name, 0))
        for index in range(50):
            accepted = index >= group_errors
            rows.append(
                {
                    "case_id": f"{group_name}_{index}",
                    "language_group": group_name,
                    "ready_for_runtime": accepted,
                    "exact_reconstruction": accepted,
                    "schema_errors_json": "[]" if accepted else '["parse failed"]',
                    "verifier_errors_json": "[]",
                    "parser_api_error": None,
                    "repair_api_error": None,
                    "verifier_api_error": None,
                }
            )
    return rows
