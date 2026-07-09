from __future__ import annotations

import json

from llm.src.parser.bank_profile_parse_v2 import (
    BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
    BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
    BANK_PROFILE_PARSE_V2_TASK,
    apply_bank_profile_parse_v2_deterministic_guard,
    build_bank_profile_parse_v2_prompt,
    deterministic_bank_profile_parse_v2_value,
    parse_bank_profile_parse_v2_extraction_payload,
    parse_bank_profile_parse_v2_verifier_payload,
    run_bank_profile_parse_v2_quality,
    validate_bank_profile_parse_v2_extraction_payload,
)


def test_bank_profile_parse_v2_accepts_word_number_payload(sample_benchmark) -> None:
    user_text = (
        "annual income seventy two; family size two; average card spend one point five; "
        "education level three; mortgage zero; securities account no; CD account yes; "
        "online banking yes; credit card no."
    )
    payload = {
        "task": BANK_PROFILE_PARSE_V2_TASK,
        "stage": BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
        "status": "complete",
        "fields": {
            "Income": _field(72, "annual income seventy two"),
            "Family": _field(2, "family size two"),
            "CCAvg": _field(1.5, "average card spend one point five"),
            "Education": _field(3, "education level three"),
            "Mortgage": _field(0, "mortgage zero"),
            "SecuritiesAccount": _field(0, "securities account no"),
            "CDAccount": _field(1, "CD account yes"),
            "Online": _field(1, "online banking yes"),
            "CreditCard": _field(0, "credit card no"),
        },
        "missing_fields": [],
        "conflicts": [],
        "notes": [],
    }

    result = run_bank_profile_parse_v2_quality(
        message_text=json.dumps(payload),
        benchmark_spec=sample_benchmark,
        user_text=user_text,
        numeric_bound_fields=["Income", "CCAvg", "Mortgage"],
    )

    assert result.schema_validation.is_valid is True
    assert result.normalized.parsed_json["task"] == "extract_cf_request"
    assert result.normalized.parsed_json["cf_request"]["Income"] == 72
    assert result.normalized.parsed_json["cf_request"]["CCAvg"] == 1.5
    assert result.normalized.parsed_json["field_evidence"]["Income"]["source_text"] == "annual income seventy two"
    assert result.metadata["flags"]["deterministic_recovery_applied"] is False


def test_bank_profile_parse_v2_rejects_non_substring_evidence(sample_benchmark) -> None:
    payload = {
        "task": BANK_PROFILE_PARSE_V2_TASK,
        "stage": BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
        "status": "partial",
        "fields": {
            "Income": _field(72, "annual income seventy two"),
        },
        "missing_fields": ["CCAvg", "Family", "Education", "Mortgage", "CDAccount", "Online", "SecuritiesAccount", "CreditCard"],
        "conflicts": [],
        "notes": [],
    }

    result = run_bank_profile_parse_v2_quality(
        message_text=json.dumps(payload),
        benchmark_spec=sample_benchmark,
        user_text="annual income seventy one",
        numeric_bound_fields=["Income", "CCAvg", "Mortgage"],
    )

    assert result.schema_validation.is_valid is False
    assert "fields.Income.evidence_quote is not an exact substring of the input." in result.schema_validation.errors
    assert result.normalized.parsed_json is None


def test_bank_profile_parse_v2_rejects_non_integral_income_before_canonicalization(sample_benchmark) -> None:
    payload = {
        "task": BANK_PROFILE_PARSE_V2_TASK,
        "stage": BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
        "status": "partial",
        "fields": {
            "Income": _field(40.8, "Income 40.8"),
        },
        "missing_fields": ["CCAvg", "Family", "Education", "Mortgage", "CDAccount", "Online", "SecuritiesAccount", "CreditCard"],
        "conflicts": [],
        "notes": [],
    }

    result = run_bank_profile_parse_v2_quality(
        message_text=json.dumps(payload),
        benchmark_spec=sample_benchmark,
        user_text="Income 40.8",
        numeric_bound_fields=["Income", "CCAvg", "Mortgage"],
    )

    assert result.schema_validation.is_valid is False
    assert "fields.Income.value must be integer-valued, got 40.8." in result.schema_validation.errors
    assert result.normalized.parsed_json is None


def test_bank_profile_parse_v2_rejects_complete_extraction_with_omitted_fields(sample_benchmark) -> None:
    user_text = "Income 68; Family 2; Online yes; CreditCard no."
    payload = {
        "task": BANK_PROFILE_PARSE_V2_TASK,
        "stage": BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
        "status": "complete",
        "fields": {
            "Income": {"evidence_quote": "Income 68", "confidence": 1.0, "reason": "exact digits"},
            "Family": {"evidence_quote": "Family 2", "confidence": 1.0, "reason": "exact digits"},
            "Online": {"evidence_quote": "Online yes", "confidence": 1.0, "reason": "explicit yes"},
            "CreditCard": {"evidence_quote": "CreditCard no", "confidence": 1.0, "reason": "explicit no"},
        },
        "missing_fields": [],
        "conflicts": [],
        "notes": [],
    }

    errors = validate_bank_profile_parse_v2_extraction_payload(
        payload,
        benchmark_spec=sample_benchmark,
        user_text=user_text,
    )

    assert "complete extraction omitted field entries: CCAvg, Education, Mortgage, SecuritiesAccount, CDAccount" in errors


def test_bank_profile_parse_v2_deterministic_value_guard_handles_field_aware_word_numbers() -> None:
    assert deterministic_bank_profile_parse_v2_value("Income", "thu nhap nam muoi ba").value == 13
    assert deterministic_bank_profile_parse_v2_value("Income", "thu nhap nam nam muoi ba").value == 53
    assert (
        deterministic_bank_profile_parse_v2_value(
            "CCAvg",
            "chi tieu the trung binh khong phay chin moi thang",
        ).value
        == 0.9
    )
    assert deterministic_bank_profile_parse_v2_value("Mortgage", "mortgage two hundred twenty one").value == 221
    assert deterministic_bank_profile_parse_v2_value("CDAccount", "hien tai khong co CD account").value == 0


def test_bank_profile_parse_v2_deterministic_guard_corrects_income_prefix_collision() -> None:
    payload = {
        "task": BANK_PROFILE_PARSE_V2_TASK,
        "stage": BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
        "status": "partial",
        "fields": {
            "Income": _field(53, "thu nhap nam muoi ba"),
        },
        "missing_fields": ["CCAvg", "Family", "Education", "Mortgage", "CDAccount", "Online", "SecuritiesAccount", "CreditCard"],
        "conflicts": [],
        "notes": [],
    }

    corrected, corrections = apply_bank_profile_parse_v2_deterministic_guard(payload)

    assert corrected["fields"]["Income"]["value"] == 13
    assert corrections == [
        {
            "field_name": "Income",
            "old_value": 53,
            "new_value": 13,
            "evidence_quote": "thu nhap nam muoi ba",
            "reason": "field-aware integer word evidence parse",
            "failure_family": "field_aware_number_prefix_collision",
        }
    ]
    assert payload["fields"]["Income"]["value"] == 53


def test_bank_profile_parse_v2_quality_corrects_deterministic_disagreement_before_canonicalization(sample_benchmark) -> None:
    user_text = "thu nhap nam muoi ba"
    payload = {
        "task": BANK_PROFILE_PARSE_V2_TASK,
        "stage": BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
        "status": "partial",
        "fields": {
            "Income": _field(53, "thu nhap nam muoi ba"),
        },
        "missing_fields": ["CCAvg", "Family", "Education", "Mortgage", "CDAccount", "Online", "SecuritiesAccount", "CreditCard"],
        "conflicts": [],
        "notes": [],
    }

    result = run_bank_profile_parse_v2_quality(
        message_text=json.dumps(payload),
        benchmark_spec=sample_benchmark,
        user_text=user_text,
        numeric_bound_fields=["Income", "CCAvg", "Mortgage"],
    )

    assert result.schema_validation.is_valid is True
    assert result.normalized.parsed_json["cf_request"]["Income"] == 13
    assert result.v2_payload["fields"]["Income"]["value"] == 13
    assert result.metadata["flags"]["deterministic_recovery_applied"] is True
    assert result.metadata["deterministic_corrections"][0]["failure_family"] == "field_aware_number_prefix_collision"


def test_bank_profile_parse_v2_prompt_and_verifier_contract(sample_benchmark) -> None:
    extraction_prompt = build_bank_profile_parse_v2_prompt(
        sample_benchmark,
        user_text="Income 68; Family mot; CCAvg zero point five.",
        stage=BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
        retry_errors=["Missing extracted fields after evidence extraction: Mortgage"],
        previous_output='{"task":"extract_bank_profile_v2"}',
    )
    normalization_prompt = build_bank_profile_parse_v2_prompt(
        sample_benchmark,
        user_text="Income 68; Family mot; CCAvg zero point five.",
        stage=BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
        extracted_evidence={
            "Income": {"evidence_quote": "Income 68", "confidence": 1.0, "reason": "exact digits"},
        },
    )
    extraction_payload = {
        "task": BANK_PROFILE_PARSE_V2_TASK,
        "stage": BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
        "status": "partial",
        "fields": {
            "Income": {"evidence_quote": "Income 68", "confidence": 1.0, "reason": "exact digits"},
        },
        "missing_fields": ["CCAvg", "Family", "Education", "Mortgage", "CDAccount", "Online", "SecuritiesAccount", "CreditCard"],
        "conflicts": [],
        "notes": [],
    }
    verifier_payload = {
        "stage": "semantic_verification",
        "verdict": "PASS",
        "candidate": {
            "task": BANK_PROFILE_PARSE_V2_TASK,
            "stage": BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
            "status": "partial",
            "fields": {"Income": _field(68, "Income 68")},
            "missing_fields": ["CCAvg", "Family", "Education", "Mortgage", "CDAccount", "Online", "SecuritiesAccount", "CreditCard"],
            "conflicts": [],
            "notes": [],
        },
        "errors": [],
        "notes": [],
    }

    parsed_extraction, extraction_errors = parse_bank_profile_parse_v2_extraction_payload(json.dumps(extraction_payload))
    parsed_verifier, verifier_errors = parse_bank_profile_parse_v2_verifier_payload(json.dumps(verifier_payload))

    assert "EvidenceExtraction" in extraction_prompt
    assert "vi_tens_teen_exact_quote" in extraction_prompt
    assert "vi_reordered_word_numbers_all_fields" in extraction_prompt
    assert "en_reordered_word_numbers_all_fields" in extraction_prompt
    assert "Normalize the extracted evidence" in normalization_prompt
    assert parsed_extraction["stage"] == BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE
    assert extraction_errors == []
    assert parsed_verifier["verdict"] == "PASS"
    assert verifier_errors == []


def test_bank_profile_parse_v2_extraction_retry_prompt_includes_evidence_hints(sample_benchmark) -> None:
    prompt = build_bank_profile_parse_v2_prompt(
        sample_benchmark,
        user_text="income ninety four; currently CD account no.",
        stage=BANK_PROFILE_PARSE_V2_EXTRACTION_STAGE,
        retry_errors=["Missing extracted fields after evidence extraction: Income, CDAccount"],
        previous_output='{"task":"extract_bank_profile_v2"}',
        extraction_hints={
            "Income": [{"evidence_quote": "income ninety four", "source": "alias_nearby_text"}],
            "CDAccount": [{"evidence_quote": "currently CD account no", "source": "boolean_phrase"}],
        },
    )

    assert "evidence_hints" in prompt
    assert "income ninety four" in prompt
    assert "currently CD account no" in prompt
    assert "Do not invent values, paraphrase quotes, or require the input fields to appear in canonical order." in prompt


def test_bank_profile_parse_v2_verifier_payload_propagates_reported_errors(sample_benchmark) -> None:
    del sample_benchmark
    verifier_payload = {
        "stage": "semantic_verification",
        "verdict": "FAIL",
        "candidate": {
            "task": BANK_PROFILE_PARSE_V2_TASK,
            "stage": BANK_PROFILE_PARSE_V2_NORMALIZATION_STAGE,
            "status": "partial",
            "fields": {"Income": _field(68, "Income 68")},
            "missing_fields": ["CCAvg"],
            "conflicts": [],
            "notes": [],
        },
        "errors": ["Income evidence is not entailed."],
        "notes": [],
    }

    parsed, errors = parse_bank_profile_parse_v2_verifier_payload(json.dumps(verifier_payload))

    assert parsed["verdict"] == "FAIL"
    assert "verifier reported: Income evidence is not entailed." in errors


def _field(value, evidence_quote: str) -> dict:
    return {
        "value": value,
        "evidence_quote": evidence_quote,
        "confidence": 1.0,
        "normalization_note": "semantic extraction",
    }
