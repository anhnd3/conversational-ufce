from __future__ import annotations

from llm.src.conversation.canonical_validator import BankCanonicalValidator
from llm.src.conversation.types import serialize_normalized_parse_payload
from llm.src.orchestration.parse_then_validate import parse_then_validate
from llm.src.parser.parser_quality import (
    ALIAS_KEY_NORMALIZED,
    AMBIGUOUS_CONSTRAINT_PHRASE,
    AMBIGUOUS_PREFERENCE_PHRASE,
    BINARY_STRING_COERCED,
    CONFLICTING_EXPLICIT_FIELD,
    CONSTRAINT_PHRASE_RECOVERED,
    CONSTRAINT_SPEC_ABSENT,
    CONSTRAINT_SPEC_RECOVERED,
    NUMERIC_STRING_COERCED,
    PREFERENCE_PHRASE_RECOVERED,
    PROFILE_FIELD_RECOVERED,
    run_parser_quality,
)


FULL_PROFILE_JSON = (
    '{"task":"extract_cf_request","status":"complete","cf_request":'
    '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
    '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
    '"missing_fields":[],"conflicts":[],"notes":[]}'
)
FULL_PROFILE_TEXT = (
    "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
    "SecuritiesAccount yes, CDAccount yes, Online yes, CreditCard no."
)


def test_run_parser_quality_normalizes_alias_keys_and_string_values(sample_benchmark):
    quality_result = run_parser_quality(
        message_text=(
            '{"task":"extract_cf_request","status":"partial","cf_request":'
            '{"income":"40.5","family":"2","online":"yes"},'
            '"missing_fields":["cc avg","securities account","credit card"],"conflicts":[],"notes":[]}'
        ),
        benchmark_spec=sample_benchmark,
    )

    assert quality_result.schema_validation.is_valid is True
    assert quality_result.normalized.parsed_json["cf_request"] == {
        "Income": 40.5,
        "Family": 2,
        "Online": 1,
    }
    assert quality_result.normalized.parsed_json["missing_fields"] == [
        "CCAvg",
        "Education",
        "Mortgage",
        "SecuritiesAccount",
        "CDAccount",
        "CreditCard",
    ]
    assert ALIAS_KEY_NORMALIZED in quality_result.metadata.reason_codes
    assert BINARY_STRING_COERCED in quality_result.metadata.reason_codes
    assert NUMERIC_STRING_COERCED in quality_result.metadata.reason_codes


def test_run_parser_quality_does_not_rewrite_existing_valid_constraint_spec(sample_benchmark):
    quality_result = run_parser_quality(
        message_text=(
            '{"task":"extract_cf_request","status":"complete","cf_request":'
            '{"Income":140,"Family":2,"CCAvg":7.7376709303,"Education":2,"Mortgage":32,'
            '"SecuritiesAccount":1,"CDAccount":1,"Online":1,"CreditCard":0},'
            '"constraint_spec":{"disallowed_changes":["Mortgage"]},"missing_fields":[],"conflicts":[],"notes":[]}'
        ),
        benchmark_spec=sample_benchmark,
        user_text=FULL_PROFILE_TEXT + " Do not change Income.",
    )

    assert quality_result.schema_validation.is_valid is True
    assert quality_result.normalized.parsed_json["constraint_spec"] == {
        "disallowed_changes": ["Mortgage"]
    }
    assert CONSTRAINT_SPEC_RECOVERED not in quality_result.metadata.reason_codes


def test_parse_then_validate_uses_user_text_to_recover_missing_constraint_spec(sample_benchmark):
    normalized, schema_validation = parse_then_validate(
        message_text=FULL_PROFILE_JSON,
        benchmark=sample_benchmark,
        user_text=FULL_PROFILE_TEXT + " Do not change Income.",
    )
    canonical_validation = BankCanonicalValidator().validate(
        candidate=normalized.parsed_json,
        schema_validation=schema_validation,
    )

    assert schema_validation.is_valid is True
    assert normalized.parsed_json["constraint_spec"] == {
        "disallowed_changes": ["Income"]
    }
    assert canonical_validation.ready_for_runtime is True


def test_run_parser_quality_recovers_deterministic_constraint_phrase_families(sample_benchmark):
    cases = [
        (
            FULL_PROFILE_TEXT + " Do not change Income.",
            {"disallowed_changes": ["Income"]},
            {CONSTRAINT_SPEC_RECOVERED, CONSTRAINT_PHRASE_RECOVERED},
        ),
        (
            FULL_PROFILE_TEXT + " Mortgage must stay under 100.",
            {"numeric_bounds": {"Mortgage": {"max": 100.0}}},
            {CONSTRAINT_SPEC_RECOVERED, CONSTRAINT_PHRASE_RECOVERED},
        ),
        (
            FULL_PROFILE_TEXT + " Change at most one thing.",
            {"max_changed_features": 1},
            {CONSTRAINT_SPEC_RECOVERED, CONSTRAINT_PHRASE_RECOVERED},
        ),
        (
            FULL_PROFILE_TEXT + " Prefer fewer changes.",
            {"prefer_fewer_changes": True},
            {CONSTRAINT_SPEC_RECOVERED, PREFERENCE_PHRASE_RECOVERED},
        ),
    ]

    for user_text, expected_constraint_spec, expected_reasons in cases:
        quality_result = run_parser_quality(
            message_text=FULL_PROFILE_JSON,
            benchmark_spec=sample_benchmark,
            user_text=user_text,
        )

        assert quality_result.schema_validation.is_valid is True
        assert quality_result.normalized.parsed_json["constraint_spec"] == expected_constraint_spec
        for reason_code in expected_reasons:
            assert reason_code in quality_result.metadata.reason_codes


def test_run_parser_quality_records_ambiguous_constraint_and_preference_phrases(sample_benchmark):
    quality_result = run_parser_quality(
        message_text=FULL_PROFILE_JSON,
        benchmark_spec=sample_benchmark,
        user_text=FULL_PROFILE_TEXT + " Do not change something. Prefer a simpler option.",
    )

    assert quality_result.schema_validation.is_valid is True
    assert "constraint_spec" not in quality_result.normalized.parsed_json
    assert AMBIGUOUS_CONSTRAINT_PHRASE in quality_result.metadata.reason_codes
    assert AMBIGUOUS_PREFERENCE_PHRASE in quality_result.metadata.reason_codes
    assert CONSTRAINT_SPEC_ABSENT in quality_result.metadata.reason_codes
    assert quality_result.metadata.constraint_extraction_absent is True


def test_run_parser_quality_does_not_silently_coerce_malformed_value_tokens(sample_benchmark):
    quality_result = run_parser_quality(
        message_text=(
            '{"task":"extract_cf_request","status":"partial","cf_request":{"Income":"forty","Online":"maybe"},'
            '"missing_fields":["Family","CCAvg","Education","Mortgage","SecuritiesAccount","CDAccount","CreditCard"],'
            '"conflicts":[],"notes":[]}'
        ),
        benchmark_spec=sample_benchmark,
    )

    assert quality_result.schema_validation.is_valid is False
    assert BINARY_STRING_COERCED not in quality_result.metadata.reason_codes
    assert NUMERIC_STRING_COERCED not in quality_result.metadata.reason_codes
    assert quality_result.normalized.parsed_json["cf_request"]["Income"] == "forty"
    assert quality_result.normalized.parsed_json["cf_request"]["Online"] == "maybe"


def test_run_parser_quality_emits_conflict_reason_without_overwriting_parser_value(sample_benchmark):
    quality_result = run_parser_quality(
        message_text=FULL_PROFILE_JSON,
        benchmark_spec=sample_benchmark,
        user_text=(
            "Income 140, Income 155, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
            "SecuritiesAccount yes, CDAccount yes, Online yes, CreditCard no."
        ),
    )

    assert quality_result.schema_validation.is_valid is True
    assert quality_result.normalized.parsed_json["cf_request"]["Income"] == 140
    assert quality_result.field_provenance["Income"] == "conflict"
    assert CONFLICTING_EXPLICIT_FIELD in quality_result.metadata.reason_codes


def test_run_parser_quality_recovers_subthreshold_explicit_ccavg(sample_benchmark):
    quality_result = run_parser_quality(
        message_text=(
            '{"task":"extract_cf_request","status":"complete","cf_request":'
            '{"Income":82,"Family":3,"Education":1,"Mortgage":309,'
            '"CDAccount":0,"Online":0,"SecuritiesAccount":0,"CreditCard":0},'
            '"missing_fields":["CCAvg"],"conflicts":[],"notes":[]}'
        ),
        benchmark_spec=sample_benchmark,
        user_text="Income 82, Family 3, CCAvg 1, Education 1, Mortgage 309. Do not change Income.",
    )

    assert quality_result.schema_validation.is_valid is True
    assert quality_result.normalized.parsed_json["cf_request"] == {
        "Income": 82,
        "Family": 3,
        "CCAvg": 1.0,
        "Education": 1,
        "Mortgage": 309,
    }
    assert quality_result.normalized.parsed_json["missing_fields"] == [
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]
    assert quality_result.field_provenance["CCAvg"] == "deterministic_extractor"
    assert PROFILE_FIELD_RECOVERED in quality_result.metadata.reason_codes
    assert quality_result.metadata.deterministic_recovery_applied is True


def test_run_parser_quality_does_not_overwrite_existing_parser_ccavg_with_subthreshold_fallback(sample_benchmark):
    quality_result = run_parser_quality(
        message_text=(
            '{"task":"extract_cf_request","status":"partial","cf_request":'
            '{"Income":82,"Family":3,"CCAvg":1.0,"Education":1,"Mortgage":309},'
            '"missing_fields":["SecuritiesAccount","CDAccount","Online","CreditCard"],'
            '"conflicts":[],"notes":[]}'
        ),
        benchmark_spec=sample_benchmark,
        user_text="Income 82, Family 3, CCAvg 1.8, Education 1, Mortgage 309.",
    )

    assert quality_result.schema_validation.is_valid is True
    assert quality_result.normalized.parsed_json["cf_request"]["CCAvg"] == 1.0
    assert quality_result.field_provenance["CCAvg"] == "parser"
    assert PROFILE_FIELD_RECOVERED not in quality_result.metadata.reason_codes


def test_serialize_normalized_parse_payload_uses_stable_parser_quality_structure():
    payload = serialize_normalized_parse_payload(
        {"task": "extract_cf_request", "status": "partial"},
        None,
        None,
    )

    assert payload["_parser_quality"] == {
        "reason_codes": [],
        "flags": {
            "deterministic_recovery_applied": False,
            "post_quality_schema_valid": False,
            "canonical_pass_after_quality": False,
            "repair_invoked": False,
            "still_failed_after_quality": False,
            "constraint_extraction_absent": False,
        },
        "semantic_buckets": {
            "profile_facts": {},
            "hard_constraints": {},
            "soft_preferences": {},
        },
    }
    assert serialize_normalized_parse_payload(None, None, None) is None
