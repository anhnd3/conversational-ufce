from __future__ import annotations

import pytest

from llm.src.conversation.bank_profile_extractor import (
    FIELD_PROVENANCE_CONFLICT,
    FIELD_PROVENANCE_DETERMINISTIC,
    FIELD_PROVENANCE_PARSER,
    FIELD_PROVENANCE_PARSER_AGREE,
    extract_explicit_bank_values,
    recover_explicit_labeled_bank_fields,
    recover_dense_bank_profile_candidate,
)
from llm.src.conversation.canonical_validator import BankCanonicalValidator


def _bank_policy():
    validator = BankCanonicalValidator()
    return validator.context.policy, list(validator.required_fields)


def _candidate(cf_request: dict[str, object], *, status: str = "partial") -> dict[str, object]:
    policy, required_fields = _bank_policy()
    del policy
    return {
        "task": "extract_cf_request",
        "status": status,
        "cf_request": dict(cf_request),
        "missing_fields": [field for field in required_fields if field not in cf_request],
        "conflicts": [],
        "notes": [],
    }


@pytest.mark.parametrize(
    ("user_input", "expected_values"),
    [
        (
            (
                "Income 68, Family 1, CCAvg 1.5, Education 2, Mortgage 0, "
                "SecuritiesAccount no, CDAccount no, Online no, CreditCard no."
            ),
            {
                "Income": 68.0,
                "Family": 1,
                "CCAvg": 1.5,
                "Education": 2,
                "Mortgage": 0.0,
                "SecuritiesAccount": 0,
                "CDAccount": 0,
                "Online": 0,
                "CreditCard": 0,
            },
        ),
        (
            (
                "Income: 83, Family=1, CCAvg 2.8, Education is 2, Mortgage 0, "
                "SecuritiesAccount 0, CDAccount 0, Online 1, CreditCard 1."
            ),
            {
                "Income": 83.0,
                "Family": 1,
                "CCAvg": 2.8,
                "Education": 2,
                "Mortgage": 0.0,
                "SecuritiesAccount": 0,
                "CDAccount": 0,
                "Online": 1,
                "CreditCard": 1,
            },
        ),
        (
            (
                "income 84, family 1, cc avg 1.3, education 3, mortgage 0, "
                "securities account yes, cd account no, online banking yes, credit card no."
            ),
            {
                "Income": 84.0,
                "Family": 1,
                "CCAvg": 1.3,
                "Education": 3,
                "Mortgage": 0.0,
                "SecuritiesAccount": 1,
                "CDAccount": 0,
                "Online": 1,
                "CreditCard": 0,
            },
        ),
    ],
)
def test_extract_explicit_bank_values_handles_dense_profile_formats(user_input, expected_values):
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=user_input,
        policy=policy,
        target_fields=required_fields,
    )

    assert result.values == expected_values
    assert result.conflicts == []


def test_extract_explicit_bank_values_rejects_malformed_values():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "Income 68, Family 1.5, CCAvg nope, Education 2, Mortgage 0, "
            "SecuritiesAccount no, CDAccount maybe, Online no, CreditCard no."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert "Family" not in result.values
    assert "CCAvg" not in result.values
    assert "CDAccount" not in result.values
    assert result.values["Online"] == 0


def test_extract_explicit_bank_values_detects_conflicting_duplicates():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "Income 68, Family 1, CCAvg 1.5, Education 2, Mortgage 0, "
            "SecuritiesAccount no, CDAccount no, Online yes, Online no, CreditCard no."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert "Online" not in result.values
    assert result.conflict_fields == ["Online"]
    assert result.conflicts == ["Explicit field 'Online' has conflicting values in the same turn."]


def test_extract_explicit_bank_values_handles_coordinated_negation_and_own_language():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "I do not have a securities account or a CD account, "
            "but I do use online banking and own a bank credit card."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert result.values["SecuritiesAccount"] == 0
    assert result.values["CDAccount"] == 0
    assert result.values["Online"] == 1
    assert result.values["CreditCard"] == 1
    assert result.conflicts == []


def test_recover_dense_bank_profile_candidate_marks_extractor_only_and_agreement_provenance():
    policy, required_fields = _bank_policy()
    candidate = _candidate(
        {
            "Income": 84,
            "Family": 1,
            "Education": 3,
            "Mortgage": 0,
            "SecuritiesAccount": 1,
            "CreditCard": 0,
        }
    )

    result = recover_dense_bank_profile_candidate(
        user_input=(
            "Income 84, Family 1, CCAvg 1.3, Education 3, Mortgage 0, "
            "SecuritiesAccount yes, CDAccount no, Online yes, CreditCard no."
        ),
        candidate=candidate,
        policy=policy,
        required_fields=required_fields,
    )

    assert result.candidate is not None
    assert result.candidate["status"] == "complete"
    assert result.candidate["cf_request"]["CCAvg"] == 1.3
    assert result.candidate["cf_request"]["CDAccount"] == 0
    assert result.candidate["cf_request"]["Online"] == 1
    assert result.field_provenance["CCAvg"] == FIELD_PROVENANCE_DETERMINISTIC
    assert result.field_provenance["CDAccount"] == FIELD_PROVENANCE_DETERMINISTIC
    assert result.field_provenance["Online"] == FIELD_PROVENANCE_DETERMINISTIC
    assert result.field_provenance["SecuritiesAccount"] == FIELD_PROVENANCE_PARSER_AGREE
    assert result.field_provenance["Income"] == FIELD_PROVENANCE_PARSER_AGREE


def test_recover_dense_bank_profile_candidate_keeps_parser_only_provenance_when_not_dense():
    policy, required_fields = _bank_policy()
    candidate = _candidate({"Income": 40, "Online": 1})

    result = recover_dense_bank_profile_candidate(
        user_input="Income 40 and Online yes.",
        candidate=candidate,
        policy=policy,
        required_fields=required_fields,
    )

    assert result.candidate == candidate
    assert result.field_provenance == {
        "Income": FIELD_PROVENANCE_PARSER,
        "Online": FIELD_PROVENANCE_PARSER,
    }


def test_recover_dense_bank_profile_candidate_marks_conflicts_without_overwrite():
    policy, required_fields = _bank_policy()
    candidate = _candidate(
        {
            "Income": 68,
            "Family": 1,
            "CCAvg": 1.5,
            "Education": 2,
            "Mortgage": 0,
            "SecuritiesAccount": 0,
            "CDAccount": 0,
            "Online": 0,
            "CreditCard": 0,
        },
        status="complete",
    )

    result = recover_dense_bank_profile_candidate(
        user_input=(
            "Income 68, Family 1, CCAvg 1.5, Education 2, Mortgage 0, "
            "SecuritiesAccount no, CDAccount no, Online yes, CreditCard no."
        ),
        candidate=candidate,
        policy=policy,
        required_fields=required_fields,
    )

    assert result.candidate is not None
    assert result.candidate["status"] == "conflict"
    assert result.candidate["cf_request"]["Online"] == 0
    assert result.field_provenance["Online"] == FIELD_PROVENANCE_CONFLICT
    assert (
        "Explicit field 'Online' disagrees between parser output and deterministic extraction."
        in result.candidate["conflicts"]
    )


@pytest.mark.parametrize(
    ("user_input", "expected_ccavg"),
    [
        (
            "Income 82, Family 3, CCAvg 1, Education 1, Mortgage 309. Do not change Income.",
            1.0,
        ),
        (
            "Income 81, Family 3, credit card avg 1.8, Education 2, Mortgage 0. Change at most one thing.",
            1.8,
        ),
    ],
)
def test_recover_explicit_labeled_bank_fields_recovers_ccavg_below_dense_threshold(user_input, expected_ccavg):
    policy, required_fields = _bank_policy()
    candidate = _candidate(
        {
            "Income": 82,
            "Family": 3,
            "Education": 1,
            "Mortgage": 309,
        }
    )

    result = recover_explicit_labeled_bank_fields(
        user_input=user_input,
        candidate=candidate,
        policy=policy,
        required_fields=required_fields,
        target_fields=("CCAvg",),
    )

    assert result.candidate is not None
    assert result.candidate["cf_request"]["CCAvg"] == expected_ccavg
    assert result.field_provenance["CCAvg"] == FIELD_PROVENANCE_DETERMINISTIC
    assert result.recovery_applied is True
    assert result.recovered_fields == ("CCAvg",)
    assert result.candidate["missing_fields"] == [
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]


def test_recover_explicit_labeled_bank_fields_does_not_recover_unlabeled_numeric_value():
    policy, required_fields = _bank_policy()
    candidate = _candidate(
        {
            "Income": 82,
            "Family": 3,
            "Education": 1,
            "Mortgage": 309,
        }
    )

    result = recover_explicit_labeled_bank_fields(
        user_input="Income 82, Family 3, value 1.8, Education 1, Mortgage 309.",
        candidate=candidate,
        policy=policy,
        required_fields=required_fields,
        target_fields=("CCAvg",),
    )

    assert result.candidate == candidate
    assert "CCAvg" not in result.field_provenance
    assert result.recovery_applied is False
    assert result.recovered_fields == ()


def test_recover_explicit_labeled_bank_fields_rejects_malformed_ccavg_value():
    policy, required_fields = _bank_policy()
    candidate = _candidate(
        {
            "Income": 82,
            "Family": 3,
            "Education": 1,
            "Mortgage": 309,
        }
    )

    result = recover_explicit_labeled_bank_fields(
        user_input="Income 82, Family 3, CCAvg nope, Education 1, Mortgage 309.",
        candidate=candidate,
        policy=policy,
        required_fields=required_fields,
        target_fields=("CCAvg",),
    )

    assert result.candidate == candidate
    assert result.recovery_applied is False
    assert result.recovered_fields == ()


def test_recover_explicit_labeled_bank_fields_marks_conflicting_ccavg_without_inventing_value():
    policy, required_fields = _bank_policy()
    candidate = _candidate(
        {
            "Income": 82,
            "Family": 3,
            "Education": 1,
            "Mortgage": 309,
        }
    )

    result = recover_explicit_labeled_bank_fields(
        user_input="Income 82, Family 3, CCAvg 1.8, CCAvg 2.1, Education 1, Mortgage 309.",
        candidate=candidate,
        policy=policy,
        required_fields=required_fields,
        target_fields=("CCAvg",),
    )

    assert result.candidate is not None
    assert "CCAvg" not in result.candidate["cf_request"]
    assert result.candidate["status"] == "conflict"
    assert result.field_provenance["CCAvg"] == FIELD_PROVENANCE_CONFLICT
    assert (
        "Explicit field 'CCAvg' has conflicting values in the same turn."
        in result.candidate["conflicts"]
    )
