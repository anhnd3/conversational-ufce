from __future__ import annotations

from llm.src.conversation.bank_completeness_guard import (
    apply_bank_boolean_completeness_guard,
    detect_explicit_bank_boolean_fields,
)


def test_detect_explicit_bank_boolean_fields_recognizes_yes_no_language():
    explicit = detect_explicit_bank_boolean_fields(
        "CD account yes, online no, I do not have a securities account, and I want a credit card."
    )

    assert explicit == ["SecuritiesAccount", "CDAccount", "Online", "CreditCard"]


def test_apply_bank_boolean_completeness_guard_downgrades_implicit_boolean_defaults():
    guard_result = apply_bank_boolean_completeness_guard(
        user_text="I want Income 140, Family 2, CCAvg 7.7376709303, Education 2, and Mortgage 32.",
        candidate={
            "task": "extract_cf_request",
            "status": "complete",
            "cf_request": {
                "Income": 140,
                "Family": 2,
                "CCAvg": 7.7376709303,
                "Education": 2,
                "Mortgage": 32,
                "SecuritiesAccount": 0,
                "CDAccount": 0,
                "Online": 0,
                "CreditCard": 0,
            },
            "missing_fields": [],
            "conflicts": [],
            "notes": [],
        },
    )

    assert guard_result.downgraded_fields == ["SecuritiesAccount", "CDAccount", "Online", "CreditCard"]
    assert guard_result.candidate["status"] == "partial"
    assert guard_result.candidate["cf_request"] == {
        "Income": 140,
        "Family": 2,
        "CCAvg": 7.7376709303,
        "Education": 2,
        "Mortgage": 32,
    }
    assert guard_result.candidate["missing_fields"] == [
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]


def test_apply_bank_boolean_completeness_guard_keeps_explicit_boolean_fields():
    guard_result = apply_bank_boolean_completeness_guard(
        user_text=(
            "Income 140, Family 2, CCAvg 7.7376709303, Education 2, Mortgage 32, "
            "SecuritiesAccount yes, CDAccount yes, Online yes, CreditCard no."
        ),
        candidate={
            "task": "extract_cf_request",
            "status": "complete",
            "cf_request": {
                "Income": 140,
                "Family": 2,
                "CCAvg": 7.7376709303,
                "Education": 2,
                "Mortgage": 32,
                "SecuritiesAccount": 1,
                "CDAccount": 1,
                "Online": 1,
                "CreditCard": 0,
            },
            "missing_fields": [],
            "conflicts": [],
            "notes": [],
        },
    )

    assert guard_result.downgraded_fields == []
    assert guard_result.candidate["status"] == "complete"


def test_detect_explicit_bank_boolean_fields_handles_coordinated_negation_and_own_language():
    explicit = detect_explicit_bank_boolean_fields(
        "I do not have a securities account or a CD account, but I do use online banking and own a bank credit card."
    )

    assert explicit == ["SecuritiesAccount", "CDAccount", "Online", "CreditCard"]


def test_apply_bank_boolean_completeness_guard_keeps_coordinated_negation_and_own_language():
    guard_result = apply_bank_boolean_completeness_guard(
        user_text=(
            "I have an annual income of $65,000, a family of 2, and spend about $1.5k monthly on my credit cards. "
            "My education is level 2 (graduate). I have no mortgage, I do not have a securities account or a CD account, "
            "but I do use online banking and own a bank credit card."
        ),
        candidate={
            "task": "extract_cf_request",
            "status": "complete",
            "cf_request": {
                "Income": 65000,
                "Family": 2,
                "CCAvg": 1.5,
                "Education": 2,
                "Mortgage": 0,
                "SecuritiesAccount": 0,
                "CDAccount": 0,
                "Online": 1,
                "CreditCard": 1,
            },
            "missing_fields": [],
            "conflicts": [],
            "notes": [],
        },
    )

    assert guard_result.downgraded_fields == []
    assert guard_result.candidate["status"] == "complete"
