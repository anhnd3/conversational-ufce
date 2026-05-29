from __future__ import annotations

from types import SimpleNamespace

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
from llm.src.runtime.datasets.bank.metadata import BANK_ALIASES, BANK_FEATURE_TYPES, BANK_REQUIRED_FIELD_ORDER


def _bank_policy():
    return SimpleNamespace(
        dataset_name="bank",
        feature_type_map=dict(BANK_FEATURE_TYPES),
        conversation_aliases={field: list(aliases) for field, aliases in BANK_ALIASES.items()},
    ), list(BANK_REQUIRED_FIELD_ORDER)


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


def test_extract_explicit_bank_values_handles_coordinated_positive_services():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input="I want online banking and a credit card, but no CD account or securities account.",
        policy=policy,
        target_fields=required_fields,
    )

    assert result.values["Online"] == 1
    assert result.values["CreditCard"] == 1
    assert result.values["CDAccount"] == 0
    assert result.values["SecuritiesAccount"] == 0
    assert result.conflicts == []


def test_extract_explicit_bank_values_handles_explicit_semantic_prose_profile():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "I make 65, have a household of 2, card spending around 1.5, education level 2, "
            "and no mortgage. No CD account or securities account, but I use online banking "
            "and own a bank credit card."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert result.values == {
        "Income": 65.0,
        "Family": 2,
        "CCAvg": 1.5,
        "Education": 2,
        "Mortgage": 0.0,
        "SecuritiesAccount": 0,
        "CDAccount": 0,
        "Online": 1,
        "CreditCard": 1,
    }
    assert result.conflicts == []


def test_extract_explicit_bank_values_handles_mortgage_target_prose():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "I earn 90, spend about 5.5 on my credit card each month, live in a family of 5, "
            "and my education category is 3. My mortgage target is 250. I want online banking "
            "and a credit card, but no CD account and no securities account."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert result.values["Mortgage"] == 250.0
    assert result.values["Online"] == 1
    assert result.values["CreditCard"] == 1


def test_extract_explicit_bank_values_handles_synonym_prose_profile():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "My salary is 58, my household size is 4, credit card spending is 2.1, "
            "education category is 1, and mortgage is 35. I have no securities account, "
            "without a certificate of deposit, use online banking, and do not own a bank credit card."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert result.values == {
        "Income": 58.0,
        "Family": 4,
        "CCAvg": 2.1,
        "Education": 1,
        "Mortgage": 35.0,
        "SecuritiesAccount": 0,
        "CDAccount": 0,
        "Online": 1,
        "CreditCard": 0,
    }
    assert result.conflicts == []


def test_extract_explicit_bank_values_does_not_invent_from_qualitative_numeric_language():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input="I need high income, low mortgage, and I use online banking.",
        policy=policy,
        target_fields=required_fields,
    )

    assert "Income" not in result.values
    assert "Mortgage" not in result.values
    assert result.values["Online"] == 1


def test_extract_explicit_bank_values_converts_supported_units_with_timeframe_guard():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "I have an annual income of $65,000, a household of 2, spend about $1.5k per month on cards, "
            "education level 2, and no mortgage. I use online banking."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert result.values["Income"] == 65.0
    assert result.values["CCAvg"] == 1.5
    assert result.values == {
        "Income": 65.0,
        "Family": 2,
        "CCAvg": 1.5,
        "Education": 2,
        "Mortgage": 0.0,
        "Online": 1,
    }


def test_extract_explicit_bank_values_keeps_timeframe_ambiguous_unit_out():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "My income is $65k, household 2, spend $1.5k on cards, education level 2, "
            "and no mortgage."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert "Income" not in result.values
    assert "CCAvg" not in result.values
    assert result.values["Family"] == 2
    assert result.values["Education"] == 2
    assert result.values["Mortgage"] == 0.0


def test_extract_explicit_bank_values_maps_education_taxonomy_labels():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input="I earn 48, family 2, CCAvg 1.2, undergraduate level, mortgage 35, and online banking yes.",
        policy=policy,
        target_fields=required_fields,
    )

    assert result.values["Education"] == 1


def test_extract_explicit_bank_values_converts_vnd_amounts():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "Thu nhap nam cua toi la 1.75 ty VND, gia dinh 2 nguoi, "
            "chi tieu the 37.5 trieu VND/thang, hoc van thac si, the chap 6.25 ty VND."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert result.values["Income"] == 70.0
    assert result.values["CCAvg"] == 1.5
    assert result.values["Mortgage"] == 250.0
    assert result.values["Education"] == 2
    assert result.field_evidence["Income"]["evidence_kind"] == "unit_converted"
    assert result.field_evidence["CCAvg"]["evidence_kind"] == "unit_converted"


def test_extract_explicit_bank_values_converts_vnd_amounts_with_diacritics_without_false_conflict():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "Thu nh\u1eadp n\u0103m c\u1ee7a t\u00f4i l\u00e0 1.75 t\u1ef7 VND, gia \u0111\u00ecnh 2 ng\u01b0\u1eddi, "
            "chi ti\u00eau th\u1ebb 37.5 tri\u1ec7u VND/th\u00e1ng, h\u1ecdc v\u1ea5n th\u1ea1c s\u0129, th\u1ebf ch\u1ea5p 6.25 t\u1ef7 VND."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert result.conflicts == []
    assert result.values["Income"] == 70.0
    assert result.values["CCAvg"] == 1.5
    assert result.values["Mortgage"] == 250.0
    assert result.field_evidence["Income"]["evidence_kind"] == "unit_converted"
    assert result.field_evidence["Income"]["language"] in {"vi", "mixed"}
    assert result.field_evidence["CCAvg"]["evidence_kind"] == "unit_converted"
    assert result.field_evidence["CCAvg"]["language"] in {"vi", "mixed"}


def test_extract_explicit_bank_values_converts_usd_thousand_wording_in_vietnamese():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "T\u00f4i c\u00f3 thu nh\u1eadp 70 ngh\u00ecn \u0111\u00f4 m\u1ed7i n\u0103m, gia \u0111\u00ecnh 3 ng\u01b0\u1eddi, "
            "chi ti\u00eau th\u1ebb 1.5 ngh\u00ecn \u0111\u00f4 m\u1ed7i th\u00e1ng, h\u1ecdc v\u1ea5n \u0111\u1ea1i h\u1ecdc, kh\u00f4ng c\u00f3 th\u1ebf ch\u1ea5p."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert result.values["Income"] == 70.0
    assert result.values["CCAvg"] == 1.5
    assert result.values["Education"] == 1
    assert result.values["Mortgage"] == 0.0


def test_extract_explicit_bank_values_detects_prose_numeric_conflict():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input="I earn 68, actually I earn 75, and I use online banking.",
        policy=policy,
        target_fields=required_fields,
    )

    assert "Income" not in result.values
    assert result.conflict_fields == ["Income"]


def test_extract_explicit_bank_values_detects_boolean_prose_conflict():
    policy, required_fields = _bank_policy()

    result = extract_explicit_bank_values(
        user_input=(
            "I use online banking, but also no online banking. I earn 70, family of 2, "
            "spend 2.0 on cards, education level 2, mortgage 0."
        ),
        policy=policy,
        target_fields=required_fields,
    )

    assert "Online" not in result.values
    assert result.conflict_fields == ["Online"]
    assert "Explicit field 'Online' has conflicting values in the same turn." in result.conflicts


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


def test_recover_dense_bank_profile_candidate_prefers_deterministic_value_on_parser_disagreement():
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
    assert result.candidate["status"] == "complete"
    assert result.candidate["cf_request"]["Online"] == 1
    assert result.field_provenance["Online"] == FIELD_PROVENANCE_DETERMINISTIC
    assert result.candidate["conflicts"] == []


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
