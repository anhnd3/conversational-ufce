from __future__ import annotations

from llm_eval.scoring import attach_stability_scores, score_prediction
from llm_eval.validation import validate_prediction


def test_score_prediction_exact_match_ignores_dict_key_order(sample_benchmark):
    case = sample_benchmark.case_map["B01"]
    predicted = {
        "status": "partial",
        "task": "extract_cf_request",
        "cf_request": {"Online": 1, "Income": 40},
        "missing_fields": [
            "CCAvg",
            "Family",
            "Education",
            "Mortgage",
            "CDAccount",
            "SecuritiesAccount",
            "CreditCard",
        ],
        "conflicts": [],
        "notes": [],
    }
    validation = validate_prediction(predicted, sample_benchmark)

    scoring = score_prediction(sample_benchmark, case, predicted, validation)

    assert scoring["exact_match"] is True
    assert scoring["missing_fields_correct"] is True


def test_score_prediction_treats_list_order_as_strict(sample_benchmark):
    case = sample_benchmark.case_map["B01"]
    predicted = {
        "task": "extract_cf_request",
        "status": "partial",
        "cf_request": {"Income": 40, "Online": 1},
        "missing_fields": [
            "Family",
            "CCAvg",
            "Education",
            "Mortgage",
            "CDAccount",
            "SecuritiesAccount",
            "CreditCard",
        ],
        "conflicts": [],
        "notes": [],
    }
    validation = validate_prediction(predicted, sample_benchmark)

    scoring = score_prediction(sample_benchmark, case, predicted, validation)

    assert scoring["exact_match"] is False
    assert scoring["missing_fields_correct"] is False


def test_score_prediction_computes_field_accuracy_and_hallucinations(sample_benchmark):
    case = sample_benchmark.case_map["B01"]
    predicted = {
        "task": "extract_cf_request",
        "status": "partial",
        "cf_request": {
            "Income": 40,
            "Online": 0,
            "CDAccount": 1,
        },
        "missing_fields": [
            "CCAvg",
            "Family",
            "Education",
            "Mortgage",
            "SecuritiesAccount",
            "CreditCard",
        ],
        "conflicts": [],
        "notes": [],
    }
    validation = validate_prediction(predicted, sample_benchmark)

    scoring = score_prediction(sample_benchmark, case, predicted, validation)

    assert scoring["field_accuracy"] == 7 / 9
    assert scoring["hallucination_count"] == 1


def test_attach_stability_scores_marks_case_stability():
    rows = [
        {"case_id": "A01", "parsed_json": {"a": 1}},
        {"case_id": "A01", "parsed_json": {"a": 1}},
        {"case_id": "B01", "parsed_json": {"a": 1}},
        {"case_id": "B01", "parsed_json": {"a": 2}},
    ]

    attach_stability_scores(rows)

    assert rows[0]["stability_case"] == 1.0
    assert rows[2]["stability_case"] == 0.0
