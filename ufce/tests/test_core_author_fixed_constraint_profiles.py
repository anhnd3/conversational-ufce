from __future__ import annotations

from pathlib import Path

import pandas as pd

from ufce.core_author_fixed.constraint_profiles import (
    AUTHOR_PUBLIC,
    DOMAIN_ACTIONABLE,
    resolve_constraint_profile,
)
from ufce.core_author_fixed.data_processing import (
    get_bank_user_constraints,
    get_bupa_user_constraints,
    get_grad_user_constraints,
    get_movie_user_constraints,
    get_wine_user_constraints,
)


ROOT = Path(__file__).resolve().parents[2]

STEP_CONFIG = {
    "bank": {"Income": 1, "Family": 1, "CCAvg": 0.1, "Education": 1, "Mortgage": 1, "SecuritiesAccount": 1, "CDAccount": 1, "Online": 1, "CreditCard": 1},
    "bupa": {"Mcv": 1, "Alkphos": 1, "Sgpt": 1, "Sgot": 1, "Gammagt": 1, "Drinks": 1},
    "grad": {"GRE Score": 1, "TOEFL Score": 1, "University Rating": 1, "SOP": 1, "LOR": 1, "CGPA": 0.1, "Research": 1},
    "wine": {"fixed acidity": 0.5, "volatile acidity": 0.1, "citric acid": 0.1, "residual sugar": 0.5, "free sulfur dioxide": 1.0, "total sulfur dioxide": 1.0, "density": 0.1, "pH": 0.5, "alcohol": 0.5},
    "movie": {"Production_expense": 3, "Num_multiplex": 3, "Multiplex_coverage": 0.2, "Movie_length": 5, "Lead_Actor_Rating": 1.0, "Lead_Actress_rating": 1.0, "Director_rating": 1.0, "Producer_rating": 1.0, "Genre": 1, "Collection": 500, "Budget": 3000},
}

GETTERS = {
    "bank": get_bank_user_constraints,
    "bupa": get_bupa_user_constraints,
    "grad": get_grad_user_constraints,
    "wine": get_wine_user_constraints,
    "movie": get_movie_user_constraints,
}


def _raw_constraints(dataset: str):
    df = pd.read_csv(ROOT / "ufce" / "data" / f"{dataset}.csv")
    features, _catf, _numf, uf, f2change, *_rest = GETTERS[dataset](df)
    return features, uf, f2change, STEP_CONFIG[dataset]


def test_author_public_profile_preserves_author_constraints() -> None:
    for dataset in GETTERS:
        features, uf, f2change, step = _raw_constraints(dataset)

        result = resolve_constraint_profile(dataset, features, uf, f2change, step, AUTHOR_PUBLIC)

        assert result.uf == uf
        assert result.f2change == f2change
        assert result.step == step
        assert result.not_main_table7 is False


def test_domain_actionable_profile_extends_author_f2change_and_has_steps() -> None:
    for dataset in GETTERS:
        features, uf, f2change, step = _raw_constraints(dataset)

        result = resolve_constraint_profile(dataset, features, uf, f2change, step, DOMAIN_ACTIONABLE)

        assert set(f2change).issubset(set(result.f2change))
        assert set(result.f2change).issubset(set(result.uf.keys()))
        assert set(result.f2change).issubset(set(result.step.keys()))
        assert result.not_main_table7 is True


def test_domain_actionable_profile_dataset_rules() -> None:
    expected = {
        "bank": {"Income", "CCAvg", "Mortgage", "CDAccount", "Online", "CreditCard", "SecuritiesAccount"},
        "bupa": {"Sgpt", "Sgot", "Gammagt", "Drinks"},
        "wine": {
            "fixed acidity",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "pH",
            "alcohol",
            "density",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "sulphates",
        },
        "movie": {
            "Production_expense",
            "Multiplex_coverage",
            "Num_multiplex",
            "Movie_length",
            "Lead_Actor_Rating",
            "Lead_Actress_rating",
            "Director_rating",
            "Producer_rating",
            "Genre",
            "Collection",
            "Budget",
            "Marketing_expense",
            "3D_available",
        },
    }
    for dataset, expected_f2change in expected.items():
        features, uf, f2change, step = _raw_constraints(dataset)
        result = resolve_constraint_profile(dataset, features, uf, f2change, step, DOMAIN_ACTIONABLE)

        assert set(result.f2change) == expected_f2change

    features, uf, f2change, step = _raw_constraints("grad")
    result = resolve_constraint_profile("grad", features, uf, f2change, step, DOMAIN_ACTIONABLE)
    assert result.uf == uf
    assert result.f2change == f2change
