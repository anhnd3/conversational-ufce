"""Constraint profiles for the theory-corrected UFCE author fork."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


AUTHOR_PUBLIC = "author_public"
DOMAIN_ACTIONABLE = "domain_actionable"
CONSTRAINT_PROFILES = (AUTHOR_PUBLIC, DOMAIN_ACTIONABLE)
DOMAIN_ACTIONABLE_CHANGED_DATASETS = ("bank", "bupa", "wine", "movie")


@dataclass(frozen=True)
class ConstraintProfileResult:
    uf: Dict[str, float]
    f2change: List[str]
    step: Dict[str, float]
    inactive_features: List[str]
    not_main_table7: bool


def resolve_constraint_profile(
    dataset: str,
    features: Sequence[str],
    uf: Dict[str, float],
    f2change: Sequence[str],
    step: Dict[str, float],
    profile: str = AUTHOR_PUBLIC,
) -> ConstraintProfileResult:
    """Return effective user-feedback/actionability settings for a run profile."""

    if profile not in CONSTRAINT_PROFILES:
        raise ValueError(f"Unsupported constraint profile: {profile}")

    if profile == AUTHOR_PUBLIC:
        effective_uf = dict(uf)
        effective_f2change = list(f2change)
        effective_step = dict(step)
    else:
        effective_uf, effective_f2change, effective_step = _domain_actionable(dataset, uf, step)

    missing_step = [feature for feature in effective_f2change if feature not in effective_step]
    if missing_step:
        raise ValueError(f"Missing step config for {dataset}/{profile}: {missing_step}")

    inactive = [feature for feature in features if feature not in set(effective_f2change)]
    return ConstraintProfileResult(
        uf=effective_uf,
        f2change=effective_f2change,
        step=effective_step,
        inactive_features=inactive,
        not_main_table7=profile != AUTHOR_PUBLIC,
    )


def _domain_actionable(
    dataset: str,
    author_uf: Dict[str, float],
    author_step: Dict[str, float],
) -> Tuple[Dict[str, float], List[str], Dict[str, float]]:
    uf = dict(author_uf)
    step = dict(author_step)

    if dataset == "grad":
        f2change = list(author_uf.keys())
    elif dataset == "bank":
        f2change = ["Income", "CCAvg", "Mortgage", "CDAccount", "Online", "CreditCard", "SecuritiesAccount"]
    elif dataset == "bupa":
        f2change = ["Sgpt", "Sgot", "Gammagt", "Drinks"]
    elif dataset == "wine":
        uf["sulphates"] = 0.2
        step["sulphates"] = 0.1
        f2change = [
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
        ]
    elif dataset == "movie":
        uf["Marketing_expense"] = 50
        uf["3D_available"] = 1
        step["Marketing_expense"] = 5
        step["3D_available"] = 1
        f2change = [
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
        ]
    else:
        raise ValueError(f"Unsupported dataset for domain_actionable profile: {dataset}")

    effective_uf = uf
    effective_step = {feature: step[feature] for feature in f2change}
    return effective_uf, f2change, effective_step
