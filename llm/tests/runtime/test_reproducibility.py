from __future__ import annotations

from llm.src.runtime.policy_registry import BANK_FROZEN_MI_FEATURE_PAIRS, PolicyRegistry
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.reproducibility import build_deterministic_seed, sort_counterfactual_candidates
from llm.src.runtime.types import CounterfactualCandidate


def test_build_deterministic_seed_is_stable_for_same_profile():
    profile = {
        "Income": 100.0,
        "Family": 1,
        "CCAvg": 2.7,
        "Education": 2,
        "Mortgage": 0.0,
        "SecuritiesAccount": 0,
        "CDAccount": 0,
        "Online": 0,
        "CreditCard": 0,
    }
    feature_order = [
        "Income",
        "Family",
        "CCAvg",
        "Education",
        "Mortgage",
        "SecuritiesAccount",
        "CDAccount",
        "Online",
        "CreditCard",
    ]
    seed_one = build_deterministic_seed(
        dataset_name="bank",
        canonical_profile=profile,
        feature_order=feature_order,
        policy_version="bank_policy_v1",
    )
    seed_two = build_deterministic_seed(
        dataset_name="bank",
        canonical_profile=dict(profile),
        feature_order=feature_order,
        policy_version="bank_policy_v1",
    )
    assert seed_one == seed_two


def test_sort_counterfactual_candidates_orders_by_method_then_rank():
    candidates = [
        CounterfactualCandidate(method="tfexp", rank=1, profile={"Income": 120.0}, changed_features=["Income"]),
        CounterfactualCandidate(method="sfexp", rank=2, profile={"Income": 110.0}, changed_features=["Income"]),
        CounterfactualCandidate(method="sfexp", rank=1, profile={"Income": 105.0}, changed_features=["Income"]),
    ]

    ordered = sort_counterfactual_candidates(candidates=candidates, feature_order=["Income"])

    assert [item.method for item in ordered] == ["sfexp", "sfexp", "tfexp"]
    assert [item.rank for item in ordered] == [1, 2, 1]


def test_sort_counterfactual_candidates_can_prefer_fewer_changes_first():
    candidates = [
        CounterfactualCandidate(
            method="sfexp",
            rank=1,
            profile={"Income": 88.0, "CCAvg": 5.1},
            changed_features=["Income", "CCAvg"],
        ),
        CounterfactualCandidate(
            method="tfexp",
            rank=1,
            profile={"Income": 90.0, "CCAvg": 4.8},
            changed_features=["Income"],
        ),
    ]

    ordered = sort_counterfactual_candidates(
        candidates=candidates,
        feature_order=["Income", "CCAvg"],
        prefer_fewer_changes=True,
    )

    assert [item.method for item in ordered] == ["tfexp", "sfexp"]


def test_policy_registry_uses_frozen_bank_mi_pairs():
    context = PolicyRegistry(ModelRegistry()).get_runtime_context("bank")
    assert context.mi_feature_pairs == BANK_FROZEN_MI_FEATURE_PAIRS
