from __future__ import annotations

import pandas as pd

from llm.src.runtime.model_registry import ModelRegistry
from llm.src.part2_eval.bank_synth_corpus import load_bank_source_analysis
from llm.src.runtime.orchestrator import RuntimeOrchestrator
from llm.src.runtime.policy_override import (
    apply_constraint_policy_override_to_runtime_request,
    derive_policy_override_from_constraint_spec,
    validate_and_apply_policy_override,
)
from llm.src.runtime.policy_registry import PolicyRegistry


def bank_policy():
    context = PolicyRegistry(ModelRegistry()).get_runtime_context("bank")
    return context.policy, list(context.bundle.feature_order)


def test_policy_override_builds_effective_policy_without_mutating_base():
    base_policy, feature_order = bank_policy()

    effective_policy, normalized, errors = validate_and_apply_policy_override(
        base_policy,
        {
            "f2change": ["Income", "CCAvg"],
            "uf": {"Income": 15},
            "step": {"CCAvg": 0.5},
        },
        feature_order=feature_order,
    )

    assert errors == []
    assert normalized == {
        "f2change": ["Income", "CCAvg"],
        "uf": {"Income": 15.0},
        "step": {"CCAvg": 0.5},
    }
    assert effective_policy.f2change == ["Income", "CCAvg"]
    assert effective_policy.uf["Income"] == 15.0
    assert effective_policy.step["CCAvg"] == 0.5
    assert base_policy.f2change == ["Income", "CCAvg", "Mortgage", "CDAccount", "Online"]
    assert base_policy.uf["Income"] == 40
    assert base_policy.step["CCAvg"] == 0.1


def test_policy_override_rejects_invalid_values_and_empty_effective_f2change():
    base_policy, feature_order = bank_policy()

    _effective_policy, _normalized, errors = validate_and_apply_policy_override(
        base_policy,
        {
            "remove_from_f2change": ["Income", "CCAvg", "Mortgage", "CDAccount", "Online"],
            "uf": {"Income": 0},
            "step": {"Online": 0.5},
        },
        feature_order=feature_order,
    )

    assert "policy_override results in empty effective f2change." in errors
    assert "policy_override.uf.Income must be > 0." in errors
    assert "policy_override.step.Online must be 1 for binary fields." in errors


def test_policy_override_rejects_unknown_fields():
    base_policy, feature_order = bank_policy()

    _effective_policy, _normalized, errors = validate_and_apply_policy_override(
        base_policy,
        {"f2change": ["BadField"]},
        feature_order=feature_order,
    )

    assert errors == ["policy_override.f2change contains unsupported fields: BadField"]


def test_policy_override_filters_effective_mi_pairs_for_generation_trace():
    registry = ModelRegistry()
    context = PolicyRegistry(registry).get_runtime_context("bank")
    profile = next(
        profile
        for profile in load_bank_source_analysis()["source_profiles"]
        if int(
            context.bundle.lr.predict(
                pd.DataFrame([profile], columns=context.bundle.feature_order)
            )[0]
        )
        != int(context.policy.desired_outcome)
    )
    result = RuntimeOrchestrator(model_registry=registry).handle(
        {
            "dataset": "bank",
            "profile": profile,
            "policy_override": {"remove_from_f2change": ["Mortgage"]},
        },
        include_debug_trace=True,
    )

    trace = result.debug_trace.to_dict()
    assert trace["effective_policy"]["f2change"] == ["Income", "CCAvg", "CDAccount", "Online"]
    assert trace["effective_mi_feature_pairs"] == [
        ["CCAvg", "Income"],
        ["CDAccount", "CCAvg"],
        ["CDAccount", "Income"],
    ]
    assert all("Mortgage" not in pair for pair in trace["effective_mi_feature_pairs"])


def test_constraint_spec_derives_generation_policy_override_for_blocked_fields_and_bounds():
    base_policy, feature_order = bank_policy()

    override = derive_policy_override_from_constraint_spec(
        base_policy=base_policy,
        feature_order=feature_order,
        profile={"Income": 72, "CCAvg": 4.8, "Mortgage": 200},
        constraint_spec={
            "disallowed_changes": ["Income"],
            "numeric_bounds": {"CCAvg": {"max": 4.85}},
        },
    )

    assert override is not None
    assert override["f2change"] == ["CCAvg", "Mortgage", "CDAccount", "Online"]
    assert abs(override["uf"]["CCAvg"] - 0.05) < 1e-9
    assert abs(override["step"]["CCAvg"] - 0.05) < 1e-9


def test_constraint_policy_override_merges_with_explicit_runtime_override():
    base_policy, feature_order = bank_policy()

    request = apply_constraint_policy_override_to_runtime_request(
        {
            "dataset": "bank",
            "profile": {"Income": 72, "CCAvg": 4.8, "Mortgage": 200},
            "constraint_spec": {"disallowed_changes": ["Mortgage"]},
            "policy_override": {"uf": {"Income": 20}},
        },
        base_policy=base_policy,
        feature_order=feature_order,
    )

    assert request["policy_override"] == {
        "f2change": ["Income", "CCAvg", "CDAccount", "Online"],
        "uf": {"Income": 20},
    }
