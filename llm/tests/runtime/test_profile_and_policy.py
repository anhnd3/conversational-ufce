from __future__ import annotations

import pandas as pd
import pytest

from llm.src.runtime.errors import RuntimeServiceError
from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import BANK_POLICY_VERSION, BANK_STEP, BANK_STEP_PROVENANCE, PolicyRegistry
from llm.src.runtime.profile_service import ProfileService
from llm.src.runtime.reason_codes import (
    INVALID_FIELD_TYPE,
    MISSING_REQUIRED_FEATURES,
    UNKNOWN_PROFILE_FIELDS,
)


@pytest.fixture
def bank_context():
    registry = ModelRegistry()
    policy_registry = PolicyRegistry(registry)
    return policy_registry.get_runtime_context("bank")


def test_bank_policy_matches_legacy_contract(bank_context):
    policy = bank_context.policy

    assert policy.f2change == ["Income", "CCAvg", "Mortgage", "CDAccount", "Online"]
    assert policy.uf["Income"] == 40
    assert policy.step == BANK_STEP
    assert policy.step_provenance == BANK_STEP_PROVENANCE
    assert policy.policy_version == BANK_POLICY_VERSION
    assert policy.expected_feature_order == bank_context.bundle.feature_order


def test_profile_service_canonicalizes_to_bundle_order(bank_context):
    service = ProfileService()
    raw_request = {
        "dataset": "bank",
        "profile": {
            "CreditCard": 0,
            "Online": 1,
            "CDAccount": 1,
            "SecuritiesAccount": 1,
            "Mortgage": 32,
            "Education": 2,
            "CCAvg": 7.7376709303,
            "Family": 2,
            "Income": 140,
        },
    }
    runtime_request = service.parse_request(raw_request, "bank")

    canonical = service.canonicalize(runtime_request, bank_context, raw_request)

    assert list(canonical.columns) == bank_context.bundle.feature_order
    assert canonical.iloc[0].to_dict()["Income"] == pytest.approx(140.0)


def test_profile_service_rejects_unknown_top_level_fields(bank_context):
    service = ProfileService()
    raw_request = {
        "dataset": "bank",
        "profile": {feature: 0 for feature in bank_context.bundle.feature_order},
        "unexpected": True,
    }
    raw_request["profile"]["Income"] = 100
    raw_request["profile"]["Family"] = 1
    raw_request["profile"]["CCAvg"] = 2.7
    raw_request["profile"]["Education"] = 2
    raw_request["profile"]["Mortgage"] = 0
    runtime_request = service.parse_request(raw_request, "bank")

    with pytest.raises(RuntimeServiceError) as exc_info:
        service.canonicalize(runtime_request, bank_context, raw_request)

    assert exc_info.value.reason_codes == (UNKNOWN_PROFILE_FIELDS,)


def test_profile_service_rejects_partial_profile(bank_context):
    service = ProfileService()
    raw_request = {
        "dataset": "bank",
        "profile": {
            "Income": 100,
            "Family": 1,
        },
    }
    runtime_request = service.parse_request(raw_request, "bank")

    with pytest.raises(RuntimeServiceError) as exc_info:
        service.canonicalize(runtime_request, bank_context, raw_request)

    assert exc_info.value.reason_codes == (MISSING_REQUIRED_FEATURES,)


def test_profile_service_rejects_bool_in_numeric_slot(bank_context):
    service = ProfileService()
    raw_request = {
        "dataset": "bank",
        "profile": {
            "Income": True,
            "Family": 1,
            "CCAvg": 2.7,
            "Education": 2,
            "Mortgage": 0,
            "SecuritiesAccount": 0,
            "CDAccount": 0,
            "Online": 0,
            "CreditCard": 0,
        },
    }
    runtime_request = service.parse_request(raw_request, "bank")

    with pytest.raises(RuntimeServiceError) as exc_info:
        service.canonicalize(runtime_request, bank_context, raw_request)

    assert exc_info.value.reason_codes == (INVALID_FIELD_TYPE,)
