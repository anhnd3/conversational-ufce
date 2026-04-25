from __future__ import annotations

import pandas as pd

from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.runtime.ufce_request_builder import UFCERequestBuilder


def test_request_builder_extracts_positive_class_pool_in_bundle_order():
    registry = ModelRegistry()
    context = PolicyRegistry(registry).get_runtime_context("bank")
    builder = UFCERequestBuilder()
    canonical_profile = pd.DataFrame(
        [
            {
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
        ],
        columns=context.bundle.feature_order,
    )

    request = builder.build("bank", canonical_profile, context)
    expected_pool = (
        context.bundle.dataset_df.loc[
            context.bundle.dataset_df[context.bundle.label_col] == context.policy.desired_outcome
        ]
        .drop(columns=[context.bundle.label_col])
        .reindex(columns=context.bundle.feature_order)
        .reset_index(drop=True)
    )

    assert request.positive_class_pool.equals(expected_pool)
    assert request.mi_feature_pairs is not context.mi_feature_pairs
    assert request.mi_feature_pairs == context.mi_feature_pairs
