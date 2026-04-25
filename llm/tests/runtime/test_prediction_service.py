from __future__ import annotations

import pandas as pd

from llm.src.runtime.model_registry import ModelRegistry
from llm.src.runtime.policy_registry import PolicyRegistry
from llm.src.runtime.prediction_service import PredictionService


def test_prediction_service_matches_bundle_model():
    registry = ModelRegistry()
    context = PolicyRegistry(registry).get_runtime_context("bank")
    service = PredictionService()
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

    prediction = service.predict("bank", canonical_profile, context)

    assert prediction.predicted_label == int(context.bundle.lr.predict(canonical_profile)[0])
    assert prediction.predicted_proba == float(context.bundle.lr.predict_proba(canonical_profile)[0][1])
    assert prediction.feature_order_used == context.bundle.feature_order
