from __future__ import annotations

import pandas as pd

from llm.src.runtime.types import PredictionResult, RuntimeContext


class PredictionService:
    def predict(self, dataset_name: str, canonical_profile: pd.DataFrame, context: RuntimeContext) -> PredictionResult:
        ordered_profile = canonical_profile.loc[:, context.bundle.feature_order]
        predicted_label = int(context.bundle.lr.predict(ordered_profile)[0])
        predicted_proba = float(context.bundle.lr.predict_proba(ordered_profile)[0][1])
        return PredictionResult(
            dataset=dataset_name,
            predicted_label=predicted_label,
            predicted_proba=predicted_proba,
            feature_order_used=list(context.bundle.feature_order),
        )
