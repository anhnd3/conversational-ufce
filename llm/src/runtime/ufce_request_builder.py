from __future__ import annotations

import pandas as pd

from llm.src.runtime.types import RuntimeContext, UFCERequest


class UFCERequestBuilder:
    def build(self, dataset_name: str, canonical_profile: pd.DataFrame, context: RuntimeContext) -> UFCERequest:
        positive_class_pool = (
            context.bundle.dataset_df.loc[
                context.bundle.dataset_df[context.bundle.label_col] == context.policy.desired_outcome
            ]
            .drop(columns=[context.bundle.label_col])
            .reindex(columns=context.bundle.feature_order)
            .reset_index(drop=True)
            .copy()
        )
        return UFCERequest(
            dataset=dataset_name,
            query_row=canonical_profile.loc[:, context.bundle.feature_order].copy(),
            feature_matrix=context.bundle.X.loc[:, context.bundle.feature_order].copy(),
            positive_class_pool=positive_class_pool,
            bundle=context.bundle,
            policy=context.policy,
            mi_feature_pairs=[list(pair) for pair in context.mi_feature_pairs],
        )
