from __future__ import annotations

import pandas as pd

from llm.src.runtime.contracts import CanonicalRecourseRequest
from llm.src.runtime.datasets.base import DatasetPackage
from llm.src.runtime.types import RuntimeContext, UFCERequest
from llm.src.runtime.ufce_request_builder import UFCERequestBuilder


class UFCERequestMapper:
    def __init__(self, builder: UFCERequestBuilder | None = None) -> None:
        self.builder = builder or UFCERequestBuilder()

    def map(
        self,
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
        context: RuntimeContext,
    ) -> UFCERequest:
        del dataset
        canonical_profile = pd.DataFrame(
            [request.profile.values],
            columns=context.bundle.feature_order,
        )
        return self.builder.build(request.dataset_id, canonical_profile, context)
