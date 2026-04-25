from __future__ import annotations

from llm.src.runtime.backend_packages.base import BackendExecutionResult
from llm.src.runtime.backend_packages.ufce.mapper import UFCERequestMapper
from llm.src.runtime.backend_packages.ufce.normalizer import UFCELegacyNormalizer
from llm.src.runtime.contracts import CanonicalRecourseRequest
from llm.src.runtime.counterfactual_service import CounterfactualService
from llm.src.runtime.datasets.base import DatasetPackage
from llm.src.runtime.types import RuntimeContext, RuntimeDebugTrace


class UFCECanonicalBackend:
    backend_id = "ufce"

    def __init__(
        self,
        *,
        mapper: UFCERequestMapper | None = None,
        normalizer: UFCELegacyNormalizer | None = None,
        service: CounterfactualService | None = None,
    ) -> None:
        self.mapper = mapper or UFCERequestMapper()
        self.normalizer = normalizer or UFCELegacyNormalizer()
        self.service = service or CounterfactualService()

    def generate(
        self,
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
        context: RuntimeContext,
        *,
        deterministic_seed_value: int | None = None,
        debug_trace: RuntimeDebugTrace | None = None,
    ) -> BackendExecutionResult:
        legacy_request = self.mapper.map(request, dataset, context)
        legacy_result = self.service.generate(
            legacy_request,
            debug_trace=debug_trace,
            deterministic_seed_value=deterministic_seed_value,
        )
        candidates = self.normalizer.normalize(
            backend_id=self.backend_id,
            legacy_result=legacy_result,
            factual_profile=dict(request.profile.values),
        )
        return BackendExecutionResult(
            backend_id=self.backend_id,
            candidates=candidates,
            reason_codes=list(legacy_result.reason_codes),
            failure_metadata={"feasible": bool(legacy_result.feasible)},
        )
