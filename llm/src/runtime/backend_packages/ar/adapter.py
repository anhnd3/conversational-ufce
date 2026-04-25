from __future__ import annotations

from llm.src.runtime.backend_packages.base import BackendExecutionResult
from llm.src.runtime.backend_packages.ufce.mapper import UFCERequestMapper
from llm.src.runtime.contracts import CanonicalCandidate, CanonicalRecourseRequest, legacy_candidate_to_canonical_candidate
from llm.src.runtime.datasets.base import DatasetPackage
from llm.src.runtime.types import RuntimeContext, RuntimeDebugTrace
from llm.src.runtime.backends import resolve_counterfactual_backend


class ARCanonicalBackend:
    backend_id = "ar"

    def __init__(self, *, mapper: UFCERequestMapper | None = None) -> None:
        self.mapper = mapper or UFCERequestMapper()
        self.legacy_backend = resolve_counterfactual_backend(self.backend_id)

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
        legacy_result = self.legacy_backend.generate(
            legacy_request,
            context,
            deterministic_seed_value=deterministic_seed_value,
            debug_trace=debug_trace,
        )
        candidates: list[CanonicalCandidate] = []
        for legacy_candidate in legacy_result.candidates:
            candidate = legacy_candidate_to_canonical_candidate(
                backend_id=self.backend_id,
                candidate=legacy_candidate,
                factual_profile=dict(request.profile.values),
            )
            metadata = dict(candidate.backend_metadata)
            metadata["rank_hint"] = [int(legacy_candidate.rank)]
            candidates.append(
                CanonicalCandidate(
                    backend_id=candidate.backend_id,
                    candidate_id=candidate.candidate_id,
                    changed_features=list(candidate.changed_features),
                    original_values=dict(candidate.original_values),
                    new_values=dict(candidate.new_values),
                    delta_summary=[dict(item) for item in candidate.delta_summary],
                    predicted_outcome=candidate.predicted_outcome,
                    raw_backend_score=candidate.raw_backend_score,
                    backend_metadata=metadata,
                )
            )
        return BackendExecutionResult(
            backend_id=self.backend_id,
            candidates=candidates,
            reason_codes=list(legacy_result.reason_codes),
            failure_metadata={"feasible": bool(legacy_result.feasible)},
        )
