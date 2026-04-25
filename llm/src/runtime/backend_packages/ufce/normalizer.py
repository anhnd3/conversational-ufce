from __future__ import annotations

from llm.src.runtime.contracts import CanonicalCandidate, legacy_candidate_to_canonical_candidate
from llm.src.runtime.types import CounterfactualResult


METHOD_PRIORITY = {"sfexp": 0, "dfexp": 1, "tfexp": 2}


class UFCELegacyNormalizer:
    def normalize(
        self,
        *,
        backend_id: str,
        legacy_result: CounterfactualResult,
        factual_profile: dict[str, object],
    ) -> list[CanonicalCandidate]:
        candidates: list[CanonicalCandidate] = []
        for legacy_candidate in legacy_result.candidates:
            candidate = legacy_candidate_to_canonical_candidate(
                backend_id=backend_id,
                candidate=legacy_candidate,
                factual_profile=factual_profile,
            )
            metadata = dict(candidate.backend_metadata)
            metadata["rank_hint"] = [
                METHOD_PRIORITY.get(str(legacy_candidate.method), 99),
                int(legacy_candidate.rank),
            ]
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
        return candidates
