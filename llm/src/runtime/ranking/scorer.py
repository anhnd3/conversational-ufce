from __future__ import annotations

from llm.src.runtime.contracts import CanonicalCandidate, CanonicalRecourseRequest
from llm.src.runtime.datasets.base import DatasetPackage


class DefaultCandidateRanker:
    def rank(
        self,
        candidates: list[CanonicalCandidate],
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
    ) -> list[CanonicalCandidate]:
        weights = dict(dataset.policy().get("ranking_weights") or {})
        prefer_fewer_changes = bool(request.soft_preferences.get("prefer_fewer_changes"))

        def score(candidate: CanonicalCandidate):
            rank_hint = tuple(candidate.backend_metadata.get("rank_hint") or [])
            backend_score = float(candidate.raw_backend_score or 0.0)
            sparsity_score = len(candidate.changed_features)
            if prefer_fewer_changes:
                return (
                    sparsity_score,
                    rank_hint,
                    -backend_score * float(weights.get("backend_score", 0.0)),
                    candidate.candidate_id,
                )
            return (
                rank_hint,
                sparsity_score * float(weights.get("sparsity", 1.0)),
                -backend_score * float(weights.get("backend_score", 0.0)),
                candidate.candidate_id,
            )

        return sorted(candidates, key=score)
