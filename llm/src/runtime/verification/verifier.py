from __future__ import annotations

from dataclasses import dataclass

from llm.src.runtime.contracts import CanonicalCandidate, CanonicalRecourseRequest, VerificationResult
from llm.src.runtime.datasets.base import DatasetPackage
from llm.src.runtime.reason_codes import DUPLICATE_DOMINATED
from llm.src.runtime.types import RuntimeContext


@dataclass(frozen=True)
class VerifiedCandidateSet:
    valid_candidates: list[CanonicalCandidate]
    verification_results: list[VerificationResult]


class CompositeCandidateVerifier:
    def __init__(self, checks: list[object]) -> None:
        self._checks = list(checks)

    def verify(
        self,
        candidate: CanonicalCandidate,
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
        context: RuntimeContext,
    ) -> VerificationResult:
        reason_codes: list[str] = []
        evidence: dict[str, object] = {}
        for check in self._checks:
            result = check.verify(candidate, request, dataset, context)
            if result.reason_codes:
                reason_codes.extend(result.reason_codes)
            if result.evidence:
                evidence.update(result.evidence)
        return VerificationResult(
            is_valid=not reason_codes,
            reason_codes=reason_codes,
            evidence=evidence,
            candidate_id=candidate.candidate_id,
        )

    def verify_all(
        self,
        candidates: list[CanonicalCandidate],
        request: CanonicalRecourseRequest,
        dataset: DatasetPackage,
        context: RuntimeContext,
    ) -> VerifiedCandidateSet:
        valid_candidates: list[CanonicalCandidate] = []
        verification_results: list[VerificationResult] = []
        for candidate in candidates:
            result = self.verify(candidate, request, dataset, context)
            verification_results.append(result)
            if result.is_valid:
                valid_candidates.append(candidate)
        filtered_candidates, duplicate_records = filter_duplicate_candidates(valid_candidates)
        verification_results.extend(duplicate_records)
        filtered_ids = {candidate.candidate_id for candidate in filtered_candidates}
        return VerifiedCandidateSet(
            valid_candidates=[candidate for candidate in filtered_candidates if candidate.candidate_id in filtered_ids],
            verification_results=verification_results,
        )


def filter_duplicate_candidates(
    candidates: list[CanonicalCandidate],
) -> tuple[list[CanonicalCandidate], list[VerificationResult]]:
    kept: list[CanonicalCandidate] = []
    duplicate_records: list[VerificationResult] = []
    seen_profiles: dict[tuple[tuple[str, object], ...], CanonicalCandidate] = {}
    for candidate in candidates:
        profile_key = tuple(sorted(candidate.new_values.items()))
        if profile_key in seen_profiles:
            duplicate_records.append(
                VerificationResult(
                    is_valid=False,
                    candidate_id=candidate.candidate_id,
                    reason_codes=[DUPLICATE_DOMINATED],
                    evidence={"dominated_by": seen_profiles[profile_key].candidate_id},
                )
            )
            continue
        seen_profiles[profile_key] = candidate
        kept.append(candidate)
    return kept, duplicate_records
